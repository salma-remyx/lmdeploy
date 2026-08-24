# Copyright (c) OpenMMLab. All rights reserved.
"""Attention-aware per-channel bit allocation for KV cache quantization.

Uniform-precision KV cache quantization (int4 / fp8 / TurboQuant's fixed
4-bit-K + 2-bit-V) minimizes reconstruction error *in the cache* and treats
every channel as equally important. The transform-coding view of KV cache
compression (AATC, arXiv:2608.14191) shows that this is the wrong target: what
actually matters is how the quantization error propagates *through attention*.
Under a white-noise quantization model the expected attention-aware distortion
decomposes into additive key and value contributions that factor across tokens
and channels, so each channel carries its own distortion weight and bits should
be spent where that weight times the channel variance is largest.

This module implements that allocator for lmdeploy's KV cache thread:

- :class:`ChannelDistortionStats` accumulates the per-channel second moments
  the decomposition needs (K channels weighted by the query second moment,
  V channels weighted by the attention mass), reusing the streaming-observer
  pattern from :mod:`lmdeploy.lite.quantization.activation`.
- :func:`waterfill_bits` is the classical reverse water-filling solver of
  transform coding: given per-channel distortion weights and a total bit
  budget, it spends bits on high-weight channels and starves low-weight ones,
  instead of spending uniformly.
- :class:`KVBitAllocation` is the resulting plan, with the compression ratio
  and the residual attention-aware distortion it implies.

The allocator is an offline tool: it consumes calibration activations and
produces a per-layer, per-channel bit plan plus the whitening transform that
diagonalizes the channel covariance. Plugging variable-width channels into the
runtime bit-packers is a separate, kernel-level change and is out of scope
here.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    'ChannelDistortionStats',
    'KVBitAllocation',
    'waterfill_bits',
    'whitening_transform',
    'allocate_kv_bits',
]

# Quantization error of a b-bit uniform quantizer acting on a unit-variance
# signal, relative to the signal power: D = 2^-2b, the standard high-rate
# approximation used in transform coding.
_DISTORTION_PER_BIT = 0.25


@dataclass
class KVBitAllocation:
    """Per-channel bit plan for one attention layer's K and V cache.

    Args:
        k_bits: bits per key channel, shape ``(num_heads, k_head_dim)``.
        v_bits: bits per value channel, shape ``(num_heads, v_head_dim)``.
        k_distortion: residual attention-aware distortion carried by K.
        v_distortion: residual attention-aware distortion carried by V.
        fp16_bits: the uniform baseline (16) the ratio is taken against.
    """

    k_bits: Tensor
    v_bits: Tensor
    k_distortion: float
    v_distortion: float
    fp16_bits: int = 16

    def total_bits(self) -> int:
        """Total bits per cached token across K and V channels."""
        return int(self.k_bits.sum().item() + self.v_bits.sum().item())

    def uniform_bits(self) -> int:
        """Bits per token the uniform fp16 cache would have spent."""
        return int(self.k_bits.numel() * self.fp16_bits + self.v_bits.numel() * self.fp16_bits)

    def compression_ratio(self) -> float:
        """Compression vs. an fp16 cache holding the same channels."""
        uniform = self.uniform_bits()
        return uniform / self.total_bits() if self.total_bits() > 0 else float('inf')

    def summary(self) -> dict:
        """Flatten the plan into a loggable/report-friendly dict."""
        return {
            'total_bits_per_token': self.total_bits(),
            'uniform_fp16_bits_per_token': self.uniform_bits(),
            'compression_ratio': round(self.compression_ratio(), 4),
            'avg_k_bits': round(float(self.k_bits.float().mean()), 4),
            'avg_v_bits': round(float(self.v_bits.float().mean()), 4),
            'k_distortion': round(self.k_distortion, 8),
            'v_distortion': round(self.v_distortion, 8),
        }


class ChannelDistortionStats:
    """Streaming accumulator for the attention-aware distortion weights.

    The decomposition in AATC factorizes the expected attention distortion
    across tokens and channels, so only per-channel second moments are needed
    and they can be accumulated over calibration batches without ever
    materializing the full cache:

    - For K, the error in channel ``d`` is seen by attention through
      ``q_d``, so its weight is the query second moment ``E[q_d^2]``
      (averaged over queries and heads that read this channel).
    - For V, the error lands on the output scaled by the attention weight of
      the token it came from, so its weight is the attention mass
      ``E[a_t]`` of that token (non-negative, sums to one over the context).

    Args:
        num_heads: number of KV heads sharing these channels.
        k_head_dim: per-head key channel count.
        v_head_dim: per-head value channel count.
    """

    def __init__(self, num_heads: int, k_head_dim: int, v_head_dim: int) -> None:
        self.num_heads = num_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim

        self._k_sq = torch.zeros(num_heads, k_head_dim, dtype=torch.float64)
        self._v_sq = torch.zeros(num_heads, v_head_dim, dtype=torch.float64)
        self._q_sq = torch.zeros(num_heads, k_head_dim, dtype=torch.float64)
        self._attn_mass = torch.zeros(num_heads, dtype=torch.float64)
        self._tokens = 0

    def observe(self, k: Tensor, v: Tensor, q: Tensor = None, attn_mass: Tensor = None) -> None:
        """Accumulate one calibration batch.

        Args:
            k: key states ``(tokens, num_heads, k_head_dim)``.
            v: value states ``(tokens, num_heads, v_head_dim)``.
            q: query states ``(tokens, num_heads_q, k_head_dim)``. Optional;
                when omitted the K weights default to uniform, which
                reproduces plain (attention-blind) transform coding.
            attn_mass: per-head attention mass over the batch, shape
                ``(num_heads,)`` or a scalar broadcast over heads. Optional;
                defaults to uniform, i.e. every token weighed equally.
        """
        k = self._as_heads(k, self.k_head_dim)
        v = self._as_heads(v, self.v_head_dim)

        self._k_sq += k.double().pow(2).sum(dim=0)
        self._v_sq += v.double().pow(2).sum(dim=0)

        if q is not None:
            q = self._as_heads(q, self.k_head_dim)
            num_q_heads = q.shape[-2]
            group = num_q_heads // self.num_heads
            q = q.reshape(*q.shape[:-2], self.num_heads, group, q.shape[-1])
            self._q_sq += q.double().pow(2).sum(dim=(0, -2))
        if attn_mass is not None:
            mass = torch.as_tensor(attn_mass, dtype=torch.float64)
            if mass.ndim == 0:
                mass = mass.expand(self.num_heads)
            self._attn_mass += mass

        self._tokens += k.shape[0]

    @staticmethod
    def _as_heads(x: Tensor, head_dim: int) -> Tensor:
        """Coerce ``(..., heads, head_dim)`` states to ``(tokens, heads, dim)``."""
        if x.ndim == 2:  # (tokens, heads*dim) -> (tokens, heads, dim)
            if x.shape[-1] % head_dim != 0:
                raise ValueError(f'last dim {x.shape[-1]} not divisible by head_dim {head_dim}')
            return x.reshape(x.shape[0], -1, head_dim)
        if x.shape[-1] != head_dim:
            raise ValueError(f'expected last dim {head_dim}, got {tuple(x.shape)}')
        return x.reshape(-1, x.shape[-2], head_dim)

    @property
    def num_tokens(self) -> int:
        """Number of calibration tokens observed."""
        return self._tokens

    def k_weights(self) -> Tensor:
        """Attention-aware distortion weight per K channel, ``(heads, dim)``."""
        return self._normalize(self._q_sq)

    def v_weights(self) -> Tensor:
        """Attention-aware distortion weight per V channel, ``(heads, dim)``.

        The attention mass is a per-head quantity; it weighs every channel of
        the head it belongs to, so it is broadcast over the head dimension.
        """
        weights = self._normalize(self._attn_mass)
        return weights.unsqueeze(-1).expand(self.num_heads, self.v_head_dim)

    def k_variance(self) -> Tensor:
        """Per-K-channel variance ``E[k_d^2]``, ``(heads, dim)``."""
        return self._mean(self._k_sq)

    def v_variance(self) -> Tensor:
        """Per-V-channel variance ``E[v_d^2]``, ``(heads, dim)``."""
        return self._mean(self._v_sq)

    def _normalize(self, weights: Tensor) -> Tensor:
        """Fall back to uniform weights when a signal was never observed."""
        if weights.numel() == 0:
            return torch.ones_like(self._k_sq[:, :1].expand_as(self._k_sq))
        total = weights.sum()
        if total <= 0:
            return torch.full_like(weights, 1.0 / weights.numel())
        return weights / total

    def _mean(self, sq: Tensor) -> Tensor:
        if self._tokens == 0:
            return torch.ones_like(sq)
        return sq / self._tokens


def whitening_transform(x: Tensor) -> tuple[Tensor, Tensor]:
    """Whitening (PCA) basis and eigenvalues of a channel covariance.

    Transform coding whitens before quantizing so that the per-channel
    distortion weights become the eigenvalues themselves and reverse
    water-filling applies directly. The basis is orthonormal, so the runtime
    inverse is its transpose -- the same contract lmdeploy's TurboQuant
    Hadamard rotation already satisfies.

    Args:
        x: activations ``(tokens, channels)`` in float.

    Returns:
        Tuple of ``(basis, eigenvalues)`` with basis sorted by descending
        eigenvalue, so low-energy channels sit at the tail where water-filling
        starves them.
    """
    if x.ndim != 2:
        raise ValueError(f'expected 2-D (tokens, channels), got {tuple(x.shape)}')
    cov = x.double().T @ x.double() / max(x.shape[0], 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    return evecs[:, order].to(x.dtype), torch.clamp(evals[order], min=0).to(x.dtype)


def waterfill_bits(weights: Tensor, total_bits: int, max_bits: int = 8, min_bits: int = 2) -> Tensor:
    """Allocate ``total_bits`` across channels by reverse water-filling.

    Classical reverse water-filling over distortion weights ``w``: channels
    with weight above the water level get bits, channels below it are dropped
    to ``min_bits``. Concretely, for a uniform quantizer the per-channel
    distortion is ``w_d * 2^-2b_d``; minimizing the sum subject to a total bit
    budget gives ``b_d = log2(w_d / theta) / 2`` for a single water level
    ``theta``, clipped to ``[min_bits, max_bits]``. Bits are then rounded to
    integers and the budget re-balanced so the total matches exactly.

    Args:
        weights: non-negative distortion weight per channel, any shape.
        total_bits: exact number of bits to distribute.
        max_bits: per-channel bit ceiling.
        min_bits: per-channel bit floor.

    Returns:
        Integer bit allocation with the same shape as ``weights``, summing to
        ``total_bits`` when the budget is attainable.
    """
    flat = torch.as_tensor(weights, dtype=torch.float64).flatten()
    if torch.any(flat < 0):
        raise ValueError('distortion weights must be non-negative')
    if min_bits > max_bits:
        raise ValueError(f'min_bits ({min_bits}) must not exceed max_bits ({max_bits})')

    num_channels = flat.numel()
    capacity = num_channels * max_bits
    if total_bits > capacity:
        raise ValueError(f'budget {total_bits} exceeds capacity {capacity}')
    if total_bits < num_channels * min_bits:
        raise ValueError(f'budget {total_bits} below floor {num_channels * min_bits}')

    # Solve for the water level by bisection on the clipped bit sum.
    lo = 0.0
    hi = float(flat.max()) if flat.max() > 0 else 1.0

    def _alloc(theta: float) -> Tensor:
        ratio = flat.clamp(min=1e-30) / max(theta, 1e-30)
        bits = 0.5 * torch.log2(ratio)
        return bits.clamp(min=min_bits, max=max_bits)

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _alloc(mid).sum() > total_bits:
            lo = mid  # level too low -> too many bits
        else:
            hi = mid  # level too high -> too few bits

    bits = _alloc(hi)
    alloc = torch.round(bits).to(torch.int64).clamp(min=min_bits, max=max_bits)

    # Repair rounding drift so the plan spends the budget exactly.
    slack = int(total_bits - alloc.sum().item())
    order = torch.argsort((alloc.to(torch.float64) - bits), descending=True)
    idx = 0
    while slack != 0 and idx < num_channels * (max_bits - min_bits + 1):
        i = int(order[idx % num_channels])
        step = 1 if slack > 0 else -1
        candidate = int(alloc[i]) + step
        if min_bits <= candidate <= max_bits:
            alloc[i] = candidate
            slack -= step
        idx += 1
    return alloc.reshape(weights.shape)


def allocate_kv_bits(stats: ChannelDistortionStats,
                     k_budget: int,
                     v_budget: int,
                     max_bits: int = 8,
                     min_bits: int = 2) -> KVBitAllocation:
    """Allocate bits for one attention layer from accumulated statistics.

    Runs reverse water-filling twice, once over the key distortion weights
    ``E[q_d^2] * E[k_d^2]`` and once over the value distortion weights
    ``E[a] * E[v_d^2]``, matching the additive key/value split of the
    attention-aware distortion decomposition.

    Args:
        stats: accumulated calibration statistics for this layer.
        k_budget: total key bits per token.
        v_budget: total value bits per token.
        max_bits: per-channel bit ceiling.
        min_bits: per-channel bit floor.

    Returns:
        The :class:`KVBitAllocation` plan for this layer.
    """
    k_weights = (stats.k_weights() * stats.k_variance()).flatten()
    v_weights = (stats.v_weights() * stats.v_variance()).flatten()

    k_bits = waterfill_bits(k_weights, k_budget, max_bits=max_bits, min_bits=min_bits)
    v_bits = waterfill_bits(v_weights, v_budget, max_bits=max_bits, min_bits=min_bits)

    k_distortion = float((k_weights * _DISTORTION_PER_BIT**k_bits.to(torch.float64)).sum())
    v_distortion = float((v_weights * _DISTORTION_PER_BIT**v_bits.to(torch.float64)).sum())
    return KVBitAllocation(
        k_bits=k_bits.reshape(stats.num_heads, stats.k_head_dim),
        v_bits=v_bits.reshape(stats.num_heads, stats.v_head_dim),
        k_distortion=k_distortion,
        v_distortion=v_distortion,
    )
