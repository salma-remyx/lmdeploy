# Copyright (c) OpenMMLab. All rights reserved.
"""Factored DoRA norm.

DoRA (Weight-Decomposed Low-Rank Adaptation) re-parameterizes a LoRA update as
``W' = m * (W + s*B^T A) / ||W + s*B^T A||_c``, where ``m`` is a learned
per-output-row magnitude, the norm is taken over the columns of the adapted
weight, and the packed adapter matrices follow lmdeploy's ``[rank, dim]``
layout. The naive way to obtain that norm materializes the dense
``[out_features, in_features]`` product ``s*B^T A`` first, which at high rank
and wide inputs is the dominant transient allocation of the adapter path.

The helpers below compute the same row norms without ever forming ``B^T A``,
following the factored formulation of "Scaling DoRA: High-Rank Adaptation via
Factored Norms and Fused Kernels" (arXiv:2603.22276). For a single adapter

    ||W + s*B^T A||_c^2 = sum_k (W + s*B^T A)_ik^2
                        = ||W_i||^2 + 2*s*(B^T A W^T)_ii + s^2*||(B^T A)_i||^2

so only two small quantities are needed: the Gram diagonal ``diag(B^T (A W^T))``
of shape ``[r, out] -> [out]`` and ``||(B^T A)_i||^2``, obtained through the
``[r, r]`` gram matrix ``A A^T`` rather than the dense product. Both are
``O(r * (out + in))`` in memory instead of ``O(out * in)``.

Only the magnitude side of DoRA is covered here: the normalized update itself
is applied at serving time by folding ``m / ||W + s*B@A||_c`` into the existing
LoRA scalings, so no extra work is added to the hot decoding path.
"""

import torch

__all__ = ["factored_dora_norm", "pack_dora_weight_norm"]


def _adapter_view(tensor: torch.Tensor, rank: int, adapter_id: int, rank_offsets: torch.Tensor):
    """Slice the rows of a packed multi-adapter tensor belonging to one adapter."""
    start = rank_offsets[adapter_id].item()
    return tensor[start : start + rank]


@torch.inference_mode()
def factored_dora_norm(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    rank: int,
    adapter_id: int,
    rank_offsets: torch.Tensor,
    reduce_group=None,
):
    """Compute per-column norms of ``weight + scaling * lora_b @ lora_a``.

    Args:
        weight: base weight of the linear layer, ``[out_features, in_features]``.
        lora_a: packed A matrices, ``[sum_rank, in_features]``.
        lora_b: packed B matrices, ``[sum_rank, out_features]``.
        scaling: the LoRA scaling of this adapter.
        rank: the rank of this adapter.
        adapter_id: index of the adapter inside the packed tensors.
        rank_offsets: cumulative rank offsets of the packed tensors.
        reduce_group: process group over which the partial sums of the squared
            norm are reduced. Needed when ``weight`` is row-sharded across
            tensor-parallel ranks, since each rank then only sees a slice of
            the columns the norm is taken over. ``None`` skips the reduction.

    Returns:
        Row norms of the adapted weight, ``[out_features]`` in float32.
    """
    weight = weight.to(torch.float32)
    lora_a = _adapter_view(lora_a, rank, adapter_id, rank_offsets).to(torch.float32)
    lora_b = _adapter_view(lora_b, rank, adapter_id, rank_offsets).to(torch.float32)

    # ||W_i||^2, summed over the columns this rank holds
    base_sq = weight.pow(2).sum(dim=1)

    # ||(B^T A)_i||^2 = (B^T (A A^T) * B).sum(dim=0). The [r, r] gram matrix
    # replaces the [out, in] dense product, so nothing of that size is formed.
    gram = lora_a @ lora_a.t()
    ba_sq = (lora_b.t() @ gram * lora_b).sum(dim=0)

    # 2*s*(B^T (A W^T))_ii -- one [r, out] matmul against W^T.
    cross = (lora_a @ weight.t() * lora_b).sum(dim=0)

    sq_norm = base_sq + 2.0 * scaling * cross + scaling * scaling * ba_sq
    if reduce_group is not None:
        torch.distributed.all_reduce(sq_norm, group=reduce_group)
    return torch.sqrt(torch.clamp(sq_norm, min=0.0))


def pack_dora_weight_norm(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    ranks: torch.Tensor,
    scalings: torch.Tensor,
    rank_offsets: torch.Tensor,
    lora_magnitude: torch.Tensor,
    reduce_group=None,
):
    """Fold DoRA magnitudes into per-adapter LoRA scalings.

    DoRA's forward applies ``m * (W + s*B@A) / ||W + s*B@A||_c``. Splitting the
    update into the base term plus a scaled LoRA term gives

        m * W / d  +  (m / d) * s * (x @ A^T @ B^T)

    with ``d = ||W + s*B@A||_c``. The first term means the base output must be
    rescaled by ``m / d`` as well, which cannot be expressed by the LoRA
    scalings alone, so this returns both factors:

    Args:
        weight: base weight of the linear layer, ``[out_features, in_features]``.
        lora_a: packed A matrices, ``[sum_rank, in_features]``.
        lora_b: packed B matrices, ``[sum_rank, out_features]``.
        ranks: rank of each adapter, ``[num_adapters]``.
        scalings: LoRA scaling of each adapter, ``[num_adapters]``.
        rank_offsets: cumulative rank offsets of the packed tensors.
        lora_magnitude: DoRA magnitude of each adapter, ``[rank, out_features]``
            with rows matching the packed layout of ``lora_b``.
        reduce_group: process group the norm sums are reduced over when the
            base weight is row-sharded across tensor-parallel ranks.

    Returns:
        A tuple ``(weight_scaling, lora_scalings)``. ``weight_scaling`` is
        ``[num_adapters, out_features]`` and multiplies the base output;
        ``lora_scalings`` is ``[num_adapters]`` and replaces the LoRA scalings.
        Rows for adapters without a magnitude are all ones.
    """
    num_adapters = ranks.numel()
    device = lora_a.device
    dtype = torch.float32

    weight = weight.to(dtype)
    weight_scaling = torch.ones((num_adapters, weight.size(0)), dtype=dtype, device=device)
    new_scalings = torch.ones((num_adapters,), dtype=dtype, device=device)

    for adapter_id in range(num_adapters):
        rank = ranks[adapter_id].item()
        if rank == 0:
            continue
        scaling = scalings[adapter_id].item()
        m = _adapter_view(lora_magnitude, rank, adapter_id, rank_offsets).to(dtype)
        # column norms of the magnitude itself, needed to rescale the base term
        m_norm = m.norm(dim=0)

        d = factored_dora_norm(
            weight, lora_a, lora_b, scaling, rank, adapter_id, rank_offsets, reduce_group=reduce_group
        )
        ratio = m_norm / torch.clamp(d, min=1e-6)

        weight_scaling[adapter_id] = ratio
        new_scalings[adapter_id] = scaling

    return weight_scaling, new_scalings
