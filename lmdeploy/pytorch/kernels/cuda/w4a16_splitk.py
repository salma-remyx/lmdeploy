# Copyright (c) OpenMMLab. All rights reserved.
"""SplitK planning for fused W4A16 dequant-GEMM kernels.

The untuned heuristic in the AWQ kernel launch path (``SPLIT_K = K // 4096``)
ignores how much parallelism is actually available in the (M, N) tile grid.
When the activation matrix is skinny, M is tiny and the tile grid alone
cannot fill the GPU, so decomposing the reduction dimension across more
CTAs is essentially free throughput. For fat batched matmuls the grid is
already saturated and a large ``SPLIT_K`` only adds atomic traffic on the
write-back. This module turns that trade-off into an explicit, testable
decision so the kernel launch picks a split that reflects the shape it was
handed.

Adapted from "Accelerating a Triton Fused Kernel for W4A16 Quantized
Inference with SplitK work decomposition" (https://arxiv.org/abs/2402.00025).
The paper's survey of skinny-M x square-K GEMMs motivates shaping the split
from the (M, N, K) problem size and the target's tile geometry; the
dequant-GEMM kernel itself is lmdeploy's existing Triton kernel.
"""

import math

# Tile shapes used by the W4A16 kernel launch path. BLOCK_SIZE_N comes from
# the autotune configs in awq_kernels; BLOCK_SIZE_M is clamped to [16, 128].
# The larger N tile is the conservative choice for occupancy estimates.
MIN_BLOCK_SIZE_M = 16
MAX_BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128

# CTAs the split-K dimension is allowed to add before the atomic write-back
# costs more than the extra parallelism buys. Calibrated against the ~10-20
# K-CTAs a streaming multiprocessor schedules concurrently.
MAX_SPLIT_K = 8

# Reduction size each K-CTA should aim for, in elements. 1024 keeps the
# per-CTA loop short enough to hide the dequant latency of the W4 payload.
SPLIT_K_TARGET_K_PER_CTA = 1024

# Targets from the paper: keep every SM busy in the compute-limited (skinny)
# regime, and stop splitting once the grid already covers the device.
TARGET_CTAS_PER_SM = 2
MIN_SPLIT_K = 1


def _num_sms(device=None) -> int:
    """Number of streaming multiprocessors on ``device`` (0 if unknown)."""
    props = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
    except Exception:
        props = None
    return getattr(props, "multi_processor_count", 0) or 0


def choose_split_k(M, N, K, num_sms=None, device=None):
    """Pick a SplitK factor for a fused W4A16 dequant-GEMM.

    Args:
        M (int): rows of the activation matrix (the "skinny" dimension).
        N (int): output columns of the GEMM.
        K (int): reduction dimension (rows of the packed weight matrix).
        num_sms (int): SM count of the target device. Probed from CUDA when
            omitted; pass an int to make the decision device-independent.
        device: device to probe when ``num_sms`` is omitted.

    Returns:
        int: SplitK factor in ``[1, MAX_SPLIT_K]``. The K dimension is split
        evenly, so the returned factor always divides ``K``.
    """
    if M <= 0 or N <= 0 or K <= 0:
        return 1

    if num_sms is None:
        num_sms = _num_sms(device)
    if num_sms <= 0:
        # No device to shape against; fall back to the shape-only heuristic.
        num_sms = 0

    block_size_m = max(MIN_BLOCK_SIZE_M, min(MAX_BLOCK_SIZE_M, _next_pow2(M)))
    tiles = math.ceil(M / block_size_m) * math.ceil(N / BLOCK_SIZE_N)
    k_blocks = K  # reduction elements per CTA; the kernel's inner loop
    # steps by BLOCK_SIZE_K = group_size, so K elements is the full extent.

    # Compute-limited (skinny) regime: the (M, N) tile grid leaves SMs idle,
    # so give each one work. Memory-bound regime (fat M): the grid already
    # covers the device, so the atomic write-back argues against splitting.
    deficit = num_sms * TARGET_CTAS_PER_SM - tiles if num_sms else 0
    if deficit <= 0:
        # Keep a minimal split so long reductions still get broken up.
        split = max(1, k_blocks // (SPLIT_K_TARGET_K_PER_CTA * 4))
    else:
        split = math.ceil(deficit / max(tiles, 1))
        split = max(split, k_blocks // SPLIT_K_TARGET_K_PER_CTA)

    # Never split past what the reduction can be divided into evenly.
    split = max(MIN_SPLIT_K, min(split, MAX_SPLIT_K, K))
    return _largest_even_divisor(K, split)


def _next_pow2(value):
    """Smallest power of two >= ``value``."""
    return 1 << max(0, (value - 1).bit_length())


def _largest_even_divisor(K, split):
    """Largest factor of ``K`` that is <= ``split`` and divides it evenly."""
    for candidate in range(min(split, K), 0, -1):
        if K % candidate == 0:
            return candidate
    return 1


def splitk_grid(M, N, K, split_k, block_size_m, block_size_n):
    """Grid the W4A16 kernel should launch with for this split.

    Returns:
        tuple: ``(grid_mn, split_k)`` matching the kernel's 2D launch grid,
        where ``grid_mn`` is the number of (M-tile, N-tile) programs.
    """
    grid_mn = math.ceil(M / block_size_m) * math.ceil(N / block_size_n)
    return grid_mn, split_k


def splitk_score(M, N, K, split_k):
    """Shape heuristic for how well ``split_k`` matches this GEMM.

    Higher is better. Used by callers that want to rank candidate splits
    without benchmarking them. The score rewards covering the device when
    the tile grid cannot, and penalizes splits that break the reduction
    unevenly or push past the point of diminishing returns.
    """
    if M <= 0 or N <= 0 or K <= 0 or split_k < 1:
        return 0.0
    if K % split_k:
        return 0.0

    parallelism = math.ceil(N / BLOCK_SIZE_N) * split_k
    coverage = 1.0 - math.exp(-parallelism / 16.0)
    k_per_cta = K / split_k
    balance = min(k_per_cta, SPLIT_K_TARGET_K_PER_CTA) / SPLIT_K_TARGET_K_PER_CTA
    overflow = max(0.0, split_k - MAX_SPLIT_K) / MAX_SPLIT_K
    return coverage * balance - overflow
