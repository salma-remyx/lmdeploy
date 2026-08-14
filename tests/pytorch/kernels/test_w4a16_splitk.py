# Copyright (c) OpenMMLab. All rights reserved.
import math

import pytest

from lmdeploy.pytorch.kernels.cuda import w4a16_splitk
from lmdeploy.pytorch.kernels.cuda.awq_kernels import get_cuda_autotune_config

A100_SMS = 108
H100_SMS = 132


def _autotune_block_n():
    """BLOCK_SIZE_N options the kernel's autotune configs can pick from."""
    return sorted({cfg.kwargs["BLOCK_SIZE_N"] for cfg in get_cuda_autotune_config()})


def test_split_matches_kernel_tile_geometry():
    """The planner's N tile must be one the kernel can actually launch with."""
    assert w4a16_splitk.BLOCK_SIZE_N in _autotune_block_n()
    assert w4a16_splitk.MIN_BLOCK_SIZE_M == 16
    assert w4a16_splitk.MAX_BLOCK_SIZE_M == 128


@pytest.mark.parametrize("M", [1, 2, 4, 8])
def test_skinny_gemm_splits_reduction(M):
    """Decode-time M x N x N GEMMs must split K across idle SMs."""
    N, K = 4096, 4096
    split = w4a16_splitk.choose_split_k(M, N, K, num_sms=A100_SMS)
    assert split > 1, "skinny GEMM should decompose the reduction dimension"
    assert K % split == 0


def test_skinny_gemm_beats_old_heuristic():
    """K//4096 returned 1 (unsplit) for the canonical decode shape."""
    N, K = 4096, 4096
    assert max(1, K // 4096) == 1
    for M in (1, 4, 16):
        assert w4a16_splitk.choose_split_k(M, N, K, num_sms=A100_SMS) > 1


def test_prefill_gemm_stays_unsplit():
    """A grid that already covers the device should not pay atomic traffic."""
    N, K = 4096, 4096
    split = w4a16_splitk.choose_split_k(1024, N, K, num_sms=A100_SMS)
    assert split == 1


@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (4, 8192, 6144), (128, 5120, 5120), (37, 4096, 11008)])
def test_split_always_divides_k(M, N, K):
    """SplitK CTAs stride over K, so the split must divide K evenly."""
    split = w4a16_splitk.choose_split_k(M, N, K, num_sms=H100_SMS)
    assert 1 <= split <= w4a16_splitk.MAX_SPLIT_K
    assert K % split == 0


@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (16, 8192, 6144), (512, 5120, 5120), (7, 4096, 11008)])
def test_grid_covers_problem(M, N, K):
    """The launch grid must tile the full (M, N) output."""
    block_m = max(16, min(128, 1 << (M - 1).bit_length()))
    split = w4a16_splitk.choose_split_k(M, N, K, num_sms=A100_SMS)
    grid_mn, grid_k = w4a16_splitk.splitk_grid(M, N, K, split, block_m, w4a16_splitk.BLOCK_SIZE_N)
    assert grid_mn >= math.ceil(M / block_m) * math.ceil(N / w4a16_splitk.BLOCK_SIZE_N)
    assert grid_k == split


@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (64, 8192, 4096), (1024, 4096, 4096)])
def test_score_ranks_skinny_split_over_flat(M, N, K):
    """The shape score should prefer splitting skinny GEMMs, not fat ones."""
    skinny = w4a16_splitk.splitk_score(M, N, K, w4a16_splitk.choose_split_k(M, N, K, num_sms=A100_SMS))
    flat = w4a16_splitk.splitk_score(M, N, K, 1)
    if M * (N // w4a16_splitk.BLOCK_SIZE_N) < A100_SMS:
        assert skinny > flat


def test_degenerate_shapes_fall_back_to_unsplit():
    for args in [(0, 4096, 4096), (1, 0, 4096), (1, 4096, 0), (-1, -1, -1)]:
        assert w4a16_splitk.choose_split_k(*args, num_sms=A100_SMS) == 1


def test_prime_k_stays_unsplit():
    """A prime K cannot be divided evenly, so the split must collapse to 1."""
    assert w4a16_splitk.choose_split_k(1, 4096, 4099, num_sms=A100_SMS) == 1


def test_no_gpu_probing_on_cpu():
    """num_sms is injectable so planning stays importable without CUDA."""
    assert w4a16_splitk._num_sms(None) in (0, A100_SMS, H100_SMS)
