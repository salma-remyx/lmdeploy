"""Tests for attention-aware KV cache bit allocation.

Covers the reverse water-filling allocator and its integration with the
calibration-observer plumbing in lmdeploy.lite.
"""

import pytest
import torch

from lmdeploy.lite.quantization import kv_bit_alloc as kvba
from lmdeploy.lite.quantization.activation import KVCacheObserver
from lmdeploy.lite.quantization.kv_bit_alloc import (
    ChannelDistortionStats,
    allocate_kv_bits,
    waterfill_bits,
    whitening_transform,
)


def _make_stats(num_heads=2, k_dim=8, v_dim=8, tokens=64, seed=0):
    """Build stats from synthetic K/V states with heterogeneous channel energy."""
    generator = torch.Generator().manual_seed(seed)
    stats = ChannelDistortionStats(num_heads, k_dim, v_dim)
    scale = torch.logspace(0, 2, k_dim)  # channel energy spans two decades
    k = torch.randn(tokens, num_heads, k_dim, generator=generator) * scale
    v = torch.randn(tokens, num_heads, v_dim, generator=generator) * scale
    stats.observe(k, v)
    return stats, scale


def test_waterfill_bits_spends_exact_budget():
    """The allocation must sum to the requested budget."""
    weights = torch.logspace(0, 2, 16)
    bits = waterfill_bits(weights, total_bits=16 * 4, min_bits=2, max_bits=8)
    assert bits.sum().item() == 16 * 4
    assert bits.dtype == torch.int64


def test_waterfill_bits_prefers_high_weight_channels():
    """Bits must flow to the channels distortion weights say matter most."""
    weights = torch.logspace(0, 2, 16)
    bits = waterfill_bits(weights, total_bits=16 * 4, min_bits=2, max_bits=8)
    assert bits[0].item() < bits[-1].item()
    assert torch.equal(bits, torch.sort(bits).values)


def test_waterfill_bits_rejects_unattainable_budget():
    """Budgets outside the [floor, ceiling] range must be rejected."""
    weights = torch.ones(8)
    with pytest.raises(ValueError):
        waterfill_bits(weights, total_bits=8 * 9, min_bits=2, max_bits=8)
    with pytest.raises(ValueError):
        waterfill_bits(weights, total_bits=8 * 1, min_bits=2, max_bits=8)


def test_waterfill_bits_rejects_negative_weights():
    with pytest.raises(ValueError):
        waterfill_bits(torch.tensor([1.0, -1.0]), total_bits=4)


def test_whitening_transform_diagonalizes_and_sorts():
    """The basis must be orthonormal and ordered by descending energy."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(512, 6, generator=generator) * torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    basis, eigenvalues = whitening_transform(x)
    assert basis.shape == (6, 6)
    assert torch.allclose(basis.T @ basis, torch.eye(6), atol=1e-4)
    assert torch.equal(eigenvalues, torch.sort(eigenvalues, descending=True).values)
    assert eigenvalues[0] > eigenvalues[-1]


def test_allocate_kv_bits_shapes_and_budgets():
    """The layer plan must respect head/dim shapes and spend both budgets."""
    stats, _ = _make_stats(num_heads=2, k_dim=8, v_dim=8)
    plan = allocate_kv_bits(stats, k_budget=2 * 8 * 4, v_budget=2 * 8 * 2)
    assert plan.k_bits.shape == (2, 8)
    assert plan.v_bits.shape == (2, 8)
    assert plan.k_bits.sum().item() == 2 * 8 * 4
    assert plan.v_bits.sum().item() == 2 * 8 * 2
    assert plan.k_distortion >= 0 and plan.v_distortion >= 0


def test_allocation_summary_reports_compression():
    """The summary must expose the compression ratio against an fp16 cache."""
    stats, _ = _make_stats()
    plan = allocate_kv_bits(stats, k_budget=2 * 8 * 4, v_budget=2 * 8 * 2)
    summary = plan.summary()
    assert summary['compression_ratio'] == pytest.approx(16 / 3, rel=1e-3)
    assert summary['uniform_fp16_bits_per_token'] == 2 * 16 * 16
    assert summary['total_bits_per_token'] == plan.total_bits()


def test_query_weighting_beats_uniform_on_skewed_attention():
    """Attention-aware weights must shift bits toward query-loaded channels.

    With a uniform K profile every channel gets the same bits. Feeding the
    allocator the actual query second moment must break that symmetry in the
    direction attention reads.
    """
    generator = torch.Generator().manual_seed(3)
    tokens, heads, dim = 256, 1, 8
    stats_uniform = ChannelDistortionStats(heads, dim, dim)
    stats_aware = ChannelDistortionStats(heads, dim, dim)

    k = torch.randn(tokens, heads, dim, generator=generator)
    v = torch.randn(tokens, heads, dim, generator=generator)
    q = torch.randn(tokens, heads * 4, dim, generator=generator)
    q[..., 0] *= 50.0  # attention reads channel 0 far harder

    stats_uniform.observe(k, v)
    stats_aware.observe(k, v, q=q)

    budget = heads * dim * 4
    uniform = allocate_kv_bits(stats_uniform, budget, budget)
    aware = allocate_kv_bits(stats_aware, budget, budget)

    assert aware.k_bits[0, 0].item() >= uniform.k_bits[0, 0].item()
    assert aware.k_bits[0, 0].item() == aware.k_bits.max().item()
    assert not torch.equal(aware.k_bits, uniform.k_bits)


def test_observer_feeds_allocator_end_to_end():
    """KVCacheObserver stats must slot into the allocator unchanged.

    lmdeploy's existing calibration observer records per-head/per-channel
    absmax; deriving second moments from it and allocating must produce a
    well-formed plan.
    """
    observer = KVCacheObserver(2, 8)
    generator = torch.Generator().manual_seed(4)
    for _ in range(3):
        observer.observe(torch.randn(1, 32, 2, 8, generator=generator))

    stats = ChannelDistortionStats(2, 8, 8)
    k = observer.absmax_val.float().unsqueeze(0)
    v = observer.absmax_val.float().unsqueeze(0)
    stats.observe(k, v)

    plan = allocate_kv_bits(stats, 2 * 8 * 4, 2 * 8 * 2)
    assert plan.k_bits.shape == (2, 8)
    assert plan.total_bits() == 2 * 8 * 4 + 2 * 8 * 2


def test_module_reuses_kv_bit_alloc_helpers():
    """The calibration API module must re-export the allocator primitives."""
    assert hasattr(kvba, 'allocate_kv_bits')
    assert hasattr(kvba, 'whitening_transform')
    assert hasattr(kvba, 'ChannelDistortionStats')
