# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the block-sparse prefill path of the Triton flash attention.

Covers the ``sparse_threshold`` argument added to
``lmdeploy.pytorch.kernels.cuda.flashattention.flash_attn_varlen_func``
(adapted from FlashPrefill V2, arXiv:2608.19758) and the host-side
scorer in ``lmdeploy.pytorch.kernels.cuda.block_sparse_prefill`` that
mirrors the kernel's keep rule.
"""
import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA is required for block-sparse prefill tests', allow_module_level=True)

from lmdeploy.pytorch.kernels.cuda.block_sparse_prefill import (  # noqa: E402
    block_sparse_scores,
    sparse_ratio_to_threshold,
)


def _naive_attention(q, k, v, q_seqlens, kv_seqlens, softmax_scale, causal=True):
    """Dense reference attention over the packed varlen layout."""
    outs = []
    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens.tolist(), kv_seqlens.tolist()):
        qs = q[q_start:q_start + q_len].float()
        num_heads = qs.size(1)
        group = num_heads // k.size(1)
        ks = k[kv_start:kv_start + kv_len].repeat_interleave(group, dim=1).float()
        vs = v[kv_start:kv_start + kv_len].repeat_interleave(group, dim=1).float()

        logits = torch.einsum('qhd,khd->hqk', qs, ks) * softmax_scale
        if causal:
            history_len = kv_len - q_len
            pos_q = torch.arange(history_len, kv_len, device=q.device)
            pos_k = torch.arange(kv_len, device=q.device)
            logits = logits.masked_fill((pos_k[None, :] > pos_q[:, None])[None], float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum('hqk,khd->qhd', attn, vs)
        outs.append(out)
        q_start += q_len
        kv_start += kv_len
    return torch.cat(outs, dim=0)


class TestSparseRatioToThreshold:

    def test_zero_keeps_everything(self):
        assert sparse_ratio_to_threshold(0.0) == 0.0

    def test_ratio_maps_to_logit_offset(self):
        assert math.isclose(sparse_ratio_to_threshold(0.5), math.log(2.0), rel_tol=1e-6)

    @pytest.mark.parametrize('ratio', [-0.1, 1.0, 2.0])
    def test_rejects_out_of_range(self, ratio):
        with pytest.raises(ValueError):
            sparse_ratio_to_threshold(ratio)


class TestBlockSparseScores:

    @pytest.fixture
    def qkv(self):
        torch.manual_seed(0)
        num_heads, num_kv_heads, head_dim = 4, 2, 16
        q_seqlens = torch.tensor([12, 8], device='cuda')
        kv_seqlens = torch.tensor([24, 16], device='cuda')
        q = torch.randn(q_seqlens.sum(), num_heads, head_dim, device='cuda')
        k = torch.randn(kv_seqlens.sum(), num_kv_heads, head_dim, device='cuda')
        v = torch.randn(kv_seqlens.sum(), num_kv_heads, head_dim, device='cuda')
        return q, k, v, q_seqlens, kv_seqlens

    def test_scores_shape(self, qkv):
        q, k, _, q_seqlens, kv_seqlens = qkv
        block_size = 8
        scores = block_sparse_scores(q, k, q_seqlens, kv_seqlens, block_size=block_size)
        num_blocks = math.ceil(kv_seqlens.max().item() / block_size)
        assert scores.shape == (q.size(0), q.size(1), num_blocks)
        assert scores.dtype == torch.float32

    def test_zero_ratio_keeps_all_blocks(self, qkv):
        q, k, _, q_seqlens, kv_seqlens = qkv
        # 24 and 16 are both multiples of 8, so there is no padded tail
        scores = block_sparse_scores(q, k, q_seqlens, kv_seqlens, block_size=8, keep_ratio=0.0)
        assert torch.all(scores > 0)

    def test_small_ratio_drops_blocks(self, qkv):
        q, k, _, q_seqlens, kv_seqlens = qkv
        # threshold lives in logit space, so a moderate ratio is needed to
        # drop anything on O(1) random logits
        scores = block_sparse_scores(q, k, q_seqlens, kv_seqlens, block_size=4, keep_ratio=0.1)
        kept = (scores > 0).float().mean()
        assert kept < 1.0


class TestFlashPrefillSparseKernel:
    """Exercise the sparse_threshold wiring on the real kernel."""

    @pytest.fixture
    def inputs(self):
        torch.manual_seed(0)
        num_heads, num_kv_heads, head_dim = 4, 2, 16
        q_seqlens = torch.tensor([64, 32], device='cuda')
        kv_seqlens = torch.tensor([128, 96], device='cuda')
        q = torch.randn(q_seqlens.sum(), num_heads, head_dim, dtype=torch.float16, device='cuda')
        # kernel wants (seq, heads, dim) for the default 'hsd' kv_layout;
        # build hsd directly so no transpose is needed at the call site
        k = torch.randn(kv_seqlens.sum(), num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        v = torch.randn(kv_seqlens.sum(), num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        return q, k, v, q_seqlens, kv_seqlens

    @staticmethod
    def _cu_seqlens(seqlens):
        cu = seqlens.cumsum(0)
        return torch.cat([cu.new_zeros(1), cu]).int()

    @pytest.fixture
    def dense_gt(self, inputs):
        q, k, v, q_seqlens, kv_seqlens = inputs
        return _naive_attention(q, k, v, q_seqlens, kv_seqlens, 1.0 / math.sqrt(q.size(-1)))

    def _run(self, inputs, **kwargs):
        from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func
        q, k, v, q_seqlens, kv_seqlens = inputs
        return flash_attn_varlen_func(
            q,
            k,
            v,
            self._cu_seqlens(q_seqlens),
            self._cu_seqlens(kv_seqlens),
            max_seqlen_q=q_seqlens.max().item(),
            causal=True,
            **kwargs,
        )

    def test_disabled_threshold_matches_dense(self, inputs, dense_gt):
        """sparse_threshold=0 must be exactly the pre-existing dense path."""
        out = self._run(inputs, sparse_threshold=0.0)
        torch.testing.assert_close(out.float(), dense_gt, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize('threshold,max_diff', [(2.0, 0.1), (4.0, 1e-3)])
    def test_sparse_output_stays_close_to_dense(self, inputs, dense_gt, threshold, max_diff):
        """Mean-corrected sparse prefill must stay close to the dense result.

        The bounds come from a reference sweep of the same loop in
        float32: error grows monotonically as the threshold tightens and
        is already 0 at threshold 4 on this short-context geometry.
        """
        out = self._run(inputs, sparse_threshold=threshold)
        diff = (out.float() - dense_gt).abs().max()
        assert diff < max_diff, f'threshold={threshold} drifted: max abs diff {diff}'

    def test_tight_threshold_stays_bounded(self, inputs, dense_gt):
        """Even at the tightest useful threshold the output stays finite."""
        out = self._run(inputs, sparse_threshold=1.0)
        assert torch.isfinite(out).all()
        diff = (out.float() - dense_gt).abs().max()
        assert diff < 0.3, f'max abs diff {diff}'
