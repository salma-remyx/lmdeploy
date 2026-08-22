# Copyright (c) OpenMMLab. All rights reserved.
"""Block-sparse prefill attention helpers.

Adapted from FlashPrefill V2: Block-Sparse Prefill Attention for
Long-Context LLM Serving (arXiv:2608.19758). The paper discovers the
important KV blocks instantaneously from the per-row max of the attention
logits, applies a dynamic threshold to drop the remaining blocks, and
corrects the dropped mass with a mean term so the approximation error
stays bounded at extreme sparsity.

Kept from the paper (the core mechanism):

- max-based dynamic thresholding: a block contributes only if the maximum
  logit it produced comes within a fixed offset of the running row max.
- mean correction: the softmax denominator gets an additive term for the
  dropped blocks, estimated from the running mean of the kept block maxima.

Deliberately substituted (target-native parts):

- the pattern is discovered inside the online-softmax loop of the existing
  Triton prefill kernel instead of by a standalone CUDA kernel with
  PackGQA memory access, warp specialization and pingpong pipelining.
- the kernel keeps streaming every KV block and zeroes the probability
  rows that miss the threshold, so no separate block-index gather pass
  and no new kernel launch are needed.

``block_sparse_scores`` is the host-side twin of that keep rule. It lets a
caller compute (or audit) the per-block decision without launching the
kernel, and it is what the unit tests use as the reference for the
threshold the kernel applies.
"""
import math

import torch
from torch import Tensor

__all__ = ['block_sparse_scores', 'sparse_ratio_to_threshold']


def sparse_ratio_to_threshold(ratio: float) -> float:
    """Map a keep ratio onto the logit-space threshold used by the kernel.

    A block is dropped when its max logit is more than ``-log(ratio)``
    below the running row max, i.e. when it contributes less than
    ``ratio`` of the dominant block's probability mass. ``ratio == 0``
    means "keep everything", so the kernel treats a non-positive
    threshold as disabled and runs the dense path.

    Args:
        ratio: fraction of the dominant block mass, in [0, 1).

    Returns:
        Threshold offset in logit space, >= 0.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f'sparse_ratio must be in [0, 1), got {ratio}')
    if ratio == 0.0:
        return 0.0
    return -math.log(ratio)


def block_sparse_scores(q: Tensor,
                        k: Tensor,
                        q_seqlens: Tensor,
                        kv_seqlens: Tensor,
                        block_size: int,
                        softmax_scale: float = None,
                        keep_ratio: float = 0.0) -> Tensor:
    """Compute the per-block max logit that drives the sparse keep rule.

    Args:
        q: packed queries, ``(total_q, num_heads, head_dim)``.
        k: packed keys, ``(total_k, num_kv_heads, head_dim)``.
        q_seqlens: per-sequence query lengths.
        kv_seqlens: per-sequence key lengths.
        block_size: KV block size to score at. The kernel scores at
            ``BLOCK_N`` granularity, which is what this should be set to.
        softmax_scale: softmax scale, defaults to ``1 / sqrt(head_dim)``.
        keep_ratio: blocks whose max logit is more than
            ``-log(keep_ratio)`` below the row's best block are zeroed.

    Returns:
        ``(total_q, num_heads, num_blocks)`` float32 block scores, 0 for
        dropped blocks and for the padding tail of a ragged last block.
    """
    if block_size < 1:
        raise ValueError(f'block_size must be >= 1, got {block_size}')

    head_dim = q.size(-1)
    num_heads = q.size(1)
    num_kv_heads = k.size(1)
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    # GQA: expand each kv head across the query-head group it serves.
    group = num_heads // num_kv_heads
    if group > 1:
        k = k.repeat_interleave(group, dim=1)

    threshold = sparse_ratio_to_threshold(keep_ratio)
    max_blocks = (kv_seqlens.max().item() + block_size - 1) // block_size
    scores = q.new_zeros(q.size(0), num_heads, max_blocks, dtype=torch.float32)

    q_start = 0
    kv_start = 0
    for q_seqlen, kv_seqlen in zip(q_seqlens.tolist(), kv_seqlens.tolist()):
        qs = q[q_start:q_start + q_seqlen].float()
        ks = k[kv_start:kv_start + kv_seqlen].float()

        # -> (num_heads, q_seqlen, kv_seqlen)
        logits = torch.einsum('qhd,khd->hqk', qs, ks) * softmax_scale

        num_blocks = (kv_seqlen + block_size - 1) // block_size
        padded = torch.nn.functional.pad(logits, (0, num_blocks * block_size - kv_seqlen),
                                         value=float('-inf'))
        # -> (num_heads, q_seqlen, num_blocks)
        block_max = padded.reshape(num_heads, q_seqlen, num_blocks, block_size).amax(dim=-1)

        if threshold > 0.0:
            row_max = block_max.amax(dim=-1, keepdim=True)
            block_max = torch.where(block_max >= row_max - threshold, block_max,
                                    torch.zeros_like(block_max))

        # (num_heads, q_seqlen, num_blocks) -> (q_seqlen, num_heads, num_blocks)
        block_max = block_max.permute(1, 0, 2)
        # a fully padded trailing block reports -inf; report it as 0
        block_max = torch.nan_to_num(block_max, nan=0.0, neginf=0.0)
        scores[q_start:q_start + q_seqlen, :, :num_blocks] = block_max
        q_start += q_seqlen
        kv_start += kv_seqlen

    return scores
