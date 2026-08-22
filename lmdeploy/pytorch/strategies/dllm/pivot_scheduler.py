# Copyright (c) OpenMMLab. All rights reserved.
"""Mid-entropy pivot commit policy for dllm parallel decoding.

``low_confidence_dynamic`` commits every masked position whose sampled
token reaches the confidence threshold, and when none does it still
commits the most confident one so that the forward pass is not wasted.
``Ripple-Pivot Search: Active Parallel Decoding for Diffusion Large
Language Models`` (arXiv:2608.11742) shows that this forced guess is
spent on the wrong position: committing a *mid-entropy* position instead
triggers a ripple that reduces uncertainty across the remaining masked
positions, so the following steps can unmask more tokens in parallel.

:class:`PivotScheduler` keeps the confidence criterion untouched and only
replaces that forced guess: when no masked position reaches the
threshold, the commit goes to the position whose entropy is closest to
the mid point of the block rather than to the most confident one.

Lookahead evaluation of the candidate token assignment (the other half of
RPS) needs extra model forwards through the agent loop and is out of
scope for this change.
"""
import math

import torch

from lmdeploy.pytorch import consts

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy in nats, reduced over the vocabulary dimension."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


def sampled_confidence(probs: torch.Tensor, token_ids: torch.Tensor, block_size: int,
                       dllm_mask: torch.Tensor) -> torch.Tensor:
    """Probability of the sampled token, ``-inf`` where not masked.

    Shaped ``[num_blocks, block_size]`` so the criterion can only ever
    select positions the commit policy is allowed to touch.

    Args:
        probs: softmax of the model logits, ``[num_tokens, vocab]``.
        token_ids: sampled candidate per token, ``[num_tokens]``.
        block_size: dllm block length.
        dllm_mask: mask per token, ``[num_tokens]``.
    """
    scores = probs.gather(-1, token_ids.unsqueeze(-1)).flatten()
    mask = dllm_mask.view(-1, block_size)
    return torch.where(mask == DLLM_MASKED, scores.view(-1, block_size), scores.new_full((1, ), -math.inf))


def pivot_distance(entropy: torch.Tensor, dllm_mask: torch.Tensor) -> torch.Tensor:
    """Distance of each masked position to the mid-entropy point of its block.

    Positions that are not masked get ``inf`` so they can never be picked
    as pivots. The mid point is the average of the lowest and highest
    entropy still masked in the block: positions already decided carry no
    ripple, while the highest-entropy position is the one most likely to
    be committed wrongly.

    Args:
        entropy: entropy per position, ``[num_blocks, block_size]``.
        dllm_mask: mask per position, same shape as ``entropy``.

    Returns:
        Distance tensor of the same shape; ``inf`` where not masked.
    """
    is_masked = dllm_mask == DLLM_MASKED
    neg_inf = entropy.new_full((1, ), -math.inf)
    masked_entropy = torch.where(is_masked, entropy, neg_inf)
    low = masked_entropy.min(dim=-1).values.clamp_min(0.0)
    high = masked_entropy.max(dim=-1).values
    # a block with nothing left to mask has no mid point; sending it to
    # -inf keeps every distance infinite so it can never produce a pivot
    has_masked = is_masked.any(dim=-1, keepdim=True)
    mid = torch.where(has_masked, (low + high) * 0.5, neg_inf)
    distance = (entropy - mid[:, None]).abs()
    return torch.where(is_masked, distance, distance.new_full((1, ), math.inf))


class PivotScheduler:
    """Ripple-Pivot commit policy on top of the confidence criterion.

    A step commits every masked position whose sampled token reaches the
    confidence threshold, exactly as ``low_confidence_dynamic`` does.
    Only when a block has no such position does the commit go to the
    ``num_pivots`` masked positions closest to the mid-entropy point of
    that block, instead of to its most confident position.

    Args:
        block_size: dllm block length.
        threshold: confidence a position must reach to be committed by
            the criterion. ``None`` makes the policy pivot-only.
        num_pivots: number of mid-entropy positions committed per block
            when the confidence criterion is idle.
    """

    def __init__(self, block_size: int, threshold: float | None, num_pivots: int = 1):
        assert num_pivots >= 1, 'num_pivots must be positive'
        self.block_size = block_size
        self.threshold = threshold
        self.num_pivots = num_pivots

    @classmethod
    def from_dllm_config(cls, dllm_config, num_pivots: int = 1) -> 'PivotScheduler':
        """Build from a :class:`~lmdeploy.pytorch.config.DLLMConfig`."""
        return cls(block_size=dllm_config.block_length,
                   threshold=dllm_config.confidence_threshold,
                   num_pivots=num_pivots)

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """Advance the mask by one commit policy step.

        Args:
            logits: model logits, ``[num_tokens, vocab]``.
            token_ids: sampled candidate per token, ``[num_tokens]``.
            dllm_mask: mask per token, ``[num_tokens]``.

        Returns:
            Tuple of the updated mask (same shape and dtype as the input)
            and the number of tokens committed in this step.
        """
        block_size = self.block_size
        mask = dllm_mask.view(-1, block_size)
        probs = logits.softmax(dim=-1)
        scores = sampled_confidence(probs, token_ids, block_size, dllm_mask)

        if self.threshold is None:
            confident = torch.zeros_like(scores, dtype=torch.bool)
        else:
            confident = scores >= self.threshold

        # blocks where the criterion is idle fall back to mid-entropy pivots
        idle = ~confident.any(dim=1)
        if bool(idle.any()):
            distance = pivot_distance(entropy_from_probs(probs).view(-1, block_size), mask)
            num_pivots = min(self.num_pivots, block_size)
            pivots = torch.zeros_like(confident).scatter(
                -1, distance.topk(num_pivots, dim=-1, largest=False).indices, True)
            pivots &= torch.isfinite(distance)
            confident = confident | (pivots & idle[:, None])

        num_committed = int(confident.sum())
        dllm_mask = torch.where(confident, mask.new_full((1, ), DLLM_UNMASKED), mask)
        return dllm_mask.view(-1), num_committed
