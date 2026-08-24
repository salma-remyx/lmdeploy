# Copyright (c) OpenMMLab. All rights reserved.
"""Entropy-guided pivot unmasking.

Ripple-Pivot Search (RPS, arXiv:2608.11742) observes that in block-diffusion
decoding, committing a *mid-entropy* position -- not the most confident one,
and not the least -- can sharply reduce the uncertainty of its neighbours.
That "ripple" lets the following steps commit more positions in parallel.

The paper picks pivots by entropy band and then resolves each pivot's token
with a lookahead forward pass per candidate. ``post_sampling`` has no budget
for extra forwards, so this port keeps the *where to decode* rule at full
fidelity (mid-entropy band over the still-masked positions) and replaces the
lookahead *what to decode* rule with a one-step proxy from the logits already
in hand: a candidate is preferred when it is settled while its neighbours are
not, i.e. when committing it carries the most information outward.
"""

import torch

from lmdeploy.pytorch import consts

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED


def position_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of each position's next-token distribution.

    Computed from log-softmax so it is stable across the whole vocabulary
    without materializing probabilities twice.
    """
    log_probs = logits.log_softmax(dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1).clamp_min(0)


def _masked_mean(values: torch.Tensor, is_masked: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the masked positions of each block."""
    count = is_masked.sum(dim=-1, keepdim=True).clamp_min(1)
    return (values * is_masked).sum(dim=-1, keepdim=True) / count


def _neighbour_mean(values: torch.Tensor) -> torch.Tensor:
    """Mean of each position's within-block neighbours (no wrap-around)."""
    padded = torch.nn.functional.pad(values, (1, 1), mode="replicate")
    return (padded[:, :-2] + padded[:, 2:]) / 2


def select_pivots(entropy: torch.Tensor, dllm_mask: torch.Tensor, num_pivots: int):
    """Select mid-entropy pivot positions per block.

    Args:
        entropy: [num_blocks, block_size] per-position entropy.
        dllm_mask: [num_blocks, block_size] mask constants.
        num_pivots: how many pivots to commit per block per step.

    Returns:
        ``(pivots, valid)`` where ``pivots`` is [num_blocks, num_pivots]
        block-local indices in decreasing preference order and ``valid`` marks
        the entries that actually point at a masked position. Blocks with
        nothing left to unmask report all-invalid.
    """
    is_masked = dllm_mask == DLLM_MASKED
    distance = (entropy - _masked_mean(entropy, is_masked)).abs()
    distance = torch.where(is_masked, distance, distance.new_full((), torch.inf))

    pivots = distance.argsort(dim=-1, stable=True)[:, :num_pivots]
    valid = torch.gather(is_masked, -1, pivots)
    return pivots, valid


def ripple_scores(entropy: torch.Tensor, dllm_mask: torch.Tensor) -> torch.Tensor:
    """One-step ripple preference for each masked position.

    A position is a promising pivot when it is settled itself but its
    neighbours are not: committing it is then what unlocks them.
    """
    return (_neighbour_mean(entropy) - entropy) * (dllm_mask == DLLM_MASKED)


def pivot_unmask(
    logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor, block_size: int, num_pivots: int
) -> torch.Tensor:
    """Commit mid-entropy pivots and return the updated dllm mask.

    Args:
        logits: [seq_len, vocab] (or broadcastable) raw logits.
        token_ids: [seq_len] currently assigned token per position. Only its
            length is used; the token assignment is the model's own argmax,
            as in the low-confidence strategies.
        dllm_mask: [seq_len] mask constants.
        block_size: block length of the diffusion model.
        num_pivots: pivots to commit per block per step.

    Returns:
        Updated [seq_len] dllm mask.
    """
    entropy = position_entropy(logits).view(-1, block_size)
    mask_2d = dllm_mask.view(-1, block_size)

    pivots, valid = select_pivots(entropy, mask_2d, num_pivots)

    # Among the band-selected candidates, commit first the one whose
    # neighbourhood is most likely to settle once it lands.
    ripple_at = torch.gather(ripple_scores(entropy, mask_2d), -1, pivots)
    order = ripple_at.argsort(dim=-1, descending=True, stable=True)
    pivots = torch.gather(pivots, -1, order)
    valid = torch.gather(valid, -1, order)

    # Re-scatter the current value where a slot points at an already-committed
    # position, so exhausted blocks are left exactly as they were.
    updates = torch.where(valid, torch.full_like(pivots, DLLM_UNMASKED), torch.gather(mask_2d, -1, pivots))
    return mask_2d.scatter(-1, pivots, updates).flatten()
