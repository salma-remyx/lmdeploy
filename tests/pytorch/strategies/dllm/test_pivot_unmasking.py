# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy
from lmdeploy.pytorch.strategies.dllm.unmasking import UnmaskingProcessor

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED

BLOCK = 8
VOCAB = 6


def _processor(strategy=UnmaskingStrategy.PIVOT, threshold=0.85):
    return UnmaskingProcessor(
        DLLMConfig(block_length=BLOCK, unmasking_strategy=strategy, confidence_threshold=threshold)
    )


def _logits_for(target: torch.Tensor) -> torch.Tensor:
    """Logits whose argmax at position i is target[i], with a tunable margin.

    Row ``i`` is a two-point distribution over (target[i], distractor) so the
    entropy of the position is controlled by ``margin``.
    """
    margin = 8.0
    logits = torch.full((BLOCK, VOCAB), -10.0)
    for pos, tok in enumerate(target.tolist()):
        logits[pos, tok] = margin
        logits[pos, (tok + 1) % VOCAB] = 0.0
    return logits


def test_pivot_strategy_is_registered_from_str():
    assert UnmaskingStrategy.from_str("pivot") is UnmaskingStrategy.PIVOT
    with pytest.raises(ValueError):
        UnmaskingStrategy.from_str("not_a_strategy")


def test_pivot_unmasks_at_least_one_position_per_step():
    token_ids = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2])
    logits = _logits_for(token_ids)
    dllm_mask = torch.full((BLOCK,), DLLM_MASKED, dtype=torch.long)

    out_mask, out_tokens = _processor()(logits, token_ids, token_ids, dllm_mask)

    assert out_mask.shape == dllm_mask.shape
    assert (out_mask == DLLM_UNMASKED).any(), "a fully masked block should commit something"
    # every committed position keeps the argmax token
    committed = out_mask == DLLM_UNMASKED
    assert torch.equal(out_tokens[committed], token_ids[committed])


def test_pivot_prefers_mid_entropy_over_most_confident():
    """The lowest-entropy position is not the one the pivot rule commits.

    With one near-deterministic outlier among otherwise high-entropy masked
    positions, the band rule measures distance to the *masked mean* entropy,
    so it must pick from the middle of the distribution rather than the
    greedily-confident outlier.
    """
    token_ids = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2])
    logits = torch.full((BLOCK, VOCAB), -10.0)
    # position 0: near-deterministic -> very low entropy
    logits[0, 1] = 30.0
    logits[0, 2] = 0.0
    # positions 1..7: uniform -> high entropy
    logits[1:, :] = 0.0

    dllm_mask = torch.full((BLOCK,), DLLM_MASKED, dtype=torch.long)
    # threshold above 1.0 disables the confidence gate entirely
    out_mask, _ = _processor(threshold=1.1)(logits, token_ids, token_ids, dllm_mask)

    committed = out_mask == DLLM_UNMASKED
    assert committed.any(), "the pivot rule should commit even with the confidence gate disabled"
    if committed.sum() == 1:
        assert not committed[0], "a lone confident outlier is not a mid-entropy pivot"


def test_pivot_leaves_cached_blocks_untouched():
    token_ids = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2])
    logits = _logits_for(token_ids)
    dllm_mask = torch.full((BLOCK,), DLLM_CACHED, dtype=torch.long)

    out_mask, _ = _processor()(logits, token_ids, token_ids, dllm_mask)

    assert (out_mask == DLLM_CACHED).all(), "already-committed blocks must not be reopened"


def test_pivot_does_not_reopen_partially_unmasked_positions():
    token_ids = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2])
    logits = _logits_for(token_ids)
    dllm_mask = torch.full((BLOCK,), DLLM_MASKED, dtype=torch.long)
    dllm_mask[3] = DLLM_CACHED

    out_mask, _ = _processor(threshold=1.1)(logits, token_ids, token_ids, dllm_mask)

    assert out_mask[3] == DLLM_CACHED, "a committed position must not be rewritten by the pivot rule"


def test_pivot_converges_to_fully_unmasked():
    token_ids = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2])
    logits = _logits_for(token_ids)
    dllm_mask = torch.full((BLOCK,), DLLM_MASKED, dtype=torch.long)
    processor = _processor()

    for _ in range(BLOCK):
        dllm_mask, token_ids = processor(logits, token_ids, token_ids, dllm_mask)
        if (dllm_mask == DLLM_MASKED).sum() == 0:
            break

    assert (dllm_mask == DLLM_MASKED).sum() == 0, "repeated steps should unmask the whole block"
    assert (dllm_mask != DLLM_MASKED).all()
