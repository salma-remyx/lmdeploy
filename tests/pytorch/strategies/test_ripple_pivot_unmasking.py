# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy
from lmdeploy.pytorch.strategies.dllm.pivot_scheduler import pivot_distance
from lmdeploy.pytorch.strategies.dllm.unmasking import UnmaskingProcessor

BLOCK = 4
VOCAB = 8


def _logits_from_confidence(confidence: list[float]) -> torch.Tensor:
    """Build logits whose softmax max probability is exactly ``confidence``."""
    rows = []
    for conf in confidence:
        noise = (1.0 - conf) / (VOCAB - 1)
        row = torch.full((VOCAB, ), noise)
        row[0] = conf
        rows.append(row.log())
    return torch.stack(rows)


def _processor(strategy: str = 'ripple_pivot', threshold: float = 0.85, num_pivots: int = 1) -> UnmaskingProcessor:
    config = DLLMConfig(block_length=BLOCK,
                        unmasking_strategy=UnmaskingStrategy.from_str(strategy),
                        denoising_steps=BLOCK,
                        confidence_threshold=threshold,
                        num_pivots=num_pivots)
    return UnmaskingProcessor(dllm_config=config)


def test_ripple_pivot_registered_in_strategy_enum():
    """The new strategy round-trips through the public config surface."""
    assert UnmaskingStrategy.from_str('ripple_pivot') is UnmaskingStrategy.RIPPLE_PIVOT
    with pytest.raises(ValueError):
        UnmaskingStrategy.from_str('not_a_strategy')


def test_ripple_pivot_commits_confident_positions_as_dynamic_does():
    """Above the threshold the criterion, not the pivot rule, commits."""
    confidence = [0.99, 0.5, 0.9, 0.5]
    logits = _logits_from_confidence(confidence)
    token_ids = torch.zeros(BLOCK, dtype=torch.long)
    dllm_mask = torch.tensor([0] * BLOCK, dtype=torch.uint8)

    out_mask, out_ids = _processor()(logits, token_ids, token_ids, dllm_mask)

    assert out_mask.tolist() == [1, 0, 1, 0]
    # confident positions keep the sampled token
    assert out_ids.tolist() == token_ids.tolist()


def test_ripple_pivot_pivots_instead_of_forcing_a_guess():
    """When no position reaches the threshold, commit the mid-entropy one.

    ``low_confidence_dynamic`` would force its most confident position
    (index 0); ripple-pivot spends that commit on the mid-entropy pivot.
    """
    confidence = [0.6, 0.2, 0.55, 0.2]
    logits = _logits_from_confidence(confidence)
    token_ids = torch.zeros(BLOCK, dtype=torch.long)
    dllm_mask = torch.tensor([0] * BLOCK, dtype=torch.uint8)

    dynamic_mask, _ = _processor(strategy='low_confidence_dynamic')(logits, token_ids, token_ids,
                                                                    dllm_mask.clone())
    pivot_mask, _ = _processor(strategy='ripple_pivot')(logits, token_ids, token_ids, dllm_mask.clone())

    # both spend the commit, but on different positions
    assert dynamic_mask.sum().item() == 1
    assert pivot_mask.sum().item() == 1
    assert dynamic_mask.argmax().item() == 0
    assert pivot_mask.argmax().item() == 2


def test_ripple_pivot_never_touches_unmasked_or_cached_positions():
    """Only masked positions may be committed.

    Entropies are [0.7083, 1.1421, 1.3762, 0.9404]; the two masked
    positions (1 and 3) sit on either side of the mid point, so the
    pivot is uniquely index 3. Positions 0 and 2 must be left alone.
    """
    logits = torch.log(torch.tensor([[0.8, 0.1, 0.05, 0.05, 0, 0, 0, 0],
                                     [0.5, 0.3, 0.15, 0.05, 0, 0, 0, 0],
                                     [0.3, 0.25, 0.25, 0.2, 0, 0, 0, 0],
                                     [0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0]]) + 1e-4)
    token_ids = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    dllm_mask = torch.tensor([1, 0, 2, 0], dtype=torch.uint8)

    out_mask, _ = _processor()(logits, token_ids, token_ids, dllm_mask)

    assert out_mask.tolist() == [1, 0, 2, 1]


def test_ripple_pivot_full_block_becomes_cached():
    """A block left fully unmasked is promoted to cached on the next call.

    This mirrors the cache transition the engine performs between steps.
    """
    confidence = [0.99, 0.99, 0.99, 0.99]
    logits = _logits_from_confidence(confidence)
    token_ids = torch.zeros(BLOCK, dtype=torch.long)
    dllm_mask = torch.tensor([1] * BLOCK, dtype=torch.uint8)

    out_mask, _ = _processor()(logits, token_ids, token_ids, dllm_mask)

    assert out_mask.tolist() == [2] * BLOCK


def test_pivot_distance_prefers_mid_entropy():
    """The pivot is the masked position closest to the block's mid entropy."""
    entropy = torch.tensor([[2.0, 0.5, 1.2, 0.1]])
    mask = torch.tensor([[0, 0, 0, 0]], dtype=torch.uint8)

    distance = pivot_distance(entropy, mask)

    assert torch.isfinite(distance).all()
    # mid of (0.1, 2.0) is 1.05 -> index 2 (1.2) is closest
    assert distance.argmin().item() == 2


def test_pivot_distance_ignores_unmasked_positions():
    """The mid point is taken over masked positions only.

    With index 0 unmasked the masked entropies are [0.5, 1.2, 0.1], so
    the mid point is 0.65 and index 1 is the closest.
    """
    entropy = torch.tensor([[2.0, 0.5, 1.2, 0.1]])
    mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.uint8)

    distance = pivot_distance(entropy, mask)

    assert torch.isinf(distance)[0].item()
    assert distance.argmin().item() == 1


def test_pivot_distance_of_a_block_with_nothing_masked_is_all_inf():
    """A fully decided block must not yield a nan pivot."""
    entropy = torch.tensor([[2.0, 0.5, 1.2, 0.1]])
    mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.uint8)

    distance = pivot_distance(entropy, mask)

    assert torch.isinf(distance).all()
    assert torch.isnan(distance).any() is False


def test_numpy_mask_dtype_is_supported():
    """The engine hands the mask over as a numpy-backed uint8 tensor."""
    confidence = [0.6, 0.6, 0.2, 0.2]
    logits = _logits_from_confidence(confidence)
    token_ids = torch.zeros(BLOCK, dtype=torch.long)
    dllm_mask = torch.from_numpy(np.zeros(BLOCK, dtype=np.uint8))

    out_mask, _ = _processor()(logits, token_ids, token_ids, dllm_mask)

    assert out_mask.dtype == torch.uint8
    assert out_mask.sum().item() >= 1
