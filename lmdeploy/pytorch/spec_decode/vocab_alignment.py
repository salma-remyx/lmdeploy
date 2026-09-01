# Copyright (c) OpenMMLab. All rights reserved.
"""Cross-vocabulary token alignment for speculative decoding.

Implements the DTW-inspired token alignment from `TokenTiming: A Dynamic
Alignment Method for Universal Speculative Decoding Model Pairs`
(https://arxiv.org/abs/2510.15545), adapted to lmdeploy's spec-decode
stack.

The paper re-encodes each drafted token sequence into the target
vocabulary and aligns the two tokenizations with Dynamic Time Warping
(DTW), so that a draft model can speculate for a target model with a
different tokenizer. lmdeploy's spec-decode engine runs the draft model in
position lockstep with the target token stream (shared ``seq_length`` /
``history_lengths``), so the alignment is built once over the two
*vocabularies* instead of per sequence: every draft token id is decoded to
its string and aligned to the best-matching target token id (exact string
match first, minimum DTW character distance otherwise). The resulting
static tables map draft ids into target-vocab space (for verification by
the reject sampler) and target ids back into draft-vocab space (for
feeding accepted tokens to the draft model).

This module is pure python so the tables can be built and validated
without torch; the proposer converts them to tensors at load time.
"""

from __future__ import annotations

from collections.abc import Sequence


def dtw_char_cost(source: str, target: str) -> int:
    """DTW distance between two strings over characters.

    Unit substitution and insertion/deletion costs, following the classic
    DTW recurrence TokenTiming uses to measure token-string mismatch.
    """
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)
    prev = list(range(len(target) + 1))
    for i, sc in enumerate(source, start=1):
        curr = [i]
        for j, tc in enumerate(target, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (sc != tc)))
        prev = curr
    return prev[-1]


def _looks_special(token: str) -> bool:
    """Heuristic for special/control tokens (e.g. ``<|im_start|>``)."""
    return len(token) >= 2 and token.startswith('<') and token.endswith('>')


def _fuzzy_match(source: str, target_tokens: Sequence[str], buckets: dict[str, list[int]], fallback_id: int,
                 max_candidates: int) -> int:
    """Find the target id minimizing the DTW character cost to ``source``.

    Candidates share the source's leading character (cross-vocab pairs
    are usually same-language, so aligned tokens share their first
    character after decode-normalization). A candidate is accepted only
    when its DTW cost is strictly below ``max(len(source), len(target))``,
    i.e. the alignment exploits at least one shared character; otherwise
    the source token is unmappable and falls back to ``fallback_id``.
    """
    candidates = buckets.get(source[:1])
    if not candidates:
        return fallback_id
    if len(candidates) > max_candidates:
        # Bound build time on very large buckets: prefer length-similar
        # tokens, which DTW favors anyway (cost is lower-bounded by the
        # length gap).
        candidates = sorted(candidates, key=lambda t: abs(len(target_tokens[t]) - len(source)))[:max_candidates]
    best_id = fallback_id
    best_cost: int | None = None
    n = len(source)
    for tid in candidates:
        target = target_tokens[tid]
        if best_cost is not None and abs(len(target) - n) >= best_cost:
            continue
        cost = dtw_char_cost(source, target)
        if best_cost is None or cost < best_cost:
            best_id, best_cost = tid, cost
    if best_cost is None or best_cost >= max(n, len(target_tokens[best_id])):
        return fallback_id
    return best_id


def build_id_mapping(source_tokens: Sequence[str],
                     target_tokens: Sequence[str],
                     fallback_id: int = 0,
                     max_candidates: int = 2048) -> list[int]:
    """Map every source token id to the best-aligned target token id.

    Args:
        source_tokens (Sequence[str]): Source token strings indexed by
            source token id (decode-normalized).
        target_tokens (Sequence[str]): Target token strings indexed by
            target token id.
        fallback_id (int): Target id used when no alignment is found
            (e.g. the target tokenizer's ``unk_token_id``).
        max_candidates (int): Cap on fuzzy-alignment candidates per source
            token, to bound build time on large vocabularies.

    Returns:
        list[int]: ``mapping[source_id] = target_id``.
    """
    exact: dict[str, int] = {}
    buckets: dict[str, list[int]] = {}
    for tid, token in enumerate(target_tokens):
        exact.setdefault(token, tid)
        if token:
            buckets.setdefault(token[:1], []).append(tid)

    mapping = []
    for token in source_tokens:
        target_id = exact.get(token)
        if target_id is None:
            if not token or _looks_special(token):
                target_id = fallback_id
            else:
                target_id = _fuzzy_match(token, target_tokens, buckets, fallback_id, max_candidates)
        mapping.append(target_id)
    return mapping


def _tokenizer_token_strings(tokenizer) -> list[str]:
    """Decode-normalized string of every token id in a HF tokenizer."""
    return [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]


class TokenVocabAligner:
    """Bidirectional token-id mapping between draft and target vocabularies.

    Attributes:
        draft_to_target (list[int]): ``draft_to_target[draft_id]`` is the
            aligned target token id.
        target_to_draft (list[int]): ``target_to_draft[target_id]`` is the
            aligned draft token id.
    """

    def __init__(self, draft_to_target: Sequence[int], target_to_draft: Sequence[int]):
        self.draft_to_target = [int(i) for i in draft_to_target]
        self.target_to_draft = [int(i) for i in target_to_draft]

    @classmethod
    def from_vocab_strings(cls,
                           draft_tokens: Sequence[str],
                           target_tokens: Sequence[str],
                           draft_fallback_id: int = 0,
                           target_fallback_id: int = 0,
                           max_candidates: int = 2048) -> 'TokenVocabAligner':
        """Build the aligner from decode-normalized token strings."""
        return cls(
            draft_to_target=build_id_mapping(draft_tokens,
                                             target_tokens,
                                             fallback_id=target_fallback_id,
                                             max_candidates=max_candidates),
            target_to_draft=build_id_mapping(target_tokens,
                                             draft_tokens,
                                             fallback_id=draft_fallback_id,
                                             max_candidates=max_candidates),
        )

    @classmethod
    def from_tokenizers(cls, draft_tokenizer, target_tokenizer, max_candidates: int = 2048) -> 'TokenVocabAligner':
        """Build the aligner from two HuggingFace tokenizers."""
        draft_tokens = _tokenizer_token_strings(draft_tokenizer)
        target_tokens = _tokenizer_token_strings(target_tokenizer)
        return cls.from_vocab_strings(
            draft_tokens,
            target_tokens,
            draft_fallback_id=draft_tokenizer.unk_token_id or 0,
            target_fallback_id=target_tokenizer.unk_token_id or 0,
            max_candidates=max_candidates,
        )

    @property
    def is_identity(self) -> bool:
        """Whether both mappings are the identity (vocabs already aligned)."""
        return (self.draft_to_target == list(range(len(self.draft_to_target)))
                and self.target_to_draft == list(range(len(self.target_to_draft))))

    @staticmethod
    def _map(table: list[int], token_ids) -> list[int]:
        last = len(table) - 1
        return [table[min(max(int(i), 0), last)] for i in token_ids]

    def map_draft_to_target(self, token_ids) -> list[int]:
        """Map draft-vocab token ids to target-vocab ids (out-of-range ids
        are clamped)."""
        return self._map(self.draft_to_target, token_ids)

    def map_target_to_draft(self, token_ids) -> list[int]:
        """Map target-vocab token ids to draft-vocab ids (out-of-range ids
        are clamped)."""
        return self._map(self.target_to_draft, token_ids)
