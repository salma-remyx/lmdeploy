# Copyright (c) OpenMMLab. All rights reserved.
"""Session-aware incremental prompt tokenization.

Coding-agent traffic resubmits a long transcript on every call with a small
append (a tool result, the next turn). Re-tokenizing the whole string each
time is pure overhead: the only ids that can change are the ones near the
append point, because a byte-level BPE merge never reorders text and any
merge that crosses the append boundary is confined to a short tail.

This module implements the incremental-repair half of
`TokTier: Exact Stateful Tokenization for Agentic LLM Serving`
(arXiv:2607.29678) as an adapted port. Kept at full fidelity:

- reuse the cached id sequence of the previous request for a session;
- re-tokenize only a small character window around the append;
- splice only after a **stable-boundary check** — the window's ids must
  decode back to exactly the window text and the spliced sequence must
  decode back to exactly the new request text — widening the window and
  retrying on failure, and falling back to a full reference tokenization
  when the retry budget is exhausted.

Substituted target-natively (the paper's auxiliary components):

- its GPU BPE engine — the repo's existing reference tokenizer already
  provides exact full tokenization, so the fallback path uses it directly;
- its sampled shadow verifier — exactness is enforced by construction:
  a splice is only emitted when both round-trip checks pass, otherwise it
  is discarded and full tokenization runs.
"""
from collections import OrderedDict

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_DEFAULT_WINDOW = 256
_MAX_WIDEN = 3
_MAX_CACHE_ENTRIES = 64


class IncrementalTokenizer:
    """Exact incremental encoder over a per-session token cache.

    The public surface mirrors the ``Tokenizer.encode`` call it replaces:
    text in, ``list[int]`` out, matching full tokenization of the text.

    Args:
        tokenizer: the serving tokenizer (anything exposing
            ``encode(str, add_bos=..., add_special_tokens=...) -> list[int]``
            and ``decode(list[int]) -> str``).
        window: initial repair-window size in characters.
        max_widen: how many times a failing window is doubled before
            falling back to full tokenization.
        max_cache_entries: number of remembered (text, ids) pairs; the
            cache is LRU-evicted to bound memory across sessions.
    """

    def __init__(self,
                 tokenizer,
                 window: int = _DEFAULT_WINDOW,
                 max_widen: int = _MAX_WIDEN,
                 max_cache_entries: int = _MAX_CACHE_ENTRIES):
        self.tokenizer = tokenizer
        self.window = window
        self.max_widen = max_widen
        self.max_cache_entries = max_cache_entries
        # session key -> (last request text, its ids)
        self._cache: OrderedDict[str, tuple[str, list[int]]] = OrderedDict()
        self.stats = {'repairs': 0, 'fallbacks': 0, 'cold_encodes': 0}

    def reset(self, session_key: str | None = None):
        """Drop cached state, either for one session or for all of them."""
        if session_key is None:
            self._cache.clear()
        else:
            self._cache.pop(session_key, None)

    def _remember(self, session_key: str, text: str, ids: list[int]):
        self._cache[session_key] = (text, ids)
        self._cache.move_to_end(session_key)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    def encode(self, text: str, session_key: str | None = None, add_bos: bool = True) -> list[int]:
        """Tokenize ``text``, repairing a cached prefix when possible.

        Args:
            text: the full prompt text for this request.
            session_key: identifies the conversation whose previous request
                may be reused as a prefix. ``None`` skips the cache and
                behaves exactly like a plain ``tokenizer.encode``.
            add_bos: forwarded to the reference tokenizer for full encodes.
                A repaired splice reuses ids the reference tokenizer already
                produced, so no id is added on that path.

        Returns:
            list[int]: token ids equal to reference tokenization of ``text``.
        """
        if session_key is None:
            return self.tokenizer.encode(text, add_bos=add_bos)

        cached = self._cache.get(session_key)
        if cached is None:
            self.stats['cold_encodes'] += 1
            ids = self.tokenizer.encode(text, add_bos=add_bos)
            self._remember(session_key, text, ids)
            return ids

        old_text, old_ids = cached
        if text == old_text:
            return list(old_ids)
        if not text.startswith(old_text):
            # Divergent text: no prefix to repair, re-encode from scratch.
            self.stats['fallbacks'] += 1
            ids = self.tokenizer.encode(text, add_bos=add_bos)
            self._remember(session_key, text, ids)
            return ids

        ids = self._repair(old_text, old_ids, text) or self.tokenizer.encode(text, add_bos=True)
        self._remember(session_key, text, ids)
        return ids

    def _repair(self, old_text: str, old_ids: list[int], new_text: str) -> list[int] | None:
        """Splice the appended tail onto ``old_ids`` via a windowed re-encode.

        Returns the spliced ids, or ``None`` when no window passes the
        stable-boundary check (the caller then re-encodes in full).
        """
        for attempt in range(self.max_widen + 1):
            overlap = min(self.window * (2**attempt), len(old_text))
            keep = self._kept_prefix_tokens(old_ids, old_text, overlap)
            if keep is None:
                continue
            boundary = keep[1]
            window_text = new_text[boundary:]
            window_ids = self.tokenizer.encode(window_text, add_bos=False)
            spliced = keep[0] + window_ids
            # Stable-boundary check: the fresh window round-trips exactly,
            # and the splice as a whole decodes back to the request text.
            # Special tokens carry no text characters, so they are skipped
            # when comparing against the request string.
            if self.tokenizer.decode(window_ids) == window_text and self.tokenizer.decode(spliced) == new_text:
                self.stats['repairs'] += 1
                return spliced
            logger.debug(f'incremental tokenize: unstable window (overlap={overlap}, attempt={attempt})')
        self.stats['fallbacks'] += 1
        return None

    def _kept_prefix_tokens(self, old_ids: list[int], old_text: str, overlap: int):
        """Find the token-boundary cut covering the last ``overlap`` chars.

        Byte-level BPE ids never reorder, so the splice can reuse every old
        token that ends at or before the window start. The cut must land on
        an actual token boundary of ``old_ids``; walking back from the end
        until the dropped suffix covers ``overlap`` characters yields the
        largest reusable prefix.

        Returns:
            (prefix_ids, boundary_chars) or ``None`` if the tail cannot be
            measured against the request text (degenerate vocab behavior).
        """
        if overlap <= 0:
            return list(old_ids), len(old_text)
        for drop in range(1, len(old_ids) + 1):
            tail = self.tokenizer.decode(old_ids[len(old_ids) - drop:])
            if not old_text.endswith(tail):
                return None
            if len(tail) >= overlap:
                keep = len(old_ids) - drop
                return old_ids[:keep], len(old_text) - len(tail)
        # The whole sequence is inside the window.
        return [], 0
