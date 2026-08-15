# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import pytest

from lmdeploy.model import BaseChatTemplate
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.serve.processors.incremental_tokenizer import IncrementalTokenizer


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class ReferenceTokenizer:
    """Minimal exact tokenizer used to test incremental repair.

    Implements a deterministic greedy longest-match BPE over a small piece
    vocabulary — the same contract the serving ``Tokenizer`` provides:
    text in -> ids out, ids decode back to the text.
    """

    def __init__(self):
        # pieces chosen so a merge ('▁ab') can straddle an append boundary
        # formed by its shorter prefixes ('▁a', '▁b')
        pieces = [
            '<s>', '▁', '▁a', '▁b', '▁ab', '▁hello', '▁world', '▁the', '▁tool', '▁result', 'c', 'd',
            '▁agent', '▁step', '.', '!', '▁x'
        ]
        self.vocab = {p: i for i, p in enumerate(pieces)}
        self.inv = {i: p for i, p in enumerate(pieces)}
        self.bos_token_id = self.vocab['<s>']
        self.eos_token_id = None

    def _match(self, s, i):
        """Longest piece matching at position i; a space maps to '▁'."""
        for length in (7, 6, 5, 4, 3, 2, 1):
            cand = s[i:i + length]
            if s[i] == ' ':
                # try the word-start marker spelling first
                piece = '▁' + cand[1:]
                if piece in self.vocab:
                    return self.vocab[piece], length
                if '▁' in self.vocab:
                    pass
            if cand in self.vocab:
                return self.vocab[cand], length
        return None, 0

    def encode(self, s, add_bos=True, add_special_tokens=True, **kwargs):
        ids = []
        i = 0
        while i < len(s):
            tid, length = self._match(s, i)
            if tid is None:
                raise AssertionError(f'unmapped character {s[i]!r} in test fixture')
            ids.append(tid)
            i += length
        if add_special_tokens and add_bos:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, t, offset=None, skip_special_tokens=True, **kwargs):
        pieces = []
        for i in t:
            if skip_special_tokens and i == self.bos_token_id:
                continue
            pieces.append(self.inv[i])
        return ''.join(pieces).replace('▁', ' ')


def _make_processor(enable=True):
    tok = ReferenceTokenizer()
    processor = MultimodalProcessor(tokenizer=tok,
                                    chat_template=BaseChatTemplate(capability='completion'),
                                    enable_incremental_tokenizer=enable)
    return processor, tok


class TestIncrementalTokenizerWiring:
    """The MultimodalProcessor call site routes through the incremental encoder."""

    def test_disabled_by_default(self):
        processor, _ = _make_processor(enable=False)
        assert processor.incremental_tokenizer is None

    def test_enabled_builds_incremental_encoder(self):
        processor, _ = _make_processor(enable=True)
        assert isinstance(processor.incremental_tokenizer, IncrementalTokenizer)

    def test_repair_matches_full_tokenization(self):
        processor, tok = _make_processor(enable=True)
        base = ' the agent'
        _run_async(processor.get_prompt_input(prompt=base, do_preprocess=True, session_key='sess-1'))
        grown = base + ' tool'
        out = _run_async(processor.get_prompt_input(prompt=grown, do_preprocess=True, session_key='sess-1'))
        assert out['input_ids'] == tok.encode(grown, add_bos=True)
        assert processor.incremental_tokenizer.stats['repairs'] >= 1


class TestIncrementalTokenizerExactness:
    """Emitted ids always equal full reference tokenization (the one contract)."""

    @staticmethod
    def _encode_pair(text_a, text_b, **kw):
        tok = ReferenceTokenizer()
        inc = IncrementalTokenizer(tok, **kw)
        first = inc.encode(text_a, session_key='s')
        second = inc.encode(text_b, session_key='s')
        return tok, inc, first, second

    def test_append_is_spliced_exactly(self):
        tok, inc, _, second = self._encode_pair(' the agent', ' the agent result')
        assert second == tok.encode(' the agent result', add_bos=True)
        assert inc.stats['repairs'] == 1

    def test_divergent_text_falls_back_to_full_encode(self):
        tok, inc, _, second = self._encode_pair(' hello', ' world')
        assert second == tok.encode(' world', add_bos=True)
        assert inc.stats['repairs'] == 0
        assert inc.stats['fallbacks'] == 1

    def test_stateless_call_matches_plain_encode(self):
        tok = ReferenceTokenizer()
        inc = IncrementalTokenizer(tok)
        assert inc.encode(' hello world', session_key=None) == tok.encode(' hello world', add_bos=True)

    def test_sessions_are_isolated(self):
        tok = ReferenceTokenizer()
        inc = IncrementalTokenizer(tok)
        inc.encode(' hello', session_key='a')
        assert inc.encode(' world', session_key='b') == tok.encode(' world', add_bos=True)

    def test_reset_clears_session_state(self):
        tok = ReferenceTokenizer()
        inc = IncrementalTokenizer(tok)
        inc.encode(' hello', session_key='a')
        inc.reset('a')
        assert inc.encode(' hello', session_key='a') == tok.encode(' hello', add_bos=True)

    def test_boundary_crossing_merge_stays_exact(self):
        # '▁ab' spans the append boundary of '▁a' + '▁b': the repaired
        # splice must still match full tokenization exactly.
        tok, inc, _, second = self._encode_pair(' a', ' ab')
        assert second == tok.encode(' ab', add_bos=True)
        assert inc.stats['repairs'] >= 1

    def test_repeated_appends_stay_exact(self):
        tok = ReferenceTokenizer()
        inc = IncrementalTokenizer(tok)
        transcript = ' the agent'
        repairs = 0
        for turn in (' step.', ' tool', ' result!'):
            transcript += turn
            got = inc.encode(transcript, session_key='s')
            assert got == tok.encode(transcript, add_bos=True)
            repairs = inc.stats['repairs']
        assert repairs >= 1  # at least one splice was served incrementally

    def test_lru_eviction_bounds_cache(self):
        tok = ReferenceTokenizer()
        inc = IncrementalTokenizer(tok, max_cache_entries=2)
        for i in range(3):
            inc.encode(' step', session_key=f's{i}')
        assert len(inc._cache) <= 2


if __name__ == '__main__':
    pytest.main([__file__])
