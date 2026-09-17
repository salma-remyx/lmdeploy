# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import importlib.util
from pathlib import Path

import pytest

# The alignment core is pure python; load it by path so these unit tests
# also run in environments without the torch/HF stack installed.
_MODULE_PATH = Path(__file__).parents[3] / 'lmdeploy/pytorch/spec_decode/vocab_alignment.py'
_spec = importlib.util.spec_from_file_location('vocab_alignment', _MODULE_PATH)
vocab_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vocab_alignment)

TokenVocabAligner = vocab_alignment.TokenVocabAligner
build_id_mapping = vocab_alignment.build_id_mapping
dtw_char_cost = vocab_alignment.dtw_char_cost


class TestDTWCharCost:

    def test_identical_strings_cost_zero(self):
        assert dtw_char_cost('hello', 'hello') == 0

    def test_empty_strings(self):
        assert dtw_char_cost('', '') == 0
        assert dtw_char_cost('', 'abc') == 3
        assert dtw_char_cost('abc', '') == 3

    def test_single_substitution(self):
        assert dtw_char_cost('abc', 'abd') == 1

    def test_insertion_and_deletion(self):
        assert dtw_char_cost('abc', 'abcd') == 1
        assert dtw_char_cost('abcd', 'abc') == 1

    def test_levenshtein_reference(self):
        # classic Levenshtein example; unit-cost DTW over chars matches it
        assert dtw_char_cost('kitten', 'sitting') == 3

    def test_disjoint_strings_pay_full_substitution(self):
        assert dtw_char_cost('abc', 'xyz') == 3


class TestBuildIdMapping:

    def test_exact_matches(self):
        mapping = build_id_mapping(['a', 'b', 'c'], ['x', 'b', 'a'], fallback_id=0)
        assert mapping == [2, 1, 0]  # 'c' has no target -> fallback

    def test_fuzzy_match_via_dtw(self):
        # 'hello' aligns to 'hallo' (cost 1 < max(5, 5))
        mapping = build_id_mapping(['hello'], ['hallo', 'world'], fallback_id=0)
        assert mapping == [0]

    def test_no_shared_prefix_falls_back(self):
        mapping = build_id_mapping(['abc'], ['xyz'], fallback_id=7)
        assert mapping == [7]

    def test_special_tokens_never_fuzzy_match(self):
        mapping = build_id_mapping(['<|im_start|>'], ['hello'], fallback_id=3)
        assert mapping == [3]

    def test_special_tokens_exact_match(self):
        mapping = build_id_mapping(['<|im_start|>'], ['a', '<|im_start|>'], fallback_id=0)
        assert mapping == [1]

    def test_duplicate_target_strings_take_first_id(self):
        mapping = build_id_mapping(['a'], ['a', 'a'], fallback_id=0)
        assert mapping == [0]


class _FakeTokenizer:
    """Minimal HF-tokenizer stand-in for the from_tokenizers seam."""

    def __init__(self, tokens, unk_token_id=0):
        self._tokens = tokens
        self.unk_token_id = unk_token_id

    def __len__(self):
        return len(self._tokens)

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return ''.join(self._tokens[i] for i in ids)


class TestTokenVocabAligner:

    def test_bidirectional_mapping(self):
        draft = ['a', 'b', 'hello']
        target = ['b', 'a', 'hallo']
        aligner = TokenVocabAligner.from_vocab_strings(draft, target)
        assert aligner.draft_to_target == [1, 0, 2]
        assert aligner.target_to_draft == [1, 0, 2]

    def test_map_helpers_clamp_out_of_range(self):
        aligner = TokenVocabAligner(draft_to_target=[4, 5], target_to_draft=[1, 0])
        assert aligner.map_draft_to_target([0, 1, 99]) == [4, 5, 5]
        assert aligner.map_target_to_draft([0, 1, 99]) == [1, 0, 0]

    def test_is_identity(self):
        identity = TokenVocabAligner([0, 1, 2], [0, 1])
        assert identity.is_identity
        shifted = TokenVocabAligner([1, 0, 2], [0, 1])
        assert not shifted.is_identity

    def test_from_tokenizers(self):
        draft_tok = _FakeTokenizer(['<unk>', 'a', 'hello'])
        target_tok = _FakeTokenizer(['<unk>', 'a', 'hallo'])
        aligner = TokenVocabAligner.from_tokenizers(draft_tok, target_tok)
        # '<unk>' is special -> fallback to target unk id; 'a' exact;
        # 'hello' -> 'hallo' via DTW
        assert aligner.draft_to_target == [0, 1, 2]


@pytest.fixture()
def cross_vocab_proposer():
    """Build a CrossVocabProposer through the spec-decode registry."""
    pytest.importorskip('torch')
    pytest.importorskip('mmengine')
    from lmdeploy.pytorch.config import SpecDecodeConfig
    from lmdeploy.pytorch.spec_decode.proposers.base import build_specdecode_proposer
    config = SpecDecodeConfig(model='dummy-draft',
                              method='cross_vocab',
                              num_speculative_tokens=2,
                              target_model_path='dummy-target')
    return build_specdecode_proposer(config, device='cpu')


class TestCrossVocabProposerWiring:
    """Integration with the existing spec-decode registry and config."""

    def test_registered_in_spec_proposers(self):
        pytest.importorskip('torch')
        from lmdeploy.pytorch.spec_decode.proposers import CrossVocabProposer  # noqa F401
        from lmdeploy.pytorch.spec_decode.proposers.base import SPEC_PROPOSERS
        assert 'cross_vocab' in SPEC_PROPOSERS.module_dict
        assert SPEC_PROPOSERS.module_dict['cross_vocab'] is CrossVocabProposer

    def test_config_carries_target_model_path(self, cross_vocab_proposer):
        assert cross_vocab_proposer.specdecode_config.target_model_path == 'dummy-target'

    def test_config_target_model_path_defaults_to_none(self):
        pytest.importorskip('torch')
        from lmdeploy.pytorch.config import SpecDecodeConfig
        assert SpecDecodeConfig(model='m', method='cross_vocab').target_model_path is None

    def test_get_outputs_maps_draft_argmax_to_target(self, cross_vocab_proposer):
        import torch
        proposer = cross_vocab_proposer
        proposer.set_vocab_aligner(TokenVocabAligner(draft_to_target=[0, 7, 2], target_to_draft=[0, 1, 2]))

        def fake_get_logits(hidden_states):
            logits = torch.full((1, hidden_states.size(1), 3), -100.0)
            logits[0, :, 1] = 100.0  # draft argmax is always draft id 1
            return logits

        proposer.get_logits = fake_get_logits
        model_outputs = {'hidden_states': torch.zeros(1, 2, 4), 'model_metas': None}
        target_ids, model_metas, hidden = asyncio.run(proposer.get_outputs(model_outputs, None))
        # draft id 1 aligns to target id 7; reject sampler only sees target ids
        assert target_ids.tolist() == [[7], [7]]
        assert model_metas is None
        assert hidden.shape == (1, 2, 4)

    def test_map_target_to_draft_ids_clamps(self, cross_vocab_proposer):
        import torch
        proposer = cross_vocab_proposer
        proposer.set_vocab_aligner(TokenVocabAligner(draft_to_target=[0, 1], target_to_draft=[1, 0]))
        ids = torch.tensor([[0], [5]])  # 5 is beyond the target table
        assert proposer.map_target_to_draft_ids(ids).tolist() == [[1], [0]]

    def test_identity_aligner_passes_ids_through(self, cross_vocab_proposer):
        import torch
        proposer = cross_vocab_proposer
        proposer.set_vocab_aligner(TokenVocabAligner([0, 1], [0, 1]))
        ids = torch.tensor([[0], [1]])
        assert proposer.map_draft_to_target_ids(ids) is ids
        assert proposer.map_target_to_draft_ids(ids) is ids
