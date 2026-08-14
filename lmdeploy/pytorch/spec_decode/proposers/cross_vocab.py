# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.utils import get_logger

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..vocab_alignment import TokenVocabAligner
from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='cross_vocab')
class CrossVocabProposer(DeepseekMTP):
    """Speculative proposer for draft models whose vocabulary differs from
    the target model's.

    Adapted from TokenTiming (https://arxiv.org/abs/2510.15545): instead of
    requiring draft and target to share a vocabulary, draft tokens are
    aligned into target-vocab space through a DTW-based token alignment
    (:class:`TokenVocabAligner`), so any off-the-shelf draft model can
    speculate for the target without retraining.

    Boundary convention: every ``ModelInputs`` / ``draft_token_ids`` tensor
    outside this proposer lives in *target* vocabulary space, so the
    ``spec_agent`` / ``reject_sampler`` contract is untouched. Ids are
    mapped into draft-vocab space only at the draft-model boundary
    (:meth:`_forward`, :meth:`embed_input_ids`), and draft argmax ids are
    mapped back into target-vocab space in :meth:`get_outputs` before they
    reach the reject sampler.
    """

    def __init__(self, specdecode_config, device: torch.device = None):
        super().__init__(specdecode_config, device=device)
        self.vocab_aligner: TokenVocabAligner | None = None
        self._aligner_tables: dict = {}

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        """Build the draft model and the draft<->target vocab alignment."""
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        self._build_vocab_aligner()

    def _build_vocab_aligner(self):
        """Build the DTW vocab alignment from the two model tokenizers."""
        target_path = getattr(self.specdecode_config, 'target_model_path', None)
        if not target_path:
            logger.warning('cross_vocab: target model path unavailable; '
                           'falling back to identity token mapping.')
            return
        try:
            from transformers import AutoTokenizer
            draft_tokenizer = AutoTokenizer.from_pretrained(self.specdecode_config.model, trust_remote_code=True)
            target_tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
            aligner = TokenVocabAligner.from_tokenizers(draft_tokenizer, target_tokenizer)
        except Exception as exc:
            logger.warning(f'cross_vocab: failed to build vocab alignment ({exc}); '
                           'falling back to identity token mapping.')
            return
        if aligner.is_identity:
            logger.info('cross_vocab: draft and target vocabularies match; using identity token mapping.')
        self.set_vocab_aligner(aligner)

    def set_vocab_aligner(self, aligner: TokenVocabAligner):
        """Attach a precomputed aligner (used by build and by tests/tools)."""
        self.vocab_aligner = aligner
        self._aligner_tables = {}

    def _get_table(self, direction: str, device: torch.device) -> torch.Tensor:
        key = (direction, str(device))
        table = self._aligner_tables.get(key)
        if table is None:
            mapping = (self.vocab_aligner.draft_to_target
                       if direction == 'd2t' else self.vocab_aligner.target_to_draft)
            table = torch.tensor(mapping, dtype=torch.long, device=device)
            self._aligner_tables[key] = table
        return table

    def map_draft_to_target_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocab ids to target-vocab ids."""
        if self.vocab_aligner is None or self.vocab_aligner.is_identity:
            return ids
        table = self._get_table('d2t', ids.device)
        return table[ids.long().clamp(min=0, max=table.numel() - 1)]

    def map_target_to_draft_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Map target-vocab ids to draft-vocab ids."""
        if self.vocab_aligner is None or self.vocab_aligner.is_identity:
            return ids
        table = self._get_table('t2d', ids.device)
        return table[ids.long().clamp(min=0, max=table.numel() - 1)]

    def _map_inputs_to_draft(self, model_inputs: ModelInputs) -> ModelInputs:
        """Translate target-vocab input ids into draft-vocab space."""
        if self.vocab_aligner is None or self.vocab_aligner.is_identity or model_inputs.input_ids is None:
            return model_inputs
        return model_inputs.clone(input_ids=self.map_target_to_draft_ids(model_inputs.input_ids))

    def _forward(self, model_inputs: ModelInputs, cache_engine=None):
        """Forward the draft model on draft-vocab input ids."""
        return super()._forward(self._map_inputs_to_draft(model_inputs), cache_engine=cache_engine)

    def embed_input_ids(self, input_ids: torch.Tensor):
        """Embed target-vocab ids with the draft embedding table."""
        return super().embed_input_ids(self.map_target_to_draft_ids(input_ids))

    async def get_outputs(self,
                          model_outputs: dict[str, torch.Tensor],
                          model_inputs: ModelInputs,
                          extra_inputs: ARSpecExtraInputs = None,
                          guided_processors: dict | None = None):
        """Get outputs, mapping draft argmax ids into target-vocab space.

        Guided decoding is not supported for cross-vocabulary drafts (the
        grammar bitmask lives in target-vocab space while the draft logits
        live in draft-vocab space); guided processors are ignored.
        """
        if guided_processors:
            logger.warning('cross_vocab: guided decoding is not supported for '
                           'cross-vocabulary drafts; ignoring guided processors.')
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if extra_inputs is not None:
            hidden_states = hidden_states[:, extra_inputs.last_token_indices]
        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        target_token_ids = self.map_draft_to_target_ids(draft_token_ids)
        return target_token_ids, model_metas, hidden_states
