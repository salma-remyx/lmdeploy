from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.nn import memory_probe
from lmdeploy.pytorch.nn.attention import Attention
from lmdeploy.pytorch.nn.memory_probe import AttnMemoryClass, attn_memory_classify, reset_attn_memory_ledger


def _paged_kv_caches(num_blocks=4, block_size=8, num_kv_heads=2, head_dim=16, corrupt=False):
    shape = (1, num_blocks, block_size, num_kv_heads, head_dim)
    k = torch.randn(*shape)
    v = torch.randn(*shape)
    if corrupt:
        # A slot-reuse style failure: one whole block written as a constant.
        k[:, 2] = 0.0
        v[:, 2] = float('nan')
    return k, v


def _attn_metadata(quant_policy=None, num_blocks=1):
    """Metadata whose block map claims ``num_blocks`` blocks starting at 0.

    The probe narrows a paged pool to the blocks this step actually reads, so
    a test that corrupts block ``i`` needs ``i`` inside this map.
    """
    return AttentionMetadata(
        is_decoding=True,
        block_offsets=torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0),
        q_seqlens=torch.tensor([1], dtype=torch.int32),
        kv_seqlens=torch.tensor([1], dtype=torch.int32),
        quant_policy=quant_policy,
    )


@pytest.fixture()
def probe_off(monkeypatch):
    monkeypatch.setattr(envs, 'attn_mem_probe_enable', False)


@pytest.fixture()
def probe_on(monkeypatch):
    monkeypatch.setattr(envs, 'attn_mem_probe_enable', True)
    return reset_attn_memory_ledger()


def _build_attention(monkeypatch, layer_id=3):
    """Build a real ``Attention`` layer with its backend impl stubbed out.

    The probe fires before the impl dispatch, so the integration under test is
    the ``Attention.forward`` wiring, not any particular backend.
    """
    import lmdeploy.pytorch.nn.attention as attention_mod

    captured = {}

    class _Impl:
        def forward(self, q, k, v, k_cache, v_cache, attn_metadata=None, **kwargs):
            captured['k_cache'] = k_cache
            captured['v_cache'] = v_cache
            return torch.zeros(1, 1, 4)

    class _Builder:
        @staticmethod
        def build(**kwargs):
            return _Impl()

    monkeypatch.setattr(attention_mod, 'get_backend',
                        lambda: SimpleNamespace(
                            get_layer_impl_builder=lambda op_type: _Builder()))
    layer = Attention(4, 16, num_kv_heads=2, layer_id=layer_id)
    return layer, captured


def test_probe_is_off_by_default(monkeypatch):
    monkeypatch.delenv('LMDEPLOY_ATTN_MEM_PROBE', raising=False)
    assert envs.attn_mem_probe_enable is False


def test_attention_forward_probes_cache_when_enabled(monkeypatch, probe_on):
    layer, _ = _build_attention(monkeypatch, layer_id=3)
    num_blocks = 4
    k_cache, v_cache = _paged_kv_caches(corrupt=True)
    query = torch.randn(1, 4, 16)
    key = torch.randn(1, 2, 16)
    value = torch.randn(1, 2, 16)

    layer(query, key, value, k_cache, v_cache, _attn_metadata(num_blocks=num_blocks))

    ledger = memory_probe.get_attn_memory_ledger()
    assert ledger.reads > 0
    assert ledger.findings > 0
    assert ledger.dominant_regime() == 'eviction_free'


def test_attention_forward_skips_probe_when_disabled(monkeypatch, probe_on):
    layer, _ = _build_attention(monkeypatch, layer_id=3)
    monkeypatch.setattr(envs, 'attn_mem_probe_enable', False)
    reset_attn_memory_ledger()
    blocks = 4
    k_cache, v_cache = _paged_kv_caches(corrupt=True)

    layer(torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16), k_cache,
          v_cache, _attn_metadata(num_blocks=blocks))

    ledger = memory_probe.get_attn_memory_ledger()
    assert ledger.reads == 0
    assert ledger.findings == 0


def test_attention_forward_clean_cache_holds_budget(monkeypatch, probe_on):
    layer, _ = _build_attention(monkeypatch, layer_id=0)
    blocks = 4
    k_cache, v_cache = _paged_kv_caches(corrupt=False)

    layer(torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16), k_cache,
          v_cache, _attn_metadata(num_blocks=blocks))

    ledger = memory_probe.get_attn_memory_ledger()
    assert ledger.findings == 0
    assert ledger.holds_budget(0.0)


def test_attention_forward_is_read_only(monkeypatch, probe_on):
    layer, captured = _build_attention(monkeypatch, layer_id=1)
    blocks = 4
    k_cache, v_cache = _paged_kv_caches(corrupt=True)
    k_before, v_before = k_cache.clone(), v_cache.clone()

    layer(torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16), k_cache,
          v_cache, _attn_metadata(num_blocks=blocks))

    # the probe must not rewrite what the impl is about to read. NaN != NaN,
    # so compare through the finite mask and the exact zero-block pattern
    # rather than value equality.
    assert torch.equal(torch.isfinite(v_cache), torch.isfinite(v_before))
    assert torch.equal(k_cache, k_before)
    # the impl still receives the original tensors, untouched
    assert captured['k_cache'] is k_cache
    assert captured['v_cache'] is v_cache


def test_ledger_regime_localizes_to_slot_reuse(monkeypatch, probe_on):
    layer, _ = _build_attention(monkeypatch, layer_id=5)
    blocks = 4
    k_cache, v_cache = _paged_kv_caches(corrupt=True)
    layer.attn_mem_regime = 'slot_reuse'

    layer(torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16), k_cache,
          v_cache, _attn_metadata(num_blocks=blocks))

    ledger = memory_probe.get_attn_memory_ledger()
    assert ledger.dominant_regime() == 'slot_reuse'
    assert ledger.regimes == {'slot_reuse': ledger.findings}


def test_ledger_tier_is_decided_by_the_machine(monkeypatch, probe_on):
    layer, _ = _build_attention(monkeypatch, layer_id=2)
    blocks = 64
    k_cache, v_cache = _paged_kv_caches(num_blocks=blocks, corrupt=True)

    layer(torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16), k_cache,
          v_cache, _attn_metadata(num_blocks=blocks))

    ledger = memory_probe.get_attn_memory_ledger()
    # every element in two blocks of sixty-four is unreadable -> beyond budget
    assert not ledger.holds_budget(1e-3)
    assert memory_probe._tier_for(1e-3, ledger.finding_rate) == 'empirical'
    assert memory_probe._tier_for(1e-3, 0.0) == 'certified'
    assert memory_probe._tier_for(1e-3, 1e-4) == 'partially_certified'


def test_layer_subset_limits_probe_scope(monkeypatch, probe_on):
    monkeypatch.setattr(envs, 'attn_mem_probe_layers', [7])
    layer, _ = _build_attention(monkeypatch, layer_id=3)
    blocks = 4
    k_cache, v_cache = _paged_kv_caches(corrupt=True)

    layer(torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16), k_cache,
          v_cache, _attn_metadata(num_blocks=blocks))

    assert memory_probe.get_attn_memory_ledger().reads == 0


@pytest.mark.parametrize(
    'shape,kwargs,expected',
    [
        ((1, 4, 8, 2, 16), {}, AttnMemoryClass.PAGED_KV),
        ((1024, 2, 1), {}, AttnMemoryClass.V4_COMPRESSED_KV),
        ((1024, 2, 16), {}, AttnMemoryClass.NSA_INDEX),
        ((1024, 2, 16), {'nsa_indices': torch.zeros(1, 1, 8)}, AttnMemoryClass.NSA_INDEX),
        ((1024, 2, 16), {'state_like': True}, AttnMemoryClass.RECURRENT_STATE),
    ])
def test_attn_memory_classify(shape, kwargs, expected):
    assert attn_memory_classify(torch.empty(*shape), **kwargs) is expected


def test_probe_reports_finite_findings_for_index_pool(monkeypatch, probe_on):
    monkeypatch.setattr(envs, 'attn_mem_probe_layers', [])
    k_cache = torch.randn(64, 2, 16)
    k_cache[3] = float('inf')
    v_cache = torch.randn(64, 2, 16)

    findings = memory_probe.probe_attention_memory(k_cache, v_cache, 0)

    assert findings > 0
    ledger = memory_probe.get_attn_memory_ledger()
    assert ledger.regimes['eviction_free'] == findings


def test_paged_probe_reads_only_live_blocks(monkeypatch, probe_on):
    """An unwritten block outside the step's block map is not a finding.

    The pool handed to ``Attention.forward`` covers every block the engine
    owns; only the blocks listed in ``block_offsets`` are claimed by a
    sequence this step, so only those are in scope for the probe.
    """
    num_blocks, block_size, heads, dim = 8, 4, 2, 16
    k_cache = torch.randn(1, num_blocks, block_size, heads, dim)
    v_cache = torch.randn(1, num_blocks, block_size, heads, dim)
    # blocks 5 and 6 belong to no sequence this step
    k_cache[:, 5] = 0.0
    k_cache[:, 6] = 0.0
    block_offsets = torch.tensor([[0, 1, 2, 3, 4, 0, 0, 0]], dtype=torch.int32)

    findings = memory_probe.probe_attention_memory(k_cache,
                                                   v_cache,
                                                   0,
                                                   block_offsets=block_offsets)

    assert findings == 0


def test_paged_probe_flags_corrupt_live_block(monkeypatch, probe_on):
    num_blocks, block_size, heads, dim = 8, 4, 2, 16
    k_cache = torch.randn(1, num_blocks, block_size, heads, dim)
    v_cache = torch.randn(1, num_blocks, block_size, heads, dim)
    # block 3 is live (in block_offsets) and corrupt
    k_cache[:, 3] = 0.0
    v_cache[:, 3] = float('nan')
    # block 7 is equally corrupt but not live
    k_cache[:, 7] = 0.0
    block_offsets = torch.tensor([[0, 1, 2, 3, 4, 0, 0, 0]], dtype=torch.int32)

    findings = memory_probe.probe_attention_memory(k_cache,
                                                   v_cache,
                                                   0,
                                                   block_offsets=block_offsets)

    assert findings > 0
    # the degenerate k block contributes one finding per slot, not per element
    live_elements = k_cache[:, 3].numel()
    assert findings < 2 * live_elements
