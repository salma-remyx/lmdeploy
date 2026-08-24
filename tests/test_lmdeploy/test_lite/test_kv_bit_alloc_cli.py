"""Integration test for the `lmdeploy lite kv_bit_alloc` CLI subcommand.

Exercises the wiring added to lmdeploy.cli.lite (parser registration and the
run handler) against the allocator in lmdeploy.lite, without needing a real
model on disk.
"""

import argparse

import torch

from lmdeploy.cli.lite import SubCliLite


def _run_handler(args: dict) -> dict:
    """Invoke SubCliLite.kv_bit_alloc through a real argparse Namespace.

    Going through argparse.Namespace keeps the real ``convert_args`` path in
    scope rather than bypassing it with a hand-rolled stand-in.
    """
    return SubCliLite.kv_bit_alloc(argparse.Namespace(**args))


def test_kv_bit_alloc_parser_registered():
    """The subcommand must be registered and accept its budget flags."""
    SubCliLite.add_parser_kv_bit_alloc()
    actions = SubCliLite.subparsers.choices['kv_bit_alloc']._actions
    names = {action.dest for action in actions}
    assert {'model', 'k_bits', 'v_bits', 'max_channel_bits', 'min_channel_bits'} <= names


def test_kv_bit_alloc_in_add_parsers():
    """add_parsers() must install the subcommand into the lite CLI."""
    try:
        SubCliLite.add_parsers()
    except argparse.ArgumentError:
        pass  # already registered by an earlier test in this session
    assert 'kv_bit_alloc' in SubCliLite.subparsers.choices


def test_cli_runs_allocation_end_to_end(tmp_path, monkeypatch):
    """The CLI handler must run a calibration and write the plan to disk."""
    from lmdeploy.lite.apis import kv_bit_alloc as api_module

    num_heads, num_q_heads, head_dim, num_layers, tokens = 2, 4, 8, 3, 128

    # collect_target_modules matches decoder layers by class name, so the
    # stub must carry the name LAYER_TYPE_MAP resolves for LlamaForCausalLM.
    class _Linear(torch.nn.Linear):
        pass

    class LlamaDecoderLayer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.q_proj = _Linear(16, num_q_heads * head_dim)
            self.k_proj = _Linear(16, num_heads * head_dim)
            self.v_proj = _Linear(16, num_heads * head_dim)

    class _Config:

        num_key_value_heads = num_heads

    class _Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(LlamaDecoderLayer() for _ in range(num_layers))

        def forward(self, x):
            # Only needs to fire each projection; outputs differ in width.
            return [(layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)) for layer in self.layers]

    model = _Model()
    model.config = _Config()
    k_proj = next(iter(model.layers)).k_proj

    monkeypatch.setattr(api_module, 'load_model_and_tokenizer',
                        lambda *a, **k: (None, None, model, None, 'LlamaForCausalLM', tmp_path))
    monkeypatch.setattr(api_module, 'get_calib_loaders',
                        lambda *a, **k: [torch.randn(tokens, k_proj.in_features)])

    out = _run_handler({
        'model': 'dummy',
        'work_dir': str(tmp_path),
        'calib_dataset': 'wikitext2',
        'calib_samples': 1,
        'calib_seqlen': tokens,
        'batch_size': 1,
        'dtype': 'auto',
        'trust_remote_code': False,
        'device': 'cpu',
        'k_bits': 4,
        'v_bits': 2,
        'max_channel_bits': 8,
        'min_channel_bits': 2,
    })

    # The CLI handler returns None (as the other lite subcommands do); the
    # deliverable is the plan written to work_dir.
    assert out is None
    assert (tmp_path / 'kv_bit_alloc.pth').exists()
    loaded = torch.load(tmp_path / 'kv_bit_alloc.pth', weights_only=False)
    assert set(loaded) == {f'layer_{i}' for i in range(num_layers)}
    for entry in loaded.values():
        assert entry['allocation'].k_bits.shape == (num_heads, head_dim)
        assert entry['allocation'].v_bits.shape == (num_heads, head_dim)
        assert entry['whitening_basis'].shape == (num_heads * head_dim, num_heads * head_dim)
        # Both budgets must be spent exactly, per the water-filling contract.
        assert entry['allocation'].k_bits.sum().item() == num_heads * head_dim * 4
        assert entry['allocation'].v_bits.sum().item() == num_heads * head_dim * 2
