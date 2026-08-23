# Copyright (c) OpenMMLab. All rights reserved.
import os
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch import nn

# Import through the call-site module (existing code), not the new module
# alone, so these tests exercise the LinearBase.forward wiring.
from lmdeploy.pytorch import envs
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.nn.linear.base import LinearBase
from lmdeploy.pytorch.nn.linear.reduced_matmul import parse_rmm_env, select_topk_contraction


def make_linear(in_features=32, out_features=16, dtype=torch.float32, bias=True):
    """Build the lightest LinearBase, bypassing tp/dist setup."""
    layer = LinearBase.__new__(LinearBase)
    nn.Module.__init__(layer)
    layer.colwise = True
    layer.tp_align_size = 1
    layer.dp_gather = False
    layer.device = torch.device('cpu')
    layer.dtype = dtype
    layer.layer_type = 'attn'
    layer.lora_adapters = nn.ModuleDict()
    layer.is_tp = False
    layer.all_reduce = False
    layer.tp_rank = 0
    layer.tp = 1
    layer.tp_mode = TPMode.DEFAULT
    layer.tp_group = None
    layer.gather_group = None
    layer.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype), requires_grad=False)
    layer.bias = nn.Parameter(torch.randn(out_features, dtype=dtype), requires_grad=False) if bias else None

    def _forward_default(x, all_reduce, tp_sizes):
        return nn.functional.linear(x, layer.weight, layer.bias)

    layer._forward_default = _forward_default
    return layer


@contextmanager
def rmm_env(**kv):
    """Run the body with LMDEPLOY_RMM_* set and envs.rmm re-parsed."""
    saved = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        with patch.object(envs, 'rmm', parse_rmm_env(os.getenv)):
            yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_disabled_by_default_uses_full_matmul():
    layer = make_linear()
    x = torch.randn(4, 32)
    with rmm_env(LMDEPLOY_RMM_ENABLE='0', LMDEPLOY_RMM_KEEP_RATIO='0.5'):
        out = layer.forward(x)
    assert torch.allclose(out, nn.functional.linear(x, layer.weight, layer.bias))


def test_ratio_one_is_identity():
    layer = make_linear()
    x = torch.randn(4, 32)
    with rmm_env(LMDEPLOY_RMM_ENABLE='1', LMDEPLOY_RMM_KEEP_RATIO='1.0'):
        out = layer.forward(x)
    assert torch.allclose(out, nn.functional.linear(x, layer.weight, layer.bias))


def test_reduced_output_tracks_full():
    layer = make_linear()
    x = torch.randn(4, 32)
    with rmm_env(LMDEPLOY_RMM_ENABLE='1', LMDEPLOY_RMM_KEEP_RATIO='0.5'):
        out = layer.forward(x)
    full = nn.functional.linear(x, layer.weight, layer.bias)
    assert out.shape == full.shape
    # Retaining the slices with the largest magnitude keeps most of the
    # output energy, but is not the full matmul.
    assert torch.cosine_similarity(out.flatten(), full.flatten(), dim=0) > 0.3
    assert not torch.allclose(out, full)


def test_selection_ranks_slices_by_max_magnitude():
    """One slice dominates only for one token: it must still be retained."""
    x = torch.zeros(2, 2, 4)
    x[0, 0, 1] = -9.0  # feature 1 matters only here
    x[1, 1, 2] = 8.0   # feature 2 matters only here
    x[:, :, 0] = 1.0   # feature 0 is mildly active everywhere
    values, indices = select_topk_contraction(x, 0.5)
    assert indices.shape == (2, )  # batch-shared slice selection
    assert set(indices.tolist()) == {1, 2}
    assert values[0, 0].tolist() == [-9.0, 0.0]  # signs preserved


def test_layer_type_filter_falls_back():
    layer = make_linear()
    layer.layer_type = 'mlp'
    x = torch.randn(4, 32)
    with rmm_env(LMDEPLOY_RMM_ENABLE='1', LMDEPLOY_RMM_KEEP_RATIO='0.5',
                 LMDEPLOY_RMM_LAYER_TYPES='attn'):
        out = layer.forward(x)
    assert torch.allclose(out, nn.functional.linear(x, layer.weight, layer.bias))


def test_packed_weight_layout_is_skipped():
    """awq-style layers (no plain self.weight) must not be reduced."""
    layer = make_linear(bias=False)
    del layer.weight
    layer.qweight = nn.Parameter(torch.randint(0, 255, (16, 32)), requires_grad=False)
    x = torch.randn(4, 32)
    # A packed layout's GEMM reads qweight, not weight, so the fallback path
    # must be the one that runs.
    called = []

    def _packed_forward(x, all_reduce, tp_sizes):
        called.append(True)
        return torch.zeros(4, 16)

    layer._forward_default = _packed_forward
    with rmm_env(LMDEPLOY_RMM_ENABLE='1', LMDEPLOY_RMM_KEEP_RATIO='0.5'):
        out = layer.forward(x)
    assert called == [True]
    assert torch.equal(out, torch.zeros(4, 16))


def test_lora_adapters_are_not_reduced():
    """Layers with LoRA adapters read the full activations; RMM steps aside."""

    class _NoopAdapter(nn.Module):

        def forward(self, x, out):
            return out

    layer = make_linear()
    layer.lora_adapters['default'] = _NoopAdapter()
    x = torch.randn(4, 32)
    with rmm_env(LMDEPLOY_RMM_ENABLE='1', LMDEPLOY_RMM_KEEP_RATIO='0.5'):
        out = layer.forward(x)
    assert torch.allclose(out, nn.functional.linear(x, layer.weight, layer.bias))
