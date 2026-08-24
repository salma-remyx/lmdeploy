# Copyright (c) OpenMMLab. All rights reserved.
"""Offline attention-aware KV cache bit allocation.

Runs lmdeploy's existing calibration pass, accumulates the per-channel second
moments that the attention-aware distortion decomposition needs, and produces
a per-layer reverse water-filling bit plan plus the whitening transform for
each K/V projection. The plan is written to ``<work_dir>/kv_bit_alloc.pth``
and a human-readable summary to stdout.

Adapted from Attention-Aware Transform Coding (AATC, arXiv:2608.14191): the
allocator and the transform are the paper's core contribution, evaluated
against lmdeploy's fixed-precision KV cache thread (int4 / fp8 / TurboQuant's
uniform 4-bit-K + 2-bit-V).
"""
from pathlib import Path

import torch

from lmdeploy.lite.apis.calibrate import load_model_and_tokenizer
from lmdeploy.lite.quantization.kv_bit_alloc import (
    ChannelDistortionStats,
    allocate_kv_bits,
    whitening_transform,
)
from lmdeploy.lite.utils import collect_target_modules, get_calib_loaders
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

# Projection names that produce K/V/Q states, keyed by the tag their captured
# activations are filed under. All supported architectures expose these as
# nn.Linear submodules of the decoder layer.
_PROJ_NAMES = ('k_proj', 'v_proj', 'q_proj')


def _kv_projections(model, layer_type: str) -> dict:
    """Map decoder-layer name to its ``{'k_proj': .., 'v_proj': .., ..}`` modules."""
    name2layer = collect_target_modules(model, layer_type)
    projections = {}
    for layer_name, layer in name2layer.items():
        layer_projs = {}
        for proj_name in _PROJ_NAMES:
            mod = getattr(layer, proj_name, None)
            if not isinstance(mod, torch.nn.Linear):
                raise RuntimeError(
                    f'{layer_name}.{proj_name} is not an nn.Linear; attention-aware '
                    f'KV bit allocation needs the standard q/k/v_proj layout.')
            layer_projs[proj_name] = mod
        projections[layer_name] = layer_projs
    if not projections:
        raise RuntimeError(f'no {layer_type} layers found; nothing to allocate over.')
    return projections


def _observe_projections(projections: dict) -> tuple[list, dict]:
    """Register forward hooks filing each projection's output under its tag.

    Returns ``(handles, captured)``, where ``captured[tag]`` holds one tensor
    per layer in layer order. The buffer is per-call, so repeated invocations
    in one process never see the previous run's activations.
    """
    captured = {name: [] for name in _PROJ_NAMES}
    handles = []

    def _make_hook(tag: str):
        def _hook(_mod, _inp, out):
            captured[tag].append(out.detach())
        return _hook

    for layer_projs in projections.values():
        for proj_name, mod in layer_projs.items():
            handles.append(mod.register_forward_hook(_make_hook(proj_name)))
    return handles, captured


def kv_bit_alloc(model: str,
                 calib_dataset: str = 'wikitext2',
                 calib_samples: int = 128,
                 calib_seqlen: int = 2048,
                 work_dir: str = './work_dir',
                 device: str = 'cuda',
                 k_bits: int = 4,
                 v_bits: int = 2,
                 max_channel_bits: int = 8,
                 min_channel_bits: int = 2,
                 dtype: str = 'auto',
                 batch_size: int = 1,
                 trust_remote_code: bool = False) -> dict:
    """Compute an attention-aware KV cache bit allocation for a model.

    Args:
        model: name or path of the model to allocate for.
        calib_dataset: calibration dataset name.
        calib_samples: number of calibration samples.
        calib_seqlen: sequence length per calibration sample.
        work_dir: output directory for ``kv_bit_alloc.pth``.
        device: device for the calibration forward passes.
        k_bits: average key bits per channel (the budget water-filling spends).
        v_bits: average value bits per channel.
        max_channel_bits: per-channel bit ceiling.
        min_channel_bits: per-channel bit floor.
        dtype: model load dtype.
        batch_size: calibration batch size.
        trust_remote_code: allow remote code when loading the model.

    Returns:
        The per-layer allocation table that was written to ``work_dir``.
    """
    from lmdeploy.lite.apis.calibrate import LAYER_TYPE_MAP

    _, _, hf_model, tokenizer, model_type, work_dir = load_model_and_tokenizer(
        model, dtype=dtype, work_dir=work_dir, trust_remote_code=trust_remote_code)
    if model_type not in LAYER_TYPE_MAP:
        raise RuntimeError(f'unsupported model type {model_type}')
    layer_type = LAYER_TYPE_MAP[model_type]

    projections = _kv_projections(hf_model, layer_type)
    num_layers = len(projections)
    logger.info('Allocating KV bits over %d %s layers.', num_layers, layer_type)

    k_proj = next(iter(projections.values()))['k_proj']
    num_heads = hf_model.config.num_key_value_heads
    head_dim = k_proj.out_features // num_heads

    handles, captured = _observe_projections(projections)
    try:
        calib_loader = get_calib_loaders(calib_dataset, tokenizer,
                                         nsamples=calib_samples,
                                         seqlen=calib_seqlen)
        all_data = torch.cat(calib_loader).to(device)
        with torch.inference_mode():
            hf_model(all_data)
    finally:
        for handle in handles:
            handle.remove()

    plan = {}
    num_captured = {tag: len(acts) for tag, acts in captured.items()}
    if len(set(num_captured.values())) != 1 or num_captured['k_proj'] == 0:
        raise RuntimeError('calibration did not capture matching q/k/v activations '
                           f'({num_captured}).')

    for layer_idx, (q_out, k_out, v_out) in enumerate(
            zip(captured['q_proj'], captured['k_proj'], captured['v_proj'])):
        stats = ChannelDistortionStats(num_heads, head_dim, head_dim)
        stats.observe(k_out, v_out, q=q_out)
        k_budget = num_heads * head_dim * k_bits
        v_budget = num_heads * head_dim * v_bits
        allocation = allocate_kv_bits(stats, k_budget, v_budget,
                                      max_bits=max_channel_bits,
                                      min_bits=min_channel_bits)
        basis, eigenvalues = whitening_transform(k_out.reshape(-1, k_proj.out_features).float())
        plan[f'layer_{layer_idx}'] = {
            'allocation': allocation,
            'whitening_basis': basis.cpu(),
            'k_eigenvalues': eigenvalues.cpu(),
        }

    out_path = Path(work_dir) / 'kv_bit_alloc.pth'
    torch.save(plan, out_path)
    logger.info('Wrote KV bit allocation for %d layers to %s', len(plan), out_path)

    for name, entry in plan.items():
        summary = entry['allocation'].summary()
        logger.info('%s: %.2fx compression, avg k/v bits %.2f/%.2f', name,
                    summary['compression_ratio'], summary['avg_k_bits'], summary['avg_v_bits'])
    return plan


if __name__ == '__main__':
    import fire

    fire.Fire(kv_bit_alloc)
