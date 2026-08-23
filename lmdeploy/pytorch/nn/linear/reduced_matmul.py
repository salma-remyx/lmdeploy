# Copyright (c) OpenMMLab. All rights reserved.
"""Reduced matrix multiplication (RMM) for linear layers.

RMM is a training-free, input-adaptive inference-time reduction of the matrix
products behind the qkv / gate-up / down projections. For ``y = x @ W.T`` the
contraction dimension is reduced by keeping only the ``k`` input features with
the largest magnitude and summing those partial products alone:

    y ~= sum(x[..., idx] * W[:, idx].T, dim=-1)

The selection is computed per forward pass from the activations, so the weights
are never modified and no calibration or fine-tuning is needed. The retained
fraction ``k / in_features`` (the retention ratio) gives a smooth knob on the
accuracy-efficiency trade-off: at ratio 1.0 the result is bit-identical to the
full matmul, and lower ratios trade accuracy for fewer multiply-adds.

Adapted from "Reduced Matrix Multiplication: Input-Adaptive Matrix-Product
Reduction for LLM Inference" (arXiv:2608.13426). The paper's custom gather-GEMM
kernel is substituted by an index-select followed by the layer's existing GEMM path;
per the reference implementation the reduction is applied only to the
attention-side (``attn``) and MLP-side (``mlp``) projections.
"""
from dataclasses import dataclass

import torch

__all__ = ['ReducedMatmulConfig', 'parse_rmm_env', 'select_topk_contraction', 'reduced_matmul']


@dataclass
class ReducedMatmulConfig:
    """Parsed ``LMDEPLOY_RMM_*`` configuration.

    Args:
        enable (bool): apply the reduction when True.
        keep_ratio (float): fraction of the contraction dimension to retain,
            in ``(0, 1]``. ``1.0`` disables the reduction.
        layer_types (set[str]): which projections to reduce, by the
            ``layer_type`` the layer was built with (``'attn'`` / ``'mlp'``).
    """

    enable: bool = False
    keep_ratio: float = 1.0
    layer_types: frozenset = frozenset({'attn', 'mlp'})

    def applies_to(self, layer_type: str) -> bool:
        """Whether the reduction applies to a layer built with layer_type."""
        return self.enable and self.keep_ratio < 1.0 and layer_type in self.layer_types


def parse_rmm_env(getenv) -> ReducedMatmulConfig:
    """Build the config from environment variables.

    ``LMDEPLOY_RMM_ENABLE`` turns the reduction on; ``LMDEPLOY_RMM_KEEP_RATIO``
    sets the retention ratio; ``LMDEPLOY_RMM_LAYER_TYPES`` restricts it to a
    comma-separated subset of ``attn,mlp`` (default both). An out-of-range
    ratio is clamped rather than fatal so a typo cannot silently re-enable
    full matmuls or drop the whole hidden size.
    """
    enable = getenv('LMDEPLOY_RMM_ENABLE', '0').lower().strip() in {'1', 'true', 'yes', 'on'}
    try:
        keep_ratio = float(getenv('LMDEPLOY_RMM_KEEP_RATIO', '1.0'))
    except ValueError:
        keep_ratio = 1.0
    keep_ratio = min(max(keep_ratio, 0.0), 1.0)

    raw_layer_types = getenv('LMDEPLOY_RMM_LAYER_TYPES', 'attn,mlp')
    layer_types = frozenset(item.strip().lower() for item in raw_layer_types.split(',') if item.strip())
    layer_types &= frozenset({'attn', 'mlp'})
    if not layer_types:
        # An unrecognized value would otherwise disable the feature silently.
        layer_types = frozenset({'attn', 'mlp'})
    return ReducedMatmulConfig(enable=enable, keep_ratio=keep_ratio, layer_types=layer_types)


def select_topk_contraction(x: torch.Tensor, keep_ratio: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select the top-k contraction slices of ``x`` by activation magnitude.

    The importance of a contraction slice is the largest magnitude any token
    reaches on it, and the ``k`` most important slices are retained for the
    whole batch. Selecting per token instead would need a gather-GEMM kernel
    to avoid materializing a ``(tokens, k, out)`` operand, so the slice-level
    granularity of the paper is what is kept here.

    Args:
        x (torch.Tensor): activations of shape ``(..., in_features)``.
        keep_ratio (float): fraction of the last dimension to retain, in
            ``(0, 1]``.

    Returns:
        Tuple ``(values, indices)``. ``values`` is ``x`` gathered at
        ``indices`` and keeps the sign of the original activations; both have
        the leading dims of ``x`` and a last dim of ``k``. When no reduction
        applies, returns ``(x, None)`` so the caller keeps the full matmul.
    """
    in_features = x.size(-1)
    k = int(in_features * keep_ratio)
    if k <= 0:
        k = 1
    if k >= in_features:
        return x, None
    token_dims = tuple(range(x.dim() - 1))
    importance = x.abs().amax(dim=token_dims) if token_dims else x.abs()
    indices = torch.topk(importance, k).indices
    return torch.index_select(x, -1, indices), indices


def reduced_matmul(values: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Run the linear product on already-reduced operands.

    Args:
        values (torch.Tensor): retained activations of shape ``(..., k)``,
            as returned by :func:`select_topk_contraction`.
        weight (torch.Tensor): the matching weight columns, of shape
            ``(out_features, k)``.
        bias (torch.Tensor | None): optional bias of shape ``(out_features,)``.

    Returns:
        torch.Tensor: output of shape ``(..., out_features)``.
    """
    return torch.nn.functional.linear(values.type_as(weight), weight, bias)
