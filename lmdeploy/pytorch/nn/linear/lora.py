# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.lora import AdapterInfo
from lmdeploy.pytorch.distributed import get_tp_world_rank


class LoRA(nn.Module):
    """LoRA layer."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ranks: torch.Tensor,
                 scalings: torch.Tensor,
                 lora_a: torch.Tensor,
                 lora_b: torch.Tensor,
                 base_slice: slice,
                 ctx_mgr: Any = None,
                 colwise: bool = True,
                 is_tp: bool = True,
                 lora_b_spliter: Any = None,
                 use_dora: bool = False,
                 weight: torch.Tensor = None):
        super().__init__()
        self.adapter_info = AdapterInfo(
            in_features=in_features,
            out_features=out_features,
            ranks=ranks,
            scalings=scalings,
            base_slice=base_slice,
        )
        impl_builder = get_backend().get_layer_impl_builder(OpType.LoRA)
        self.impl = impl_builder.build()

        lora_A = nn.Parameter(lora_a, requires_grad=False)
        lora_B = nn.Parameter(lora_b, requires_grad=False)
        self.register_parameter('lora_A', lora_A)
        self.register_parameter('lora_B', lora_B)
        lora_A.weight_loader = self.weight_loader_A
        lora_B.weight_loader = self.weight_loader_B
        self.is_tp = is_tp
        self.ctx_mgr = ctx_mgr
        self.colwise = colwise
        self.lora_b_spliter = lora_b_spliter
        # dora (weight-decomposed low-rank adaptation) keeps a per-row
        # magnitude and rescales the base output by m/||W+s*BA||_c. The base
        # weight is stored unregistered so it is not part of the adapter's
        # trainable parameters; it is only read once the magnitude is loaded.
        self.use_dora = use_dora
        if use_dora:
            lora_m = nn.Parameter(lora_b.clone().fill_(1.0), requires_grad=False)
            self.register_parameter('lora_magnitude', lora_m)
            lora_m.weight_loader = self.weight_loader_magnitude
            self.weight = weight
        else:
            self.weight = None

    def forward(self, x, base_output=None):
        """Forward of loraA@loraB."""
        return self.impl.forward(x,
                                 self.lora_A,
                                 self.lora_B,
                                 base_output,
                                 self.adapter_info,
                                 ctx_mgr=self.ctx_mgr,
                                 colwise=self.colwise,
                                 is_tp=self.is_tp)

    def finalize_dora(self):
        """Compute dora weight scaling m/||W+s*BA||_c once weights are loaded.

        The norms are computed with the factored formulation, which never
        materializes the dense BA product. The result is stored on
        ``adapter_info`` so the serving path picks it up through the existing
        lora impl without any per-step work.
        """
        from lmdeploy.pytorch.kernels.cuda.dora import pack_dora_weight_norm
        if not self.use_dora or self.adapter_info.weight_scaling is not None:
            return
        assert self.weight is not None, 'dora needs the base weight to compute ||W + s*BA||_c'

        # for rowwise-parallel layers each rank holds a slice of the columns
        # the norm is taken over, so the partial sums have to be reduced
        reduce_group = None
        if self.is_tp and not self.colwise:
            from lmdeploy.pytorch.distributed import get_tp_group
            reduce_group = get_tp_group()

        self.adapter_info.weight_scaling, _ = pack_dora_weight_norm(
            self.weight,
            self.lora_A,
            self.lora_B,
            ranks=self.adapter_info.ranks,
            scalings=self.adapter_info.scalings,
            rank_offsets=self.adapter_info.rank_offsets,
            lora_magnitude=self.lora_magnitude,
            reduce_group=reduce_group,
        )

    def weight_loader_A(self, param: nn.Parameter, loaded_weight: torch.Tensor, adapter_id: int):
        """Weight loader."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and not self.colwise:
            world_size, rank = get_tp_world_rank()
            loaded_weight = loaded_weight.to(param_r.device)
            loaded_weight = loaded_weight.chunk(world_size, dim=1)[rank]

        param_r.copy_(loaded_weight)

    def weight_loader_B(self, param: nn.Parameter, loaded_weight: torch.Tensor, adapter_id: int):
        """Weight loader."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and self.colwise:
            world_size, rank = get_tp_world_rank()
            if self.lora_b_spliter is not None:
                loaded_weights = self.lora_b_spliter(loaded_weight)
                new_weights = []
                for w in loaded_weights:
                    w = w.chunk(world_size, dim=0)[rank]
                    new_weights.append(w)
                loaded_weight = torch.cat(new_weights, dim=0)
            else:
                loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]

        param_r.copy_(loaded_weight.t())

    def weight_loader_magnitude(self, param: nn.Parameter, loaded_weight: torch.Tensor, adapter_id: int):
        """Weight loader of the dora magnitude."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and self.colwise:
            world_size, rank = get_tp_world_rank()
            if self.lora_b_spliter is not None:
                loaded_weights = self.lora_b_spliter(loaded_weight)
                new_weights = []
                for w in loaded_weights:
                    w = w.chunk(world_size, dim=1)[rank]
                    new_weights.append(w)
                loaded_weight = torch.cat(new_weights, dim=1)
            else:
                loaded_weight = loaded_weight.chunk(world_size, dim=1)[rank]

        param_r.copy_(loaded_weight)
