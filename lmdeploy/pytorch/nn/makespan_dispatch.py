# Copyright (c) OpenMMLab. All rights reserved.
"""Makespan-aware replica preference for EPLB dispatch.

In expert-parallel serving every MoE layer synchronizes at the slowest GPU, so
the dispatch policy decides the layer's block time. Expert latency is not
linear in token count: below a crossover point HBM weight streaming dominates
and cost attaches to *activated replicas*, while above it grouped GEMM rounds
tokens up to tile granularity and cost grows with tokens. A max-affine profile

    t(G, N) = max(a + b * G, c + beta * N)

captures both regimes from a single (activated-replica, token) pair per GPU.
Balancing token counts alone is only optimal in the compute-bound regime;
balancing activated replicas alone is only optimal in the memory-bound one.
Realistic decode batches sit in both at once, so the preference map should be
built against the max of the two rather than either alone.

This module implements that cost model plus a greedy makespan heuristic for
choosing, per (layer, logical expert), which replica a GPU should prefer when
several replicas of the expert exist. The result is a drop-in replacement for
the uniform-random replica fill in
``compute_logical_to_rank_dispatch_physical_map``.

Adapted from TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across
Memory- and Compute-Bound Regimes (arXiv:2608.13057). The out-of-process
solver and the in-graph count fusion of the reference system are out of scope
here; what is kept is the regime-aware cost model and the makespan objective,
applied to lmdeploy's static replica-preference map.
"""

from dataclasses import dataclass

import torch

# Grouped-GEMM M-tile granularity. Tokens routed to a replica are padded up to
# this stride, which is why per-token cost is flat until the tile fills.
TILE_TOKENS = 128

# Cost of streaming one replica's expert weights, in units of a filled tile's
# compute cost. Streaming one replica costs slightly more than one full tile,
# which puts the single-replica crossover at ~160 tokens -- the regime boundary
# measured on Hopper-class GPUs.
REPLICA_STREAM_COST = 1.25


@dataclass
class RegimeProfile:
    """Max-affine expert-time model ``t = max(a + b*G, c + beta*N)``.

    ``G`` is the number of activated replicas on a GPU and ``N`` its routed
    token count. The two branches are the memory-bound (weight streaming) and
    compute-bound (tiled GEMM) regimes; a GPU sits in whichever is slower.
    Both are expressed in tile units, i.e. multiples of the compute cost of
    one full ``TILE_TOKENS`` tile.
    """

    # memory-bound branch: a + b * G
    stream_base: float = 0.0
    stream_per_replica: float = REPLICA_STREAM_COST
    # compute-bound branch: c + beta * N
    gemm_base: float = 0.0
    tokens_per_tile: int = TILE_TOKENS

    def block_time(self, num_replicas: int, num_tokens: int) -> float:
        """Modeled time for one GPU activating ``num_replicas`` experts for ``num_tokens``."""
        streaming = self.stream_base + self.stream_per_replica * num_replicas
        compute = self.gemm_base + num_tokens / self.tokens_per_tile
        return max(streaming, compute)

    def crossover_tokens(self) -> float:
        """Token count at which one activated replica leaves the memory-bound regime."""
        return (self.stream_base + self.stream_per_replica - self.gemm_base) * self.tokens_per_tile


def compute_makespan_preferences(
    logical_to_all_physical_map: torch.Tensor,
    token_counts: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    profile: RegimeProfile | None = None,
) -> torch.Tensor:
    """Greedily build a ``(num_gpus, num_layers, num_logical)`` preference map.

    For each (layer, logical expert), GPUs that already host a replica keep
    hosting it. The remaining GPUs — which hold no local copy — each take the
    replica that minimizes their resulting modeled block time, i.e. the
    incremental makespan. This is a greedy surrogate for the fixed-charge
    makespan problem, which is NP-hard even on two fully replicated GPUs.

    ``token_counts`` is the per-(layer, logical expert) activation statistic,
    the same ``experts_statistic`` already consumed by ``rebalance_experts``.
    """
    profile = profile or RegimeProfile()
    num_layers, num_logical_experts = logical_to_all_physical_map.shape[:2]
    if token_counts is None:
        # No activation statistic available: fall back to balancing activated
        # replicas only, which is the memory-bound-regime half of the model.
        token_counts = torch.zeros(
            (num_layers, num_logical_experts), dtype=torch.float32, device=logical_to_all_physical_map.device
        )
    num_local_physical_experts = num_physical_experts // num_gpus
    replica_load = [0] * num_gpus
    token_load = [0] * num_gpus

    preference = torch.full(
        (num_gpus, num_layers, num_logical_experts),
        -1,
        dtype=logical_to_all_physical_map.dtype,
        device=logical_to_all_physical_map.device,
    )

    for layer_id in range(num_layers):
        replica_load = [0] * num_gpus
        token_load = [0] * num_gpus
        for logical_expert_id in range(num_logical_experts):
            candidates = [
                physical
                for physical in logical_to_all_physical_map[layer_id, logical_expert_id].tolist()
                if physical != -1
            ]
            if not candidates:
                continue
            tokens = float(token_counts[layer_id, logical_expert_id])

            # Local replicas pin their host first; they are the fixed charge.
            for gpu_id in range(num_gpus):
                for physical in candidates:
                    if physical // num_local_physical_experts == gpu_id:
                        preference[gpu_id, layer_id, logical_expert_id] = physical
                        replica_load[gpu_id] += 1
                        token_load[gpu_id] += tokens
                        break

            # GPUs without a local copy take the replica that least increases
            # their modeled block time.
            for gpu_id in range(num_gpus):
                if preference[gpu_id, layer_id, logical_expert_id] != -1:
                    continue
                best_physical = min(
                    candidates,
                    key=lambda physical: profile.block_time(replica_load[gpu_id] + 1, token_load[gpu_id] + tokens),
                )
                preference[gpu_id, layer_id, logical_expert_id] = best_physical
                replica_load[gpu_id] += 1
                token_load[gpu_id] += tokens

    return preference
