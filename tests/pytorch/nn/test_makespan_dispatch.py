# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.nn import eplb
from lmdeploy.pytorch.nn.makespan_dispatch import RegimeProfile


def _metadata(ep_size, physical_to_logical, logical_to_all_physical, **kwargs):
    return eplb.EPLBMetadata._init_raw(
        ep_size=ep_size,
        physical_to_logical_map=torch.tensor(physical_to_logical),
        logical_to_all_physical_map=torch.tensor(logical_to_all_physical),
        **kwargs,
    )


def test_default_dispatch_policy_is_unchanged_random_fill(monkeypatch):
    """The default policy must reproduce the pre-existing uniform-random fill."""
    monkeypatch.setattr(eplb, "_global_eplb_metadata", None)

    physical_to_logical = [[0, 1, 1]]
    logical_to_all_physical = [[[0, -1], [1, 2]]]
    metadata = _metadata(
        ep_size=1, physical_to_logical=physical_to_logical, logical_to_all_physical=logical_to_all_physical
    )

    # GPU 0 hosts physical 0 and 1, so logical 0 -> 0 and logical 1 -> 1; the
    # second replica of logical 1 never enters the map with a single GPU.
    assert metadata.logical_to_rank_dispatch_physical_map[0, 0].tolist() == [0, 1]


def test_makespan_policy_diverts_to_less_loaded_gpu():
    """Two GPUs, one hot expert with a replica on each.

    GPU 0 already carries a heavier replica load from earlier layers, so a GPU
    with no local copy should be steered to GPU 1's replica rather than piling
    onto GPU 0. The random policy cannot express this: every candidate is
    equally likely.
    """
    # Two layers, two logical experts per layer. Layer 0's expert 0 has
    # replicas on both GPUs (physical 0 on GPU 0, physical 2 on GPU 1); every
    # expert is hot except layer 1 / expert 1, which is near-idle.
    physical_to_logical = [[0, 1, 0, 1], [0, 1, 0, 1]]
    logical_to_all_physical = [
        [[0, 2, -1, -1], [1, 3, -1, -1]],  # layer 0
        [[0, 2, -1, -1], [1, 3, -1, -1]],  # layer 1
    ]
    token_counts = torch.tensor([[4096.0, 4096.0], [4096.0, 4.0]])

    makespan = _metadata(
        ep_size=2,
        physical_to_logical=physical_to_logical,
        logical_to_all_physical=logical_to_all_physical,
        token_counts=token_counts,
        dispatch_policy="makespan",
    )

    # Replicas are only distinct where an expert is replicated across GPUs.
    # The cold expert (layer 1, logical 1) is the one whose choice carries
    # signal: both GPUs want a copy, and the fill must break the tie by load
    # rather than by coin flip.
    cold_logical = 1
    preferences = makespan.logical_to_rank_dispatch_physical_map[:, 1, cold_logical].tolist()
    assert sorted(preferences) == [1, 3]


def test_makespan_policy_keeps_local_replica_pinned():
    """A GPU that hosts a replica must keep preferring it under either policy."""
    physical_to_logical = [[0, 1, 1]]
    logical_to_all_physical = [[[0, -1], [1, 2]]]
    token_counts = torch.tensor([[128.0, 128.0]])

    makespan = _metadata(
        ep_size=1,
        physical_to_logical=physical_to_logical,
        logical_to_all_physical=logical_to_all_physical,
        token_counts=token_counts,
        dispatch_policy="makespan",
    )

    assert makespan.logical_to_rank_dispatch_physical_map[0, 0].tolist() == [0, 1]


def test_regime_profile_captures_both_regimes():
    """The cost model must be flat in tokens below the crossover, linear above."""
    profile = RegimeProfile()

    # One replica streaming its weights: below the crossover, more tokens on
    # the same replica are free (they pack into the tile being streamed).
    below = profile.block_time(1, 8)
    assert below == profile.block_time(1, 100)

    # Above the crossover the compute branch takes over and grows with tokens.
    above = profile.block_time(1, profile.crossover_tokens() * 4)
    assert above > below

    # Activating extra replicas always raises the streaming branch.
    assert profile.block_time(3, 8) > profile.block_time(1, 8)


def test_regime_profile_crossover_matches_measured_boundary():
    """The default profile puts one replica's crossover near the measured 156-168."""
    assert 150 <= RegimeProfile().crossover_tokens() <= 175


def test_makespan_policy_without_token_counts_balances_replicas():
    """With no activation statistic the policy degrades to replica balancing."""
    physical_to_logical = [[0, 1, 0, 1]]
    logical_to_all_physical = [[[0, 2, -1, -1], [1, 3, -1, -1]]]

    makespan = _metadata(
        ep_size=2,
        physical_to_logical=physical_to_logical,
        logical_to_all_physical=logical_to_all_physical,
        dispatch_policy="makespan",
    )

    # Every expert is replicated once per GPU, so each GPU prefers its local
    # replica rather than being scattered across the remote one.
    assert makespan.logical_to_rank_dispatch_physical_map[0, 0].tolist() == [0, 1]
    assert makespan.logical_to_rank_dispatch_physical_map[1, 0].tolist() == [2, 3]


def test_eplb_manager_end_to_end_with_makespan_policy(monkeypatch):
    """EPLBManager dispatch must round-trip physical ids under the makespan map."""
    physical_to_logical = [[0, 1, 1]]
    logical_to_all_physical = [[[0, -1], [1, 2]]]
    metadata = _metadata(
        ep_size=1,
        physical_to_logical=physical_to_logical,
        logical_to_all_physical=logical_to_all_physical,
        token_counts=torch.tensor([[128.0, 128.0]]),
        dispatch_policy="makespan",
    )
    monkeypatch.setattr(eplb, "_global_eplb_metadata", metadata)

    info = eplb.EPLBManager.get_dispatch_info(ep_rank=0, layer_idx=0)
    topk_ids = torch.tensor([[0, 1, 1]])
    physical = eplb.EPLBManager.topk_ids_logical_to_physical(topk_ids, info)

    assert physical[0, 0].item() == 0
    assert physical[0, 1].item() in (1, 2)
    assert physical[0, 2].item() in (1, 2)
