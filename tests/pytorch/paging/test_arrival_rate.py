import time

import pytest

import lmdeploy.pytorch.paging.scheduler as scheduler_module
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.paging.arrival_rate import ArrivalRateEstimator, BurstyPrefillGate, build_bursty_prefill_gate
from lmdeploy.pytorch.paging.scheduler import Scheduler


def _make_scheduler(max_batches: int = 4, bursty_policy: str = 'wait', reference_rate: float = 1.0,
                    max_hold_sec: float = 60.0):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy

    block_size = 4
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=max_batches,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=32,
                               max_prefill_token_num=block_size * 4)
    scheduler_config = SchedulerConfig(max_batches=max_batches,
                                       max_session_len=64,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
    scheduler.bursty_prefill_gate = build_bursty_prefill_gate(bursty_policy,
                                                              reference_rate=reference_rate,
                                                              max_threshold=max_batches,
                                                              max_hold_sec=max_hold_sec)
    return scheduler, block_size


# --- estimator ---


def test_estimator_first_arrival_has_no_rate():
    estimator = ArrivalRateEstimator()

    assert estimator.observe(10.0) is None
    assert estimator.rate is None
    assert estimator.num_observations == 0


def test_estimator_inverts_interarrival_into_rate():
    estimator = ArrivalRateEstimator()
    estimator.observe(10.0)

    assert estimator.observe(10.5) == pytest.approx(2.0)
    assert estimator.rate is not None
    assert estimator.num_observations == 1


def test_estimator_ignores_non_increasing_arrivals():
    estimator = ArrivalRateEstimator()
    estimator.observe(10.0)
    estimator.observe(11.0)

    assert estimator.observe(11.0) is None
    assert estimator.observe(10.0) is None
    assert estimator.num_observations == 1


def test_estimator_rate_rises_under_burst_and_falls_under_lull():
    burst = ArrivalRateEstimator()
    lull = ArrivalRateEstimator()

    for now in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]:
        burst.observe(now)
    for now in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        lull.observe(now)

    assert burst.rate > lull.rate


def test_observe_arrivals_skips_timestamps_already_seen():
    estimator = ArrivalRateEstimator()

    estimator.observe_arrivals([10.0, 10.5, 11.0])
    first_count = estimator.num_observations

    # Waiting sequences persist across scheduling passes; re-observing the
    # same timestamps must not inflate the estimated rate.
    estimator.observe_arrivals([10.5, 11.0, 11.5])

    assert first_count == 2
    assert estimator.num_observations == 3


# --- adaptive threshold ---


def test_threshold_is_inverse_in_estimated_rate():
    fast = ArrivalRateEstimator()
    slow = ArrivalRateEstimator()
    for now in [0.0, 0.01, 0.02, 0.03, 0.04]:
        fast.observe(now)
    for now in [0.0, 10.0, 20.0, 30.0, 40.0]:
        slow.observe(now)

    fast_gate = BurstyPrefillGate(fast, reference_rate=1.0, max_threshold=8, max_hold_sec=60.0)
    slow_gate = BurstyPrefillGate(slow, reference_rate=1.0, max_threshold=8, max_hold_sec=60.0)

    # Burst: rate above the reference lowers the threshold. Lull: below the
    # reference raises it, so batches keep accumulating.
    assert fast_gate.threshold == 1
    assert slow_gate.threshold == 8
    assert slow_gate.threshold > fast_gate.threshold


def test_threshold_clamps_to_max_and_one():
    gate = BurstyPrefillGate(ArrivalRateEstimator(), reference_rate=1.0, max_threshold=3, max_hold_sec=60.0)

    # No observations yet: the rule cannot lower anything below 1.
    assert gate.threshold == 1

    for now in [0.0, 5.0, 10.0, 15.0]:
        gate.estimator.observe(now)
    assert gate.threshold == 3

    for now in [20.0, 20.01, 20.02, 20.03]:
        gate.estimator.observe(now)
    assert gate.threshold == 1


def test_gate_does_not_hold_without_rate_estimate():
    gate = BurstyPrefillGate(ArrivalRateEstimator(), reference_rate=0.001, max_threshold=8, max_hold_sec=60.0)

    class _Seq:

        def __init__(self, arrive_time):
            self.arrive_time = arrive_time

    waiting = [_Seq(0.0), _Seq(0.0)]

    assert gate.should_hold(waiting, now=1.0) is False


def test_gate_holds_when_below_threshold_and_releases_on_age():
    estimator = ArrivalRateEstimator()
    for now in [0.0, 10.0, 20.0, 30.0, 40.0]:
        estimator.observe(now)
    gate = BurstyPrefillGate(estimator, reference_rate=1.0, max_threshold=8, max_hold_sec=0.5)

    class _Seq:

        def __init__(self, arrive_time):
            self.arrive_time = arrive_time

    fresh = [_Seq(10.0), _Seq(10.1)]
    stale = [_Seq(5.0), _Seq(5.1)]

    assert gate.should_hold(fresh, now=10.2) is True
    assert gate.should_hold(stale, now=10.2) is False
    assert gate.should_hold([], now=10.2) is False


def test_build_gate_is_none_when_policy_disabled():
    assert build_bursty_prefill_gate('off', reference_rate=1.0) is None
    assert build_bursty_prefill_gate('wait', reference_rate=1.0) is not None


# --- scheduler integration ---


def test_scheduler_builds_no_gate_by_default(monkeypatch):
    monkeypatch.setattr(scheduler_module._envs, 'bursty_prefill_policy', 'off')
    monkeypatch.setattr(scheduler_module._envs, 'bursty_prefill_reference_rate', 1.0)

    scheduler, _ = _make_scheduler(bursty_policy='off')

    assert scheduler.bursty_prefill_gate is None


def test_scheduler_reads_bursty_prefill_env(monkeypatch):
    monkeypatch.setattr(scheduler_module._envs, 'bursty_prefill_policy', 'wait')
    monkeypatch.setattr(scheduler_module._envs, 'bursty_prefill_reference_rate', 2.5)

    scheduler, _ = _make_scheduler()

    assert isinstance(scheduler.bursty_prefill_gate, BurstyPrefillGate)
    assert scheduler.bursty_prefill_gate.reference_rate == 2.5
    assert scheduler.bursty_prefill_gate.max_threshold == scheduler.scheduler_config.max_batches


def test_scheduler_env_defaults_are_off():
    from lmdeploy.pytorch import envs

    assert envs.bursty_prefill_policy == 'off'
    assert envs.bursty_prefill_reference_rate == 1.0


def test_disabled_gate_leaves_prefill_admission_unchanged():
    scheduler, block_size = _make_scheduler(bursty_policy='off')
    seq = scheduler.add_session(100).add_sequence([1] * block_size)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.status == MessageStatus.READY


def test_gate_holds_prefill_until_burst_threshold_reached():
    scheduler, block_size = _make_scheduler(reference_rate=1.0, max_batches=4, max_hold_sec=60.0)
    now = time.perf_counter()

    # A lull: arrivals 10s apart put the estimated rate at 0.1, well below the
    # reference rate, so the adaptive threshold rises to its clamp and a
    # partially-filled waiting queue is held back.
    for idx in range(3):
        seq = scheduler.add_session(100 + idx).add_sequence([1] * block_size)
        seq.arrive_time = now - 30.0 + idx * 10.0

    output = scheduler.schedule(is_prefill=True)

    gate = scheduler.bursty_prefill_gate
    assert gate.estimator.rate == pytest.approx(0.1)
    assert gate.threshold == 4
    assert output.running == []
    for seq in scheduler.waiting:
        assert seq.status == MessageStatus.WAITING
        assert seq.num_blocks == 0

    # Crossing the threshold admits the accumulated batch in one prefill turn.
    extra = scheduler.add_session(200).add_sequence([2] * block_size)
    extra.arrive_time = now

    output = scheduler.schedule(is_prefill=True)
    assert len(output.running) == scheduler.scheduler_config.max_batches
    for seq in output.running:
        assert seq.status == MessageStatus.READY


def test_gate_admits_immediately_under_burst():
    scheduler, block_size = _make_scheduler(reference_rate=1.0, max_batches=4)
    now = time.perf_counter()

    # A burst: arrivals 0.05s apart put the estimated rate far above the
    # reference, so the threshold drops to 1 and prefill runs right away.
    seq = scheduler.add_session(100).add_sequence([1] * block_size)
    seq.arrive_time = now - 0.10
    later = scheduler.add_session(101).add_sequence([2] * block_size)
    later.arrive_time = now - 0.05

    output = scheduler.schedule(is_prefill=True)

    assert scheduler.bursty_prefill_gate.threshold == 1
    assert output.running == [seq, later]
    assert seq.status == MessageStatus.READY
    assert later.status == MessageStatus.READY


def test_gate_releases_held_prefill_after_max_hold():
    scheduler, block_size = _make_scheduler(reference_rate=1.0, max_hold_sec=0.05)
    now = time.perf_counter()

    # Same lull as above, so the gate holds the queue below its threshold.
    for idx in range(3):
        seq = scheduler.add_session(100 + idx).add_sequence([1] * block_size)
        seq.arrive_time = now - 30.0 + idx * 10.0

    assert scheduler.schedule(is_prefill=True).running == []
    assert scheduler.bursty_prefill_gate.threshold == 4

    # Once the oldest waiter exceeds max_hold_sec the gate opens even though
    # the estimated rate still asks the scheduler to keep accumulating.
    stale = scheduler.add_session(200).add_sequence([3] * block_size)
    stale.arrive_time = now - 300.0

    output = scheduler.schedule(is_prefill=True)
    assert len(output.running) > 0
    assert all(seq.status == MessageStatus.READY for seq in output.running)
