# Copyright (c) OpenMMLab. All rights reserved.
"""Online arrival-rate estimation for bursty prefill batching.

Adapted from "LLM Inference Under Bursty Workload Distribution: Modifying the
WAIT Algorithm" (Katageria, Rani and Sengupta, arXiv:2608.06135).

The paper's estimator observes only request arrival timestamps -- no workload
knowledge is assumed.  Interarrival times are inverted into instantaneous
rates, smoothed by an exponential moving average (EMA) to follow bursts while
retaining long-term trend, and averaged over a short window into the stable
global mean rate ``λ̂`` that its adaptive threshold ``n(t) ∝ 1/λ̂(t)`` consumes.
Under that rule a burst (high ``λ̂``) lowers the batching threshold so batches
form immediately, while a lull raises it so the engine accumulates a larger
batch instead of running tiny ones and churning KV cache.

Two adaptations to this codebase, both substituting auxiliary machinery the
paper builds around the core rule:

* WAIT batches per prompt *type* against fluid-dynamics thresholds.  This
  scheduler has a single prefill lane, so the rule is applied once to the
  whole waiting queue.
* A hold-time release is added so a slow trickle cannot hold the oldest
  request indefinitely.  The paper reports latency staying comparable; here
  that is enforced rather than measured.

The estimator itself is parameter-light and dependency-free: the paper's
Savitzky-Golay refinement denoises plots of the instantaneous rate, while the
threshold only reads the global mean, which the trailing average below already
provides.
"""

import time
from collections import deque


class ArrivalRateEstimator:
    """Estimate request arrival rate online from observed arrivals.

    Args:
        smoothing (float): EMA smoothing factor ``α``.  The paper selects
            ``α = 0.05`` by sensitivity analysis, favoring stability over
            reacting to a single fast arrival.
        average_window (int): Number of EMA values averaged into the global
            mean rate.  The paper uses 5.
    """

    def __init__(self, smoothing: float = 0.05, average_window: int = 5):
        if not 0.0 < smoothing < 1.0:
            raise ValueError(f'smoothing must be in (0, 1), got {smoothing}')
        if average_window < 1:
            raise ValueError(f'average_window must be >= 1, got {average_window}')
        self.smoothing = smoothing
        self._ema: float | None = None
        self._last_arrival: float | None = None
        self._window: deque[float] = deque(maxlen=average_window)

    @property
    def num_observations(self) -> int:
        """Number of interarrival times observed so far."""
        return len(self._window)

    def observe(self, arrive_time: float) -> float | None:
        """Record one arrival timestamp and return the local rate.

        The first arrival only anchors the timeline, so it produces no rate --
        an estimator with no interarrival observation must not report one.
        """
        if self._last_arrival is None:
            self._last_arrival = arrive_time
            return None
        if arrive_time <= self._last_arrival:
            # Zero or negative interarrival (clock granularity, out-of-order
            # arrival) carries no rate information.  Keep the previous anchor
            # so a later slow arrival is still measured against a real gap.
            return None

        local_rate = 1.0 / (arrive_time - self._last_arrival)
        self._last_arrival = arrive_time
        if self._ema is None:
            self._ema = local_rate
        else:
            self._ema = self.smoothing * local_rate + (1 - self.smoothing) * self._ema
        self._window.append(self._ema)
        return local_rate

    def observe_arrivals(self, arrive_times) -> None:
        """Record arrival timestamps, skipping ones already observed.

        A waiting request stays in the queue across scheduling passes, so its
        timestamp must feed the estimator exactly once.  Timestamps are
        monotonic in arrival order, which makes a watermark sufficient.
        """
        for arrive_time in arrive_times:
            if self._last_arrival is not None and arrive_time <= self._last_arrival:
                continue
            self.observe(arrive_time)

    @property
    def rate(self) -> float | None:
        """Estimated global mean arrival rate, or None before it is known."""
        if not self._window:
            return None
        return sum(self._window) / len(self._window)


class BurstyPrefillGate:
    """Hold prefills until the online arrival rate says a batch is worth it.

    Applies the paper's inverse-rate rule against a reference rate: the
    waiting count needed before a prefill turn runs shrinks when traffic is
    faster than the reference and grows when it is slower.

    Args:
        estimator (ArrivalRateEstimator): Source of ``λ̂``.
        reference_rate (float): Arrival rate, in requests per second, at which
            the gate stops holding and admits whatever is waiting.
        max_threshold (int): Upper clamp, so a long lull cannot make the gate
            wait for a batch larger than the engine can usefully run.
        max_hold_sec (float): Longest the oldest waiting request may be held.
            Releasing on age bounds the tail-latency cost of accumulating.
    """

    #: Interarrival observations required before the gate may hold anything.
    min_observations = 2

    def __init__(self, estimator: ArrivalRateEstimator, reference_rate: float, max_threshold: int,
                 max_hold_sec: float):
        if reference_rate <= 0:
            raise ValueError(f'reference_rate must be > 0, got {reference_rate}')
        if max_threshold < 1:
            raise ValueError(f'max_threshold must be >= 1, got {max_threshold}')
        if max_hold_sec < 0:
            raise ValueError(f'max_hold_sec must be >= 0, got {max_hold_sec}')
        self.estimator = estimator
        self.reference_rate = reference_rate
        self.max_threshold = max_threshold
        self.max_hold_sec = max_hold_sec

    @property
    def threshold(self) -> int:
        """Waiting-request count the current rate asks the scheduler to reach.

        At the reference rate this is 1, i.e. the unbatched behavior of
        admitting whatever is waiting.
        """
        rate = self.estimator.rate
        if rate is None or rate <= 0.0:
            return 1
        return max(1, min(int(self.reference_rate / rate), self.max_threshold))

    def should_hold(self, waiting: list, now: float | None = None) -> bool:
        """Whether this prefill turn should keep accumulating.

        Holds only when the estimated rate is known and the queue is below the
        adaptive threshold.  With no estimate yet the gate stays open rather
        than trading measured latency for a guess.
        """
        if len(waiting) == 0:
            return False
        if self.estimator.num_observations < self.min_observations:
            return False

        oldest_arrival = min(seq.arrive_time for seq in waiting)
        now = time.perf_counter() if now is None else now
        if now - oldest_arrival >= self.max_hold_sec:
            return False
        return len(waiting) < self.threshold


def build_bursty_prefill_gate(policy: str, reference_rate: float, max_threshold: int = 4,
                              max_hold_sec: float = 0.2) -> BurstyPrefillGate | None:
    """Build the prefill gate for a policy name, or None when disabled.

    Args:
        policy (str): ``'off'`` keeps the scheduler's default admission;
            ``'wait'`` enables arrival-rate-aware holding.
        reference_rate (float): See :class:`BurstyPrefillGate`.
        max_threshold (int): See :class:`BurstyPrefillGate`.
        max_hold_sec (float): See :class:`BurstyPrefillGate`.
    """
    if policy != 'wait':
        return None
    return BurstyPrefillGate(ArrivalRateEstimator(), reference_rate=reference_rate, max_threshold=max_threshold,
                             max_hold_sec=max_hold_sec)
