# Copyright (c) OpenMMLab. All rights reserved.
"""Runtime integrity probe for heterogeneous attention memory.

lmdeploy now serves four attention-memory classes side by side: paged KV,
the DeepSeek-V4 latent compressor, native-sparse-attention index caches and
the gated-delta recurrent state. Each carries the model's memory in a
different form, so each fails differently under compression -- and a failure
in a fused SM90 FP8/BF16 kernel or an FP8 KV cache usually shows up
downstream as a quietly wrong token rather than an error.

This module ports the *contract* part of "Runtime Observability for
Heterogeneous Attention Memory" (arXiv:2608.05863): memory reads are checked
at the attention boundary, per-layer findings compose into a request-level
risk ledger, and every composed claim carries an explicit tier
(``certified`` / ``partially_certified`` / ``empirical``) that is decided by
the checker rather than asserted by the caller. Composition inherits the
weakest tier, mirroring the paper's metric-typed contracts.

Two things are deliberately parameter-free substitutes for the paper's
machinery rather than ports of it:

* the paper's Lean-certified error metrics become structural checks (finite,
  in-dtype-range, not-all-identical) that need no estimator and no proof
  artifact;
* the paper's machine-adjudicated discrimination campaign becomes a *regime
  label* (eviction-free vs. eviction/slot-reuse) attached to each finding, so
  a burst of findings localizes to the structural boundary it occurred in
  instead of demanding a separate replay harness.

The probe is off by default; enable it with ``LMDEPLOY_ATTN_MEM_PROBE=1``.
"""

from enum import Enum
from functools import lru_cache

import torch

from lmdeploy.pytorch import envs

__all__ = [
    'AttnMemoryClass',
    'AttnMemoryLedger',
    'attn_memory_classify',
    'get_attn_memory_ledger',
    'probe_attention_memory',
    'reset_attn_memory_ledger',
]

# Findings above this rate are surfaced as a single log line per step rather
# than one line per layer, keeping the probe inside the serving noise floor.
_LOG_AGGREGATE_THRESHOLD = 0.25


class AttnMemoryClass(str, Enum):
    """The four attention-memory classes the probe knows how to read."""

    PAGED_KV = 'paged_kv'
    V4_COMPRESSED_KV = 'v4_compressed_kv'
    NSA_INDEX = 'nsa_index'
    RECURRENT_STATE = 'recurrent_state'


class AttnMemoryLedger:
    """Accumulates per-layer findings into a tiered, request-level risk ledger.

    Attributes:
        findings: total number of non-finite / out-of-range reads observed.
        reads: total number of cache elements the probe inspected.
        regimes: counter keyed by the structural regime each finding occurred
            in, used to localize failures to eviction vs. slot-reuse paths.
    """

    __slots__ = ('findings', 'reads', 'regimes')

    def __init__(self):
        self.findings = 0
        self.reads = 0
        self.regimes: dict[str, int] = {}

    def record(self, findings: int, reads: int, regime: str):
        """Record one layer's probe result."""
        self.findings += findings
        self.reads += reads
        self.regimes[regime] = self.regimes.get(regime, 0) + findings

    @property
    def finding_rate(self) -> float:
        """Fraction of inspected elements that were corrupt."""
        return self.findings / self.reads if self.reads else 0.0

    def holds_budget(self, budget: float) -> bool:
        """Whether the ledger stays inside ``budget``. A zero-read ledger
        holds any budget: absence of evidence is not a violation."""
        return self.finding_rate <= budget

    def dominant_regime(self) -> str | None:
        """The structural regime carrying the most findings, if any."""
        if not self.regimes:
            return None
        return max(self.regimes.items(), key=lambda kv: kv[1])[0]

    def as_dict(self) -> dict:
        """Snapshot for the metrics/health path."""
        return {
            'findings': self.findings,
            'reads': self.reads,
            'finding_rate': self.finding_rate,
            'regimes': dict(self.regimes),
            'dominant_regime': self.dominant_regime(),
        }

    def __repr__(self) -> str:
        regime = self.dominant_regime()
        where = f', regime={regime}' if regime else ''
        return (f'AttnMemoryLedger(findings={self.findings}, reads={self.reads}, '
                f'rate={self.finding_rate:.2e}{where})')


_LEDGER = AttnMemoryLedger()


def reset_attn_memory_ledger() -> AttnMemoryLedger:
    """Start a fresh ledger (per-request, or per-evaluation-window)."""
    global _LEDGER
    _LEDGER = AttnMemoryLedger()
    return _LEDGER


def get_attn_memory_ledger() -> AttnMemoryLedger:
    """The live process-wide ledger."""
    return _LEDGER


def attn_memory_classify(
    tensor: torch.Tensor,
    *,
    nsa_indices: torch.Tensor | None = None,
    state_like: bool = False,
) -> AttnMemoryClass:
    """Classify which attention-memory class a cache tensor belongs to.

    Shape and provenance carry the signal -- the probe never needs to know the
    model architecture. The caller passes the selectors/state hints it already
    has in scope at the attention boundary.
    """
    if state_like:
        # GatedDelta / StateCacheEngine slot pool: [num_slots, ...] with no
        # paged block dimension.
        return AttnMemoryClass.RECURRENT_STATE
    if nsa_indices is not None:
        # An explicit selector came with the read: this is an index cache.
        return AttnMemoryClass.NSA_INDEX
    if tensor.size(-1) == 1 and tensor.dim() >= 2:
        # Compressor writes latent KV as a packed per-slot row with a
        # trailing singleton. Checked before the flat-pool shape below because
        # a 3-D latent cache is also 3-D.
        return AttnMemoryClass.V4_COMPRESSED_KV
    if tensor.dim() == 3:
        # Indexer caches are flat per-slot pools ([num_slots, heads, dim]).
        return AttnMemoryClass.NSA_INDEX
    return AttnMemoryClass.PAGED_KV


@lru_cache(maxsize=64)
def _finfo_bounds(dtype_name: str) -> tuple[float, float] | None:
    """Finite representable range for a dtype, cached per dtype.

    Returns ``None`` when the dtype has no meaningful range check (integer
    caches, or float dtypes ``finfo`` cannot describe).
    """
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        return None
    try:
        info = torch.finfo(dtype)
    except TypeError:
        return None
    return -abs(float(info.max)), abs(float(info.max))


def _check_finite(tensor: torch.Tensor) -> int:
    """Count non-finite elements without materializing a float copy."""
    return int(torch.count_nonzero(~torch.isfinite(tensor)).item())


def _check_degenerate(tensor: torch.Tensor) -> int:
    """Count slots whose entire last dim is one repeated value.

    A real KV row varies along the feature dim; an unwritten or
    slot-reuse-corrupted row is a constant (typically all zeros). Constant
    rows are counted per-slot, not per-element, so a single reused slot is one
    finding rather than ``head_size`` of them.
    """
    if tensor.size(-1) < 2:
        return 0
    lead = tensor.shape[:-1].numel()
    if lead == 0:
        return 0
    rows = tensor.reshape(lead, tensor.size(-1))
    constant = (rows == rows[..., :1]).all(dim=-1)
    return int(torch.count_nonzero(constant).item())


def _select_live_blocks(tensor: torch.Tensor, block_offsets: torch.Tensor) -> torch.Tensor:
    """Narrow a paged pool down to the blocks this step actually reads.

    The pool handed to ``Attention.forward`` spans every block the engine
    owns, including ones no sequence has written yet -- and an unwritten block
    is legitimately constant, so probing the whole pool would report findings
    for memory that was never claimed. ``block_offsets`` is the per-sequence
    logical->physical map already in scope at the attention boundary.
    """
    if block_offsets is None or tensor.dim() < 3:
        return tensor
    try:
        blocks = block_offsets.reshape(-1).unique().to(torch.long)
    except (RuntimeError, ValueError):
        return tensor
    if blocks.numel() == 0:
        return tensor
    # Paged pools are [num_blocks, block_size, heads, dim] or
    # [layers, num_blocks, ...]; the block axis is the one matching the
    # offset values' range.
    for axis in (1, 0):
        if axis < tensor.dim() and int(blocks.max()) < tensor.size(axis):
            return tensor.index_select(axis, blocks)
    return tensor


def _tier_for(budget: float, rate: float) -> str:
    """Decide the tier the machine assigns to this ledger reading.

    ``certified``: the ledger is structurally exact -- every read finite, in
    range, non-degenerate, so the bound holds by construction.
    ``partially_certified``: some findings exist but the composed bound still
    holds, so the claim is certified only up to the stated budget.
    ``empirical``: the budget is violated. Nothing structural can be claimed
    about the composed chain; the reading is evidence, not a bound.
    """
    if rate == 0.0:
        return 'certified'
    return 'partially_certified' if rate <= budget else 'empirical'


def probe_attention_memory(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    layer_id: int,
    *,
    nsa_indices: torch.Tensor | None = None,
    state_like: bool = False,
    regime: str = 'eviction_free',
    budget: float = 0.0,
    block_offsets: torch.Tensor | None = None,
) -> int:
    """Probe one layer's attention-memory reads at the attention boundary.

    Returns the number of findings for this layer and records them on the
    process-wide ledger. The probe is read-only: it inspects cache tensors
    that the attention implementation is about to read, and never mutates
    them.

    Args:
        k_cache: the layer's key cache (paged, latent or index pool).
        v_cache: the layer's value cache, or the recurrent-state pool.
        layer_id: layer index, used for layer subsetting.
        nsa_indices: sparse-attention indices when this layer reads a selector
            cache.
        state_like: set when ``v_cache`` is a recurrent-state slot pool.
        regime: structural regime label, e.g. ``'eviction_free'`` or
            ``'slot_reuse'``.
        budget: composed risk budget on the finding rate.
        block_offsets: per-sequence block map used to narrow a paged pool to
            the blocks this step reads.
    """
    layers = envs.attn_mem_probe_layers
    if layers and layer_id not in layers:
        return 0

    mem_class = attn_memory_classify(k_cache, nsa_indices=nsa_indices, state_like=state_like)
    findings = 0
    reads = 0

    for tensor in (k_cache, v_cache):
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if mem_class is AttnMemoryClass.PAGED_KV:
            tensor = _select_live_blocks(tensor, block_offsets)
        probe = tensor.detach().to(torch.float32, copy=False)
        reads += probe.numel()
        findings += _check_finite(probe)
        bounds = _finfo_bounds(str(tensor.dtype).replace('torch.', ''))
        if bounds is not None:
            lo, hi = bounds
            findings += int(torch.count_nonzero((probe < lo) | (probe > hi)).item())
        if mem_class is not AttnMemoryClass.RECURRENT_STATE:
            findings += _check_degenerate(tensor)

    ledger = get_attn_memory_ledger()
    ledger.record(findings, reads, regime)
    _maybe_log(layer_id, mem_class, findings, reads, ledger, budget)
    return findings


def _maybe_log(layer_id: int, mem_class: AttnMemoryClass, findings: int, reads: int,
               ledger: AttnMemoryLedger, budget: float):
    """Surface findings on one line per step, inside the serving noise floor."""
    if findings == 0 or reads == 0:
        return
    from lmdeploy.utils import get_logger

    logger = get_logger('lmdeploy')
    rate = findings / reads
    if rate >= _LOG_AGGREGATE_THRESHOLD:
        # Widespread corruption reads as a whole-ledger event, not 64 layer
        # events; the tier names what the composed claim is still worth.
        logger.warning(
            'attn memory probe: ledger %s exceeds per-layer signal at layer %d (%s); '
            'tier=%s, budget=%.2e', ledger, layer_id, mem_class.value,
            _tier_for(budget, ledger.finding_rate), budget)
    else:
        logger.warning('attn memory probe: layer %d (%s) %d/%d unreadable elements; %s',
                       layer_id, mem_class.value, findings, reads, ledger)
