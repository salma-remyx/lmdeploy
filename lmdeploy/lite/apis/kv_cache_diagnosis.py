# Copyright (c) OpenMMLab. All rights reserved.
"""Diagnose KV-cache compression failures against a per-source full-cache control.

Adapted from KVDiagnosis (arXiv:2608.09412): instead of reporting an aggregate
score per compression setting, this harness runs every source twice -- once with
``quant_policy=NONE`` (the FullCache control) and once with the configured
quant policy -- classifies each pair as C-to-C (both correct), C-to-W (control
correct, compressed wrong) or W-to-W, and emits a per-source diagnostic record
linking the paired outputs with likelihood-based measurements.

The paper's core protocol is kept: paired per-source control, C-to-W selection
computed per method setting so no compressor defines another's test set, and a
common record format. Three of the paper's ten diagnostics are target-native
substitutes that LMDeploy can measure without instrumentation the engine does
not expose:

- ``likelihood_drift``: mean per-position logprob of the generated tokens under
  the compressed run relative to the control (the paper's likelihood drift).
- ``position_agreement``: fraction of output positions where the two runs emit
  the same token (the paper's decoding-measurement axis).
- ``prefix_divergence``: first position at which the two outputs diverge,
  normalized by output length -- the paper's structural-position-addressability
  axis, restricted to where LMDeploy can observe it (the emitted sequence).

The attention / coverage diagnostics that require per-layer attention or KV
block statistics are out of scope for this harness; they would need engine-side
instrumentation and are listed as ``applicability: not_measured`` in the
records so downstream tooling knows they were skipped, not lost.
"""

import csv
import json
import os
import time

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from lmdeploy.messages import QuantPolicy, Response

CONTROL_POLICY = QuantPolicy.NONE

CORRECT_TO_CORRECT = 'C-to-C'
CORRECT_TO_WRONG = 'C-to-W'
WRONG_TO_WRONG = 'W-to-W'

_NOT_MEASURED = {
    'attention': 'not_measured',
    'coverage': 'not_measured',
    'cache': 'not_measured',
}


def _as_policy(policy):
    """Coerce ``policy`` to a QuantPolicy, accepting names or raw values."""
    if isinstance(policy, QuantPolicy):
        return policy
    try:
        return QuantPolicy(policy)
    except ValueError:
        pass
    for member in QuantPolicy:
        if member.name.lower() == str(policy).lower():
            return member
    raise ValueError(f'invalid quant_policy: {policy!r}')


def _policy_name(policy):
    return _as_policy(policy).name.lower()


def _mean_logprob(response: Response):
    """Mean logprob of the tokens the compressed run actually emitted.

    Requires ``GenerationConfig(logprobs=1)``. Returns ``None`` when the
    engine did not attach logprobs (e.g. the control run), so drift is only
    reported when both sides are measurable.
    """
    if not response.logprobs:
        return None
    values = []
    for position in response.logprobs:
        if not position:
            continue
        top = max(position.values())
        values.append(top)
    if not values:
        return None
    return sum(values) / len(values)


def likelihood_drift(control: Response, compressed: Response):
    """Relative likelihood change of the emitted tokens (paper diagnostic).

    Negative drift means the compressed run assigns lower likelihood to its
    own output than the control assigns to its (correct) output.
    """
    control_lp = _mean_logprob(control)
    compressed_lp = _mean_logprob(compressed)
    if control_lp is None or compressed_lp is None:
        return None
    return compressed_lp - control_lp


def position_agreement(control: Response, compressed: Response):
    """Fraction of aligned output positions carrying the same token."""
    if not control.token_ids or not compressed.token_ids:
        return None
    shared = min(len(control.token_ids), len(compressed.token_ids))
    matches = sum(1 for i in range(shared) if control.token_ids[i] == compressed.token_ids[i])
    return matches / shared


def prefix_divergence(control: Response, compressed: Response):
    """First divergent output position, normalized by the longer output.

    0.0 means the runs disagree at the first token; ``None`` means they are
    identical up to the shorter one's end.
    """
    if not control.token_ids or not compressed.token_ids:
        return None
    for i in range(min(len(control.token_ids), len(compressed.token_ids))):
        if control.token_ids[i] != compressed.token_ids[i]:
            return i / max(len(control.token_ids), len(compressed.token_ids))
    if len(control.token_ids) == len(compressed.token_ids):
        return None
    return min(len(control.token_ids), len(compressed.token_ids)) / max(len(control.token_ids),
                                                                        len(compressed.token_ids))


def classify(control_ok: bool, compressed_ok: bool) -> str:
    """KVDiagnosis transition label for one control/compressed pair."""
    if control_ok and not compressed_ok:
        return CORRECT_TO_WRONG
    if control_ok and compressed_ok:
        return CORRECT_TO_CORRECT
    return WRONG_TO_WRONG


def diagnose_pair(control: Response, compressed: Response, source: str, reference: str, **metadata):
    """Build the common record for one source under one method setting.

    Args:
        control: the FullCache (``quant_policy=NONE``) response.
        compressed: the response under the configured quant policy.
        source: the prompt fed to both runs.
        reference: gold answer for correctness.
        **metadata: extra run metadata merged into the record.
    """
    control_ok = reference in control.text
    compressed_ok = reference in compressed.text
    record = {
        'source': source,
        'reference': reference,
        'control_correct': control_ok,
        'compressed_correct': compressed_ok,
        'transition': classify(control_ok, compressed_ok),
        'control_text': control.text,
        'compressed_text': compressed.text,
        'diagnostics': {
            'likelihood_drift': likelihood_drift(control, compressed),
            'position_agreement': position_agreement(control, compressed),
            'prefix_divergence': prefix_divergence(control, compressed),
            **_NOT_MEASURED,
        },
    }
    record.update(metadata)
    return record


def summarize(records: list[dict]):
    """Aggregate a record list into the failure profile for one method setting."""
    counts = {CORRECT_TO_WRONG: 0, CORRECT_TO_CORRECT: 0, WRONG_TO_WRONG: 0}
    for record in records:
        counts[record['transition']] += 1
    regressions = [r for r in records if r['transition'] == CORRECT_TO_WRONG]
    successes = [r for r in records if r['transition'] == CORRECT_TO_CORRECT]

    def _avg(rows, key):
        values = [r['diagnostics'][key] for r in rows if r['diagnostics'][key] is not None]
        return sum(values) / len(values) if values else None

    return {
        'total': len(records),
        'transitions': counts,
        'c_to_w_rate': counts[CORRECT_TO_WRONG] / len(records) if records else None,
        'regression_mean': {
            key: _avg(regressions, key) for key in ('likelihood_drift', 'position_agreement', 'prefix_divergence')
        },
        'success_mean': {
            key: _avg(successes, key) for key in ('likelihood_drift', 'position_agreement', 'prefix_divergence')
        },
    }


def _format_summary_line(label, stats):
    parts = [label]
    for key, value in stats.items():
        parts.append(f'{key}={value:.4f}' if isinstance(value, float) else f'{key}={value}')
    return '  '.join(parts)


def kv_cache_diagnosis(model_path: str,
                       sources: list[str],
                       references: list[str],
                       quant_policy=QuantPolicy.INT8,
                       session_len: int = None,
                       cache_max_entry_count: float = 0.8,
                       max_new_tokens: int = 64,
                       tp: int = 1,
                       output_dir: str = None,
                       log_level: str = 'WARNING'):
    """Run paired control/compressed inference and report failure diagnostics.

    Every source is evaluated under a per-source FullCache control
    (``quant_policy=NONE``) before the compressed run, so the C-to-W test set
    is defined by the control rather than by the compressor.

    Args:
        model_path: the model to load.
        sources: prompts, one per source.
        references: gold answers, one per source. A run counts as correct when
            its output contains the reference.
        quant_policy: the KV-cache compression setting under diagnosis.
        session_len: engine session length.
        cache_max_entry_count: engine cache budget.
        max_new_tokens: generation budget per run.
        tp: tensor parallel size.
        output_dir: if set, write ``records.json`` and ``summary.csv`` here.
        log_level: pipeline log level.

    Returns:
        dict: ``{'records': [...], 'summary': {...}}``.
    """
    if len(sources) != len(references):
        raise ValueError('sources and references must have the same length')
    quant_policy = _as_policy(quant_policy)
    gen_config = GenerationConfig(temperature=0.0, max_new_tokens=max_new_tokens, logprobs=1)
    started = time.time()

    def _engine_config(policy):
        return PytorchEngineConfig(tp=tp,
                                   session_len=session_len,
                                   cache_max_entry_count=cache_max_entry_count,
                                   quant_policy=policy,
                                   logprobs_mode='raw_logprobs')

    # control first, per source, so no compressed run defines the test set
    control_pipe = pipeline(model_path, backend_config=_engine_config(CONTROL_POLICY), log_level=log_level)
    try:
        controls = control_pipe(list(sources), gen_config=gen_config)
    finally:
        control_pipe.close()
    if isinstance(controls, Response):
        controls = [controls]

    policy_name = _policy_name(quant_policy)
    compressed_pipe = pipeline(model_path, backend_config=_engine_config(quant_policy), log_level=log_level)
    try:
        compressed = compressed_pipe(list(sources), gen_config=gen_config)
    finally:
        compressed_pipe.close()
    if isinstance(compressed, Response):
        compressed = [compressed]

    records = [
        diagnose_pair(control, compressed_run,
                      source=source,
                      reference=reference,
                      quant_policy=policy_name,
                      control_policy=_policy_name(CONTROL_POLICY),
                      model=model_path) for source, reference, control, compressed_run in zip(
                          sources, references, controls, compressed)
    ]
    summary = summarize(records)
    summary.update({
        'model': model_path,
        'quant_policy': policy_name,
        'elapsed_sec': time.time() - started,
    })

    print(f'\nKV-cache diagnosis: {model_path} quant_policy={policy_name}')
    for name, counts in summary['transitions'].items():
        print(f'  {name}: {counts}')
    for label, stats in (('C-to-W', summary['regression_mean']), ('C-to-C', summary['success_mean'])):
        print(_format_summary_line(f'  {label} mean', stats))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'records.json'), 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['transition', 'likelihood_drift', 'position_agreement', 'prefix_divergence'])
            writer.writerow(['C-to-W', *summary['regression_mean'].values()])
            writer.writerow(['C-to-C', *summary['success_mean'].values()])
    return {'records': records, 'summary': summary}
