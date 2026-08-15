# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the KV-cache diagnosis harness (lmdeploy lite kv_diagnose).

The diagnostics themselves are exercised against the engine-independent
record format built on ``lmdeploy.messages.Response``; the CLI wiring is
checked through the lite subcommand parser.
"""

import pytest

from lmdeploy.lite.apis.kv_cache_diagnosis import (
    CORRECT_TO_CORRECT,
    CORRECT_TO_WRONG,
    WRONG_TO_WRONG,
    classify,
    diagnose_pair,
    kv_cache_diagnosis,
    likelihood_drift,
    position_agreement,
    prefix_divergence,
    summarize,
)
from lmdeploy.messages import QuantPolicy, Response


def _response(text, token_ids, logprobs=None):
    return Response(text=text,
                    generate_token_len=len(token_ids),
                    input_token_len=8,
                    token_ids=list(token_ids),
                    logprobs=logprobs)


def test_classify_transitions():
    assert classify(True, False) == CORRECT_TO_WRONG
    assert classify(True, True) == CORRECT_TO_CORRECT
    assert classify(False, False) == WRONG_TO_WRONG
    # a control-wrong run is never attributed to the compressor
    assert classify(False, True) == WRONG_TO_WRONG


def test_likelihood_drift_negative_when_compressed_degrades():
    control = _response('gold', [1, 2], logprobs=[{1: -0.1}, {2: -0.1}])
    degraded = _response('gild', [1, 3], logprobs=[{1: -0.2}, {3: -4.0}])
    assert likelihood_drift(control, degraded) < 0
    assert likelihood_drift(control, control) == pytest.approx(0.0)
    # unmeasurable sides report None instead of a fabricated number
    assert likelihood_drift(_response('a', [1]), control) is None


def test_position_and_prefix_diagnostics():
    control = _response('gold answer', [10, 11, 12, 13])
    compressed = _response('gold answer', [10, 11, 55, 13])
    assert position_agreement(control, compressed) == pytest.approx(0.75)
    assert prefix_divergence(control, compressed) == pytest.approx(2 / 4)
    assert prefix_divergence(control, control) is None


def test_diagnose_pair_marks_c_to_w_with_metadata():
    control = _response('the capital is Paris', [5, 6, 7])
    compressed = _response('the capital is Lyon', [5, 6, 8])
    record = diagnose_pair(control,
                           compressed,
                           source='capital of France?',
                           reference='Paris',
                           quant_policy='int8')
    assert record['transition'] == CORRECT_TO_WRONG
    assert record['control_correct'] is True
    assert record['compressed_correct'] is False
    assert record['quant_policy'] == 'int8'
    # diagnostics that need engine instrumentation are explicitly marked
    assert record['diagnostics']['attention'] == 'not_measured'
    assert record['diagnostics']['coverage'] == 'not_measured'
    assert record['diagnostics']['cache'] == 'not_measured'


def test_summarize_separates_regressions_from_successes():
    good_control = _response('Paris', [1, 2])
    good_compressed = _response('Paris', [1, 2])
    bad_compressed = _response('Lyon', [1, 3])
    records = [
        diagnose_pair(good_control, bad_compressed, 's0', 'Paris'),
        diagnose_pair(good_control, good_compressed, 's1', 'Paris'),
    ]
    summary = summarize(records)
    assert summary['transitions'][CORRECT_TO_WRONG] == 1
    assert summary['transitions'][CORRECT_TO_CORRECT] == 1
    assert summary['c_to_w_rate'] == pytest.approx(0.5)
    # drift/agreement means are reported per population
    assert set(summary['regression_mean']) == {'likelihood_drift', 'position_agreement', 'prefix_divergence'}
    assert set(summary['success_mean']) == {'likelihood_drift', 'position_agreement', 'prefix_divergence'}


def test_harness_validates_policy_and_dataset():
    from lmdeploy.lite.apis import kv_cache_diagnosis as module

    assert module._as_policy('int8') == QuantPolicy.INT8
    assert module._as_policy(42) == QuantPolicy.TURBO_QUANT
    with pytest.raises(ValueError):
        module._as_policy('nope')
    with pytest.raises(ValueError):
        kv_cache_diagnosis('model', sources=['a'], references=[], quant_policy=QuantPolicy.INT8)


def test_lite_cli_registers_kv_diagnose():
    import lmdeploy.cli.lite as lite_cli
    lite_cli.SubCliLite.add_parser_kv_diagnose()
    choices = lite_cli.SubCliLite.subparsers.choices
    assert 'kv_diagnose' in choices
    kv_parser = choices['kv_diagnose']
    args = kv_parser.parse_args(['Qwen/Qwen3-8B', '--dataset', 'rows.json'])
    assert args.model == 'Qwen/Qwen3-8B'
    assert args.dataset == 'rows.json'
    assert args.quant_policy == QuantPolicy.INT8
    assert hasattr(args, 'run') and args.run == lite_cli.SubCliLite.kv_diagnose
