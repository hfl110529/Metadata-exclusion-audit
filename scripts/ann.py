"""Recompute ANN omission bounds, recall and paired labels from saved scores.

Requires NumPy; does not build graphs, execute searches or encode text.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

EFS, DEPTHS, MODES, PHASES = [8, 32, 128], [5, 10, 20, 50], ['included', 'excluded'], [0, 1]
EPS = 1e-6
LABELS = ['supported', 'no_detectable_effect', 'insufficient']
METHODS = ['safe_unknown', 'naive_ann_cutoff', 'scan_certified']


def classify(low, high):
    return ('supported' if low > EPS or high < -EPS else
            'no_detectable_effect' if low >= -EPS and high <= EPS else 'insufficient')


def score_interval(row, target, method):
    if target in row['indices']:
        value = row['scores'][row['indices'].index(target)]
        return value, value
    upper = (math.inf if method == 'safe_unknown' else row[
        'cutoff' if method == 'naive_ann_cutoff' else 'max_omitted'])
    return -math.inf, upper


def paired(before, after, target, method):
    lo0, hi0 = score_interval(before, target, method)
    lo1, hi1 = score_interval(after, target, method)
    lo, hi = lo1 - hi0, hi1 - lo0
    return classify(lo, hi), lo, hi


def json_bound(value):
    return value if math.isfinite(value) else ('-inf' if value < 0 else '+inf')


def stats_ns(values):
    return {'n': len(values), 'min_ms': min(values) / 1e6, 'median_ms': float(np.median(values)) / 1e6,
            'p95_ms': float(np.percentile(values, 95)) / 1e6, 'max_ms': max(values) / 1e6}


def aggregate(rows, decisions):
    result = {'responses': len(rows), 'paired_cells': len(decisions),
              'mean_recall_at_k': float(np.mean([r['recall_at_k'] for r in rows])),
              'strict_bound_violations': sum(r['strict_bound_violation'] for r in rows),
              'epsilon_bound_violations': sum(r['epsilon_bound_violation'] for r in rows),
              'boundary_tie_responses': sum(r['exact_boundary_tie_count'] > 1 for r in rows),
              'ann_search_timing': stats_ns([r['search_ns'] for r in rows]), 'methods': {}}
    for method in METHODS:
        counts = Counter(d[method]['label'] for d in decisions)
        result['methods'][method] = {**{label: counts[label] for label in LABELS},
            'false_supported': sum(d[method]['label'] == 'supported' and d['oracle_label'] != 'supported' for d in decisions),
            'false_no_detectable_effect': sum(d[method]['label'] == 'no_detectable_effect' and d['oracle_label'] != 'no_detectable_effect' for d in decisions),
            'target_containment_failures': sum(not d[method]['contains_target_scores'] for d in decisions)}
    return result


def check(root):
    if not __debug__:
        raise RuntimeError('Run without -O: scientific checks require assertions.')
    base = Path(root) / 'data/ann'
    inputs = json.loads((base / 'inputs.json').read_text())
    protocol = json.loads((base / 'protocol.json').read_text())
    assert protocol['grid'] == {'efSearch': EFS, 'k': DEPTHS, 'policies': MODES, 'phases': PHASES}
    assert protocol['epsilon'] == EPS
    positions = {ident: i for i, ident in enumerate(inputs['document_ids'])}
    query_positions = {ident: i for i, ident in enumerate(inputs['query_ids'])}
    assert len(positions) == len(inputs['document_ids']) == 5183
    assert len(query_positions) == len(inputs['query_ids']) == 16
    assert len(inputs['comparisons']) == len({(p['query_id'], p['target_id']) for p in inputs['comparisons']}) == 18
    full = np.load(base / 'native_full_scores.npz', allow_pickle=False)
    assert set(full.files) == {f'{m}_{p}' for m in MODES for p in PHASES}
    for state in full.files:
        assert full[state].shape == (16, 5183) and np.isfinite(full[state]).all()
    rows = [json.loads(line) for line in (base / 'responses.jsonl').read_text().splitlines()]
    by_key = {(r['policy'], r['phase'], r['efSearch'], r['k'], r['query_id']): r for r in rows}
    expected_keys = {(mode, phase, ef, k, qid) for mode in MODES for phase in PHASES for ef in EFS for k in DEPTHS for qid in query_positions}
    assert len(rows) == len(by_key) == 768 and set(by_key) == expected_keys
    for row in rows:
        exact = full[f"{row['policy']}_{row['phase']}"][query_positions[row['query_id']]]
        assert len(set(row['indices'])) == row['k'] and len(row['scores']) == row['k']
        assert all(0 <= i < len(exact) for i in row['indices'])
        assert all(s == float(exact[i]) for i, s in zip(row['indices'], row['scores']))
        mask = np.ones(len(exact), dtype=bool)
        mask[row['indices']] = False
        maximum = float(np.max(exact[mask]))
        assert maximum == row['max_omitted'] and min(row['scores']) == row['cutoff']
        assert row['strict_bound_violation'] == (maximum > row['cutoff'])
        assert row['epsilon_bound_violation'] == (maximum > row['cutoff'] + EPS)
        rank = np.lexsort((np.arange(len(exact)), -exact))
        assert row['recall_at_k'] == len(set(row['indices']).intersection(rank[:row['k']].tolist())) / row['k']
        assert row['exact_boundary_tie_count'] == int(np.count_nonzero(exact == exact[rank[row['k'] - 1]]))
    decisions = []
    for mode in MODES:
        for ef in EFS:
            for k in DEPTHS:
                for pair in inputs['comparisons']:
                    qi, ti = query_positions[pair['query_id']], positions[pair['target_id']]
                    before, after = [by_key[(mode, phase, ef, k, pair['query_id'])] for phase in PHASES]
                    value0, value1 = [float(full[f'{mode}_{phase}'][qi, ti]) for phase in PHASES]
                    delta = value1 - value0
                    decision = {'policy': mode, 'efSearch': ef, 'k': k, **pair,
                                'oracle_scores': [value0, value1], 'oracle_delta': delta, 'oracle_label': classify(delta, delta),
                                'visible': [ti in before['indices'], ti in after['indices']]}
                    for method in METHODS:
                        label, low, high = paired(before, after, ti, method)
                        contained = all(score_interval(r, ti, method)[0] <= value <= score_interval(r, ti, method)[1]
                                        for r, value in ((before, value0), (after, value1)))
                        decision[method] = {'label': label, 'delta_interval': [json_bound(low), json_bound(high)],
                                            'contains_target_scores': contained}
                        if method != 'naive_ann_cutoff':
                            assert contained and (label == 'insufficient' or label == decision['oracle_label'])
                    decisions.append(decision)
    assert len(decisions) == 432
    assert decisions == json.loads((base / 'decisions.json').read_text())
    saved = json.loads((base / 'summary.json').read_text())
    overall = aggregate(rows, decisions)
    assert overall == saved['overall']
    by_ef = {}
    for ef in EFS:
        by_ef[str(ef)] = aggregate([r for r in rows if r['efSearch'] == ef], [d for d in decisions if d['efSearch'] == ef])
        assert by_ef[str(ef)] == saved['by_efSearch'][str(ef)]
        for k in DEPTHS:
            actual = aggregate([r for r in rows if r['efSearch'] == ef and r['k'] == k], [d for d in decisions if d['efSearch'] == ef and d['k'] == k])
            assert actual == saved['by_efSearch_k'][f'{ef}_{k}']
    for mode in MODES:
        assert aggregate([r for r in rows if r['policy'] == mode], [d for d in decisions if d['policy'] == mode]) == saved['by_policy'][mode]
    scans = [json.loads(line) for line in (base / 'scans.jsonl').read_text().splitlines()]
    direct = [json.loads(line) for line in (base / 'direct_targets.jsonl').read_text().splitlines()]
    assert len(scans) == 64 and len(direct) == 72
    for row in direct:
        assert row['score'] == float(full[row['state']][query_positions[row['query_id']], positions[row['target_id']]])
    assert stats_ns([r['scan_ns'] for r in scans]) == saved['full_native_scan_timing']
    assert stats_ns([r['direct_ns'] for r in direct]) == saved['direct_target_scoring_timing']
    assert stats_ns([r['omitted_max_ns'] for r in rows]) == saved['omitted_max_extraction_timing']
    scan_map = {(r['state'], r['query_id']): r['scan_ns'] for r in scans}
    assert len(scan_map) == 64
    assert stats_ns([scan_map[(f"{r['policy']}_{r['phase']}", r['query_id'])] + r['omitted_max_ns'] for r in rows]) == saved['per_response_exhaustive_fallback_cost_reconstructed_from_shared_scan']
    return {'passed': True, 'responses': len(rows), 'paired_cells': len(decisions),
            'by_efSearch': {ef: {k: result[k] for k in ['mean_recall_at_k', 'epsilon_bound_violations', 'methods']} for ef, result in by_ef.items()},
            'scope': 'Saved ANN responses checked against saved complete native scores; no new ANN search or native scorer execution.'}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
