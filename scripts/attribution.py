"""Recompute attribution rules from saved per-comparison scores, ranks and intervals."""
from collections import Counter
import json
import math
from pathlib import Path

EPSILON = 1e-6


def verdict(low, high):
    return ('supported' if low > EPSILON or high < -EPSILON else
            'no_detectable_effect' if -EPSILON <= low <= high <= EPSILON else 'insufficient')


def check(root):
    base = root / 'data/attribution'
    rows = [json.loads(line) for line in (base / 'episodes.jsonl').read_text().splitlines()]
    summary = json.loads((base / 'summary.json').read_text())
    assert len(rows) == summary['denominators']['paired_episodes'] == 1152
    aliases = {'rank_change': 'rank_change', 'pair_gap_change': 'pair_gap_change',
               'target_zero': 'target_delta_unguarded', 'target_point_epsilon': 'target_delta_guarded_visible',
               'target_interval': 'target_delta_guarded_bound',
               'target_interval_null_gate': 'target_delta_guarded_bound_null_gate'}
    nulls = {(r['block_id'], r['query_id'], r['top_k']): r for r in rows if r['cohort'] == 'equalnull'}
    for r in rows:
        assert not r['query_vector_changed']
        assert r['body_equal'] or r['cohort'] == 'bodybias'
        assert r['score_effect_gt'] == (abs(r['full_target_delta']) > EPSILON)
        a, b = r['target0'], r['target1']
        rank = 'insufficient' if not a['visible'] and not b['visible'] else (
            'supported' if (a['rank'] if a['visible'] else r['top_k'] + 1) !=
                           (b['rank'] if b['visible'] else r['top_k'] + 1) else 'no_detectable_effect')
        assert r['rank_change'] == rank
        if a['visible'] and b['visible']:
            delta = b['score'] - a['score']
            assert r['target_delta'] == delta
            assert r['target_delta_unguarded'] == ('supported' if delta != 0 else 'no_detectable_effect')
            assert r['target_delta_guarded_visible'] == verdict(delta, delta)
        else:
            assert r['target_delta_unguarded'] == r['target_delta_guarded_visible'] == 'insufficient'
        if all(r[x]['visible'] for x in ['target0', 'target1', 'reference0', 'reference1']):
            delta = (b['score'] - r['reference1']['score']) - (a['score'] - r['reference0']['score'])
            assert r['pair_gap_change'] == verdict(delta, delta)
        else:
            assert r['pair_gap_change'] == 'insufficient'
        low, high = r['target_delta_interval']
        low = -math.inf if low is None else low
        high = math.inf if high is None else high
        assert low <= r['full_target_delta'] <= high
        assert r['target_delta_guarded_bound'] == verdict(low, high)
        control = nulls[r['block_id'], r['query_id'], r['top_k']]['target_delta_guarded_bound']
        assert r['null_control_verdict'] == control
        assert r['target_delta_guarded_bound_null_gate'] == (
            r['target_delta_guarded_bound'] if control == 'no_detectable_effect' else 'insufficient')
    output = []
    for depth in [10, 50]:
        group = [r for r in rows if r['top_k'] == depth]
        assert len(group) == 576
        for method, field in aliases.items():
            counts = {label: sum(r[field] == label for r in group)
                      for label in ['supported', 'no_detectable_effect', 'insufficient']}
            target = summary['by_k'][str(depth)]['methods'][method]
            assert all(v == target['verdicts'].get(k, 0) for k, v in counts.items())
            derived = {'true_supports': sum(r[field] == 'supported' and r['score_effect_gt'] for r in group),
                       'false_supports': sum(r[field] == 'supported' and not r['score_effect_gt'] for r in group),
                       'false_no_effect_claims': sum(r[field] == 'no_detectable_effect' and r['score_effect_gt'] for r in group)}
            assert all(target[k] == v for k, v in derived.items())
            output.append({'depth': depth, 'method': method, **counts, **derived})
    return {'passed': True, 'comparisons': len(rows), 'rows': output,
            'scope': 'Saved per-comparison values; raw service collection is not repeated.'}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
