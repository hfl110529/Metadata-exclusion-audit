"""Recompute depth/precision results and tolerance sensitivity from all saved intervals."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

EPSILON = Fraction(1, 1000000)


def verdict(low, high, epsilon=EPSILON):
    assert low <= high and epsilon >= 0
    return ('supported' if low > epsilon or high < -epsilon else
            'no_detectable_effect' if -epsilon <= low <= high <= epsilon else 'insufficient')


def aggregate(rows, method):
    counts = Counter(r[method] for r in rows)
    return {'episodes': len(rows), 'effect_positive': sum(r['score_effect_gt'] for r in rows),
            'effect_null': sum(not r['score_effect_gt'] for r in rows),
            **{name: counts[name] for name in ['supported', 'no_detectable_effect', 'insufficient']},
            'true_supports': sum(r[method] == 'supported' and r['score_effect_gt'] for r in rows),
            'false_supports': sum(r[method] == 'supported' and not r['score_effect_gt'] for r in rows),
            'false_no_effect_claims': sum(r[method] == 'no_detectable_effect' and r['score_effect_gt'] for r in rows)}


def check(root):
    output = {}
    for name, count in [('natural', 1728), ('engineered', 528)]:
        base = root / 'data/precision'
        rows = [json.loads(line) for line in (base / f'{name}.jsonl').read_text().splitlines()]
        summary = json.loads((base / f'{name}_summary.json').read_text())
        assert len(rows) == 5 * count
        for r in rows:
            low, high = map(Fraction, r['interval_bounds'])
            delta = Fraction(r['exact_delta'])
            assert r['valid'] and low <= delta <= high
            assert delta == Fraction(r['full_target_score1']) - Fraction(r['full_target_score0'])
            assert r['interval'] == verdict(low, high)
            assert r['score_effect_gt'] == (abs(delta) > EPSILON)
            point = r['rounded_point_delta']
            assert r['rounded_point'] == ('insufficient' if point is None else
                                           verdict(Fraction(point), Fraction(point)))
            assert r['point_zero'] == ('insufficient' if point is None else
                                       'supported' if Fraction(point) != 0 else 'no_detectable_effect')
            assert r['rounded_cutoff_exact'] == verdict(*map(Fraction, r['rounded_cutoff_exact_bounds']))
        table = []
        for precision in ['full', '6', '4', '3', '2']:
            group = [r for r in rows if r['precision'] == precision]
            assert len(group) == count
            methods = summary['projections']['by_precision'][precision]['methods']
            for method in ['interval', 'rounded_point', 'rounded_cutoff_exact', 'point_zero']:
                assert all(methods[method][k] == v for k, v in aggregate(group, method).items())
            counts = aggregate(group, 'interval')
            assert counts['false_supports'] == counts['false_no_effect_claims'] == 0
            table.append({'precision': precision, **counts,
                          'point_false_no_effect': aggregate(group, 'rounded_point')['false_no_effect_claims'],
                          'controls': {m: aggregate(group, m) for m in ['rounded_point', 'rounded_cutoff_exact', 'point_zero']}})
        sensitivity = []
        for precision in ['full', '6', '4', '3', '2']:
            group = [r for r in rows if r['precision'] == precision]
            for text in ['0', '1e-8', '1e-7', '1e-6', '1e-5', '1e-4']:
                epsilon = Fraction(text)
                labels = [verdict(*map(Fraction, r['interval_bounds']), epsilon) for r in group]
                effects = [abs(Fraction(r['exact_delta'])) > epsilon for r in group]
                assert all(label == 'insufficient' or (label == 'supported') == effect
                           for label, effect in zip(labels, effects))
                sensitivity.append({'precision': precision, 'epsilon': text,
                                    'effects': sum(effects), **dict(Counter(labels))})
        output[name] = {'episodes': count, 'precision_rows': table, 'tolerance_sensitivity': sensitivity}
        if name == 'natural':
            depth_rows = []
            for size in [128, 512, 1024]:
                for depth in [5, 10, 50]:
                    group = [r for r in rows if r['precision'] == 'full' and
                             r['corpus_size'] == size and r['top_k'] == depth]
                    assert len(group) == 192
                    methods = summary['projections']['by_size_precision_k'][f'{size}/full/{depth}']['methods']
                    for method in ['interval', 'rounded_point', 'rounded_cutoff_exact', 'point_zero']:
                        assert all(methods[method][k] == v for k, v in aggregate(group, method).items())
                    point, interval = aggregate(group, 'rounded_point'), aggregate(group, 'interval')
                    depth_rows.append({'corpus_size': size, 'depth': depth,
                                       'point_T': point['true_supports'], 'interval_T': interval['true_supports'],
                                       'point_I': point['insufficient'], 'interval_I': interval['insufficient']})
            output[name]['depth_rows'] = depth_rows
    return {'passed': True, 'epsilon': float(EPSILON), **output}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
