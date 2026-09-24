"""Recompute public saved-evidence transitions, recall bounds and rank margins."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

EPS = Fraction(1, 1_000_000)


def label(bounds):
    lo, hi = [float(v) if v in ('-inf', '+inf') else Fraction(v) for v in bounds]
    return 'T' if lo > EPS or hi < -EPS else 'N' if lo >= -EPS and hi <= EPS else 'I'


def check(root):
    def read(name):
        return json.loads((root/'data'/name).read_text())
    def lines(name):
        return [json.loads(line) for line in (root/'data'/name).read_text().splitlines()]
    ann = lines('ann/responses.jsonl')
    bound = read('ann/recall_bounds.json')
    for summary in bound['strata']+bound['mixed_depth_by_efSearch']+[bound['overall']]:
        rows=[r for r in ann if ('efSearch' not in summary or r['efSearch']==summary['efSearch']) and ('k' not in summary or r['k']==summary['k'])]
        misses=[round(r['k']*(1-r['recall_at_k'])) for r in rows]
        assert len(rows)==summary['responses']
        assert sum(misses)==summary['missed_true_topk_total']
        assert dict(Counter(map(str,misses)))==summary['missed_true_topk_histogram']
        assert Fraction(summary['empirical_bound_exact'])==min(1,Fraction(sum(misses),len(rows)))
        assert sum(r['epsilon_bound_violation'] for r in rows)==summary['epsilon_cutoff_invalid']
        assert sum(r['strict_bound_violation'] for r in rows)==summary['strict_cutoff_invalid']
        assert all(not r['epsilon_bound_violation'] or r['strict_bound_violation'] for r in rows)
        assert all(not r['strict_bound_violation'] or m>=1 for r,m in zip(rows,misses))
    transitions=lines('scifact/transitions.jsonl')
    for row in transitions:
        for state in ['included','excluded']:
            r=row[state]
            delta=Fraction.from_float(r['native_scores'][1])-Fraction.from_float(r['native_scores'][0])
            assert delta==Fraction(r['native_delta_exact'])
            assert r['truth']==label([delta,delta])
            assert sum(r['visible'])==r['visible_count']
            for method in r['methods'].values():
                assert method['label']==label(method['delta_interval_exact'])
    for summary in read('scifact/transitions_summary.json')['summaries']:
        rows=[r for r in transitions if (r['channel'],r['efSearch'],r['k'])==(summary['channel'],summary['efSearch'],summary['k'])]
        assert len(rows)==summary['pairs']==18
        visibility=[[sum(r['included']['visible_count']==i and r['excluded']['visible_count']==j for r in rows) for j in range(3)] for i in range(3)]
        assert visibility==summary['visibility_transition']
        for name,method in summary['methods'].items():
            matrix=[[sum(r['included']['methods'][name]['label']==i and r['excluded']['methods'][name]['label']==j for r in rows) for j in 'TNI'] for i in 'TNI']
            assert matrix==method['transition']
    margins=read('reader_repair/rank_margins.json')
    for stage in margins['stages']:
        rows=stage['per_query']
        assert len(rows)==24
        for r in rows:
            assert r['score_bound_pass']==(Fraction(r['max_change_exact'])<=EPS)
            assert r['strict_margin_pass']==(Fraction(r['baseline_gap_exact'])>2*EPS)
            assert r['equation_2_pass']==(r['score_bound_pass'] and r['strict_margin_pass'])
        assert sum(r['equation_2_pass'] for r in rows)==stage['equation_2_pass_queries']
        assert sum(r['same_top10_membership'] for r in rows)==stage['same_top10_membership_queries']
    geometry=read('geometry/analysis.json')['rows']
    saved=read('geometry/double_omissions.json')
    for engine, summary in [(None,saved['overall']),*saved['by_engine'].items()]:
        rows=[r for r in geometry if (engine is None or r['engine']==engine) and all(e is not None for e in r['bound_evidence'])]
        assert len(rows)==summary['double_omission_cells']
        assert len({(r['query_id'],r['target_id']) for r in rows})==summary['double_omission_unique_query_target_pairs']
        for method in summary['methods']:
            assert dict(Counter(r['labels'][method] for r in rows))==summary['methods'][method]['double_omission_labels']
            assert sum(r['difference_intervals'][method]!=r['difference_intervals']['A_range'] for r in rows)==summary['methods'][method]['double_omission_narrower_difference_intervals']
        assert sum(r['difference_intervals']['C_joint'][0]<=0<=r['difference_intervals']['C_joint'][1] for r in rows)==summary['double_omission_C_intervals_containing_zero']
    return {'passed': True, 'ann_strata':12, 'scifact_transition_cells':len(transitions),
            'rank_margin_queries':120, 'geometry_double_omissions':saved['overall']['double_omission_cells']}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]),indent=2))
