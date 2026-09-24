"""Cross-check nine numerical tables supported by public evidence and figure exports."""
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import re


def check(root):
    def read(name):
        return json.loads((root / name).read_text())

    tables = read('results/paper_tables.json')
    checked = []

    def expected(label, rows):
        def norm(value):
            text = str(value).strip().replace('---', '—')
            if re.fullmatch(r'[+-]?\d[\d,]*(?:\.\d*)?(?:e[+-]?\d+)?', text):
                text = format(float(text.replace(',', '')), '.15g')
            return text
        assert [[norm(cell) for cell in row] for row in rows] == tables[label]['rows'], label
        checked.append(label)

    data = read('data/attribution/summary.json'); rows = []
    for label, key in [('Rank change','rank_change'),('Pair-gap change','pair_gap_change'),('Target, nonzero','target_zero'),('Target, guarded point','target_point_epsilon'),('Target, interval','target_interval'),('Interval + null gate','target_interval_null_gate')]:
        row = [label]
        for depth in ['10', '50']:
            r = data['by_k'][depth]['methods'][key]
            row += [r['true_supports'], r['false_supports'], r['verdicts']['insufficient']]
        rows.append(row)
    expected('tab:paired-audit', rows)
    data = read('data/precision/natural_summary.json'); rows = []
    for size in [128, 512, 1024]:
        for depth in [5, 10, 50]:
            r = data['projections']['by_size_precision_k'][f'{size}/full/{depth}']['methods']
            point, interval = r['rounded_point'], r['interval']
            rows.append([size, depth, point['true_supports'], interval['true_supports'], point['insufficient'], interval['insufficient']])
    expected('tab:boundary-depth', rows)
    rows = []
    for precision in ['full', '6', '4', '3', '2']:
        r = data['projections']['by_precision'][precision]['methods']; interval = r['interval']
        rows.append([precision, interval['true_supports'], interval['no_detectable_effect'], interval['insufficient'], r['rounded_point']['false_no_effect_claims']])
    expected('tab:precision', rows)
    rows = [['Exact native scores', '0', 'epsilon >= 0']]
    for decimals in [8, 6, 4]:
        assert 2 * Fraction(1, 2 * 10**decimals) == Fraction(1, 10**decimals)
        rows.append([f'Nearest rounding, {decimals} decimals', f'10^-{decimals}', f'epsilon >= 10^-{decimals}'])
    rows.append(['Same rounding step h, additional errors eta_0, eta_1', 'h+eta_0+eta_1', 'epsilon >= w'])
    expected('tab:precision-guidance', rows)
    data = read('data/scifact/decisions.json'); rows = []
    for size in [128, 512, 5183]:
        row = [size]; extra = 0
        for depth in [5, 10, 20, 50]:
            counts = []
            for engine in ['haystack_cosine', 'faiss_flatip']:
                group = [r for r in data if r['engine'] == engine and r['load'] == f'n{size}' and r['depth'] == depth]
                assert len(group) == 36
                counts.append((sum(r['point'] != 'insufficient' for r in group), sum(r['interval'] != 'insufficient' for r in group), sum(r['extra_supported'] for r in group)))
            assert counts[0] == counts[1]
            row.append(f'{counts[0][0]} / {counts[0][1]}'); extra += counts[0][2]
        rows.append(row + [extra])
    expected('tab:scifact', rows)
    data = read('data/geometry/analysis.json'); rows = []
    for engine, label in [('faiss_flatip', 'FlatIP'), ('faiss_hnsw', 'HNSW')]:
        for depth in [5, 10, 20, 50]:
            r = data['by_engine_depth'][f'{engine}_k{depth}']; counts = r['counts']['A_range']
            assert r['cells'] == 610 and all(r['counts'][m] == counts for m in ['B_single', 'P_simplex', 'C_joint'])
            available = r['anchor_availability']
            rows.append([label, depth, ' / '.join(str(counts.get(k, 0)) for k in ['supported', 'no_detectable_effect', 'insufficient']), available['missing_target_phase_scores'], available['with_visible_anchors']])
    expected('tab:geometry-main', rows)
    sql = read('data/access_sql/summary.json')['routes']
    qdrant = read('data/access_qdrant/summary.json')['costs']
    rows = []
    for route, role, title in [
        ('batch_top_ten','P','Top ten + valid cutoffs'),
        ('batch_target_scores','P+','Target scores'),
        ('batch_filtered_full_scores','P+','Full scores, then target filter'),
        ('batch_full_scores','P+','Full scores'),
        ('batch_target_vectors','P+','Target-vector export + cosine')]:
        old,new = [next(r for r in sql if r['route']==route and r['stage']==stage) for stage in ['old','new']]
        rows.append([role,title,f"{old['labels'].get('effect',0)} / {new['labels'].get('no_effect',0)}",new['calls'],
                     '---' if route.endswith('vectors') else new['returned_scores'],new['json_value_bytes'],f"{new['median_seconds']:.3f}"])
    for route,title in [('exact10','Exact top ten + cutoffs'),('query_batch','Batched target scores')]:
        old,new = [next(r for r in qdrant if r['route']==route and r['variant']==stage) for stage in ['old','new']]
        rows.append(['Q',title,f"{old['decisions'].get('effect',0)} / {new['decisions'].get('no_effect',0)}",
                     new['requests'],new['native_score_values'],new['response_body_bytes'],f"{new['client_seconds']:.3f}"])
    expected('tab:access-routes',rows)
    bounds=read('data/ann/recall_bounds.json')['strata']
    expected('tab:ann-recall-bound',[[r['k'],100*r['mean_recall_at_k'],f"{r['epsilon_cutoff_invalid']}/64",100*r['empirical_bound_min_1_mean_X']]
                                   for r in bounds if r['efSearch']==128])
    episodes=[json.loads(line) for line in (root/'data/precision/natural.jsonl').read_text().splitlines()]
    rows=[]
    for tolerance in ['1e-8','1e-7','1e-6','1e-5','1e-4']:
        epsilon=Fraction(tolerance); row=[tolerance]
        for precision in ['full','6']:
            counts=Counter()
            for r in episodes:
                if r['precision']!=precision: continue
                low,high=map(Fraction,r['interval_bounds'])
                counts['T' if low>epsilon or high< -epsilon else 'N' if -epsilon<=low<=high<=epsilon else 'I']+=1
            row.append('/'.join(str(counts[k]) for k in 'TNI'))
        rows.append(row)
    expected('tab:epsilon-sensitivity',rows)
    assert set(checked) == set(tables)
    ann = read('data/ann/summary.json')
    with (root / 'results/ann_cutoffs.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    for row, ef in zip(rows, [8, 32, 128]):
        source = ann['by_efSearch'][str(ef)]
        assert int(row['efSearch']) == ef and int(row['responses']) == source['responses'] == 256
        assert float(row['mean_recall_at_k']) == source['mean_recall_at_k']
        assert int(row['invalid_cutoffs']) == source['epsilon_bound_violations']
        assert float(row['invalid_cutoff_fraction']) == source['epsilon_bound_violations'] / 256
    return {'passed': True, 'numerical_tables': len(checked), 'table_labels': checked, 'ann_figure_rows': 3}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
