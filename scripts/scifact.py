"""Recompute the SciFact exact-search results from saved evidence (NumPy).

Checks source selection, encoder captures, complete scores, public cutoffs and
all paired decisions. Does not re-encode text, execute searches or load a tokenizer.
"""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np

EPS = Fraction(1, 1000000)
MODES = ['included', 'excluded']
ENGINES = ['haystack_cosine', 'faiss_flatip']
PHASES = [0, 1]
DEPTHS = [5, 10, 20, 50]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def verdict(low, high):
    if low > EPS or high < -EPS:
        return 'supported'
    if low >= -EPS and high <= EPS:
        return 'no_detectable_effect'
    return 'insufficient'


def check(root):
    base = Path(root) / 'data/scifact'
    manifest = read(base / 'manifest.json')
    selection = read(base / 'selection.json')
    need(manifest['modes'] == MODES and manifest['engines'] == ENGINES
         and manifest['phases'] == PHASES and manifest['depths'] == DEPTHS
         and manifest['epsilon'] == float(EPS), 'Declared computation grid differs.')
    need(manifest['field'] == 'Title' and manifest['body_prefix_wordpiece_tokens'] == 160
         and manifest['title_prefix_wordpiece_tokens'] == 48, 'Field/prefix protocol differs.')
    for key in ['queries', 'comparisons', 'loads']:
        need(manifest[key] == selection[key], f'Selection differs from manifest: {key}')
    with zipfile.ZipFile(base / 'scifact.zip') as archive:
        original = {d['_id']: d for d in map(json.loads, archive.read('scifact/corpus.jsonl').splitlines())}
        source_queries = {q['_id']: q for q in map(json.loads, archive.read('scifact/queries.jsonl').splitlines())}
        positive = defaultdict(set)
        for row in csv.DictReader(io.StringIO(archive.read('scifact/qrels/test.tsv').decode()), delimiter='\t'):
            if int(row['score']) > 0:
                positive[row['query-id']].add(row['corpus-id'])
    expected_queries = [{'id': q, 'text': source_queries[q]['text'], 'target_ids': sorted(positive[q], key=int)[:2]}
                        for q in sorted(positive, key=int)[:16]]
    need(manifest['queries'] == expected_queries, 'Source qrel/ID selection rule was not followed.')
    expected_comparisons = [{'query_id': q['id'], 'target_id': target} for q in expected_queries for target in q['target_ids']]
    need(manifest['comparisons'] == expected_comparisons, 'Qrel target selection rule was not followed.')
    source_targets = {x['target_id'] for x in expected_comparisons}
    ordered_source = sorted(source_targets, key=int) + sorted(set(original) - source_targets, key=int)
    expected_loads = [{'name': f'n{n}', 'size': n, 'document_ids': sorted(ordered_source[:n], key=int)} for n in [128, 512, 5183]]
    need(manifest['loads'] == expected_loads, 'Nested candidate sets differ from source ID selection.')
    counts = {'corpus_records': len(original), 'all_source_queries': len(source_queries),
              'positive_test_queries': len(positive), 'queries': len(expected_queries),
              'target_query_pairs': len(expected_comparisons), 'unique_targets': len(source_targets)}
    need(counts == selection['counts'], 'Source selection counts differ.')
    documents = manifest['documents']
    ids = [d['id'] for d in documents]
    need(ids == sorted(original, key=int) and len(set(ids)) == 5183, 'Corpus identity or ordering changed.')
    id_pos = {ident: i for i, ident in enumerate(ids)}
    for d in documents:
        need(set(d) == {'id', 'content', 'original_title', 'Title'}, 'Unexpected document metadata.')
        source = original[d['id']]
        title = d['Title'][1]
        need(d['original_title'] == source['title'] and d['content']
             and source['text'].startswith(d['content']) and title and source['title'].startswith(title),
             f'Wrong original title/body source prefix: {d["id"]}')
        need(d['Title'] == ['Administrative record.' if d['id'] in source_targets else title, title],
             f'Wrong field contrast: {d["id"]}')
    # Saved evidence checks source prefixes; tokenizer boundaries require the encoder environment.
    need(0 < manifest['max_input_tokens'] <= 256, 'Declared encoder input length exceeds model limit.')
    query_ids = [q['id'] for q in manifest['queries']]
    query_pos = {ident: i for i, ident in enumerate(query_ids)}
    loads = {x['name']: x for x in manifest['loads']}
    load_pos = {name: {ident: i for i, ident in enumerate(x['document_ids'])} for name, x in loads.items()}
    traces = read(base / 'encoder_trace.json')
    vectors = np.load(base / 'vectors.npz', allow_pickle=False)
    need(set(vectors.files) == {f'{m}_{p}' for m in MODES for p in PHASES} | {'queries'}, 'Incomplete saved vector grid.')
    need(len(traces) == 20, 'Unexpected number of native embed calls.')
    for array in vectors.values():
        need(array.dtype == np.float32 and np.isfinite(array).all(), 'Vectors must be finite captured float32 values.')
    for i, (mode, phase) in enumerate(itertools.product(MODES, PHASES)):
        trace, array = traces[i], vectors[f'{mode}_{phase}']
        need(trace['kind'] == 'documents' and trace['mode'] == mode and trace['phase'] == phase, 'Document trace binding differs.')
        need(trace['texts'] == [(d['Title'][phase] + '\n' if mode == 'included' else '') + d['content'] for d in documents],
             'Captured native encoder text differs from the frozen composed input.')
        need(array.shape == (5183, 384) and trace['shape'] == list(array.shape)
             and trace['vectors_sha256'] == hashlib.sha256(array.tobytes()).hexdigest(), 'Captured document vector hash/shape differs.')
    need(vectors['queries'].shape == (16, 384), 'Query vector shape differs.')
    for i, q in enumerate(manifest['queries']):
        trace, array = traces[4 + i], vectors['queries'][i:i + 1]
        need(trace['kind'] == 'query' and trace['query_id'] == q['id'] and trace['texts'] == [q['text']], 'Captured query binding differs.')
        need(trace['shape'] == list(array.shape) and trace['vectors_sha256'] == hashlib.sha256(array.tobytes()).hexdigest(),
             'Captured query vector hash/shape differs.')
    inventory = read(base / 'write_inventory.json')
    inventory_keys = [(x['mode'], x['phase'], x['load']) for x in inventory]
    need(len(inventory_keys) == len(set(inventory_keys)) == 12 and set(inventory_keys) == set(itertools.product(MODES, PHASES, loads)),
         'Write inventory has duplicate or missing states.')
    for x in inventory:
        need(x['haystack_count'] == x['faiss_count'] == x['expected'] == loads[x['load']]['size'], 'Native stored counts differ.')

    full = np.load(base / 'full_scores.npz', allow_pickle=False)
    full_keys = {f'{e}__{m}__{p}__{load}' for e, m, p, load in itertools.product(ENGINES, MODES, PHASES, loads)}
    need(set(full.files) == full_keys, 'Saved complete score matrices do not cover exactly the declared grid.')
    max_reconstruction_error = {e: 0.0 for e in ENGINES}
    normalized_queries = vectors['queries'].astype(np.float64)
    normalized_queries /= np.linalg.norm(normalized_queries, axis=1, keepdims=True)
    for engine, mode, phase, load in itertools.product(ENGINES, MODES, PHASES, loads):
        recorded = full[f'{engine}__{mode}__{phase}__{load}']
        need(recorded.shape == (16, loads[load]['size']) and np.isfinite(recorded).all(), 'Complete scores are missing or nonfinite.')
        matrix = vectors[f'{mode}_{phase}'][[id_pos[x] for x in loads[load]['document_ids']]].astype(np.float64)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        error = float(np.max(np.abs(recorded - normalized_queries @ matrix.T)))
        max_reconstruction_error[engine] = max(max_reconstruction_error[engine], error)
        need(error <= float(EPS), f'Saved {engine} complete scores disagree with frozen vectors: {error}')
        need((recorded >= -1).all() and (recorded <= 1).all(), 'Score table falls outside declared interval domain.')

    public = read(base / 'public_responses.json')
    responses = {}
    cutoff_ties = Counter()
    expected_public = set(itertools.product(ENGINES, MODES, PHASES, loads, query_ids, DEPTHS))
    for r in public:
        key = tuple(r[x] for x in ['engine', 'mode', 'phase', 'load', 'query_id', 'depth'])
        need(key in expected_public and key not in responses, 'Public response has an undeclared or duplicate identity.')
        returned = r['documents']
        returned_ids = [x['id'] for x in returned]
        need(len(returned_ids) == len(set(returned_ids)) == r['depth'], 'Public response length/identity differs.')
        table = full[f'{r["engine"]}__{r["mode"]}__{r["phase"]}__{r["load"]}'][query_pos[r['query_id']]]
        positions = load_pos[r['load']]
        need(set(returned_ids) <= set(positions), 'Public response contains an unknown document.')
        need(all(d['score'] == float(table[positions[d['id']]]) for d in returned), 'Public score differs from its native complete score.')
        need(all(a['score'] >= b['score'] for a, b in zip(returned, returned[1:])), 'Public result is not score sorted.')
        cutoff = returned[-1]['score']
        need(all(float(table[i]) <= cutoff for ident, i in positions.items() if ident not in returned_ids), 'Exact top-k cutoff bound is violated.')
        tied = int(np.count_nonzero(table == cutoff)) > 1
        cutoff_ties[r['engine']] += tied
        responses[key] = (r, {d['id']: d['score'] for d in returned}, cutoff, tied)
    need(set(responses) == expected_public, 'Public retrieval grid is incomplete.')
    decisions = []
    for engine, mode, load, depth, comparison in itertools.product(ENGINES, MODES, loads, DEPTHS, manifest['comparisons']):
        qid, target = comparison['query_id'], comparison['target_id']
        scores, bounds, visible, cutoffs, ties = [], [], [], [], []
        for phase in PHASES:
            r, returned, cutoff, tied = responses[(engine, mode, phase, load, qid, depth)]
            exact = Fraction(float(full[f'{engine}__{mode}__{phase}__{load}'][query_pos[qid], load_pos[load][target]]))
            is_visible = target in returned
            interval = (Fraction(returned[target]),) * 2 if is_visible else (Fraction(-1), Fraction(cutoff))
            need(interval[0] <= exact <= interval[1], 'A target score falls outside its public evidence interval.')
            scores.append(exact); bounds.append(interval); visible.append(is_visible); cutoffs.append(cutoff); ties.append(tied)
        delta = scores[1] - scores[0]
        low, high = bounds[1][0] - bounds[0][1], bounds[1][1] - bounds[0][0]
        outcome = verdict(low, high)
        gold = verdict(delta, delta)
        point = gold if all(visible) else 'insufficient'
        need(low <= delta <= high, 'Oracle delta falls outside the difference interval.')
        distances = [abs(float(scores[i]) - cutoffs[i]) for i in PHASES]
        decisions.append({'engine': engine, 'mode': mode, 'load': load, 'depth': depth, 'query_id': qid, 'target_id': target,
             'target_visible': visible, 'native_target_scores': [float(x) for x in scores], 'cutoffs': cutoffs,
             'cutoff_ties': ties, 'target_cutoff_distances': distances,
             'difference_interval': [float(low), float(high)], 'exact_fraction_interval': [str(low), str(high)],
             'native_delta': float(delta), 'gold': gold, 'interval': outcome, 'point': point,
             'extra_supported': outcome == 'supported' and point == 'insufficient',
             'extra_no_effect': outcome == 'no_detectable_effect' and point == 'insufficient'})
    need(len(decisions) == 864, 'Wrong number of paired judgment cells.')

    def aggregate(rows):
        added = [r for r in rows if r['extra_supported'] or r['extra_no_effect']]
        counts = {'cells': len(rows), 'interval': dict(Counter(r['interval'] for r in rows)),
                  'point': dict(Counter(r['point'] for r in rows)), 'gold': dict(Counter(r['gold'] for r in rows)),
                  'extra_supported': sum(r['extra_supported'] for r in rows),
                  'extra_no_effect': sum(r['extra_no_effect'] for r in rows),
                  'false_supported': sum(r['interval'] == 'supported' and r['gold'] != 'supported' for r in rows),
                  'false_no_effect': sum(r['interval'] == 'no_detectable_effect' and r['gold'] != 'no_detectable_effect' for r in rows),
                  'both_visible': sum(all(r['target_visible']) for r in rows),
                  'gained_query_ids': sorted({r['query_id'] for r in added}, key=int),
                  'gained_pairs': sorted({(r['query_id'], r['target_id']) for r in added}, key=lambda p: (int(p[0]), int(p[1]))),
                  'nonzero_nontied_extra_supported': sum(r['extra_supported'] and min(r['target_cutoff_distances']) > 0 and not any(r['cutoff_ties']) for r in rows)}
        counts['point_decidable'] = counts['cells'] - counts['point'].get('insufficient', 0)
        counts['interval_decidable'] = counts['cells'] - counts['interval'].get('insufficient', 0)
        counts['point_decidable_percent'] = counts['point_decidable'] / counts['cells'] * 100
        counts['interval_decidable_percent'] = counts['interval_decidable'] / counts['cells'] * 100
        return counts

    def grouped(keys):
        groups = defaultdict(list)
        for r in decisions:
            groups[tuple(r[k] for k in keys)].append(r)
        return [{**dict(zip(keys, values)), **aggregate(rows)} for values, rows in groups.items()]

    need(decisions == read(base / 'decisions.json'), 'Recomputed paired decisions differ from saved results.')
    summary = {
        'dataset': 'BEIR SciFact', 'counts': counts, 'epsilon': float(EPS),
        'public_responses': len(public), 'complete_responses': len(full_keys) * len(query_ids),
        'paired_cells': len(decisions),
        'cutoff_tied_public_responses_by_engine': {e: cutoff_ties[e] for e in ENGINES},
        'overall': aggregate(decisions), 'by_engine': grouped(['engine']),
        'by_engine_mode': grouped(['engine', 'mode']),
        'by_engine_load_depth': grouped(['engine', 'load', 'depth']),
        'by_engine_query': grouped(['engine', 'query_id']),
        'by_engine_pair': grouped(['engine', 'query_id', 'target_id'])}
    saved = read(base / 'summary.json')
    # JSON stores gained-pair tuples as arrays.
    summary = json.loads(json.dumps(summary))
    for key, value in summary.items():
        need(value == saved[key], f'Recomputed summary differs: {key}')
    need(summary['public_responses'] == 1536 and summary['complete_responses'] == 384,
         'Unexpected response counts.')
    need(summary['overall']['false_supported'] == summary['overall']['false_no_effect'] == 0,
         'Public verdicts disagree with the native complete-score oracle.')
    return {'passed': True, 'counts': counts, 'public_responses': len(public),
            'complete_responses': summary['complete_responses'], 'paired_cells': len(decisions),
            'overall': summary['overall'], 'by_engine': summary['by_engine'],
            'maximum_native_score_reconstruction_error': max_reconstruction_error,
            'model_reencoded': False, 'tokenizer_boundaries_recomputed': False}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
