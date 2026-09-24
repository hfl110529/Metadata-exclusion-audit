#!/usr/bin/env python3
"""Independent checker: saved HTTP evidence only; no server or collector import."""
import collections
import gzip
import hashlib
import json
import math
import pathlib
import statistics

BASE = pathlib.Path(__file__).resolve().parents[1] / 'data/access_qdrant'
EPS = 1e-6


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def points(record):
    result = record['response']['result']
    return result if isinstance(result, list) else result['points']


def cosine(a, b):
    return math.fsum(x*y for x, y in zip(a, b))/math.sqrt(math.fsum(x*x for x in a)*math.fsum(x*x for x in b))


def verdict(a, b):
    low, high = b[0]-a[1], b[1]-a[0]
    return 'effect' if low > EPS or high < -EPS else 'no_effect' if low >= -EPS and high <= EPS else 'insufficient'


def value_counts(value):
    scores = vectors = 0
    if isinstance(value, dict):
        scores += int('score' in value)
        if isinstance(value.get('vector'), list):
            vectors += len(value['vector'])
        for k, v in value.items():
            if k != 'vector':
                a, b = value_counts(v)
                scores += a
                vectors += b
    elif isinstance(value, list):
        for v in value:
            a, b = value_counts(v)
            scores += a
            vectors += b
    return scores, vectors


def check(root):
    global BASE
    BASE = pathlib.Path(root)/'data/access_qdrant'
    protocol = read(BASE/'protocol.json')
    records=[json.loads(line) for line in gzip.decompress((BASE/'responses.jsonl.gz').read_bytes()).splitlines()]
    saved = read(BASE/'summary.json')
    assert len(records) == saved['records']
    for i, r in enumerate(records):
        assert r['sequence'] == i
        try:
            parsed = json.loads(r['response_raw'])
        except json.JSONDecodeError:
            parsed = {'raw_text': r['response_raw']}
        r['response'] = parsed
        assert len(r['response_raw'].encode()) == r['response_body_bytes']
        request_bytes = 0 if r['request'] is None else len(json.dumps(r['request'], separators=(',', ':')).encode())
        assert request_bytes == r['request_body_bytes']
    auth = [r for r in records if r['phase'] == 'auth']
    assert len(auth) == 10 and all(r['status'] in [401, 403] for r in auth)
    formal = [r for r in records if r['phase'] == 'formal']
    assert all(r['status'] == 200 and r['role'] == 'reader' for r in formal)
    identity = read(BASE/'identity.json')
    targets = {q['query_id']: q['target_ids'] for q in identity['queries']}
    queries = {q['query_id']: q['embedding'] for q in read(BASE.parent/'reader_repair/snapshots/queries.json')}
    truth = {(r['variant'], r['world'], r['query_id']): {p['id']: p['score'] for p in points(r)} for r in records if r['phase'] == 'truth'}
    assert len(truth) == 96 and all(len(p) == 192 for p in truth.values())
    states = {(r['phase'], r['variant'], r['world']): points(r) for r in records if r['phase'].startswith('state_')}
    assert len(states) == 8
    for variant in ['old', 'new']:
        for w in range(2):
            assert states['state_before', variant, w] == states['state_after', variant, w]
            source = read(BASE.parent/f'reader_repair/snapshots/world{w}_{variant}.json')
            source_by_uid = {d['meta']['SOPInstanceUID']: d['embedding'] for d in source}
            native = states['state_before', variant, w]
            assert {p['payload']['uid'] for p in native} == set(source_by_uid)
            for p in native:
                vector = source_by_uid[p['payload']['uid']]
                norm = math.sqrt(math.fsum(x*x for x in vector))
                assert len(p['vector']) == len(vector) == 384
                assert max(abs(a-b/norm) for a, b in zip(p['vector'], vector)) < 1e-6
    repaired_vectors = [{p['id']: p['vector'] for p in states['state_before', 'new', w]} for w in range(2)]
    assert all(repaired_vectors[0][t] == repaired_vectors[1][t] for ts in targets.values() for t in ts)
    ground_truth = {}
    for variant in ['old', 'new']:
        ground_truth[variant] = dict(collections.Counter(verdict((truth[variant, 0, q][t],)*2, (truth[variant, 1, q][t],)*2) for q, ts in targets.items() for t in ts))
    summary = []
    max_native_error = max_vector_error = 0.0
    for variant in ['old', 'new']:
        for route in protocol['routes']:
            runs = []
            for rep in range(protocol['repetitions']):
                selected = [r for r in formal if (r['variant'], r['route'], r['repetition']) == (variant, route, rep)]
                observed = {}
                vectors = {}
                for r in selected:
                    if route == 'query_batch':
                        assert len(r['response']['result']) == len(r['query_ids']) == 24
                        for q, result in zip(r['query_ids'], r['response']['result']):
                            observed[r['world'], q] = result['points']
                    elif route in ['fetch_targets', 'scroll_all']:
                        vectors[r['world']] = {p['id']: p['vector'] for p in points(r)}
                    else:
                        observed[r['world'], r['query_id']] = points(r)
                        if route == 'id_filter_vectors':
                            vectors.setdefault(r['world'], {}).update({p['id']: p['vector'] for p in points(r)})
                        if route.startswith('top') or route == 'exact10':
                            pts = points(r)
                            assert len(pts) == r['request']['limit'] == len({p['id'] for p in pts})
                            assert all(pts[i]['score'] >= pts[i+1]['score'] for i in range(len(pts)-1))
                            cutoff = pts[-1]['score']
                            ids = {p['id'] for p in pts}
                            assert all(score <= cutoff for pid, score in truth[variant, r['world'], r['query_id']].items() if pid not in ids)
                decisions = collections.Counter()
                direct = vector_pairs = 0
                local_decisions = collections.Counter()
                for q, ts in targets.items():
                    for target in ts:
                        intervals = []
                        hit = []
                        for world in range(2):
                            rows = observed.get((world, q), [])
                            scores = {p['id']: p['score'] for p in rows}
                            for pid, score in scores.items():
                                error = abs(score-truth[variant, world, q][pid])
                                max_native_error = max(max_native_error, error)
                                assert error <= EPS
                            if target in scores:
                                intervals.append((scores[target], scores[target]))
                                hit.append(True)
                            else:
                                intervals.append((-1.0, min(scores.values()) if scores else 1.0))
                                hit.append(False)
                        direct += int(all(hit))
                        if observed:
                            decision = verdict(*intervals)
                            decisions[decision] += 1
                            if decision != 'insufficient':
                                expected = verdict((truth[variant, 0, q][target],)*2, (truth[variant, 1, q][target],)*2)
                                assert decision == expected
                        if vectors and all(target in vectors[w] for w in range(2)):
                            vector_pairs += 1
                            reconstructed = [cosine(queries[q], vectors[w][target]) for w in range(2)]
                            for w in range(2):
                                max_vector_error = max(max_vector_error, abs(reconstructed[w]-truth[variant, w, q][target]))
                            local_decisions[verdict((reconstructed[0],)*2, (reconstructed[1],)*2)] += 1
                score_values, vector_values = map(sum, zip(*(value_counts(r['response']['result']) for r in selected)))
                runs.append({'requests': len(selected), 'request_body_bytes': sum(r['request_body_bytes'] for r in selected), 'response_body_bytes': sum(r['response_body_bytes'] for r in selected), 'native_score_values': score_values, 'vector_values': vector_values, 'client_seconds': sum(r['client_seconds'] for r in selected), 'server_seconds': sum(r['server_seconds'] for r in selected), 'paired_native_score_coverage': direct, 'paired_exported_vector_coverage': vector_pairs, 'decisions': dict(decisions), 'local_vector_decisions_diagnostic': dict(local_decisions)})
            for field in ['requests', 'request_body_bytes', 'native_score_values', 'vector_values', 'paired_native_score_coverage', 'paired_exported_vector_coverage', 'decisions', 'local_vector_decisions_diagnostic']:
                assert all(r[field] == runs[0][field] for r in runs), (variant, route, field)
            row = {'variant': variant, 'route': route, **runs[0]}
            for field in ['client_seconds', 'server_seconds', 'response_body_bytes']:
                row[field] = statistics.median(r[field] for r in runs)
                row[field+'_range'] = [min(r[field] for r in runs), max(r[field] for r in runs)]
            row['repetitions'] = runs
            summary.append(row)
    assert max_native_error <= EPS and max_vector_error <= EPS
    assert summary == saved['costs']
    assert ground_truth == saved['ground_truth']
    assert max_native_error == saved['max_native_route_score_error']
    assert max_vector_error == saved['max_exported_cosine_reconstruction_error']
    return {'passed': True, 'records': len(records), 'formal_requests': len(formal),
            'auth_probes': len(auth), 'routes': len(summary),
            'max_exported_cosine_error': max_vector_error}


if __name__ == '__main__':
    print(json.dumps(check(pathlib.Path(__file__).resolve().parents[1]), indent=2))
