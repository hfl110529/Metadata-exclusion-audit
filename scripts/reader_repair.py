"""Recompute reader-repair results from saved native responses (standard library)."""
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalized(vector):
    norm = math.sqrt(math.fsum(x * x for x in vector))
    need(norm > 0, 'zero vector')
    return [x / norm for x in vector]


def response(record):
    raw = record['raw_body'].encode()
    need(record['status'] == 200 and len(raw) == record['response_bytes'] and
         hashlib.sha256(raw).hexdigest() == record['body_sha256'], 'response integrity failure')
    if 'request' in record:
        request = json.dumps(record['request'], ensure_ascii=False).encode()
        need(len(request) == record['request_bytes'] and hashlib.sha256(request).hexdigest() == record['request_sha256'], 'request integrity failure')
    return json.loads(raw)


def documents(record):
    docs = response(record)['result']['retriever']['documents']
    scores = [d['score'] for d in docs]
    need(bool(docs) and all(math.isfinite(x) for x in scores) and scores == sorted(scores, reverse=True), 'invalid score order')
    need(len({d['meta']['SOPInstanceUID'] for d in docs}) == len(docs), 'duplicate identity')
    need(all(d['embedding'] is None for d in docs), 'public response exposes vector')
    return docs


def label(bounds, epsilon):
    low, high = bounds
    return ('supported' if low > epsilon or high < -epsilon else
            'no_detectable_effect' if -epsilon <= low <= high <= epsilon else 'insufficient')


def check(root):
    base = Path(root) / 'data/reader_repair'
    manifest = read(base / 'public_manifest.json')
    queries = read(base / 'snapshots/queries.json')
    reported = read(base / 'reported_results.json')
    metadata = {d['SOPInstanceUID']: d for d in manifest['documents']}
    ids = set(metadata)
    need(len(ids) == manifest['N'] == 192 and len(queries) == 24, 'workload inventory mismatch')
    need(manifest['depths'] == [10, 20, 50, 192] and manifest['epsilon'] == 1e-6, 'protocol mismatch')
    need([{k: v for k, v in q.items() if k != 'embedding'} for q in queries] == manifest['queries'], 'query identity mismatch')
    pairs = [(q['query_id'], uid) for q in queries for uid, d in metadata.items()
             if d['role'] != 'background' and d['topic'] == q['topic']]
    need(len(pairs) == 96, 'target/control comparison inventory mismatch')
    snapshots, public, scores, volumes = {}, {}, {}, {}
    mapping, writes = [], []
    prefix_checks = 0
    maximum_error = 0.0
    for world in (0, 1):
        for generation, stage in [('old', 'stale_reader'), ('new', 'corrected_reader')]:
            docs = read(base / f'snapshots/world{world}_{generation}.json')
            snapshots[world, generation] = docs
            uid_map = {d['meta']['SOPInstanceUID']: d for d in docs}
            need(set(uid_map) == ids, 'snapshot membership mismatch')
            if generation == 'old':
                mapping.extend({'world': world, 'logical_uid': d['meta']['SOPInstanceUID'], 'native_id': d['id']} for d in docs)
            inputs = read(base / f'snapshots/world{world}_{generation}_encoder_inputs.json')
            need(inputs == [d['content'] if generation == 'new' else d['meta']['PatientComments'] + '\n' + d['content'] for d in docs], 'encoding-input mismatch')
            observed = read(base / f'world{world}/{generation}_write_observations.json')
            receipts = lines(base / f'world{world}/{generation}_receipts.jsonl')
            need(len(observed) == len(receipts) == 6, 'missing write batch')
            for i, (observation, receipt) in enumerate(zip(observed, receipts)):
                batch = docs[i * 32:(i + 1) * 32]
                need(observation == {'documents': 32, 'documents_sha256': digest(batch), 'documents_written': 32}, 'write observation mismatch')
                response(receipt)
                request = json.dumps({'documents': batch}, ensure_ascii=False).encode()
                need(len(request) == receipt['request_bytes'] and hashlib.sha256(request).hexdigest() == receipt['request_sha256'], 'write request mismatch')
                need(response(receipt)['result']['writer']['documents_written'] == 32, 'write incomplete')
            writes.append({'world': world, 'generation': generation, 'documents_written': 192})
            route = (base / f'world{world}/{generation}_reader.yaml').read_text()
            need(f'index: r12-{generation}' in route and 'embedding_similarity_function: cosine' in route and 'scale_score: false' in route, 'reader/scorer binding mismatch')
            full = lines(base / f'world{world}/{stage}/k192.jsonl')
            need(len(full) == len(queries), 'missing complete response')
            normalized_docs = {uid: normalized(d['embedding']) for uid, d in uid_map.items()}
            for q, record in zip(queries, full):
                need(record['query_id'] == q['query_id'] and record['request']['query_embedding'] == q['embedding'], 'query binding mismatch')
                ds = documents(record)
                need({d['meta']['SOPInstanceUID'] for d in ds} == ids and len(ds) == 192, 'complete response inventory mismatch')
                query = normalized(q['embedding'])
                for d in ds:
                    uid = d['meta']['SOPInstanceUID']
                    need(d['id'] == uid_map[uid]['id'], 'native/logical identity mismatch')
                    expected = math.fsum(a * b for a, b in zip(normalized_docs[uid], query))
                    error = abs(expected - d['score'])
                    maximum_error = max(maximum_error, error)
                    need(error < 1e-12, 'native cosine mismatch')
                scores[world, stage, q['query_id']] = {d['meta']['SOPInstanceUID']: d['score'] for d in ds}
            for depth in manifest['depths']:
                records = lines(base / f'world{world}/{stage}/k{depth}.jsonl')
                need(len(records) == 24, 'missing public response')
                volume = volumes.setdefault(f'{stage}_k{depth}', Counter())
                for q, record, complete in zip(queries, records, full):
                    ds = documents(record)
                    need(record['query_id'] == q['query_id'] and record['request']['query_embedding'] == q['embedding'] and record['request']['top_k'] == depth, 'public request mismatch')
                    need(ds == documents(complete)[:depth], 'public response is not complete-response prefix')
                    public[world, stage, depth, q['query_id']] = ds
                    volume.update(calls=1, response_bytes=record['response_bytes'], score_values=len(ds))
                    prefix_checks += 1
        need([d['id'] for d in snapshots[world, 'old']] == [d['id'] for d in snapshots[world, 'new']], 'within-world IDs changed')
    need(mapping == read(base / 'identity_mapping.json'), 'identity mapping mismatch')
    for generation in ('old', 'new'):
        for a, b in zip(snapshots[0, generation], snapshots[1, generation]):
            uid = a['meta']['SOPInstanceUID']
            need(uid == b['meta']['SOPInstanceUID'] and a['content'] == b['content'], 'paired identity/body mismatch')
            need({k: v for k, v in a['meta'].items() if k not in ['url', 'PatientComments']} ==
                 {k: v for k, v in b['meta'].items() if k not in ['url', 'PatientComments']}, 'paired non-field metadata mismatch')
            need((a['meta']['PatientComments'] != b['meta']['PatientComments']) == (metadata[uid]['role'] == 'target'), 'field-world intervention mismatch')
            if generation == 'new':
                need(a['embedding'] == b['embedding'], 'repaired paired vectors mismatch')
    aggregate, first, max_rank = {}, {}, []
    for stage in manifest['stages']:
        for depth in manifest['depths']:
            for query_id, uid in pairs:
                ds = [public[w, stage, depth, query_id] for w in (0, 1)]
                visible = [{d['meta']['SOPInstanceUID']: d['score'] for d in row} for row in ds]
                bounds = [(v[uid], v[uid]) if uid in v else (-math.inf, row[-1]['score']) for v, row in zip(visible, ds)]
                low, high = bounds[1][0] - bounds[0][1], bounds[1][1] - bounds[0][0]
                verdict = label((low, high), manifest['epsilon'])
                delta = scores[1, stage, query_id][uid] - scores[0, stage, query_id][uid]
                effect = abs(delta) > manifest['epsilon']
                need(low - 1e-12 <= delta <= high + 1e-12, 'true difference outside interval')
                need(not (verdict == 'supported' and not effect) and not (verdict == 'no_detectable_effect' and effect), 'unsound decision')
                role = metadata[uid]['role']
                count = aggregate.setdefault(f'{stage}_k{depth}_{role}', Counter())
                count.update(episodes=1, full_effect=int(effect), **{verdict: 1}, point_supported=int(all(uid in v for v in visible) and effect))
                if stage == 'corrected_reader' and role == 'target' and verdict == 'no_detectable_effect':
                    first[query_id, uid] = min(first.get((query_id, uid), depth), depth)
    for query_id, uid in first:
        max_rank.append(max(next(i + 1 for i, d in enumerate(public[w, 'corrected_reader', 192, query_id]) if d['meta']['SOPInstanceUID'] == uid) for w in (0, 1)))
    projected = []
    volume = volumes.setdefault('trusted_targets', Counter())
    for world in (0, 1):
        rows = lines(base / f'world{world}/corrected_reader/trusted_targets.jsonl')
        need(len(rows) == 24, 'missing trusted response')
        for q, record in zip(queries, rows):
            ds = documents(record)
            targets = {uid for uid, d in metadata.items() if d['role'] == 'target' and d['topic'] == q['topic']}
            need(record['query_id'] == q['query_id'] and record['request']['query_embedding'] == q['embedding'] and {d['meta']['SOPInstanceUID'] for d in ds} == targets, 'trusted target binding mismatch')
            for d in ds:
                uid = d['meta']['SOPInstanceUID']
                need(abs(d['score'] - scores[world, 'corrected_reader', q['query_id']][uid]) < 1e-12, 'trusted/full score mismatch')
                projected.append({'world': world, 'query_id': q['query_id'], 'uid': uid, 'score': d['score']})
            volume.update(calls=1, response_bytes=record['response_bytes'], score_values=len(ds))
    need(projected == read(base / 'trusted_target_projection.json'), 'trusted projection mismatch')
    target_scores = {(r['world'], r['query_id'], r['uid']): r['score'] for r in projected}
    need(all(abs(target_scores[1, q, uid] - target_scores[0, q, uid]) <= manifest['epsilon'] for q, uid in first), 'trusted paired effect after repair')
    volume['published_projection_bytes'] = (base / 'trusted_target_projection.json').stat().st_size
    need(volumes == reported['volumes'], 'reported response volumes mismatch')
    need(aggregate == reported['aggregate'], 'reported decision counts mismatch')
    coverage = [aggregate[f'corrected_reader_k{k}_target']['no_detectable_effect'] for k in manifest['depths']]
    need(coverage == [0, 2, 24, 48] and len(first) == 48, 'repair coverage mismatch')
    need(volumes['trusted_targets']['response_bytes'] == 121694 and volumes['corrected_reader_k192']['response_bytes'] == 11171866, 'main volume comparison mismatch')
    target_pair_rows = []
    for query_id, uid in sorted(first):
        ranks = [next(i + 1 for i, d in enumerate(public[w, 'corrected_reader', 192, query_id]) if d['meta']['SOPInstanceUID'] == uid) for w in (0, 1)]
        need(ranks[0] == ranks[1], 'repaired target ranks differ between worlds')
        deltas = {stage: float(Fraction(scores[1, stage, query_id][uid]) - Fraction(scores[0, stage, query_id][uid]))
                  for stage in ('stale_reader', 'corrected_reader')}
        target_pair_rows.append({'query_id': query_id, 'target_uid': uid, 'old_delta': deltas['stale_reader'],
                                 'repaired_delta': deltas['corrected_reader'], 'repaired_rank_both_worlds': ranks[0]})
    with (Path(root) / 'results/repair_pairs.csv').open(newline='') as handle:
        published = [{**r, 'old_delta': float(r['old_delta']), 'repaired_delta': float(r['repaired_delta']),
                      'repaired_rank_both_worlds': int(r['repaired_rank_both_worlds'])} for r in csv.DictReader(handle)]
    need(target_pair_rows == published, 'published target-pair figure data mismatch')
    with (Path(root) / 'results/repair_acquisition.csv').open(newline='') as handle:
        acquisition = list(csv.DictReader(handle))
    need(len(acquisition) == 5, 'published acquisition row count mismatch')
    for row, route, resolved in zip(acquisition, [f'k={k}' for k in manifest['depths']] + ['Trusted targets'], coverage + [48]):
        key = 'trusted_targets' if route == 'Trusted targets' else 'corrected_reader_k' + route[2:]
        need(row['route'] == route and int(row['resolved']) == resolved, 'published acquisition route/coverage mismatch')
        for field in ['calls', 'response_bytes', 'score_values']:
            need(int(row[field]) == volumes[key][field], 'published acquisition volume mismatch')
        projection = row['published_projection_bytes']
        need((int(projection) if projection else None) == volumes[key].get('published_projection_bytes'), 'published projection volume mismatch')
    return {'passed': True, 'scope': 'saved native response, snapshot cosine, identity, write-receipt and interval replay',
            'target_comparisons': 48, 'target_pair_rows': target_pair_rows, 'aggregate': aggregate, 'repaired_no_effect_by_depth': dict(zip(map(str, manifest['depths']), coverage)),
            'target_rank_range': [min(max_rank), max(max_rank)], 'writes': writes, 'volumes': volumes,
            'trusted_body_reduction_percent': 100 * (1 - 121694 / 11171866), 'exact_prefix_checks': prefix_checks,
            'public_decision_checks': len(pairs) * 8, 'max_independent_cosine_error': maximum_error}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
