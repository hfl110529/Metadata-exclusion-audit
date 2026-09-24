"""Stdlib-only saved-evidence check: independent cosine, identities and paired decisions."""
from collections import Counter
import hashlib
import copy
import json
import math
from pathlib import Path
import statistics


def read(path): return json.loads(path.read_text())
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def need(value, message):
    if not value: raise ValueError(message)


def bound_body(record, request, ready, channel, uid_map):
    need(record['status'] == 200, 'HTTP failure')
    body = record['body']
    for key in ['world', 'generation', 'descriptor', 'query_id', 'query_digest', 'target_ids']:
        need(body[key] == request[key], 'binding mismatch: ' + key)
    need(body['descriptor'] == ready['descriptor'] and body['epoch'] == ready['epoch'], 'bootstrap epoch mismatch')
    need(body['channel'] == channel, 'channel mismatch')
    docs = body['documents']; ids = [d['uid'] for d in docs]
    need(len(ids) == len(set(ids)) == (10 if channel == 'ordinary' else len(request['target_ids'])), 'wrong result identities/count')
    need(all(d['uid'] in uid_map and d['id'] == uid_map[d['uid']]['id'] for d in docs), 'native/logical identity mismatch')
    need(all(math.isfinite(d['score']) for d in docs), 'invalid score')
    need([d['score'] for d in docs] == sorted((d['score'] for d in docs), reverse=True), 'unsorted response')
    if channel == 'trusted':
        need(set(ids) == set(request['target_ids']), 'wrong selected target')
    return body


def check(root):
    BASE = Path(root) / 'data/equal_access'
    protocol = read(BASE / 'protocol.json')
    manifest = read(BASE / 'inputs/manifest.json'); queries = read(BASE / 'inputs/queries.json')
    expected = read(BASE / 'expected_descriptors.json'); specification = read(BASE / 'descriptor_specification.json')
    targets = {q['query_id']: [d['SOPInstanceUID'] for d in manifest['documents'] if d['role'] == 'target' and d['topic'] == q['topic']] for q in queries}
    obligations = {(q, uid) for q, ids in targets.items() for uid in ids}
    need(len(queries) == 24 and len(obligations) == 48, 'wrong comparison inventory')
    maps = {}; oracle = {}; ready = {}
    for w in (0, 1):
        for g in ('old', 'new'):
            docs = read(BASE / f'inputs/world{w}_{g}.json')
            maps[w, g] = {d['meta']['SOPInstanceUID']: d for d in docs}
            ready[w, g] = read(BASE / f'services/w{w}_{g}_ready.json')
            state = {'world': w, 'generation': g, 'documents': docs, 'loaded_queries': queries,
                **{k: specification[k] for k in ['logical_mapping', 'query_provenance', 'scorer', 'base_filter', 'precision']},
                'runtime_scorer': specification['runtime_scorer_expected']}
            need(digest(state) == expected[f'w{w}_{g}'] == ready[w, g]['descriptor'], 'loaded-state descriptor mismatch')
            need(ready[w, g]['store_handle'] == ready[w, g]['retriever_store_handle'] and ready[w, g]['loaded_documents'] == 192
                 and ready[w, g]['loaded_query_vectors'] == 24 and ready[w, g]['mutation_routes'] == [], 'measured reader/store inspection mismatch')
            need(ready[w, g]['startup_descriptor_sensitivity_checks'] == ['retriever_scale_score', 'store_similarity', 'loaded_query_vector'], 'missing runtime sensitivity checks')
            for q in queries:
                norm_q = math.sqrt(math.fsum(x * x for x in q['embedding']))
                for uid, d in maps[w, g].items():
                    norm_d = math.sqrt(math.fsum(x * x for x in d['embedding']))
                    oracle[w, g, q['query_id'], uid] = math.fsum((a / norm_d) * (b / norm_q) for a, b in zip(d['embedding'], q['embedding']))
    for g in ('old', 'new'):
        need(set(maps[0, g]) == set(maps[1, g]), 'paired membership mismatch')
        for uid, a in maps[0, g].items():
            b = maps[1, g][uid]
            need(a['content'] == b['content'] and {k: v for k, v in a['meta'].items() if k not in ['url', 'PatientComments']} ==
                 {k: v for k, v in b['meta'].items() if k not in ['url', 'PatientComments']}, 'paired non-field content mismatch')
            if g == 'new':
                need(a['embedding'] == b['embedding'], 'repaired paired vectors mismatch')
    need(len({r['epoch'] for r in ready.values()}) == len({r['pid'] for r in ready.values()}) == 4, 'service reuse across states')
    batches = [json.loads(x) for x in (BASE / 'batches.jsonl').read_text().splitlines()]
    need([(b['generation'], b['repeat'], b['method']) for b in batches] ==
        [(g, r, m) for g in ('old', 'new') for r in range(-1, 6)
         for m in (['point_interval', 'direct_rescore'] if r >= 0 and r % 2 else ['direct_rescore', 'point_interval'])], 'unplanned/missing timing round')
    summaries = {}; max_error = 0.; score_checks = 0; evidence = {}; paired = {}; mismatches = 0
    for b in batches:
        g = b['generation']; m = b['method']; records = b['responses']
        need(len(records) == 48 and b['seconds'] > 0, 'missing request or invalid timing')
        need(b['warmup'] == (b['repeat'] == -1), 'warmup label mismatch')
        need(b['started_at_utc'] < b['ended_at_utc'], 'invalid batch timing window')
        need([x['world'] for x in b['inspections']] == [0, 1], 'missing reader inspection')
        for inspection in b['inspections']:
            w = inspection['world']
            need(inspection == {'world': w, 'served_generation': g, 'descriptor': expected[f'w{w}_{g}'], 'desired_generation': 'new', 'matches_desired_generation': g == 'new'}, 'incorrect reader/version diagnostic')
        scores = {}; response_bodies = []
        for (w, q), record in zip(((w, q) for w in (0, 1) for q in queries), records):
            req = {'world': w, 'generation': g, 'descriptor': expected[f'w{w}_{g}'], 'query_id': q['query_id'], 'query_digest': digest(q['embedding']), 'target_ids': targets[q['query_id']]}
            need(record['request'] == req and record['method'] == 'POST' and record['url'] == ready[w, g]['endpoint'] + '/trusted-target', 'changed request/access path')
            need(record['request_sha256'] == digest(req) and record['response_sha256'] == hashlib.sha256(record['raw_response'].encode()).hexdigest()
                 and json.loads(record['raw_response']) == record['body'], 'raw byte/hash mismatch')
            body = bound_body(record, req, ready[w, g], 'trusted', maps[w, g])
            response_bodies.append(record['raw_response'])
            for d in body['documents']:
                error = abs(d['score'] - oracle[w, g, q['query_id'], d['uid']]); max_error = max(max_error, error)
                need(error < 1e-12, 'native score inconsistent with frozen snapshot')
                scores[w, q['query_id'], d['uid']] = d['score']; score_checks += 1
        need(set((d['query_id'], d['uid']) for d in b['decisions']) == obligations and len(b['decisions']) == 48, 'changed obligations')
        counts = Counter()
        for d in b['decisions']:
            q, uid = d['query_id'], d['uid']; delta = scores[1, q, uid] - scores[0, q, uid]
            oracle_delta = oracle[1, g, q, uid] - oracle[0, g, q, uid]
            verdict = 'supported_effect' if abs(delta) > protocol['epsilon'] else 'no_detectable_effect'
            need(d['delta'] == delta and d['verdict'] == verdict and (abs(oracle_delta) > protocol['epsilon']) == (verdict == 'supported_effect'), 'invalid verdict')
            counts[verdict] += 1
        need(dict(counts) == b['counts'], 'wrong aggregate')
        volume = {'http_calls': len(records), 'returned_scores': len(scores), 'response_body_bytes': sum(len(x.encode()) for x in response_bodies), 'request_body_bytes': sum(len(canonical(r['request'])) for r in records)}
        need(volume == b['volume'] and volume['http_calls'] == 48 and volume['returned_scores'] == 96, 'wrong evidence volume')
        if g in evidence: need(response_bodies == evidence[g], 'methods/rounds received different evidence')
        else: evidence[g] = response_bodies
        key = (g, b['repeat'])
        if key in paired:
            need(b['decisions'] == paired[key], 'equal-access verdict disagreement')
        else: paired[key] = b['decisions']
        if not b['warmup']:
            group = summaries.setdefault(g + '_' + m, {'counts': dict(counts), 'volume': volume, 'batch_seconds': []})
            need(group['counts'] == dict(counts) and group['volume'] == volume, 'unstable batch result')
            group['batch_seconds'].append(b['seconds'])
    for group in summaries.values():
        group.update(median_seconds=statistics.median(group['batch_seconds']), min_seconds=min(group['batch_seconds']), max_seconds=max(group['batch_seconds']))
    ordinary = {}
    for g in ('old', 'new'):
        rows = []; table = {}
        for w in (0, 1):
            wd = BASE / f'binding/w{w}_{g}'; old_ready = read(wd / 'ready.json')
            rs = [json.loads(x) for x in (wd / 'responses.jsonl').read_text().splitlines()]
            selected = [r for r in rs if r['body']['channel'] == 'ordinary']; need(len(selected) == 24, 'missing frozen ordinary responses')
            for r in selected:
                q = next(q for q in queries if q['query_id'] == r['request']['query_id'])
                need(r['request']['target_ids'] == [] and r['request']['query_digest'] == digest(q['embedding']), 'ordinary query mismatch')
                need(r['response_sha256'] == hashlib.sha256(r['raw_response'].encode()).hexdigest() and json.loads(r['raw_response']) == r['body'], 'ordinary raw mismatch')
                b = bound_body(r, r['request'], old_ready, 'ordinary', maps[w, g]); table[w, q['query_id']] = b['documents']
                visible = {d['uid'] for d in b['documents']}; cutoff = b['documents'][-1]['score']
                need(all(oracle[w, g, q['query_id'], uid] <= cutoff + 1e-12 for uid in maps[w, g] if uid not in visible), 'ordinary cutoff invalid')
                rows.append(r)
        counts = Counter()
        for q, uid in obligations:
            bounds = []
            for w in (0, 1):
                ds = table[w, q]; visible = {d['uid']: d['score'] for d in ds}
                bounds.append((visible[uid], visible[uid]) if uid in visible else (-math.inf, ds[-1]['score']))
            lo = bounds[1][0] - bounds[0][1]; hi = bounds[1][1] - bounds[0][0]
            label = 'supported_effect' if lo > protocol['epsilon'] or hi < -protocol['epsilon'] else 'no_detectable_effect' if -protocol['epsilon'] <= lo and hi <= protocol['epsilon'] else 'insufficient'
            counts[label] += 1
        ordinary[g] = {'counts': dict(counts), 'http_calls': len(rows), 'returned_scores': sum(len(r['body']['documents']) for r in rows), 'response_body_bytes': sum(len(r['raw_response'].encode()) for r in rows), 'new_timing': None, 'source': 'saved ordinary top-10 responses; not timed in this experiment'}
    reported = read(BASE / 'reported_results.json')
    need(summaries == reported['results'], 'reported timing, decision or volume mismatch')
    for generation in ordinary:
        for key in ['counts', 'http_calls', 'returned_scores', 'response_body_bytes']:
            need(ordinary[generation][key] == reported['ordinary_weaker_capability_reference'][generation][key], 'ordinary reference mismatch')
    # Validate all binding channels and independently replay client-side rejection cases.
    binding_records, binding_ready = {}, {}
    binding_counts = {}
    for w in (0, 1):
        for g in ('old', 'new'):
            wd = BASE / f'binding/w{w}_{g}'
            rdy = read(wd / 'ready.json'); binding_ready[w, g] = rdy
            need(rdy['descriptor'] == expected[f'w{w}_{g}'], 'binding descriptor mismatch')
            need(rdy['store_handle'] == rdy['retriever_store_handle'] and rdy['loaded_documents'] == 192
                 and rdy['loaded_query_vectors'] == 24 and rdy['mutation_routes'] == [], 'reader/store isolation mismatch')
            records = [json.loads(s) for s in (wd / 'responses.jsonl').read_text().splitlines()]
            need(len(records) == 48, 'missing binding response')
            for q in queries:
                for channel, path in [('ordinary', '/ordinary'), ('trusted', '/trusted-target')]:
                    selected = [r for r in records if r['request']['query_id'] == q['query_id'] and r['url'] == rdy['endpoint'] + path]
                    need(len(selected) == 1, 'duplicate/missing binding response')
                    record = selected[0]
                    req = {'world': w, 'generation': g, 'descriptor': expected[f'w{w}_{g}'], 'query_id': q['query_id'],
                           'query_digest': digest(q['embedding']), 'target_ids': targets[q['query_id']] if channel == 'trusted' else []}
                    need(record['request'] == req and record['request_sha256'] == digest(req), 'binding request mismatch')
                    need(hashlib.sha256(record['raw_response'].encode()).hexdigest() == record['response_sha256']
                         and json.loads(record['raw_response']) == record['body'], 'binding response integrity mismatch')
                    body = bound_body(record, req, rdy, channel, maps[w, g])
                    for d in body['documents']:
                        need(abs(d['score'] - oracle[w, g, q['query_id'], d['uid']]) < 1e-12, 'binding native score mismatch')
                    binding_records[w, g, q['query_id'], channel] = record
    need(len({r['epoch'] for r in binding_ready.values()}) == len({r['native_store_index'] for r in binding_ready.values()}) == 4, 'binding service state reuse')
    for g in ('old', 'new'):
        for channel in ('ordinary', 'trusted'):
            counts = Counter()
            for query_id, uid in obligations:
                bounds = []
                for w in (0, 1):
                    docs = binding_records[w, g, query_id, channel]['body']['documents']
                    found = {d['uid']: d['score'] for d in docs}
                    bounds.append((found[uid], found[uid]) if uid in found else (-math.inf, docs[-1]['score']))
                lo, hi = bounds[1][0] - bounds[0][1], bounds[1][1] - bounds[0][0]
                verdict = 'supported_effect' if lo > protocol['epsilon'] or hi < -protocol['epsilon'] else 'no_detectable_effect' if -protocol['epsilon'] <= lo and hi <= protocol['epsilon'] else 'insufficient'
                counts[verdict] += 1
            binding_counts[g + '_' + channel] = dict(counts)
    need(binding_counts == {'old_ordinary': {'supported_effect': 48}, 'old_trusted': {'supported_effect': 48},
                            'new_ordinary': {'insufficient': 48}, 'new_trusted': {'no_detectable_effect': 48}}, 'binding decision mismatch')
    negatives = read(BASE / 'binding/negative_checks.json')
    negative_counts = Counter(n['name'] for n in negatives)
    need(len(negatives) == 52 and set(negative_counts) == set(specification['negative_tests']) and all(n['rejected'] for n in negatives), 'negative coverage mismatch')
    wire_negatives = [n for n in negatives if 'record' in n]
    need(len(wire_negatives) == 36 and all(n['record']['status'] in (405, 409) for n in wire_negatives), 'missing HTTP rejection')
    for n in wire_negatives:
        r = n['record']
        need(r['request_sha256'] == digest(r['request']) and hashlib.sha256(r['raw_response'].encode()).hexdigest() == r['response_sha256']
             and json.loads(r['raw_response']) == r['body'], 'negative response integrity mismatch')
    rejected = 0
    for w in (0, 1):
        q = queries[0]; req = binding_records[w, 'old', q['query_id'], 'trusted']['request']; rdy = binding_ready[w, 'old']
        variants = [(binding_records[w, 'new', q['query_id'], 'trusted'], 'trusted')]
        for key, value in [('query_id', 'other-query'), ('world', 1 - w), ('epoch', 'other-epoch')]:
            r = copy.deepcopy(binding_records[w, 'old', q['query_id'], 'trusted']); r['body'][key] = value; variants.append((r, 'trusted'))
        r = copy.deepcopy(binding_records[w, 'old', q['query_id'], 'trusted']); r['body']['documents'][0]['uid'] = 'undeclared-target'; variants.append((r, 'trusted'))
        for channel, key, value in [('trusted', 'id', 'incorrect-native-id'), ('ordinary', 'id', 'incorrect-native-id'), ('ordinary', 'uid', 'undeclared-public-uid')]:
            r = copy.deepcopy(binding_records[w, 'old', q['query_id'], channel]); r['body']['documents'][0][key] = value; variants.append((r, channel))
        for r, channel in variants:
            try:
                bound_body(r, dict(req, target_ids=[] if channel == 'ordinary' else req['target_ids']), rdy, channel, maps[w, 'old'])
            except ValueError:
                rejected += 1
            else:
                raise ValueError('client accepted mismatched evidence')
    need(rejected == 16, 'client rejection replay incomplete')
    return {'passed': True, 'scope': 'saved matched-access benchmark and cooperative local binding evidence',
        'measured_batches': 24, 'warmup_batches': 4, 'total_target_http_calls': sum(len(b['responses']) for b in batches),
        'native_scores_checked': score_checks, 'max_independent_cosine_error': max_error,
        'equal_access_disagreements': mismatches, 'raw_response_bodies_identical_between_methods_and_rounds': True,
        'shared_startup_seconds': read(BASE / 'startup_times.json'), 'results': summaries,
        'ordinary_weaker_capability_reference': ordinary, 'binding_aggregate': binding_counts,
        'binding_rejected_cases': len(negatives), 'binding_http_rejections': len(wire_negatives),
        'independently_replayed_client_rejections': rejected, 'binding_negative_cases_by_name': dict(negative_counts)}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
