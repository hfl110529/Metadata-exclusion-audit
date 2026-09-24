"""Independent stdlib checker: reconstruct all results from retained native responses."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

BASE = Path(__file__).resolve().parents[1] / 'data/backend_updates'


def read(name):
    if name.startswith('observations/'):
        return OBSERVATIONS[name.split('/', 1)[1]]
    return json.loads((BASE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values), 'values': values}


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))


def scores(query):
    return {p['id']: p['score'] for p in query['full']}


def check(root):
    global BASE, OBSERVATIONS
    BASE = Path(root) / 'data/backend_updates'
    OBSERVATIONS = json.loads(gzip.decompress((BASE/'observations.json.gz').read_bytes()))
    protocol = read('protocol.json')
    epsilon = protocol['design']['epsilon']
    docs = {d['id']: d for d in protocol['documents']}
    queries = {q['id']: q for q in protocol['queries']}
    targets = {i for i, d in docs.items() if d['role'] == 'target'}
    pairs, summaries, control_rows, deletion_identity = [], [], [], []
    maximum_cosine_residual = 0.0
    metadata_vector_checks = 0
    for model in protocol['models']:
        encoded_path = BASE / 'embeddings' / (model['key'] + '.json')
        encoded = json.loads(encoded_path.read_text())
        assert encoded['device'] == 'cpu'
        assert encoded['calibration']['max_cosine_drift'] <= epsilon
        for backend in ['qdrant', 'chroma']:
            label = model['key'] + '_' + backend
            calibration = read('observations/' + label + '_calibration.json')
            baseline = [{p['id']: p['score'] for p in response} for response in calibration['before']]
            calibration_drift = max(abs(p['score'] - baseline[i][p['id']])
                for control in ['repeat', 'metadata_noop'] for i, response in enumerate(calibration[control]) for p in response)
            assert calibration_drift <= epsilon
            states = {}
            for world in [0, 1]:
                for stage in protocol['stages']:
                    state = read('observations/' + label + '_world' + str(world) + '_' + stage + '.json')
                    state['by_id'] = {r['id']: r for r in state['records']}
                    state['by_query'] = {q['query_id']: q for q in state['queries']}
                    assert set(state['by_id']) == set(docs)
                    assert set(state['by_query']) == set(queries)
                    for row in state['records']:
                        assert row['float32_sha256'] == hashlib.sha256(struct.pack('<' + 'f' * len(row['vector']), *row['vector'])).hexdigest()
                        doc = docs[row['id']]
                        text = doc['body']
                        if doc['role'] == 'target' and stage not in ['repaired', 'excluded_note_edit']:
                            text += '\nprotected_note: ' + doc[f'note_world{world}']
                        assert max(abs(x-y) for x,y in zip(row['vector'], encoded['vectors'][text])) <= epsilon
                    for qid, result in state['by_query'].items():
                        assert len(result['full']) == len(docs)
                        assert set(scores(result)) == set(docs)
                        assert all(result['full'][i]['score'] >= result['full'][i+1]['score'] for i in range(len(docs)-1))
                        for k in protocol['design']['depths']:
                            response = result['topk'][str(k)]
                            assert len(response) == len({p['id'] for p in response}) == k
                            assert {p['id'] for p in response} <= set(docs)
                            assert max(abs(p['score']-scores(result)[p['id']]) for p in response) <= epsilon
                        for p in result['full']:
                            residual = abs(p['score'] - cosine(encoded['vectors'][queries[qid]['text']], state['by_id'][p['id']]['vector']))
                            maximum_cosine_residual = max(maximum_cosine_residual, residual)
                            assert residual <= epsilon, (label, world, stage, qid, p['id'], residual)
                    states[world, stage] = state
                for stage in ['same_input_repeat', 'audit_tag_only', 'metadata_removed', 'excluded_note_edit']:
                    reference = 'repaired' if stage == 'excluded_note_edit' else 'included'
                    a, b = states[world, reference], states[world, stage]
                    unchanged = sum(a['by_id'][i]['float32_sha256'] == b['by_id'][i]['float32_sha256'] for i in docs)
                    coordinate_drift = max(abs(x-y) for i in docs for x,y in
                        zip(a['by_id'][i]['vector'], b['by_id'][i]['vector']))
                    assert coordinate_drift <= epsilon
                    drift = max(abs(scores(a['by_query'][qid])[i] - scores(b['by_query'][qid])[i]) for qid in queries for i in docs)
                    assert drift <= epsilon
                    list_changes = {str(k): sum([p['id'] for p in a['by_query'][qid]['topk'][str(k)]] !=
                        [p['id'] for p in b['by_query'][qid]['topk'][str(k)]] for qid in queries) for k in protocol['design']['depths']}
                    control_rows.append({'model': model['key'], 'backend': backend, 'world': world,
                        'stage': stage, 'reference': reference, 'bitwise_unchanged_returned_vectors': unchanged,
                        'maximum_vector_coordinate_drift': coordinate_drift,
                        'maximum_all_candidate_score_drift': drift, 'native_topk_order_changes': list_changes})
                removed = states[world, 'metadata_removed']
                assert all(removed['by_id'][i]['float32_sha256'] ==
                    states[world, 'audit_tag_only']['by_id'][i]['float32_sha256'] for i in docs)
                deletion_identity.append({'model': model['key'], 'backend': backend, 'world': world,
                    'reference': 'audit_tag_only', 'stage': 'metadata_removed',
                    'bitwise_equal_returned_vectors': len(docs), 'total_candidates': len(docs)})
                assert all('protected_note' not in removed['by_id'][i]['metadata'] and
                    removed['by_id'][i]['metadata']['encoder_fields'] == 'body' for i in targets)
                assert removed['logical_encoder_fields'] == ['body']
                metadata_vector_checks += len(targets)
                repaired = states[world, 'repaired']
                assert all('protected_note' not in repaired['by_id'][i]['metadata'] for i in targets)
                assert all('protected_note' in states[world, 'excluded_note_edit']['by_id'][i]['metadata'] for i in targets)
            stage_summaries = {}
            for stage in protocol['stages']:
                stage_pairs = []
                for qid, query in queries.items():
                    a, b = [states[w, stage]['by_query'][qid] for w in [0, 1]]
                    sa, sb = scores(a), scores(b)
                    target = query['target_id']
                    delta = sb[target] - sa[target]
                    max_delta = max(abs(sb[i]-sa[i]) for i in docs)
                    row = {'model': model['key'], 'backend': backend, 'stage': stage, 'query_id': qid,
                        'target_id': target, 'topic': docs[target]['topic'], 'score0': sa[target], 'score1': sb[target],
                        'signed_delta': delta, 'absolute_delta': abs(delta), 'exceeds_epsilon': abs(delta) > epsilon,
                        'rank0': [p['id'] for p in a['full']].index(target)+1,
                        'rank1': [p['id'] for p in b['full']].index(target)+1,
                        'max_all_candidate_delta': max_delta,
                        'maximum_background_delta': max(abs(sb[i]-sa[i]) for i in docs if i not in targets),
                        'native_topk': {}, 'exact_score_margin': {}}
                    for k in protocol['design']['depths']:
                        aa, bb = [[p['id'] for p in r['topk'][str(k)]] for r in [a,b]]
                        row['native_topk'][str(k)] = {'target_visible0': target in aa, 'target_visible1': target in bb,
                            'target_visibility_changed': (target in aa) != (target in bb),
                            'order_changed': aa != bb, 'set_changed': set(aa) != set(bb),
                            'symmetric_difference_size': len(set(aa) ^ set(bb))}
                        gaps = [r['full'][k-1]['score'] - r['full'][k]['score'] for r in [a,b]]
                        row['exact_score_margin'][str(k)] = {'gap0': gaps[0], 'gap1': gaps[1],
                            'sufficient_membership_condition': max_delta <= epsilon and gaps[0] > 2*epsilon,
                            'scope': 'Complete candidate scores only; never a certificate for ANN output.'}
                    row['weak_gate_accepts'] = stage == 'metadata_removed'
                    row['weak_gate_false_acceptance'] = row['weak_gate_accepts'] and row['exceeds_epsilon']
                    repaired_scores = [scores(states[w, 'repaired']['by_query'][qid]) for w in [0,1]]
                    row['maximum_target_residual_from_repaired'] = max(abs(s[target]-r[target]) for s,r in zip([sa,sb],repaired_scores))
                    stage_pairs.append(row)
                    pairs.append(row)
                stage_summaries[stage] = {
                    'registered_pairs': len(stage_pairs), 'violations': sum(r['exceeds_epsilon'] for r in stage_pairs),
                    'absolute_delta': distribution([r['absolute_delta'] for r in stage_pairs]),
                    'signed_delta': distribution([r['signed_delta'] for r in stage_pairs]),
                    'positive_deltas': sum(r['signed_delta'] > epsilon for r in stage_pairs),
                    'negative_deltas': sum(r['signed_delta'] < -epsilon for r in stage_pairs),
                    'within_epsilon': sum(r['absolute_delta'] <= epsilon for r in stage_pairs),
                    'rank_changes': sum(r['rank0'] != r['rank1'] for r in stage_pairs),
                    'weak_gate_false_acceptances': sum(r['weak_gate_false_acceptance'] for r in stage_pairs),
                    'maximum_background_delta': max(r['maximum_background_delta'] for r in stage_pairs),
                    'maximum_target_residual_from_repaired': max(r['maximum_target_residual_from_repaired'] for r in stage_pairs),
                    'topk': {str(k): {field: sum(r['native_topk'][str(k)][field] for r in stage_pairs)
                        for field in ['target_visible0', 'target_visible1', 'target_visibility_changed', 'order_changed', 'set_changed']}
                        for k in protocol['design']['depths']},
                    'margins': {str(k): {'world'+str(w): distribution([r['exact_score_margin'][str(k)]['gap'+str(w)] for r in stage_pairs])
                        for w in [0,1]} for k in protocol['design']['depths']}}
            summaries.append({'model': model['key'], 'backend': backend, 'calibration_drift': calibration_drift, 'stages': stage_summaries})
    expected = read('results.json')
    for key, value in [('configurations', summaries), ('controls', control_rows),
                       ('metadata_removal_vector_identity', deletion_identity),
                       ('all_registered_pair_rows', pairs)]:
        assert value == expected[key], key
    assert maximum_cosine_residual == expected['maximum_native_score_vs_independent_double_cosine_residual']
    return {'passed': True, 'native_states': 48, 'paired_rows': len(pairs),
            'max_cosine_residual': maximum_cosine_residual}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
