"""Independently check native refresh receipts, persisted vectors, and all scores."""
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'data/backend_updates/policy_refresh'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root):
    global BASE
    BASE = Path(root) / 'data/backend_updates/policy_refresh'
    protocol = read(BASE/'protocol.json')
    raw = read(BASE/'observations/results.json')
    corpus = read(BASE.parent/'protocol.json')
    embeddings = read(BASE.parent/'embeddings/minilm.json')['vectors']
    query = next(q for q in corpus['queries'] if q['id'] == protocol['query_id'])
    qv = embeddings[query['text']]
    states = {(r['world'],r['stage']):r for r in raw['states']}
    assert len(states) == 12
    vectors = {}
    for key,state in states.items():
        vector_file = BASE/'observations'/state['vector_file']
        vectors[key] = read(vector_file)['embedding_dict']
        assert set(vectors[key]) == {'1-node-0','101-node-0','201-node-0'}
        assert set(state['native_result']['ids']) == set(vectors[key])
        for node,score in zip(state['native_result']['ids'],state['native_result']['scores']):
            v = vectors[key][node]
            computed = sum(a*b for a,b in zip(v,qv))/math.sqrt(sum(a*a for a in v)*sum(b*b for b in qv))
            assert abs(computed-score) <= protocol['epsilon']
        assert state['incoming_document_hash'] == state['stored_document_hash']
        if state['stage'] == 'included':
            assert len(state['embedding_calls']) == 3
        elif state['stage'] == 'forced_update_repair':
            assert state['embedding_calls'] == [state['incoming_embedding_text']]
        else:
            assert state['receipt'] == [False]
            assert state['embedding_calls'] == []
    for world in [0,1]:
        baseline = states[world,'included']
        assert len({states[world,stage]['incoming_document_hash'] for stage in protocol['stages']}) == 1
        assert all(states[world,stage]['incoming_metadata'] == baseline['incoming_metadata'] for stage in protocol['stages'])
        for stage in ['same_input_refresh','llm_policy_only_refresh','embed_policy_only_refresh']:
            assert vectors[world,stage] == vectors[world,'included']
            assert states[world,stage]['native_result'] == baseline['native_result']
        changed = states[world,'embed_policy_only_refresh']
        assert changed['incoming_embed_exclusion'] == ['protected_note']
        assert changed['incoming_embedding_text'] != baseline['incoming_embedding_text']
        assert changed['stored_nodes']['1-node-0']['excluded_embed_metadata_keys'] == []
        for stage in ['forced_update_repair','repaired_refresh']:
            state = states[world,stage]
            assert state['stored_nodes']['1-node-0']['excluded_embed_metadata_keys'] == ['protected_note']
            assert vectors[world,stage]['1-node-0'] == embeddings[state['incoming_embedding_text']]
    rows = []
    for stage in protocol['stages']:
        scores = [dict(zip(states[w,stage]['native_result']['ids'],states[w,stage]['native_result']['scores'])) for w in [0,1]]
        rows.append({'stage':stage,'score0':scores[0]['1-node-0'],'score1':scores[1]['1-node-0'],
            'signed_delta':scores[1]['1-node-0']-scores[0]['1-node-0'],
            'absolute_delta':abs(scores[1]['1-node-0']-scores[0]['1-node-0']),
            'embedding_calls_by_world':[len(states[w,stage]['embedding_calls']) for w in [0,1]],
            'refresh_receipt_by_world':[states[w,stage]['receipt'] for w in [0,1]]})
    assert rows == read(BASE/'results.json')['rows']
    return {'passed': True, 'states': len(states), 'stages': rows}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
