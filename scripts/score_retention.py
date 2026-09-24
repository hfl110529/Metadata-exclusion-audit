"""Verify paired native-score retention and deliberate corruption controls."""
import copy
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parents[1] / 'data/score_retention'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(canonical(value))


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(name, value):
    (HERE / name).write_bytes(canonical(value) + b'\n')


def check_body(body, contract, channel='trusted', identities=None):
    key = f'w{body["world"]}_{body["generation"]}'
    require(key in contract['states'], 'undeclared world/generation')
    state = contract['states'][key]
    require(body['descriptor'] == state['descriptor'], 'descriptor mismatch')
    require(body['epoch'] == state['epoch'], 'epoch mismatch')
    require(body['query_id'] in contract['queries'], 'undeclared query')
    query = contract['queries'][body['query_id']]
    require(body['query_digest'] == query['digest'], 'query digest mismatch')
    require(body['channel'] == channel, 'channel mismatch')
    selected = query['targets'] if channel == 'trusted' else []
    require(body['target_ids'] == selected, 'selected target mismatch')
    docs = body['documents']
    uids = [d['uid'] for d in docs]
    require(len(uids) == len(set(uids)) == (2 if channel == 'trusted' else 10),
            'response cardinality/duplicate identity')
    if channel == 'trusted':
        require(set(uids) == set(selected), 'target coverage mismatch')
    identities = identities if identities is not None else state['native_ids']
    require(all(d['uid'] in identities and d['id'] == identities[d['uid']] for d in docs),
            'native/logical identity mismatch')
    scores = [d['score'] for d in docs]
    require(all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                for v in scores), 'nonfinite/invalid score')
    require(scores == sorted(scores, reverse=True), 'response score order')


def entries(payloads, contract):
    previous = digest({'log_id': contract['log_id'], 'contract_sha256': digest(contract)})
    for sequence, payload in enumerate(payloads, 1):
        entry = {'sequence': sequence, 'previous': previous, 'payload': payload}
        entry['sha256'] = digest(entry)
        previous = entry['sha256']
        yield entry


def checkpoints(rows, contract):
    return [{'log_id': contract['log_id'], 'contract_sha256': digest(contract),
             'stage': stage, 'records': n, 'root': rows[n - 1]['sha256']}
            for stage, n in [('old', 48), ('new', 96)]]


def encode_rows(rows):
    return b''.join(canonical(row) + b'\n' for row in rows)


def verify_log(data, contract, expected_checkpoints):
    """Checkpoints are independent inputs, never derived from the supplied log."""
    rows = [json.loads(line) for line in data.splitlines()]
    require(data == encode_rows(rows), 'noncanonical JSONL')
    require(len(rows) == 96, 'log record coverage')
    expected_keys = [(s, w, q) for s in contract['stages'] for w in contract['worlds']
                     for q in contract['queries']]
    previous = digest({'log_id': contract['log_id'], 'contract_sha256': digest(contract)})
    payloads = []
    for i, (row, expected_key) in enumerate(zip(rows, expected_keys), 1):
        require(set(row) == {'sequence', 'previous', 'payload', 'sha256'}, 'log row schema')
        require(row['sequence'] == i and row['previous'] == previous, 'sequence/hash linkage')
        require(row['sha256'] == digest({k: v for k, v in row.items() if k != 'sha256'}),
                'row content hash')
        previous = row['sha256']
        p = row['payload']; b = p['body']
        require((b['generation'], b['world'], b['query_id']) == expected_key,
                'stage/world/query coverage/order')
        check_body(b, contract)
        require(p['source_file'] in contract['source_files']
                and p['source_file_sha256'] == contract['source_files'][p['source_file']],
                'source file commitment')
        expected_source = f'runs/round15_20260918/binding/attempt02/formal/w{b["world"]}_{b["generation"]}/responses.jsonl'
        require(p['source_file'] == expected_source, 'source path/state binding')
        require(p['source_response_sha256'] == digest(b), 'response commitment')
        start, end = (datetime.fromisoformat(p[k]) for k in
                      ('captured_started_at_utc', 'captured_ended_at_utc'))
        require(start.tzinfo is not None and start <= end, 'capture time interval')
        payloads.append(p)
    require(checkpoints(rows, contract) == expected_checkpoints, 'independent checkpoint mismatch')
    return payloads


def verify_source_replay(payloads, source_payloads):
    require(payloads == source_payloads, 'frozen-source replay mismatch')


def negative_checks(rows, contract, original_checkpoints, source_payloads):
    results = []
    for name in read(HERE / 'negative_cases.json'):
        changed = copy.deepcopy(rows)
        b = changed[48]['payload']['body']
        if name in ('score_byte_edit', 'rehashed_rewrite_with_original_checkpoint',
                    'rehashed_rewrite_with_replaced_checkpoint'):
            b['documents'][0]['score'] += 0.01
        elif name == 'drop_middle': del changed[20]
        elif name == 'drop_tail': changed.pop()
        elif name == 'duplicate': changed.insert(20, copy.deepcopy(changed[20]))
        elif name == 'reorder': changed[20], changed[21] = changed[21], changed[20]
        elif name == 'wrong_query_digest': b['query_digest'] = '0' * 64
        elif name == 'wrong_target_id': b['documents'][0]['uid'] = 'undeclared-target'
        elif name == 'wrong_native_id': b['documents'][0]['id'] = 'wrong-native-id'
        elif name == 'wrong_world': b['world'] = 1
        elif name == 'old_state_as_new':
            changed[48]['payload']['body'] = copy.deepcopy(rows[0]['payload']['body'])
            changed[48]['payload']['body']['generation'] = 'new'
        elif name == 'wrong_epoch': b['epoch'] = 'wrong-epoch'
        elif name == 'wrong_descriptor': b['descriptor'] = '0' * 64
        else: raise ValueError('unimplemented frozen negative: ' + name)
        rehash = name not in ('score_byte_edit', 'drop_middle', 'drop_tail', 'duplicate', 'reorder')
        if rehash:
            for row in changed:
                row['payload']['source_response_sha256'] = digest(row['payload']['body'])
            changed = list(entries([r['payload'] for r in changed], contract))
        replace_checkpoint = name == 'rehashed_rewrite_with_replaced_checkpoint'
        supplied = checkpoints(changed, contract) if replace_checkpoint else original_checkpoints
        try:
            recovered = verify_log(encode_rows(changed), contract, supplied)
            accepted, reason = True, 'log-only checks pass; source truth is not authenticated'
        except (ValueError, KeyError, TypeError) as exc:
            accepted, reason = False, str(exc)
        require(accepted == replace_checkpoint, 'unexpected negative outcome: ' + name)
        result = {'case': name, 'log_only_accepted': accepted, 'reason': reason,
                  'rehashed': rehash, 'checkpoint_replaced': replace_checkpoint}
        if accepted:
            try: verify_source_replay(recovered, source_payloads)
            except ValueError: result['frozen_source_replay_accepted'] = False
            else: raise ValueError('frozen-source check accepted falsified score')
        results.append(result)
    return results


def check(root):
    global HERE
    HERE = Path(root) / 'data/score_retention'
    contract = read(HERE/'contract.json')
    data = (HERE/'scores.jsonl').read_bytes()
    supplied = read(HERE/'checkpoints.json')
    payloads = verify_log(data, contract, supplied)
    expected_payloads = []
    ordinary = {}
    for stage in contract['stages']:
        for world in contract['worlds']:
            path = Path(root)/f'data/equal_access/binding/w{world}_{stage}/responses.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for qid in contract['queries']:
                r = next(r for r in rows if r['body']['query_id'] == qid and r['body']['channel'] == 'trusted')
                original_path = next(p for p in contract['source_files'] if f'/w{world}_{stage}/' in p)
                expected_payloads.append({'body': r['body'], 'captured_started_at_utc': r['started_at_utc'],
                    'captured_ended_at_utc': r['ended_at_utc'], 'source_file': original_path,
                    'source_file_sha256': contract['source_files'][original_path],
                    'source_response_sha256': r['response_sha256']})
                ordinary[stage,world,qid] = next(r['body'] for r in rows if r['body']['query_id'] == qid and r['body']['channel'] == 'ordinary')
    verify_source_replay(payloads, expected_payloads)
    negatives = negative_checks([json.loads(line) for line in data.splitlines()], contract, supplied, expected_payloads)
    assert negatives == read(HERE/'negative_checks.json')
    scoremap = {(p['body']['generation'],p['body']['world'],p['body']['query_id']):
                {d['uid']:d['score'] for d in p['body']['documents']} for p in payloads}
    for stage in contract['stages']:
        deltas = [scoremap[stage,1,q][t] - scoremap[stage,0,q][t] for q,v in contract['queries'].items() for t in v['targets']]
        assert len(deltas) == 48
        assert all(abs(d)>1e-6 for d in deltas) if stage=='old' else all(d==0 for d in deltas)
    # Every repaired target is omitted in both ordinary responses. Without a
    # cross-stage score constraint, old scores cannot narrow these new scores.
    assert all(t not in {d['uid'] for d in ordinary['new',w,q]['documents']}
               for q,v in contract['queries'].items() for t in v['targets'] for w in (0,1))
    result = read(HERE/'results.json')
    assert len(data) == result['bytes']['single_log'] == 123341
    assert len((HERE/'contract.json').read_bytes()) == result['bytes']['contract']
    assert len((HERE/'checkpoints.json').read_bytes()) == result['bytes']['checkpoints']
    assert median(r['total_seconds'] for r in result['timing_repetitions']) == result['timing_medians_seconds']['total_seconds']
    return {'passed': True, 'records': len(payloads), 'scores': 192, 'bytes': len(data),
            'rejected_controls': sum(not r['log_only_accepted'] for r in negatives),
            'repaired_acceptances': 48, 'old_log_plus_new_top10_insufficient': 48}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
