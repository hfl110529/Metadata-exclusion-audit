"""Check the reported migration aggregate and synthetic workload, not raw ingestion."""
import json
from pathlib import Path


def check(root):
    base = Path(root) / 'data/migration'
    result = json.loads((base / 'reported_results.json').read_text())
    truth = json.loads((base / 'synthetic_records.json').read_text())
    docs = truth['documents']
    if not (len(docs) == len({d['SOPInstanceUID'] for d in docs}) == result['N'] == 192
            and len(truth['queries']) == result['queries'] == 24 and len(truth['topics']) == result['topics'] == 12):
        raise ValueError('synthetic workload inventory mismatch')
    for role in ['target', 'unchanged_null']:
        pairs = [(q, d) for q in truth['queries'] for d in docs if d['role'] == role and d['topic'] == q['topic']]
        if len(pairs) != 48:
            raise ValueError('target/control inventory mismatch')
    for d in docs:
        if (d['comments'][0] != d['comments'][1]) != (d['role'] == 'target'):
            raise ValueError('field-world intervention mismatch')
    expected = {'included': (48, 48, 0), 'excluded_skip': (48, 48, 0), 'excluded_overwrite': (0, 0, 48)}
    for stage, values in expected.items():
        a = result['aggregate'][stage]['target']
        if (a['full_score_effects'], a['public_supported'], a['insufficient']) != values or a['episodes'] != 48:
            raise ValueError('reported migration result mismatch')
        control = result['aggregate'][stage]['unchanged_null']
        if (control['episodes'], control['full_score_effects'], control['public_supported'], control['no_detectable_effect'], control['insufficient']) != (48, 0, 0, 42, 6):
            raise ValueError('reported control result mismatch')
    stages = result['stage_checks']
    if len(stages) != 6 or {(s['phase'], s['stage']) for s in stages} != {(w, g) for w in (0, 1) for g in expected}:
        raise ValueError('reported stage inventory mismatch')
    for s in stages:
        written = 0 if s['stage'] == 'excluded_skip' else 192
        if not (s['documents_attempted'] == 192 and s['documents_written'] == written
                and s['changed_vectors_from_included'] == (192 if s['stage'] == 'excluded_overwrite' else 0)
                and s['config_excludes'] == s['current_inputs_exclude'] == (s['stage'] != 'included')):
            raise ValueError('reported migration write-policy mismatch')
    if sum(s['documents_attempted'] for s in stages) != result['ingestion_documents_attempted'] or sum(s['documents_written'] for s in stages) != result['ingestion_documents_written']:
        raise ValueError('reported write total mismatch')
    return {'passed': True, 'scope': 'aggregate-only consistency check plus synthetic-content inventory; raw ingestion traces are not redistributed',
            'N': result['N'], 'queries': result['queries'], 'aggregate': result['aggregate'],
            'stage_checks': stages, 'data_volume_by_stage': result['data_volume_by_stage'],
            'raw_ingestion_recomputed': False}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
