"""Verify a frozen public audit before opening separate privileged target truth."""
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import importlib.util

_spec = importlib.util.spec_from_file_location("geometry_bounds", Path(__file__).with_name("geometry_bounds.py"))
geometry_bounds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry_bounds)

METHODS=('A_range','B_single','P_simplex','C_joint')
KEYS=('engine','mode','depth','query_id','target_id')

def key(row):return tuple(row[k] for k in KEYS)

def directed(value,upper):
    answer=float(value);represented=Fraction.from_float(answer)
    if upper and represented<value:answer=math.nextafter(answer,math.inf)
    if not upper and represented>value:answer=math.nextafter(answer,-math.inf)
    return answer

def verdict(lower,upper,epsilon):
    if lower>epsilon or upper< -epsilon:return 'supported'
    if lower>=-epsilon and upper<=epsilon:return 'no_detectable_effect'
    return 'insufficient'

def verify_public(packet,analysis):
    epsilon=Fraction(*packet['epsilon_rational']);R=packet['norm_bound'];eta=Fraction.from_float(packet['score_error'])
    assert math.isfinite(R) and R>0 and eta>=0 and epsilon>=0
    maximum=directed(Fraction.from_float(R)**2+eta,True)
    queries=dict(zip(packet['query_ids'],packet['query_vectors']));groups={};checks=0;started=time.perf_counter()
    assert len(queries)==len(packet['query_ids'])==len(packet['query_vectors'])>0
    assert len({len(q) for q in queries.values()})==1 and len(next(iter(queries.values())))>0
    assert all(math.isfinite(v) for q in queries.values() for v in q)
    assert all(sum(Fraction.from_float(float(v))**2 for v in q)<=Fraction.from_float(R)**2 for q in queries.values()),'Public query norm contract failed'
    assert not set(packet['grid']['engines'])-{'faiss_flatip','faiss_hnsw'} and packet['grid']['phases']==[0,1]
    assert packet['engine_permissions']=={e:('exact_cutoff' if e=='faiss_flatip' else 'unknown') for e in packet['grid']['engines']}
    states={(s['engine'],s['mode'],s['phase']):s for s in packet['states']}
    assert len(states)==len(packet['states']) and set(states)==set(itertools.product(packet['grid']['engines'],packet['grid']['modes'],packet['grid']['phases']))
    for row in packet['responses']:
        assert row['state_id']==states[(row['engine'],row['mode'],row['phase'])]['state_id']
        response_key=tuple(row[k] for k in ['engine','mode','depth','phase','query_id'])
        assert response_key not in groups and row['query_id'] in queries and row['depth'] in packet['grid']['depths']
        scores={d['id']:d['score'] for d in row['documents']}
        assert len(scores)==len(row['documents'])==row['depth'] and all(math.isfinite(v) and abs(v)<=maximum for v in scores.values())
        if row['engine']=='faiss_flatip':assert row['exact_cutoff']==min(scores.values())
        groups[response_key]=(row,scores)
    expected={(e,m,d,p['query_id'],p['target_id']) for e,m,d,p in itertools.product(packet['grid']['engines'],packet['grid']['modes'],packet['grid']['depths'],packet['comparisons'])}
    assert len(analysis['rows'])==len(expected) and {key(r) for r in analysis['rows']}==expected,'Missing/duplicated expected audit cells'
    for row in analysis['rows']:
        engine,mode,depth,qid,target=key(row)
        for phase in [0,1]:
            current=groups.get((engine,mode,depth,phase,qid));evidence=row['bound_evidence'][phase]
            intervals={m:[-maximum,maximum] for m in METHODS}
            if current is None:assert any(f['phase']==phase for f in row['failures'])
            elif target in current[1]:
                score=current[1][target];intervals={m:[score,score] for m in METHODS};assert evidence is None
            else:
                base=[-maximum,min(maximum,current[0]['exact_cutoff']) if engine=='faiss_flatip' else maximum]
                intervals={m:base for m in METHODS}
                anchor_ids=[other for other in packet['query_ids'] if other!=qid and (engine,mode,depth,phase,other) in groups and target in groups[(engine,mode,depth,phase,other)][1]]
                assert evidence['anchors']==anchor_ids,'Anchor selection/state mismatch'
                if evidence.get('status')=='invalid_certificate_input':assert any(f['phase']==phase for f in row['failures'])
                elif anchor_ids:
                    A=[queries[a] for a in anchor_ids];values=[groups[(engine,mode,depth,phase,a)][1][target] for a in anchor_ids]
                    limits=[[directed(Fraction.from_float(v)-eta,False),directed(Fraction.from_float(v)+eta,True)] for v in values]
                    identity=json.dumps([states[(engine,mode,phase)]['index_sha256'],engine,mode,depth,phase,qid,target],separators=(',',':'));previous=base
                    for method,section in [('B_single','single_anchor'),('P_simplex','simplex'),('C_joint','multi_anchor')]:
                        result=evidence[section]
                        for side in ['lower','upper']:
                            certificate=result[side+'_certificate']
                            assert certificate[side]==result[side]
                            assert geometry_bounds.verify_certificate(certificate,A,limits,queries[qid],R,identity),'Invalid or mismatched certificate'
                            checks+=1
                        lower=directed(Fraction.from_float(result['lower'])-eta,False);upper=directed(Fraction.from_float(result['upper'])+eta,True)
                        previous=[max(previous[0],lower),min(previous[1],upper)];assert previous[0]<=previous[1]
                        intervals[method]=previous
                else:assert evidence.get('status')=='no_anchor'
            for method in METHODS:assert row['score_intervals'][method][phase]==intervals[method],'Score interval does not follow public evidence'
        for method in METHODS:
            before,after=row['score_intervals'][method]
            lower=Fraction.from_float(after[0])-Fraction.from_float(before[1]);upper=Fraction.from_float(after[1])-Fraction.from_float(before[0])
            assert row['difference_intervals'][method]==[directed(lower,False),directed(upper,True)]
            assert row['labels'][method]==verdict(lower,upper,epsilon),'Label does not follow exact guard'
    return {'certificates_rechecked':checks,'public_check_seconds':time.perf_counter()-started}

def evaluate(packet,analysis,truth):
    started=time.perf_counter();epsilon=Fraction(*packet['epsilon_rational']);oracle={key(r):r for r in truth}
    assert len(oracle)==len(truth)==len(analysis['rows']) and set(oracle)=={key(r) for r in analysis['rows']}
    failures={m:{'score_containment':[],'difference_containment':[],'false_decisive':[]} for m in METHODS};gold={};valid={}
    for row in analysis['rows']:
        actual=oracle[key(row)];before,after=map(Fraction.from_float,actual['scores']);delta=after-before
        gold[key(row)]=verdict(delta,delta,epsilon)
        assert actual['label']==gold[key(row)],'Privileged label differs from the declared exact epsilon'
        for method in METHODS:
            valid[(key(row),method)]=not row['failures']
            for phase,score in enumerate(actual['scores']):
                lo,hi=row['score_intervals'][method][phase]
                if not lo<=score<=hi:
                    failures[method]['score_containment'].append([*key(row),phase]);valid[(key(row),method)]=False
            lo,hi=row['difference_intervals'][method]
            if not Fraction.from_float(lo)<=delta<=Fraction.from_float(hi):
                failures[method]['difference_containment'].append(key(row));valid[(key(row),method)]=False
            label=row['labels'][method]
            if label!='insufficient' and label!=gold[key(row)]:failures[method]['false_decisive'].append(key(row))
    def metrics(rows):
        result={'cells':len(rows),'unique_query_targets':len({(r['query_id'],r['target_id']) for r in rows}),'methods':{},'sound_gains':{}}
        hidden=[(r,p,e) for r in rows for p,e in enumerate(r['bound_evidence']) if e is not None]
        result['anchor_availability']={'missing_target_phase_scores':len(hidden),'with_visible_anchors':sum(bool(e['anchors']) for r,p,e in hidden),
            'anchor_count_histogram':dict(Counter(len(e['anchors']) for r,p,e in hidden))}
        result['narrowed_hidden_phase_scores']={m:sum(r['score_intervals'][m][p][0]>r['score_intervals']['A_range'][p][0] or r['score_intervals'][m][p][1]<r['score_intervals']['A_range'][p][1] for r,p,e in hidden) for m in METHODS[1:]}
        for method in METHODS:
            counts=Counter(r['labels'][method] for r in rows)
            result['methods'][method]={'counts':dict(counts),'abstention_rate':counts['insufficient']/len(rows) if rows else 0.}
        result['D_privileged_oracle']={'counts':dict(Counter(gold[key(r)] for r in rows)),'access':'Private native target scores; evaluation only, unavailable to every auditor.'}
        for old,new in [('A_range','B_single'),('A_range','P_simplex'),('A_range','C_joint'),('B_single','C_joint'),('P_simplex','C_joint')]:
            selected=[r for r in rows if r['labels'][old]=='insufficient' and r['labels'][new]!='insufficient' and r['labels'][new]==gold[key(r)] and valid[(key(r),new)]]
            result['sound_gains'][f'{new}_vs_{old}']={'cells':len(selected),'unique_query_targets':len({(r['query_id'],r['target_id']) for r in selected}),
                'abstention_rate_reduction':result['methods'][old]['abstention_rate']-result['methods'][new]['abstention_rate']}
        return result
    return {'overall':metrics(analysis['rows']),'by_engine':{e:metrics([r for r in analysis['rows'] if r['engine']==e]) for e in packet['grid']['engines']},
            'by_engine_depth':{f'{e}_k{d}':metrics([r for r in analysis['rows'] if r['engine']==e and r['depth']==d]) for e,d in itertools.product(packet['grid']['engines'],packet['grid']['depths'])},
            'failures':failures,'analysis_failure_cells':sum(bool(r['failures']) for r in analysis['rows']),
            'passed':not any(v for m in failures.values() for v in m.values()) and not any(r['failures'] for r in analysis['rows']),
            'private_evaluation_seconds':time.perf_counter()-started}


def check(root):
    """Recheck public certificates first, then compare with held-out target scores."""
    if not __debug__:
        raise RuntimeError('Run without -O: scientific checks require assertions.')
    base = Path(root) / 'data/geometry'
    raw = (base / 'analysis.json').read_bytes()
    analysis = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert (base / 'analysis.sha256').read_text().strip() == digest
    packet_raw = (base / 'public_packet.json').read_bytes()
    assert hashlib.sha256(packet_raw).hexdigest() == analysis['public_packet_sha256']
    packet = json.loads(packet_raw)
    assert len(packet['query_ids']) == 284 and len(packet['comparisons']) == 305
    assert len(packet['states']) == 8 and len(analysis['rows']) == 4880
    public = verify_public(packet, analysis)
    assert public['certificates_rechecked'] == 5988
    oracle_raw = (base / 'oracle.json').read_bytes()
    recorded = json.loads((base / 'evaluation.json').read_text())
    assert recorded['analysis_sha256_before_private_evaluation'] == digest
    assert recorded['oracle_sha256'] == hashlib.sha256(oracle_raw).hexdigest()
    report = evaluate(packet, analysis, json.loads(oracle_raw))
    for name in ['overall', 'by_engine', 'by_engine_depth', 'failures', 'analysis_failure_cells', 'passed']:
        assert json.loads(json.dumps(report[name])) == recorded[name], name
    assert report['passed']
    return {'passed': True, 'queries': 284, 'query_target_pairs': 305,
            'paired_cells': len(analysis['rows']), 'certificates_rechecked': public['certificates_rechecked'],
            'counts': report['overall']['methods']['A_range']['counts'],
            'narrowed_intervals': report['overall']['narrowed_hidden_phase_scores'],
            'additional_decisions': report['overall']['sound_gains']['C_joint_vs_A_range']['cells'],
            'scope': 'Saved public certificate replay and separate target-score evaluation; no new encoding, search, or optimization.'}


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
