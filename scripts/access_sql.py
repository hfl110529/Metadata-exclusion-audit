"""Independently recompute route decisions/costs from saved SQL responses (stdlib)."""
from collections import Counter
import hashlib
import gzip
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parents[1] / 'data/access_sql'


def read(name):
    if name == 'inputs/queries.json':
        return json.loads((BASE.parent/'reader_repair/snapshots/queries.json').read_text())
    if name == 'inputs/public_manifest.json':
        return json.loads((BASE.parent/'reader_repair/public_manifest.json').read_text())
    name = name.removeprefix('formal/')
    if name.endswith('_decisions.json'):
        return json.loads((BASE/'decisions.json').read_text())[name]
    return json.loads((BASE / name).read_text())


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def label(left,right):
    lo=right[0]-left[1]; hi=right[1]-left[0]
    if lo>1e-6 or hi < -1e-6:
        return 'effect'
    if lo>=-1e-6 and hi<=1e-6:
        return 'no_effect'
    return 'insufficient'


def check(root):
    global BASE
    BASE = Path(root)/'data/access_sql'
    records=[json.loads(x) for x in gzip.decompress((BASE/'responses.jsonl.gz').read_bytes()).splitlines()]
    summary=read('formal/summary.json'); batches=read('formal/batches.json')
    queries=read('inputs/queries.json'); manifest=read('inputs/public_manifest.json')
    pairs=[(q['query_id'],d['SOPInstanceUID']) for q in queries for d in manifest['documents']
           if d['role']=='target' and d['topic']==q['topic']]
    assert len(pairs)==len(set(pairs))==48
    qmap={q['query_id']:q['embedding'] for q in queries}
    for r in records:
        assert r['json_value_bytes']==len(canonical(r.get('rows',[])))
    denied=[r for r in records if r.get('category')=='denied']
    assert len(denied)==19 and all(not r['ok'] for r in denied)
    assert all(r['ok'] for r in records if 'trial' in r)
    analysis={}; checks=0
    for b in batches:
        trial,stage,route=b['trial'],b['stage'],b['route']
        rows=[r for r in records if (r.get('trial'),r.get('stage'),r.get('route'))==(trial,stage,route)]
        assert len(rows)==b['calls']
        assert sum(r['json_value_bytes'] for r in rows)==b['json_value_bytes']
        maps={}; vector_route=route.endswith('vectors')
        for r in rows:
            if vector_route:
                vector_rows=r['rows'] if route.startswith('batch_') else \
                    [[f"{stage}_w{r['world']}",*v] for v in r['rows']]
                for epoch,uid,v in vector_rows:
                    for q in queries:
                        if route.endswith('target_vectors') and (q['query_id'],uid) not in pairs:
                            continue
                        w=int(epoch[-1]); x=q['embedding']
                        score=math.fsum(a*c for a,c in zip(v,x))/math.sqrt(
                            math.fsum(a*a for a in v)*math.fsum(c*c for c in x))
                        maps.setdefault((w,q['query_id']),{})[uid]=score
            elif route.startswith('batch_'):
                for epoch,qid,uid,s in r['rows']:
                    maps.setdefault((int(epoch[-1]),qid),{})[uid]=s
            else:
                maps[r['world'],r['qid']]=dict(r['rows'])
        native_scores=0 if vector_route else sum(len(r['rows']) for r in rows)
        assert native_scores==b['returned_scores']
        labels=[]; visible_labels=[]
        for q,u in pairs:
            a=[]; both=True
            for w in (0,1):
                scores=maps[w,q]; both &= u in scores
                a.append((scores[u],scores[u]) if u in scores else (-math.inf,min(scores.values())))
            decision=label(*a)
            labels.append(decision); visible_labels.append(decision if both else 'insufficient')
        assert dict(Counter(labels))==b['labels']
        assert dict(Counter(visible_labels))==b['visible_pair_labels']
        if trial==0:
            analysis[route,stage]=maps
            saved=read(f'formal/{route}_{stage}_decisions.json')
            assert [r['label'] for r in saved]==labels
            assert [(r['qid'],r['uid']) for r in saved]==pairs
        elif trial>0:
            assert analysis[route,stage]==maps
        checks+=len(pairs)
    for s in summary['routes']:
        bs=[b for b in batches if b['route']==s['route'] and b['stage']==s['stage'] and b['trial']>=0]
        assert len(bs)==3
        for k in ['calls','returned_scores','json_value_bytes','labels','visible_pair_labels']:
            assert all(b[k]==s[k] for b in bs)
        assert s['median_seconds']==statistics.median(b['seconds'] for b in bs)
        assert s['min_seconds']==min(b['seconds'] for b in bs)
        assert s['max_seconds']==max(b['seconds'] for b in bs)
    errors=[]
    for stage in ('old','new'):
        oracle=analysis['full_scores',stage]
        for route in ('top_ten','target_scores','full_scores','export_vectors','target_vectors','filtered_full_scores'):
            assert analysis[route,stage]==analysis['batch_'+route,stage]
        for key,full in oracle.items():
            assert analysis['top_ten',stage][key]==dict(sorted(full.items(),key=lambda p:(-p[1],p[0]))[:10])
            for u,s in analysis['target_scores',stage][key].items():
                assert full[u]==s
            for u,s in analysis['export_vectors',stage][key].items():
                errors.append(abs(full[u]-s))
        for q,u in pairs:
            oracle_label='no_effect' if abs(oracle[1,q][u]-oracle[0,q][u])<=1e-6 else 'effect'
            for route in {s['route'] for s in summary['routes']}:
                row=read(f'formal/{route}_{stage}_decisions.json')
                d=next(r['label'] for r in row if r['qid']==q and r['uid']==u)
                assert d=='insufficient' or d==oracle_label
    assert len(errors)==18432 and max(errors)==summary['max_export_reconstruction_error']<1e-6
    setup=read('formal/setup.json')
    grants={(g[0],g[1],g[2]) for g in setup['function_grants'] if g[3]=='EXECUTE'}
    limited={g[1:] for g in grants if g[0]=='limited_auditor'}
    assert limited=={('approved','top_ten'),('approved','state_receipt')}
    report={'passed':True,'saved_response_records':len(records),'decision_recomputations':checks,
            'measured_batches':len([b for b in batches if b['trial']>=0]),
            'denied_database_probes':len(denied),'native_score_reconstructions':len(errors),
            'max_reconstruction_error':max(errors),'summary_sha256':hashlib.sha256(
                (BASE/'summary.json').read_bytes()).hexdigest(),
            'scope':'Saved consistency; does not independently authenticate original DB execution'}
    return report


if __name__=='__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
