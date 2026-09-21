"""Independently check saved native states, cosine scores and frozen inputs (stdlib)."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data/upstream_update"

def read(name):
    return json.loads((HERE/name).read_text())

def sha(name):
    return hashlib.sha256((HERE/name).read_bytes()).hexdigest()

def cosine(a,b):
    return math.fsum(x*y for x,y in zip(a,b))/math.sqrt(math.fsum(x*x for x in a)*math.fsum(y*y for y in b))

def check(root):
    global HERE
    HERE = root / "data/upstream_update"
    fixture=read('fixture.json');embeddings=read('embeddings.json')
    vectors={r['text']:r['vector'] for r in embeddings['records']}
    query=vectors[fixture['query']]
    assert all(len(v)==384 for v in vectors.values())
    assert max(r['tokens'] for r in embeddings['records'])<=embeddings['max_seq_length']
    required_ids={d['id'] for d in fixture['documents']};summary=[];score_checks=0
    for role,version in [('affected','0.12.27'),('fixed','0.12.28')]:
        path=role;result=read(f'{path}/result.json');fixed=role=='fixed'
        assert result['version']==version and result['assertions_passed']
        assert result['embeddings_sha256']==sha('embeddings.json')
        before,after=result['upstream']
        assert before['processed']==1 and after['processed']==int(fixed)
        assert (before['input_hash']!=after['input_hash'])==fixed
        for row in result['upstream']:
            raw=read(f"{path}/upstream_{row['stage']}_docstore.json")
            stored=raw['docstore/data']['0']['__data__']['metadata']
            assert stored==row['stored_metadata']
        assert after['stored_metadata']==fixture['upstream']['after' if fixed else 'before']
        scores={}
        for row in result['vector_stages']:
            stage=row['stage'];ds=read(f'{path}/{stage}_docstore.json');vs=read(f'{path}/{stage}_vector_store.json')
            assert set(ds['docstore/data'])==set(vs['embedding_dict'])==required_ids
            response=row['response'];assert set(response['ids'])==required_ids
            assert len(response['ids'])==len(response['similarities'])==2
            assert response['similarities']==sorted(response['similarities'],reverse=True)
            assert row['ingest_seconds']>=0 and row['query_seconds']>=0
            for d in fixture['documents']:
                ident=d['id'];stored=ds['docstore/data'][ident]['__data__']
                assert stored['text']==d['text'] and stored['metadata']==row['stored_metadata'][ident]
                assert ('Title' in vs['metadata_dict'][ident])==('Title' in stored['metadata'])
                for key,value in stored['metadata'].items():assert vs['metadata_dict'][ident][key]==value
                assert ds['docstore/metadata'][ident]['doc_hash']==row['stored_hashes'][ident]
                metadata='\n'.join(f'{k}: {v}' for k,v in stored['metadata'].items())
                text=(metadata+'\n\n'+d['text']).strip() if metadata else d['text']
                assert vs['embedding_dict'][ident]==vectors[text]
                measured=response['similarities'][response['ids'].index(ident)]
                assert abs(cosine(query,vs['embedding_dict'][ident])-measured)<1e-12
                score_checks+=1
            scores[stage]=response['similarities'][response['ids'].index(fixture['target'])]
        stages={r['stage']:r for r in result['vector_stages']}
        assert stages['initial']['processed']==stages['clean_rebuild']['processed']==2
        assert stages['metadata_edit']['processed']==int(fixed) and stages['unchanged_resubmit']['processed']==0
        assert len(stages['metadata_edit']['embedding_inputs'])==int(fixed)
        assert stages['unchanged_resubmit']['embedding_inputs']==[]
        assert scores['metadata_edit']==scores['clean_rebuild' if fixed else 'initial']
        assert scores['unchanged_resubmit']==scores['metadata_edit']
        delta=scores['metadata_edit']-scores['clean_rebuild']
        summary.append({'version':version,'edit_processed':int(fixed),'target_score_before':scores['initial'],
          'target_score_after_edit':scores['metadata_edit'],'clean_rebuild_score':scores['clean_rebuild'],
          'residual_score_difference':delta,'absolute_residual_above_epsilon':abs(delta)>fixture['epsilon'],
          'stored_title_after_edit':stages['metadata_edit']['stored_metadata'][fixture['target']].get('Title')})
    assert summary[0]['target_score_before']==summary[1]['target_score_before']
    assert summary[0]['clean_rebuild_score']==summary[1]['clean_rebuild_score']
    return {'passed':True,'versions':2,'documents':2,'queries':1,'native_cosine_scores_checked':score_checks,
      'epsilon':fixture['epsilon'],'results':summary,'model_reencoded':False,'native_workflow_rerun':False,
      'scope':'Saved native in-process pipeline/store/query reproduction, not a deployment or answer-level audit.'}

if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
