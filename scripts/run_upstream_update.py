"""Run unchanged official native ingestion/update/query APIs; preserve all states."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery

HERE = Path(__file__).resolve().parents[1] / 'data/upstream_update'

class SavedMiniLM(BaseEmbedding):
    """Exact-input adapter for actual frozen MiniLM output, not a synthetic embedding."""
    vectors: dict[str,list[float]]
    calls: list[str]=[]

    def _get_text_embedding(self,text):
        self.calls.append(text)
        return self.vectors[text]

    def _get_query_embedding(self,text):
        return self.vectors[text]

    async def _aget_query_embedding(self,text):
        return self._get_query_embedding(text)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Preserve previous/failed measurements'
    args.output.mkdir(parents=True)
    fixture=json.loads((HERE/'fixture.json').read_text())
    embeddings=json.loads(args.embeddings.read_text())
    version=importlib.metadata.version('llama-index-core')
    assert version in ['0.12.27','0.12.28']
    result={'version':version,
      'embeddings_sha256':hashlib.sha256(args.embeddings.read_bytes()).hexdigest(),'upstream':[],'vector_stages':[]}
    def write(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    minimal=fixture['upstream'];store=SimpleDocumentStore()
    pipeline=IngestionPipeline(docstore=store,transformations=[],docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY)
    document=Document(id_=minimal['id'],text=minimal['text'],metadata=minimal['before'])
    for stage in ['before','after']:
        document.metadata=dict(minimal[stage])
        nodes=pipeline.run(documents=[document])
        result['upstream'].append({'stage':stage,'input_hash':document.hash,'input_metadata':dict(document.metadata),
          'processed':len(nodes),'stored_metadata':dict(store.docs[minimal['id']].metadata),
          'stored_hash':store.get_document_hash(minimal['id'])})
        store.persist(str(args.output/f'upstream_{stage}_docstore.json'))
    adapter=SavedMiniLM(vectors={r['text']:r['vector'] for r in embeddings['records']})
    def native_pipeline():
        ds=SimpleDocumentStore();vs=SimpleVectorStore()
        return IngestionPipeline(docstore=ds,vector_store=vs,transformations=[adapter],docstore_strategy=DocstoreStrategy.UPSERTS),ds,vs
    def docs(stage):
        return [Document(id_=d['id'],text=d['text'],metadata=dict(d[stage])) for d in fixture['documents']]
    def measure(stage,pipeline,ds,vs,documents):
        calls=len(adapter.calls);start=time.perf_counter()
        nodes=pipeline.run(documents=documents)
        ingest_seconds=time.perf_counter()-start;start=time.perf_counter()
        response=vs.query(VectorStoreQuery(query_embedding=adapter.get_query_embedding(fixture['query']),similarity_top_k=len(fixture['documents'])))
        query_seconds=time.perf_counter()-start
        value={'stage':stage,'input_hashes':{d.id_:d.hash for d in documents},'processed':len(nodes),
          'embedding_inputs':adapter.calls[calls:],'ingest_seconds':ingest_seconds,'query_seconds':query_seconds,
          'response':{'ids':response.ids,'similarities':response.similarities},
          'stored_metadata':{k:dict(v.metadata) for k,v in ds.docs.items()},
          'stored_hashes':{d['id']:ds.get_document_hash(d['id']) for d in fixture['documents']}}
        result['vector_stages'].append(value)
        ds.persist(str(args.output/f'{stage}_docstore.json'))
        vs.persist(str(args.output/f'{stage}_vector_store.json'))
        write('result.json',result)
    pipeline,ds,vs=native_pipeline()
    measure('initial',pipeline,ds,vs,docs('before'))
    changed=[d for d in docs('after') if d.id_==fixture['target']]
    measure('metadata_edit',pipeline,ds,vs,changed)
    measure('unchanged_resubmit',pipeline,ds,vs,[d for d in docs('after') if d.id_==fixture['target']])
    clean,clean_ds,clean_vs=native_pipeline()
    measure('clean_rebuild',clean,clean_ds,clean_vs,docs('after'))
    fixed=version=='0.12.28'
    assert result['upstream'][1]['processed']==int(fixed)
    assert result['upstream'][1]['stored_metadata']==minimal['after' if fixed else 'before']
    assert result['vector_stages'][1]['processed']==int(fixed)
    assert result['vector_stages'][2]['processed']==0
    assert result['vector_stages'][1]['stored_metadata'][fixture['target']]==fixture['documents'][0]['after' if fixed else 'before']
    result['assertions_passed']=True;write('result.json',result)
    print(json.dumps({'version':version,'assertions_passed':True,'metadata_edit_processed':int(fixed),'output':str(args.output)}))

if __name__=='__main__':main()
