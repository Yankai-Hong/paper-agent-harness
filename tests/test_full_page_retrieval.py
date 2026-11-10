import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from paper_agent.full_page_retrieval import FullColQwenPageRetriever
from paper_agent.resumable_colqwen import ResumableColQwenStore


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_search_scores_every_page_and_checks_cache_and_source(tmp_path):
    images=tmp_path/'images';images.mkdir()
    pages=[]
    for i in (1,2):
        path=images/f'{i}.png';path.write_bytes(str(i).encode())
        pages.append({'page_id':str(i),'paper_id':'p','page_number':i,'image_path':path.name,'image_sha256':sha(path)})
    manifest=tmp_path/'pages.jsonl';manifest.write_text(''.join(json.dumps(p)+'\n' for p in pages))
    contract={'model_revision':'model','base_revision':'base','weight_sha256':{'x':'y'},'max_visual_tokens':256,
              'source_hash':sha(manifest),'dimension':2}
    store=ResumableColQwenStore(tmp_path/'index',contract,create=True)
    store.add_page('1','p',pages[0]['image_sha256'],np.array([[1,0]],dtype=np.float32))
    store.add_page('2','p',pages[1]['image_sha256'],np.array([[0,1]],dtype=np.float32));store.seal()
    cache=tmp_path/'cache';cache.mkdir();(cache/'cache-pages').mkdir()
    matrix=np.array([[1,0],[0,1]],dtype=np.float32);np.save(cache/'vectors.npy',matrix)
    seal=sha(store.final);(cache/'plan.json').write_text(json.dumps({'index_seal_sha256':seal,'source_pages_sha256':sha(manifest)}))
    for i,page in enumerate(pages):
        marker={'seal':seal,'sha256':hashlib.sha256(matrix[i:i+1].tobytes()).hexdigest()}
        (cache/'cache-pages'/(page['page_id']+'.json')).write_text(json.dumps(marker))
    encoder=SimpleNamespace(model_revision='model',base_revision='base',weight_sha256={'x':'y'},
                            max_visual_tokens=256,encode_query=lambda q:[[0.,1.]])
    retriever=FullColQwenPageRetriever(store,encoder,manifest,images,cache,device='cpu',batch_pages=1)
    result=retriever.search_text('query',top_k=2)
    assert result['scope']=='exact_full_corpus' and result['pages_scored']==2
    assert [x['page_id'] for x in result['hits']]==['2','1']
    (images/'2.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='image changed'): retriever.search_text('query',top_k=1)
    marker_path=cache/'cache-pages'/'1.json'
    marker=json.loads(marker_path.read_text());marker['sha256']='bad';marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='checksum'): FullColQwenPageRetriever(store,encoder,manifest,images,cache,device='cpu')
