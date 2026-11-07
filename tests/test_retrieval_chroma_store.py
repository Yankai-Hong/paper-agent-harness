import json
import subprocess
import sys

import numpy as np
import pytest

from paper_agent.vector_store import ChromaVectorStore, ChromaIndexAdapter


def contract(dimension=2):
    return {'collection':'paper_test_vectors_v1','kind':'test','dimension':dimension,
            'normalization':'l2','metric':'cosine','corpus_hash':'corpus-v1','model':'model-v1'}


def test_persistent_build_resume_upsert_delete_and_distance(tmp_path):
    path=tmp_path/'retrieval'; c=contract()
    store=ChromaVectorStore(path,c,create=True)
    store.upsert(['a','b'],[[1,0],[0,1]])
    store.upsert(['a'],[[1,0]])
    store.delete(['b'])
    with pytest.raises(ValueError,match='incomplete'): ChromaVectorStore(path,c)
    resumed=ChromaVectorStore(path,c,allow_building=True)
    resumed.upsert(['b'],[[0,1]])
    resumed.seal(['a','b'])
    reopened=ChromaVectorStore(path,c)
    assert reopened.collection.count()==2
    scores=ChromaIndexAdapter(reopened,['a','b']).search(np.array([[1.,0.]],dtype=np.float32),2)
    assert scores[1].tolist()==[[0,1]]
    assert scores[0][0,0]==pytest.approx(1.0,abs=1e-5)
    assert scores[0][0,1]==pytest.approx(0.0,abs=1e-5)
    with pytest.raises(ValueError,match='immutable'): reopened.delete(['a'])
    with pytest.raises(ValueError,match='contract'): ChromaVectorStore(path,{**c,'model':'changed'})
    with pytest.raises(ValueError,match='normalized'):
        ChromaVectorStore(tmp_path/'bad',c,create=True).upsert(['x'],[[2,0]])
    # A distinct interpreter must read the sealed on-disk collection.
    script=('import json,sys; from paper_agent.vector_store import ChromaVectorStore; '
            's=ChromaVectorStore(sys.argv[1],json.loads(sys.argv[2])); '
            'print(s.collection.count())')
    process=subprocess.run([sys.executable,'-c',script,str(path),json.dumps(c)],capture_output=True,text=True,check=True)
    assert process.stdout.strip().endswith('2')


def test_id_mapping_mismatch_and_unsealed_failure(tmp_path):
    store=ChromaVectorStore(tmp_path/'idx',contract(),create=True)
    store.upsert(['a'],[[1.,0.]])
    with pytest.raises(ValueError,match='incomplete'): store.search([1.,0.],1)
    with pytest.raises(ValueError,match='incomplete or unexpected'): store.seal(['a','b'])
    store.seal(['a'])
    with pytest.raises(ValueError,match='mapping'): ChromaIndexAdapter(store,['other'])


def test_cosine_similarity_conversion_and_vector_tamper(tmp_path):
    store=ChromaVectorStore(tmp_path/'idx',contract(),create=True)
    store.upsert(['exact','near'],[[1.,0.],[.95,float(np.sqrt(1-.95**2))]])
    store.seal(['exact','near'])
    results=dict(store.search([1.,0.],2))
    assert results['exact']==pytest.approx(1.,abs=1e-5)
    assert results['near']==pytest.approx(.95,abs=1e-5)
    store.collection.upsert(ids=['near'],embeddings=[[0.,1.]])
    with pytest.raises(ValueError,match='vector contents'):
        ChromaVectorStore(tmp_path/'idx',contract())


def test_full_ranking_chroma_limit_uses_observable_exact_fallback(tmp_path,monkeypatch):
    store=ChromaVectorStore(tmp_path/'idx',contract(),create=True)
    store.upsert(['a','b'],[[1.,0.],[0.,1.]]);store.seal(['a','b'])
    calls=[]
    def broken(**kwargs):
        calls.append(1);raise RuntimeError('Cannot return the results in a contigious 2D array')
    monkeypatch.setattr(store.collection,'query',broken)
    assert store.search([1.,0.],2)==[('a',pytest.approx(1.)),('b',pytest.approx(0.))]
    assert store.last_search_mode=='chroma_exact_full_scan_after_hnsw_limit'
    assert store.search([0.,1.],2)[0][0]=='b' and len(calls)==1
    with pytest.raises(RuntimeError):store.search([1.,0.],1)
