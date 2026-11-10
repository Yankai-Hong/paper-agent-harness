import json
import numpy as np
import pytest
from paper_agent.batched_colqwen import read_pages, seal_batched, add_pages_batched
from paper_agent.resumable_colqwen import ResumableColQwenStore


def test_batched_read_matches_verified_individual_pages_and_detects_tampering(tmp_path):
    contract={'model_revision':'m','source_hash':'s','dimension':2}
    store=ResumableColQwenStore(tmp_path/'index',contract,create=True)
    store.add_page('a','p','a'*64,np.array([[1.,0.],[0.,1.]],dtype=np.float32))
    store.add_page('b','p','b'*64,np.array([[0.,1.]],dtype=np.float32))
    store.seal()
    records=[json.loads((store.committed/(x+'.json')).read_text()) for x in ('a','b')]
    pages=read_pages(store,records)
    for record, page in zip(records,pages):
        assert np.allclose(page,store.get_page(record['page_id']))
    with pytest.raises(ValueError,match='Duplicate'):
        read_pages(store,[records[0],records[0]])
    store.collection.update(ids=['a:000000'],metadatas=[{'page_id':'wrong'}])
    with pytest.raises(ValueError,match='provenance'):
        read_pages(store,[records[0]])


def test_batched_seal_verifies_unfinished_and_tampered_pages(tmp_path):
    import hashlib
    store=ResumableColQwenStore(tmp_path/'index',{'model_revision':'m','source_hash':'s','dimension':2},create=True)
    store.add_page('a','p','a'*64,np.array([[1.,0.]],dtype=np.float32))
    provenance=store.root/'patch-provenance.jsonl'; provenance.write_bytes(b'provenance')
    fingerprint=hashlib.sha256(provenance.read_bytes()).hexdigest()
    store.collection.update(ids=['a:000000'],metadatas=[{'page_id':'wrong'}])
    with pytest.raises(ValueError,match='provenance'):
        seal_batched(store,patch_provenance_sha256=fingerprint)
    assert not store.final.exists()
    store.collection.update(ids=['a:000000'],metadatas=[{'page_id':'a'}])
    seal_batched(store,patch_provenance_sha256=fingerprint)
    assert store.final.is_file()


def test_batched_import_recovers_after_interrupted_token_write(tmp_path, monkeypatch):
    import hashlib
    store=ResumableColQwenStore(tmp_path/'index',{'model_revision':'m','source_hash':'s','dimension':2},create=True)
    pages=[{'page_id':key,'paper_id':'p','image_sha256':key*64,
            'vectors':np.array([[1.,0.],[0.,1.]],dtype=np.float32)} for key in ('a','b','c')]
    original=store.collection.upsert
    calls=[]
    def interrupted(**kwargs):
        calls.append(1)
        if len(calls)==2: raise RuntimeError('simulated interruption')
        return original(**kwargs)
    monkeypatch.setattr(store.collection,'upsert',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):
        add_pages_batched(store,pages,token_batch=2)
    assert len(list(store.committed.glob('*.json')))==0
    monkeypatch.setattr(store.collection,'upsert',original)
    add_pages_batched(store,pages,token_batch=2)
    assert len(list(store.committed.glob('*.json')))==3
    provenance=store.root/'patch-provenance.jsonl'; provenance.write_bytes(b'provenance')
    seal_batched(store,patch_provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest())
    assert len(read_pages(store,[json.loads((store.committed/'a.json').read_text())]))==1
