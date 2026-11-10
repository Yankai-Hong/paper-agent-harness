"""Exact full-corpus text-to-page MaxSim over a verified lossless cache."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from .colqwen_store import _matrix
from .full_maxsim import rank_pages
from .maxsim_batch import score_maxsim_batch


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class FullColQwenPageRetriever:
    def __init__(self, store, encoder, source_manifest, image_root, cache_dir, *, device='cuda:0', batch_pages=64):
        if not store.final.is_file() or not 1<=batch_pages<=256:
            raise ValueError('Sealed index and valid batch size required')
        for key in ('model_revision','base_revision','weight_sha256','max_visual_tokens'):
            if getattr(encoder,key,None)!=store.contract.get(key): raise ValueError('Query model differs from page index')
        source=Path(source_manifest); self.root=Path(image_root).resolve(); self.cache=Path(cache_dir)
        if _sha(source)!=store.contract['source_hash']: raise ValueError('Page source manifest mismatch')
        with source.open(encoding='utf-8') as f:self.pages=[json.loads(x) for x in f if x.strip()]
        seal=json.loads(store.final.read_text(encoding='utf-8'))
        plan=json.loads((self.cache/'plan.json').read_text(encoding='utf-8'))
        if (len(self.pages)!=seal['pages'] or len({x['page_id'] for x in self.pages})!=len(self.pages)
                or plan['index_seal_sha256']!=_sha(store.final) or plan['source_pages_sha256']!=_sha(source)):
            raise ValueError('Full index or cache contract mismatch')
        self.offsets=[0]
        for page in self.pages:
            record=json.loads((store.committed/(page['page_id']+'.json')).read_text(encoding='utf-8'))
            if record['paper_id']!=page['paper_id'] or record['image_sha256']!=page['image_sha256']:
                raise ValueError('Page source mismatch')
            self.offsets.append(self.offsets[-1]+record['count'])
        self.matrix=np.load(self.cache/'vectors.npy',mmap_mode='r',allow_pickle=False)
        if self.matrix.shape!=(seal['tokens'],store.contract['dimension']) or self.offsets[-1]!=seal['tokens']:
            raise ValueError('Full page cache is incomplete')
        for i,page in enumerate(self.pages):
            marker=json.loads((self.cache/'cache-pages'/(page['page_id']+'.json')).read_text(encoding='utf-8'))
            if (marker['seal']!=plan['index_seal_sha256'] or marker['sha256']!=hashlib.sha256(
                    self.matrix[self.offsets[i]:self.offsets[i+1]].tobytes()).hexdigest()):
                raise ValueError('Full page cache checksum mismatch')
        self.cache_mtime=(self.cache/'vectors.npy').stat().st_mtime_ns
        self.encoder,self.device,self.batch_pages,self.dimension=encoder,device,batch_pages,store.contract['dimension']

    def search_text(self, query, *, top_k=20):
        if not isinstance(query,str) or not query.strip() or type(top_k) is not int or not 1<=top_k<=len(self.pages):
            raise ValueError('Invalid query or result limit')
        if (self.cache/'vectors.npy').stat().st_mtime_ns!=self.cache_mtime: raise ValueError('Page cache changed')
        started=time.perf_counter(); q=_matrix(self.encoder.encode_query(query),self.dimension)
        encoded=time.perf_counter(); scores=[]
        for start in range(0,len(self.pages),self.batch_pages):
            stop=min(start+self.batch_pages,len(self.pages))
            docs=[self.matrix[self.offsets[i]:self.offsets[i+1]] for i in range(start,stop)]
            scores.extend(score_maxsim_batch(q,docs,device=self.device))
        order=rank_pages([p['page_id'] for p in self.pages],scores)
        hits=[]
        for i in order[:top_k]:
            source=self.pages[i]; path=(self.root/source['image_path']).resolve()
            if not path.is_relative_to(self.root) or _sha(path)!=source['image_sha256']:
                raise ValueError('Retrieved page image changed')
            hits.append({**{k:source.get(k) for k in ('paper_id','page_id','page_number','image_path','image_sha256')},
                         'score':float(scores[i]),'evidence_type':'page','citation_granularity':'page'})
        if (self.cache/'vectors.npy').stat().st_mtime_ns!=self.cache_mtime: raise ValueError('Page cache changed')
        return {'scope':'exact_full_corpus','pages_scored':len(self.pages),'hits':hits,
                'timing_seconds':{'query_encoding':encoded-started,'page_scoring_and_validation':time.perf_counter()-encoded}}
