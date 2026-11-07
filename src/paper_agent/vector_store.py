"""Immutable, separately persisted Chroma retrieval vectors and source contracts.

Collections hold only IDs, scalar provenance and vectors. Source text, boxes and
other citation data remain in independently hashed authoritative sidecars.
"""
import json
import hashlib
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .storage import digest, write_json


class ChromaVectorStore:
    _lock = threading.RLock()
    SCHEMA = 1

    def __init__(self, directory, contract, create=False, allow_building=False):
        import chromadb
        from chromadb.config import Settings
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory/'vector-store.json'
        self.contract = contract
        self.signature = digest(contract)
        self._exact_cache = None
        self.last_search_mode = None
        self.dimension = contract.get('dimension')
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError('Invalid vector dimension')
        if contract.get('normalization') != 'l2' or contract.get('metric') != 'cosine':
            raise ValueError('Unsupported vector normalization or metric')
        name = contract.get('collection')
        if not isinstance(name,str) or not name.startswith('paper_') or len(name)>63:
            raise ValueError('Invalid retrieval collection name')
        if create:
            self.directory.mkdir(parents=True,exist_ok=False)
            self.manifest = {'schema':self.SCHEMA, 'contract':contract, 'signature':self.signature,
                             'state':'building', 'created_utc':datetime.now(timezone.utc).isoformat()}
            write_json(self.manifest_path,self.manifest)
        else:
            self.manifest=json.loads(self.manifest_path.read_text(encoding='utf8'))
            if (self.manifest.get('schema')!=self.SCHEMA or self.manifest.get('contract')!=contract
                    or self.manifest.get('signature')!=self.signature):
                raise ValueError('Vector collection corpus/model contract mismatch')
            if self.manifest.get('state')!='sealed' and not allow_building:
                raise ValueError('Vector collection build is incomplete')
        with self._lock:
            self.client=chromadb.PersistentClient(path=(self.directory/'chroma').as_posix(),
                                                  settings=Settings(anonymized_telemetry=False,allow_reset=False))
            if create:
                self.collection=self.client.create_collection(name,embedding_function=None,
                    metadata={'hnsw:space':'cosine','contract_hash':self.signature,
                              'schema':self.SCHEMA,'dimension':self.dimension})
            else:
                self.collection=self.client.get_collection(name,embedding_function=None)
        meta=self.collection.metadata or {}
        if (meta.get('contract_hash')!=self.signature or meta.get('hnsw:space')!='cosine'
                or meta.get('dimension')!=self.dimension or meta.get('schema')!=self.SCHEMA):
            raise ValueError('Chroma collection metadata mismatch')
        if self.manifest.get('state')=='sealed':
            ids=self.collection.get(include=[])['ids']
            if len(ids)!=self.manifest.get('count') or digest(sorted(ids))!=self.manifest.get('ids_hash'):
                raise ValueError('Chroma collection IDs or count mismatch')
            if self._vector_fingerprint(ids)!=self.manifest.get('vectors_hash'):
                raise ValueError('Chroma vector contents or normalization mismatch')

    def _vector_fingerprint(self, ids):
        pairs=[]; ordered=sorted(ids)
        for start in range(0,len(ordered),256):
            block=ordered[start:start+256]
            result=self.collection.get(ids=block,include=['embeddings'])
            returned=result['ids']; values=np.asarray(result['embeddings'],dtype=np.float32)
            if (len(returned)!=len(block) or set(returned)!=set(block)
                    or values.shape!=(len(block),self.dimension) or not np.isfinite(values).all()):
                raise ValueError('Chroma vector contents or dimensions mismatch')
            norms=np.linalg.norm(values,axis=1)
            if np.any(norms<=0) or not np.allclose(norms,1.,atol=1e-4,rtol=1e-4):
                raise ValueError('Chroma vector normalization mismatch')
            indexed={key:values[i] for i,key in enumerate(returned)}
            pairs.extend((key,hashlib.sha256(np.asarray(indexed[key],dtype='<f4').tobytes()).hexdigest()) for key in block)
        return digest(pairs)

    def _require_building(self):
        if self.manifest.get('state')!='building':
            raise ValueError('Sealed collection is immutable; build a new version')

    def upsert(self, ids, vectors, metadatas=None):
        self._require_building()
        if len(ids)!=len(set(ids)) or not ids or any(not isinstance(x,str) or not x for x in ids):
            raise ValueError('Vector IDs must be nonempty and unique within a batch')
        values=np.asarray(vectors,dtype=np.float32)
        if values.shape!=(len(ids),self.dimension) or not np.isfinite(values).all():
            raise ValueError('Vector dimensions or values invalid')
        norms=np.linalg.norm(values,axis=1)
        if np.any(norms<=0) or not np.allclose(norms,1.0,atol=1e-4,rtol=1e-4):
            raise ValueError('Vectors must be L2 normalized')
        if metadatas is not None and len(metadatas)!=len(ids):
            raise ValueError('Vector metadata count mismatch')
        # Chroma 0.6 has bounded batch sizes. Explicit chunks also bound RAM.
        for start in range(0,len(ids),512):
            end=min(start+512,len(ids)); kwargs={'ids':ids[start:end], 'embeddings':values[start:end].tolist()}
            if metadatas is not None: kwargs['metadatas']=metadatas[start:end]
            self.collection.upsert(**kwargs)

    def delete(self, ids):
        self._require_building()
        if ids: self.collection.delete(ids=list(ids))

    def seal(self, expected_ids):
        self._require_building()
        expected=list(expected_ids)
        actual=self.collection.get(include=[])['ids']
        if len(expected)!=len(set(expected)) or set(expected)!=set(actual):
            raise ValueError('Cannot seal incomplete or unexpected vector IDs')
        self.manifest.update(state='sealed',count=len(actual),ids_hash=digest(sorted(actual)),
                             vectors_hash=self._vector_fingerprint(actual),
                             sealed_utc=datetime.now(timezone.utc).isoformat())
        write_json(self.manifest_path,self.manifest)

    def search(self, vector, n_results):
        if self.manifest.get('state')!='sealed': raise ValueError('Cannot query an incomplete collection')
        value=np.asarray(vector,dtype=np.float32)
        if value.shape!=(self.dimension,) or not np.isfinite(value).all():
            raise ValueError('Query vector dimension or values invalid')
        norm=np.linalg.norm(value)
        if not math.isfinite(float(norm)) or norm<=0: raise ValueError('Invalid query norm')
        value=value/norm
        count=self.collection.count()
        if not count: return []
        if type(n_results) is not int or n_results<1: raise ValueError('n_results must be positive')
        requested=min(n_results,count)
        if requested==count and self._exact_cache is not None:
            return self._exact_search(value,requested)
        try:
            result=self.collection.query(query_embeddings=[value.tolist()],n_results=requested,include=['distances'])
            self.last_search_mode='chroma_hnsw_cosine'
        except RuntimeError as exc:
            # Chroma 0.6.3 may be unable to return k=count from HNSW even
            # though all vectors are present. For a requested full ranking,
            # use an explicitly reported exact scan of the Chroma-stored
            # vectors; do not turn this into an empty/silent success.
            if requested!=count or 'contigious 2D array' not in str(exc):
                raise
            return self._exact_search(value,requested)
        ids=result['ids'][0]; distances=result['distances'][0]
        if len(ids)!=len(distances): raise ValueError('Chroma returned inconsistent distances')
        found=[]
        for key,distance in zip(ids,distances):
            if not isinstance(distance,(float,int)) or not math.isfinite(distance) or not -1e-4<=distance<=2.0001:
                raise ValueError('Invalid cosine distance')
            # For normalized vectors cosine similarity is 1 - Chroma cosine distance.
            found.append((key,max(-1.,min(1.,1.-float(distance)))))
        return found

    def _exact_search(self, vector, n_results):
        if self._exact_cache is None:
            result=self.collection.get(include=['embeddings'])
            ids=result['ids']; values=np.asarray(result['embeddings'],dtype=np.float32)
            if len(ids)!=self.manifest.get('count') or values.shape!=(len(ids),self.dimension):
                raise ValueError('Chroma exact scan vector count mismatch')
            order=np.argsort(np.asarray(ids,dtype=str),kind='stable')
            self._exact_cache=([ids[int(i)] for i in order],np.ascontiguousarray(values[order]))
        ids,values=self._exact_cache
        similarities=values@vector
        order=np.argsort(-similarities,kind='stable')[:n_results]
        self.last_search_mode='chroma_exact_full_scan_after_hnsw_limit'
        return [(ids[int(i)],max(-1.,min(1.,float(similarities[int(i)])))) for i in order]


class ChromaIndexAdapter:
    """Minimal index API preserving the existing retrieval caller's ID order."""
    def __init__(self,store,ordered_ids):
        self.store=store; self.ordered_ids=list(ordered_ids)
        if (len(self.ordered_ids)!=len(set(self.ordered_ids)) or len(self.ordered_ids)!=store.manifest['count']
                or digest(sorted(self.ordered_ids))!=store.manifest['ids_hash']):
            raise ValueError('Vector ID mapping differs from source snapshot')
        self.positions={key:i for i,key in enumerate(self.ordered_ids)}
        self.d=store.dimension; self.ntotal=len(self.ordered_ids)

    def search(self,vectors,n_results):
        vectors=np.asarray(vectors,dtype=np.float32)
        if vectors.ndim!=2 or vectors.shape[1]!=self.d: raise ValueError('Vector query dimension mismatch')
        scores=[]; positions=[]
        for vector in vectors:
            found=self.store.search(vector,n_results)
            scores.append([score for _,score in found]); positions.append([self.positions[key] for key,_ in found])
        return np.asarray(scores,dtype=np.float32),np.asarray(positions,dtype=np.int64)
