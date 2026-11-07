"""Located text representations in the aligned CLIP space, separate from BGE."""
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

from .storage import digest, read_jsonl, write_json, write_jsonl
from .visual_retrieval import _normalize


def representation_vector_id(rep):
    return 'textrep-'+digest([rep['paper_id'],rep['chunk_id'],rep['start'],rep['end']])


def build_clip_text_index(source, directory, encoder, backend='chroma'):
    from .pipeline import _valid_located_text
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Refusing to overwrite an existing CLIP text index')
    sources = read_jsonl(source)
    keys = [(x.get('paper_id'), x.get('chunk_id')) for x in sources]
    if not sources or any(not all(k) for k in keys) or len(keys) != len(set(keys)):
        raise ValueError('Source evidence identities must be nonempty and unique')
    located = [x for x in sources if x.get('evidence_type') in (None, 'text')
               and isinstance(x.get('text'), str) and x['text'].strip() and _valid_located_text(x)]
    if not located:
        raise ValueError('No located text evidence')
    worker = Path(__file__).resolve().parents[2] / 'scripts' / 'clip_worker.py'
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clip-text-index-', dir=directory.parent) as tmp:
        tmp = Path(tmp)
        request, result = tmp/'split-request.json', tmp/'spans.json'
        request.write_text(json.dumps({'model':encoder.text_model_path, 'inputs':[x['text'] for x in located],
                                       'output':str(result)}, ensure_ascii=False), encoding='utf-8')
        subprocess.run([encoder.python, str(worker), 'split', str(request)], check=True, timeout=1800)
        split = json.loads(result.read_text(encoding='utf-8'))
        representations = []
        for item, spans in zip(located, split['spans']):
            for start, end in spans:
                representations.append({'paper_id':item['paper_id'], 'chunk_id':item['chunk_id'],
                                        'start':start, 'end':end, 'text':item['text'][start:end],
                                        'kind':'caption' if item.get('docling_label') == 'caption' else 'text',
                                        'extraction_status':'parser_output_not_independently_verified'})
        if not representations:
            raise ValueError('No tokenized text representations')
        # Encoder performs internal batches while loading the model once.
        vectors = _normalize(encoder.encode_texts([x['text'] for x in representations]))
        if len(vectors) != len(representations):
            raise ValueError('CLIP text vector count mismatch')
        if backend not in {'chroma','faiss'}: raise ValueError('Unknown CLIP text vector backend')
        directory.mkdir(parents=True,exist_ok=False)
        count,dimension=len(vectors),vectors.shape[1]
        metadata = {'schema':1, 'source_hash':digest(sources), 'representation_hash':digest(representations),
                    'model_provenance':encoder.provenance, 'dimension':dimension, 'count':count, 'backend':backend,
                    'source_count':len(sources), 'located_count':len(located),
                    'skipped_count':len(sources)-len(located), 'normalization':'l2', 'metric':'inner_product',
                    'preprocessing':{'tokenizer':'local_fast_offsets', 'max_seq_length':split['limit'],
                                     'window_tokens':split['window'], 'overlap_tokens':split['overlap'],
                                     'text':'exact_original_character_slice'}}
        if backend=='chroma':
            from .vector_store import ChromaVectorStore
            contract={'collection':'paper_clip_text_v1','kind':'clip_located_text', 'source_hash':metadata['source_hash'],
                      'representation_hash':metadata['representation_hash'],'model_provenance':encoder.provenance,
                      'dimension':dimension,'normalization':'l2','metric':'cosine','preprocessing':metadata['preprocessing']}
            store=ChromaVectorStore(directory/'vectors',contract,create=True)
            ids=[representation_vector_id(r) for r in representations]
            store.upsert(ids,vectors); store.seal(ids)
            metadata['vector_contract']=contract
        else:
            import faiss
            idx=faiss.IndexFlatIP(dimension); idx.add(vectors)
            faiss.write_index(idx,str(directory/'text.faiss'))
            metadata['index_sha256']=hashlib.sha256((directory/'text.faiss').read_bytes()).hexdigest()
        write_jsonl(directory/'sources.jsonl', sources)
        write_jsonl(directory/'representations.jsonl', representations)
        write_json(directory/'metadata.json', metadata)
    return metadata


class ClipTextRetriever:
    def __init__(self, directory, encoder, expected_source_hash=None):
        self.directory, self.encoder = Path(directory), encoder
        self.metadata = json.loads((self.directory/'metadata.json').read_text(encoding='utf-8'))
        self.sources = read_jsonl(self.directory/'sources.jsonl')
        self.representations = read_jsonl(self.directory/'representations.jsonl')
        m = self.metadata
        if (m.get('schema') != 1 or m.get('source_hash') != digest(self.sources)
                or m.get('representation_hash') != digest(self.representations)
                or (expected_source_hash is not None and m.get('source_hash') != expected_source_hash)):
            raise ValueError('CLIP text corpus version or hash mismatch')
        if m.get('model_provenance') != encoder.provenance:
            raise ValueError('CLIP model provenance differs from text index')
        if m.get('normalization') != 'l2' or m.get('metric') != 'inner_product':
            raise ValueError('Unsupported CLIP text index version or metric')
        if m.get('backend','faiss')=='chroma':
            from .vector_store import ChromaVectorStore,ChromaIndexAdapter
            contract=m.get('vector_contract')
            if (not isinstance(contract,dict) or contract.get('source_hash')!=m['source_hash']
                    or contract.get('representation_hash')!=m['representation_hash']
                    or contract.get('model_provenance')!=encoder.provenance or contract.get('preprocessing')!=m.get('preprocessing')
                    or contract.get('dimension')!=m.get('dimension')):
                raise ValueError('CLIP text vector contract mismatch')
            self.index=ChromaIndexAdapter(ChromaVectorStore(self.directory/'vectors',contract),
                                          [representation_vector_id(r) for r in self.representations])
        elif m.get('backend','faiss')=='faiss':
            import faiss
            path = self.directory/'text.faiss'
            if hashlib.sha256(path.read_bytes()).hexdigest() != m.get('index_sha256'):
                raise ValueError('CLIP text index checksum mismatch')
            self.index = faiss.read_index(str(path))
            if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError('CLIP text index metric mismatch')
        else: raise ValueError('Unknown CLIP text vector backend')
        if (self.index.d != m.get('dimension') or self.index.ntotal != m.get('count') or self.index.ntotal != len(self.representations)):
            raise ValueError('CLIP text dimensions or count mismatch')
        self.by_key = {(s['paper_id'],s['chunk_id']):s for s in self.sources}
        for rep in self.representations:
            source = self.by_key.get((rep['paper_id'],rep['chunk_id']))
            if (source is None or not isinstance(rep.get('start'), int) or not isinstance(rep.get('end'), int)
                    or not 0 <= rep['start'] < rep['end'] <= len(source['text'])
                    or source['text'][rep['start']:rep['end']] != rep['text']):
                raise ValueError('CLIP representation source mapping mismatch')

    def search_vector(self, vector, top_k=5, paper_ids=None, kind=None):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be positive')
        if kind not in (None, 'caption', 'text'):
            raise ValueError('Unknown CLIP text source kind')
        start = time.perf_counter()
        vector = _normalize(vector)
        if vector.shape != (1, self.index.d):
            raise ValueError('CLIP query dimension mismatch')
        scores, positions = self.index.search(vector, self.index.ntotal)
        allowed = set(paper_ids) if paper_ids is not None else None
        hits, seen = [], set()
        for pos, score in zip(positions[0], scores[0]):
            rep = self.representations[int(pos)]; key = (rep['paper_id'],rep['chunk_id'])
            if key in seen or (allowed is not None and key[0] not in allowed):
                continue
            source_kind = 'caption' if self.by_key[key].get('docling_label') == 'caption' else 'text'
            if kind is not None and source_kind != kind:
                continue
            seen.add(key)
            hits.append({**self.by_key[key], 'rank':len(hits)+1, 'clip_text_score':float(score),
                         'matched_span':[rep['start'],rep['end']], 'matched_text':rep['text']})
            if len(hits) == top_k:
                break
        return {'hits':hits, 'timing_ms':{'clip_text_retrieve':(time.perf_counter()-start)*1000},
                'source_hash':self.metadata['source_hash']}

    def search_image(self, path, top_k=5, paper_ids=None, kind=None):
        from .query_images import validate_query_image
        item = validate_query_image(path)
        return self.search_vector(self.encoder.encode_images([item['path']]), top_k, paper_ids, kind)
