"""Text + CLIP candidate generation and candidate-only ColQwen page search."""
import hashlib
import json
from pathlib import Path

from .page_fusion import fuse_page_rankings


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ClipPageCandidates:
    """Full page CLIP candidates from an independently bound Chroma index."""
    def __init__(self, directory, encoder, sources, source_hash):
        from .vector_store import ChromaVectorStore
        from .storage import digest
        root = Path(directory)
        metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
        mapping_path = root / 'mapping.jsonl'
        contract = metadata.get('vector_contract') or {}
        if (metadata.get('backend') != 'chroma' or metadata.get('pages_sha256') != source_hash
                or metadata.get('mapping_sha256') != _sha(mapping_path)
                or metadata.get('vector_manifest_sha256') != _sha(root / 'vectors' / 'vector-store.json')
                or metadata.get('model_provenance') != encoder.provenance
                or contract.get('source_hash') != source_hash
                or contract.get('mapping_hash') != metadata.get('mapping_sha256')
                or contract.get('model_provenance') != encoder.provenance
                or contract.get('dimension') != metadata.get('dimension')
                or contract.get('kind') != 'clip_page_image'):
            raise ValueError('CLIP page index provenance mismatch')
        with mapping_path.open(encoding='utf-8') as stream:
            mapping = [json.loads(line) for line in stream if line.strip()]
        ids = [item['page_id'] for item in mapping]
        if len(ids) != len(set(ids)) or any(
                item['page_id'] not in sources or any(item.get(k) != sources[item['page_id']].get(k)
                    for k in ('paper_id', 'image_sha256')) for item in mapping):
            raise ValueError('CLIP page mapping differs from the source manifest')
        self.store = ChromaVectorStore(root / 'vectors', metadata['vector_contract'])
        if self.store.manifest['count'] != len(ids) or self.store.manifest['ids_hash'] != digest(sorted(ids)):
            raise ValueError('CLIP index IDs or count mismatch')
        self.encoder = encoder

    def search(self, query, limit=50):
        from .visual_retrieval import _normalize
        vector = _normalize(self.encoder.encode_texts([query]))[0]
        # Match the frozen benchmark's full-corpus page ranking; the store
        # reports when Chroma needs its exact full-scan fallback.
        ranking = self.store.search(vector, self.store.manifest['count'])
        return {'pages': [page for page, _ in ranking[:limit]], 'search_mode': self.store.last_search_mode}


class HybridColQwenPageSearch:
    """Run text and CLIP independently, then rerank their page union."""
    def __init__(self, text_retriever, clip_pages, colqwen, *, text_source_hash, depth=50, visual_weight=0):
        if type(depth) is not int or depth < 1 or 2 * depth > colqwen.max_candidates:
            raise ValueError('Invalid candidate depth')
        if not text_source_hash or text_source_hash != colqwen.store.contract.get('source_hash'):
            raise ValueError('Text and visual source manifests differ')
        self.text, self.clip, self.colqwen = text_retriever, clip_pages, colqwen
        self.depth, self.visual_weight = depth, visual_weight

    def search(self, query, *, limit=20):
        if type(limit) is not int or limit < 1:
            raise ValueError('Invalid result limit')
        total = len(self.text.chunks)
        if not total:
            raise ValueError('Empty text index')
        text = self.text.search(query, mode='hybrid', top_k=min(100, total), candidate_k=total)
        pages, seen, first_evidence, unknown, refs = [], set(), {}, [], []
        for rank, chunk_id in enumerate(text['candidate_ids'], 1):
            item = self.text.by_id[chunk_id]
            page = item.get('page_id')
            source = self.colqwen.sources.get(page)
            if not source or item.get('paper_id') != source.get('paper_id'):
                unknown.append({**item, 'candidate_rank': rank})
                refs.append({'paper_id': item.get('paper_id'), 'chunk_id': chunk_id,
                             'page_id': None, 'rank': rank, 'page_mapping': 'unknown'})
                continue
            refs.append({'paper_id': item['paper_id'], 'chunk_id': chunk_id, 'page_id': page,
                         'rank': rank, 'page_mapping': 'verified'})
            first_evidence.setdefault(page, {**item, 'candidate_rank': rank})
            if page not in seen:
                pages.append(page)
                seen.add(page)
        clip = self.clip.search(query, limit=self.depth)
        candidates = list(dict.fromkeys(pages[:self.depth] + clip['pages']))
        visual = self.colqwen.search_text(query, candidates)
        visual_by_page = {item['page_id']: item for item in visual['hits']}
        ranked = fuse_page_rankings([pages, [item['page_id'] for item in visual['hits'][:20]]],
                                    [1, self.visual_weight], k=60)
        results = [{'paper_id': self.colqwen.sources[page]['paper_id'], 'page_id': page,
                    'page': self.colqwen.sources[page]['page_number'],
                    'text_evidence': first_evidence.get(page), 'page_evidence': visual_by_page.get(page)}
                   for page in ranked[:limit]]
        return {'query': query, 'scope': 'ColQwen reranks only text/CLIP candidate union',
                'text_candidate_pages': pages[:self.depth], 'clip_candidate_pages': clip['pages'],
                'clip_search_mode': clip.get('search_mode'), 'candidate_count': len(candidates),
                'visual_weight': self.visual_weight, 'unknown_text_locations': unknown,
                'text_candidate_refs': refs, 'text_hits': text['hits'], 'colqwen_hits': visual['hits'], 'pages': results}
