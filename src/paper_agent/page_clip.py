"""Independent CLIP page candidates bound to an authoritative page manifest."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ClipPageCandidates:
    def __init__(self, directory, encoder, sources, source_hash):
        from .storage import digest
        from .vector_store import ChromaVectorStore
        root = Path(directory)
        metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
        mapping_path = root / 'mapping.jsonl'
        contract = metadata.get('vector_contract') or {}
        if (metadata.get('backend') != 'chroma' or metadata.get('pages_sha256') != source_hash
                or metadata.get('mapping_sha256') != file_sha256(mapping_path)
                or metadata.get('vector_manifest_sha256') != file_sha256(root / 'vectors' / 'vector-store.json')
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
        self.store = ChromaVectorStore(root / 'vectors', contract)
        if self.store.manifest['count'] != len(ids) or self.store.manifest['ids_hash'] != digest(sorted(ids)):
            raise ValueError('CLIP index IDs or count mismatch')
        self.encoder = encoder

    def search(self, query, limit=50):
        from .visual_retrieval import _normalize
        vector = _normalize(self.encoder.encode_texts([query]))[0]
        ranking = self.store.search(vector, self.store.manifest['count'])
        return {'pages': [page for page, _ in ranking[:limit]],
                'hits': [{'page_id': page, 'score': float(score), 'rank': rank}
                         for rank, (page, score) in enumerate(ranking[:limit], 1)],
                'search_mode': self.store.last_search_mode}
