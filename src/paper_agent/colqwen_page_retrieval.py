"""Candidate-only page retrieval using a sealed ColQwen multi-vector store."""
import hashlib
import json
import re
from pathlib import Path

from .colqwen_store import _matrix
from .maxsim_batch import score_maxsim_batch


class ColQwenCandidateRetriever:
    """Rerank supplied page candidates; this is not full-corpus retrieval.

    The encoder must return active query-token vectors after masking padding.
    Source records and the image root are authoritative inputs, not inferred
    from vector scores. Each returned page is page-level evidence only.
    """

    def __init__(self, store, encoder, source_manifest, image_root, *, device='cpu', max_candidates=100):
        if not store.final.is_file():
            raise ValueError('A sealed page-token collection is required')
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError('Invalid candidate limit')
        if any(getattr(encoder, key, None) != store.contract.get(key) for key in
               ('model_revision', 'base_revision', 'weight_sha256', 'max_visual_tokens')):
            raise ValueError('Query encoder and page index model revisions differ')
        source_path = Path(source_manifest)
        h = hashlib.sha256()
        with source_path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != store.contract.get('source_hash'):
            raise ValueError('Page source manifest differs from index contract')
        with source_path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        self.sources = {item['page_id']: item for item in rows}
        if len(self.sources) != len(rows):
            raise ValueError('Duplicate page in source manifest')
        self.store, self.encoder = store, encoder
        self.image_root = Path(image_root).resolve()
        self.device, self.max_candidates = device, max_candidates

    def search_text(self, query, page_ids):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be nonempty')
        page_ids = list(dict.fromkeys(page_ids))
        if len(page_ids) > self.max_candidates:
            raise ValueError('Candidate limit exceeded')
        if not page_ids:
            return {'hits': [], 'scope': 'candidate_only', 'candidate_count': 0}
        documents, evidence = [], []
        for page_id in page_ids:
            if not isinstance(page_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', page_id):
                raise ValueError('Invalid page identity')
            source = self.sources.get(page_id)
            marker = self.store.committed / (page_id + '.json')
            if not isinstance(source, dict) or not marker.is_file():
                raise ValueError('Candidate page is absent from source or index')
            record = json.loads(marker.read_text(encoding='utf-8'))
            if (record.get('page_id') != page_id or record.get('paper_id') != source.get('paper_id')
                    or record.get('image_sha256') != source.get('image_sha256')
                    or type(source.get('page_number')) is not int or source['page_number'] < 1):
                raise ValueError('Candidate page provenance mismatch')
            path = (self.image_root / source['image_path']).resolve()
            if not path.is_relative_to(self.image_root) or not path.is_file():
                raise ValueError('Candidate image is outside the source root or missing')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != source['image_sha256']:
                raise ValueError('Candidate image changed')
            documents.append(self.store.get_page(page_id))
            evidence.append({'paper_id': source['paper_id'], 'page_id': page_id,
                             'page': source['page_number'],
                             'image_path': source['image_path'], 'image_sha256': source['image_sha256'],
                             'pdf_sha256': source.get('pdf_sha256'),
                             'evidence_type': 'page', 'citation_granularity': 'page'})
        query_vectors = _matrix(self.encoder.encode_query(query), self.store.contract['dimension'])
        scores = score_maxsim_batch(query_vectors, documents, device=self.device)
        hits = [{**item, 'score': float(score), 'candidate_rank': rank}
                for rank, (item, score) in enumerate(zip(evidence, scores), 1)]
        hits.sort(key=lambda item: (-item['score'], item['page_id']))
        return {'hits': hits, 'scope': 'candidate_only', 'candidate_count': len(page_ids)}
