"""Text + CLIP candidate generation and candidate-only ColQwen page search."""
from .page_clip import ClipPageCandidates, file_sha256 as _sha
from .page_fusion import fuse_page_rankings


class HybridColQwenPageSearch:
    """Run text and CLIP independently, then rerank their page union."""
    def __init__(self, text_retriever, clip_pages, colqwen, *, text_source_hash, depth=50, visual_weight=0,
                 visual_sources=None, visual_root=None):
        if type(depth) is not int or depth < 1 or 2 * depth > colqwen.max_candidates:
            raise ValueError('Invalid candidate depth')
        if not text_source_hash or text_source_hash != colqwen.store.contract.get('source_hash'):
            raise ValueError('Text and visual source manifests differ')
        self.text, self.clip, self.colqwen = text_retriever, clip_pages, colqwen
        self.depth, self.visual_weight = depth, visual_weight
        if (visual_sources is None) != (visual_root is None):
            raise ValueError('Docling visual sources and their image root must be supplied together')
        self.visual_sources, self.visual_root = visual_sources, visual_root
        locators = [(x.get('paper_id'), x.get('pdf_sha256'), x.get('page_number'), page)
                    for page, x in colqwen.sources.items() if x.get('pdf_sha256')]
        self.pdf_pages = {(paper, pdf, number): page for paper, pdf, number, page in locators}
        if len(self.pdf_pages) != len(locators):
            raise ValueError('Ambiguous PDF page mapping')

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
            if not page and item.get('pdf_sha256') and type(item.get('page')) is int:
                page = self.pdf_pages.get((item.get('paper_id'), item['pdf_sha256'], item['page']))
            source = self.colqwen.sources.get(page)
            if (not source or item.get('paper_id') != source.get('paper_id')
                    or (source.get('pdf_sha256') and item.get('pdf_sha256') != source['pdf_sha256'])):
                unknown.append({**item, 'candidate_rank': rank})
                refs.append({'paper_id': item.get('paper_id'), 'chunk_id': chunk_id,
                             'page_id': None, 'rank': rank, 'page_mapping': 'unknown'})
                continue
            refs.append({'paper_id': item['paper_id'], 'chunk_id': chunk_id, 'page_id': page,
                         'rank': rank, 'page_mapping': 'verified'})
            original = {**item, 'candidate_rank': rank}
            if isinstance(item.get('parser'), dict) and item['parser'].get('name') == 'docling':
                original.setdefault('parse_status', 'docling_extracted_unverified')
            first_evidence.setdefault(page, original)
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
        unresolved = []
        if self.visual_sources is not None:
            from .docling_page_evidence import resolve_page_visual_evidence
            resolved = resolve_page_visual_evidence([x['page_id'] for x in results], self.colqwen.sources,
                                                     self.visual_sources, self.visual_root)
            for item in results:
                item['page_source_evidence'] = resolved['by_page'][item['page_id']]
            unresolved = resolved['unknown']
        return {'query': query, 'scope': 'ColQwen reranks only text/CLIP candidate union',
                'text_candidate_pages': pages[:self.depth], 'clip_candidate_pages': clip['pages'],
                'clip_search_mode': clip.get('search_mode'), 'candidate_count': len(candidates),
                'visual_weight': self.visual_weight, 'unknown_text_locations': unknown,
                'text_candidate_refs': refs, 'text_hits': text['hits'], 'colqwen_hits': visual['hits'],
                'unresolved_visual_evidence': unresolved, 'pages': results}
