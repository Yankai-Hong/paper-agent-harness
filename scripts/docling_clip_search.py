"""Text and CLIP page retrieval over a source-bound Docling PDF corpus."""
import argparse
import json
from pathlib import Path

from paper_agent.docling_page_evidence import resolve_page_visual_evidence
from paper_agent.page_clip import ClipPageCandidates, file_sha256
from paper_agent.page_fusion import fuse_page_rankings
from paper_agent.retrieval import BGEEncoder, Retriever
from paper_agent.storage import read_jsonl
from paper_agent.visual_retrieval import SubprocessEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('query')
    for key in ('prepared', 'text-index', 'text-binding', 'encoder', 'clip-index', 'clip-python',
                'clip-image-model', 'clip-text-model'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--visual-weight', type=float, default=0)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--evidence-jsonl')
    p.add_argument('--evidence-root')
    a = p.parse_args()
    if not a.query.strip() or a.limit < 1 or bool(a.evidence_jsonl) != bool(a.evidence_root):
        raise ValueError('Query, limit or visual evidence arguments invalid')
    source, text_index = Path(a.prepared), Path(a.text_index)
    binding = json.loads(Path(a.text_binding).read_text(encoding='utf-8'))
    metadata = json.loads((text_index / 'metadata.json').read_text(encoding='utf-8'))
    if (binding.get('kind') != 'docling_pdf_pages_v1' or binding.get('status') != 'sealed'
            or binding.get('pages_sha256') != file_sha256(source / 'pages.jsonl')
            or binding.get('text_chunks_sha256') != file_sha256(text_index / 'chunks.jsonl')
            or binding.get('text_corpus_hash') != metadata.get('corpus_hash')):
        raise ValueError('Docling text and PDF page source mismatch')
    rows = read_jsonl(source / 'pages.jsonl')
    pages = {x['page_id']: x for x in rows}
    locators = {(x['paper_id'], x['pdf_sha256'], x['page_number']): x['page_id'] for x in rows}
    if len(pages) != len(rows) or len(locators) != len(rows):
        raise ValueError('Ambiguous PDF page source')
    text = Retriever(text_index, BGEEncoder(a.encoder, a.device, 2, metadata['max_length']))
    clip_encoder = SubprocessEncoder(a.clip_python, a.clip_image_model, a.clip_text_model, device=a.device)
    clip = ClipPageCandidates(a.clip_index, clip_encoder, pages, binding['pages_sha256'])
    response = text.search(a.query, mode='hybrid', top_k=min(100, len(text.chunks)), candidate_k=len(text.chunks))
    ranked_pages, seen, first, refs, unknown = [], set(), {}, [], []
    for rank, chunk_id in enumerate(response['candidate_ids'], 1):
        item = text.by_id[chunk_id]
        key = (item.get('paper_id'), item.get('pdf_sha256'), item.get('page'))
        page_id = locators.get(key)
        refs.append({'paper_id': item.get('paper_id'), 'chunk_id': chunk_id, 'page_id': page_id,
                     'rank': rank, 'page_mapping': 'verified' if page_id else 'unknown'})
        if page_id is None:
            unknown.append({**item, 'candidate_rank': rank})
            continue
        entry = {**item, 'candidate_rank': rank}
        entry.setdefault('parse_status', 'docling_extracted_unverified')
        first.setdefault(page_id, entry)
        if page_id not in seen:
            seen.add(page_id); ranked_pages.append(page_id)
    visual = clip.search(a.query, limit=len(pages))
    visual_lookup = {x['page_id']: x for x in visual['hits']}
    combined = fuse_page_rankings([ranked_pages, visual['pages']], [1, a.visual_weight], k=60)
    results = []
    for page_id in combined[:a.limit]:
        page = pages[page_id]
        path = (source / page['image_path']).resolve()
        if not path.is_relative_to(source.resolve()) or file_sha256(path) != page['image_sha256']:
            raise ValueError('Rendered page image changed')
        results.append({'paper_id': page['paper_id'], 'page_id': page_id, 'page': page['page_number'],
                        'text_evidence': first.get(page_id), 'clip_rank': visual_lookup.get(page_id),
                        'page_evidence': {**page, 'citation_granularity': 'page'}})
    unresolved = []
    if a.evidence_jsonl:
        resolved = resolve_page_visual_evidence([x['page_id'] for x in results], pages,
                                                 read_jsonl(a.evidence_jsonl), a.evidence_root)
        for item in results:
            item['page_source_evidence'] = resolved['by_page'][item['page_id']]
        unresolved = resolved['unknown']
    print(json.dumps({'query': a.query, 'visual_weight': a.visual_weight, 'text_candidate_refs': refs,
                      'unknown_text_locations': unknown, 'text_hits': response['hits'],
                      'clip_hits': visual['hits'], 'clip_search_mode': visual['search_mode'],
                      'unresolved_visual_evidence': unresolved, 'pages': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
