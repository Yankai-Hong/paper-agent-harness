"""Measure bounded filtering and pagination against exact real development queries."""
import argparse
import json
import time
from pathlib import Path


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from paper_agent.bounded_search import search_grouped
    from paper_agent.retrieval import BGEEncoder
    from paper_agent.vector_store import ChromaVectorStore
    index = Path('indexes/irpapers-chroma-text-baseline-v2-20261106')
    chunks = list(rows(index / 'chunks.jsonl'))
    source = {c['chunk_id']: (c['paper_id'], c['page_id']) for c in chunks}
    paper = 'irp-98'  # Fixed before inspecting these questions; never derived from gold.
    page_filter = sorted({c['page_id'] for c in chunks if c['paper_id'] == paper})[:3]
    if len(page_filter) < 3:
        raise ValueError('Fixed non-gold scope lacks pages')
    meta = json.loads((index / 'metadata.json').read_text(encoding='utf-8'))
    store = ChromaVectorStore(index / 'vectors', meta['vector_contract'])
    questions = [r for r in rows('runs/irpapers-chroma-baseline-v2-20261106/text-results.jsonl') if r['partition'] == 'dev'][:3]
    encoder = BGEEncoder('models/bge-m3', 'cuda:0', 2, 768)
    vectors = encoder.encode([r['question'] for r in questions])
    result = []
    for row, vector in zip(questions, vectors):
        for label, pages, offset, limit in (('paper-first', None, 0, 5), ('paper-next', None, 5, 5),
                                             ('page-filter', page_filter, 0, 3)):
            start = time.perf_counter()
            approx = search_grouped(store, vector, source, limit=limit, offset=offset, paper_ids=[paper], page_ids=pages,
                                    initial_depth=100, max_depth=1000, exact_fallback=True)
            elapsed = time.perf_counter() - start
            exact = search_grouped(store, vector, source, limit=limit, offset=offset, paper_ids=[paper], page_ids=pages,
                                   initial_depth=store.manifest['count'], max_depth=store.manifest['count'])
            got = [(x['paper_id'], x['page_id']) for x in approx['results']]
            expected = [(x['paper_id'], x['page_id']) for x in exact['results']]
            result.append({'case_id': row['case_id'], 'scope': label, 'offset': offset, 'limit': limit,
                           'retrieved': got, 'exact': expected, 'exact_order_match': got == expected,
                           'satisfied': approx['satisfied'], 'reason': approx['reason'], 'depths': approx['depths'],
                           'search_mode': approx['search_mode'], 'seconds': elapsed})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump({'paper_scope': paper, 'page_scope': page_filter, 'source': 'fixed independent of labels', 'results': result},
                  stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(out), 'results': result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
