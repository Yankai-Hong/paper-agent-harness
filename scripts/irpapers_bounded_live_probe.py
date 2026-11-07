"""Measure opt-in bounded page retrieval end-to-end on three dev questions."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import psutil
    import torch
    from paper_agent.bounded_retriever import BoundedPageRetriever
    from paper_agent.retrieval import BGEEncoder, Retriever
    path = Path('runs/irpapers-chroma-baseline-v2-20261106/text-results.jsonl')
    with path.open(encoding='utf-8') as stream:
        cases = [json.loads(line) for line in stream if '"partition": "dev"' in line][:3]
    if len(cases) != 3:
        raise ValueError('Development cases missing')
    torch.cuda.reset_peak_memory_stats()
    init = time.perf_counter()
    encoder = BGEEncoder('models/bge-m3', 'cuda:0', 2, 768)
    retriever = BoundedPageRetriever(Retriever('indexes/irpapers-chroma-text-baseline-v2-20261106', encoder))
    startup = time.perf_counter() - init
    records = []
    for case in cases:
        start = time.perf_counter()
        found = retriever.search(case['question'], limit=20, initial_depth=300, max_depth=300)
        records.append({'case_id': case['case_id'], 'seconds': time.perf_counter() - start, 'timing_ms': found['timing_ms'],
                        'mode': found['search_mode'], 'satisfied': found['satisfied'],
                        'top5_exact_order_match': [x['page_id'] for x in found['results'][:5]] == case['dense']['pages'][:5]})
    result = {'scope': 'three development questions; dense page retrieval only; no BM25/fusion or generation',
              'startup_seconds': startup, 'queries': records, 'median_seconds': statistics.median(x['seconds'] for x in records),
              'rss_bytes_final': psutil.Process().memory_info().rss, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
