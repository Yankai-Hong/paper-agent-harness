"""Resume/import real smoke vectors into Chroma and rerank those candidates."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
from colqwen_smoke import sha, rows
from paper_agent.resumable_colqwen import ResumableColQwenStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    smoke = Path(args.smoke)
    result = json.loads((smoke / 'result.json').read_text(encoding='utf-8'))
    if (result.get('status') != 'ok' or not result.get('maxsim_checked_against_processor')
            or sha(smoke / 'vectors.npz') != result.get('vectors_sha256')):
        raise ValueError('Real smoke result is missing, failed, or changed')
    prepared = Path('data/raw/irpapers-prepared-20261106')
    pages = {row['page_id']: row for row in rows(prepared / 'pages.jsonl')}
    contract = {'model_revision': result['model_revision'], 'base_revision': result['base_revision'],
                'source_hash': sha(prepared / 'pages.jsonl'), 'dimension': result['dimension'],
                'max_visual_tokens': result['max_visual_tokens'], 'weight_sha256': result['weight_sha256']}
    index = Path(args.index)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    store = ResumableColQwenStore(index, contract, create=not index.exists())
    arrays = np.load(smoke / 'vectors.npz', allow_pickle=False)
    started = time.perf_counter()
    if not store.final.exists():
        for i, record in enumerate(result['pages']):
            page = pages[record['page_id']]
            if page['image_sha256'] != record['image_sha256'] or sha(prepared / page['image_path']) != record['image_sha256']:
                raise ValueError('Page source image changed')
            store.add_page(record['page_id'], page['paper_id'], record['image_sha256'], arrays[f'page_{i}'])
        store.seal()
    import_seconds = time.perf_counter() - started
    rerank_start = time.perf_counter()
    ranked = store.rerank(arrays['query'], [r['page_id'] for r in result['pages']])
    rerank_seconds = time.perf_counter() - rerank_start
    original = {r['page_id']: r['score'] for r in result['pages']}
    if any(not np.isclose(score, original[page], atol=1e-3) for page, score in ranked):
        raise ValueError('Persisted MaxSim differs from the real model smoke')
    report = {'scope': 'real smoke candidates only, not an IRPAPERS benchmark', 'ranked_pages': ranked,
              'pages': len(ranked), 'tokens': store.collection.count(), 'elapsed_seconds': time.perf_counter() - started,
              'import_and_seal_seconds': import_seconds, 'rerank_seconds': rerank_seconds,
              'rss_before': rss_before, 'rss_after': process.memory_info().rss,
              'index_bytes': sum(p.stat().st_size for p in index.rglob('*') if p.is_file()),
              'smoke_vectors_sha256': result['vectors_sha256'], 'contract': contract}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
