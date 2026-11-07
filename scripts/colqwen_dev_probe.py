"""Measure real candidate-only MaxSim on one already committed dev case."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
from colqwen_candidate_eval import read_vector
from colqwen_smoke import rows, sha
from paper_agent.colqwen_store import maxsim
from paper_agent.maxsim_batch import score_maxsim_batch
from paper_agent.resumable_colqwen import ResumableColQwenStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--case', default='standard-006')
    p.add_argument('--score-device', default='cuda:0')
    p.add_argument('--verify-all', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source, index = Path(args.encoded), Path(args.index)
    config = json.loads((source / 'contract.json').read_text(encoding='utf-8'))
    if config['partition'] != 'dev':
        raise ValueError('Probe is restricted to development cases')
    matches = [r for r in rows('runs/irpapers-colqwen-candidates-20261106/results.jsonl') if r['case_id'] == args.case and r['partition'] == 'dev']
    baseline = [r for r in rows('runs/irpapers-chroma-baseline-v2-20261106/text-results.jsonl') if r['case_id'] == args.case and r['partition'] == 'dev']
    if len(matches) != 1 or len(baseline) != 1:
        raise ValueError('Development case not unique')
    case, original = matches[0], baseline[0]
    import hashlib
    query = read_vector(source / 'queries', args.case,
                        {'case_id': args.case, 'question_sha256': hashlib.sha256(original['question'].encode()).hexdigest()})
    if query is None:
        raise ValueError('Query has not been encoded')
    contract = json.loads((index / 'contract.json').read_text(encoding='utf-8'))['contract']
    if (contract['candidate_plan_sha256'] != config['candidate_plan_sha256']
            or contract['model_revision'] != config['model_revision'] or contract['source_hash'] != config['source_pages_sha256']):
        raise ValueError('Index and real encoding contracts differ')
    process = psutil.Process()
    store = ResumableColQwenStore(index, contract)
    missing = [page for page in case['pages'] if not (store.committed / (page + '.json')).is_file()]
    if missing:
        raise ValueError(f'Candidate pages not yet committed: {len(missing)}')
    import torch
    if args.score_device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    documents = [store.get_page(page) for page in case['pages']]
    read_seconds = time.perf_counter() - start
    scoring = time.perf_counter()
    values = score_maxsim_batch(query, documents, device=args.score_device)
    scoring_seconds = time.perf_counter() - scoring
    verify_count = len(documents) if args.verify_all else min(5, len(documents))
    max_difference = max(abs(values[i] - maxsim(query, documents[i])) for i in range(verify_count))
    if max_difference > 1e-3:
        raise ValueError('Batched MaxSim differs from per-page reference')
    scores = list(zip(case['pages'], values))
    scores.sort(key=lambda x: (-x[1], x[0]))
    ranked = [page for page, score in scores]
    gold_rank = ranked.index(original['gold_page']) + 1 if original['gold_status'] == 'consistent' and original['gold_page'] in ranked else None
    result = {'scope': 'one development question, committed candidates only; no sealed full evaluation',
              'case_id': args.case, 'candidate_count': len(ranked), 'candidate_gold_rank': gold_rank,
              'text_gold_rank': original['hybrid']['gold_rank'], 'top20': ranked[:20],
              'rerank_seconds': time.perf_counter() - start, 'read_seconds': read_seconds,
              'batch_scoring_seconds': scoring_seconds, 'score_device': args.score_device,
              'reference_pages_checked': verify_count, 'reference_max_abs_difference': max_difference,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if args.score_device.startswith('cuda') else None,
              'rss_bytes_final': process.memory_info().rss,
              'index_bytes': sum(p.stat().st_size for p in index.rglob('*') if p.is_file())}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
