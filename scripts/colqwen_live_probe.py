"""Probe an actual query against the first two frozen candidate pages."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from paper_agent.colqwen_page_retrieval import ColQwenCandidateRetriever
from paper_agent.colqwen_subprocess import SubprocessColQwenQueryEncoder
from paper_agent.resumable_colqwen import ResumableColQwenStore


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--encoded', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    index = Path(args.index)
    contract = json.loads((index / 'contract.json').read_text(encoding='utf-8'))['contract']
    store = ResumableColQwenStore(index, contract)
    case = next(x for x in rows(Path(args.baseline) / 'text-results.jsonl') if x['case_id'] == args.case)
    candidate = next(x for x in rows(Path(args.candidates) / 'results.jsonl') if x['case_id'] == args.case)
    encoder = SubprocessColQwenQueryEncoder(args.python, 'scripts/colqwen_query_worker.py', args.base, args.adapter, contract)
    started = time.perf_counter()
    vectors = np.asarray(encoder.encode_query(case['question']), dtype=np.float32)
    elapsed = time.perf_counter() - started
    marker = json.loads((Path(args.encoded) / 'queries' / (args.case + '.json')).read_text(encoding='utf-8'))
    with np.load(Path(args.encoded) / 'queries' / marker['vector_file'], allow_pickle=False) as archive:
        reference = np.asarray(archive['vectors'], dtype=np.float32)
    if vectors.shape != reference.shape:
        raise ValueError('Query shape differs from saved checkpoint')
    difference = float(np.max(np.abs(vectors - reference)))
    # Reuse the actual query output to isolate page search time from model loading.
    encoder.encode_query = lambda query: vectors
    search = ColQwenCandidateRetriever(store, encoder, Path(args.prepared) / 'pages.jsonl', args.prepared, device='cuda:0')
    started = time.perf_counter()
    result = search.search_text(case['question'], candidate['pages'][:2])
    report = {'case_id': args.case, 'candidate_selection': 'first two frozen candidate pages, independent of gold',
              'query_shape': list(vectors.shape), 'query_checkpoint_max_abs_difference': difference,
              'query_seconds_including_subprocess_and_model_load': elapsed,
              'page_search_seconds': time.perf_counter() - started, 'result': result}
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
