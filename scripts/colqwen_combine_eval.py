"""Combine completed, hash-checked dev and test ColQwen candidate results."""
import argparse
import json
import statistics
from pathlib import Path

from colqwen_smoke import rows, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    dev_summary = json.loads(Path(args.dev + '.summary.json').read_text(encoding='utf-8'))
    test_summary = json.loads(Path(args.test + '.summary.json').read_text(encoding='utf-8'))
    if (selection.get('status') != 'frozen_before_test_evaluation'
            or selection['development_summary_sha256'] != sha(args.dev + '.summary.json')
            or dev_summary['results_sha256'] != sha(args.dev) or test_summary['results_sha256'] != sha(args.test)
            or test_summary.get('selection_sha256') != sha(args.selection)
            or dev_summary['candidate_plan_sha256'] != selection['candidate_plan_sha256']
            or test_summary['candidate_plan_sha256'] != selection['candidate_plan_sha256']):
        raise ValueError('Development, test or frozen selection mismatch')
    records = rows(args.dev) + rows(args.test)
    if len(records) != 210 or len({r['case_id'] for r in records}) != 210:
        raise ValueError('Expected the 210 distinct frozen cases')
    data = {}
    for group, expected in (('standard', 180), ('visual', 30)):
        subset = [r for r in records if r['group'] == group]
        if len(subset) != expected:
            raise ValueError('Case group count differs from frozen benchmark')
        n = len(subset)
        metrics = {'n': n, 'candidate_coverage': sum(r['candidate_hit'] for r in subset) / n,
                   'median_candidate_count': statistics.median(r['candidate_count'] for r in subset),
                   'rerank_median_seconds': statistics.median(r['rerank_seconds'] for r in subset)}
        for key in ('bm25_rank', 'dense_rank', 'text_rank', 'clip_rank', 'equal_fusion_rank', 'reranked_with_text_tail_rank'):
            ranks = [r[key] for r in subset]
            metrics[key] = {'hit1': sum(x is not None and x <= 1 for x in ranks) / n,
                            'hit5': sum(x is not None and x <= 5 for x in ranks) / n,
                            'hit20': sum(x is not None and x <= 20 for x in ranks) / n,
                            'mrr': sum(1 / x if x else 0 for x in ranks) / n}
        for k in (1, 5, 20):
            def hit(row, field): return row[field] is not None and row[field] <= k
            metrics[f'paired_hit{k}'] = {'improved': [r['case_id'] for r in subset if hit(r, 'reranked_with_text_tail_rank') and not hit(r, 'text_rank')],
                                        'regressed': [r['case_id'] for r in subset if hit(r, 'text_rank') and not hit(r, 'reranked_with_text_tail_rank')]}
        data[group] = metrics
    report = {'status': 'complete', 'selection_sha256': sha(args.selection), 'development_results_sha256': sha(args.dev),
              'test_results_sha256': sha(args.test), 'strict_blind_test': False,
              'scope': 'candidate-only MaxSim with remaining pages in text rank order', 'metrics': data}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
