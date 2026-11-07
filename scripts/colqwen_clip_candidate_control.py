"""Compare existing CLIP and ColQwen scores on the same frozen page candidates."""
import argparse
import json
from pathlib import Path

from colqwen_smoke import rows, sha
from irpapers_optimize import read_frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('baseline', 'candidates', 'dev', 'test', 'selection', 'output'):
        p.add_argument('--' + option, required=True)
    a = p.parse_args()
    root, candidate_dir = Path(a.baseline), Path(a.candidates)
    freeze = read_frozen(root)
    selection = json.loads(Path(a.selection).read_text(encoding='utf-8'))
    candidate_path = candidate_dir / 'results.jsonl'
    if (selection.get('candidate_results_sha256') != sha(candidate_path)
            or selection.get('candidate_plan_sha256') != sha(candidate_dir / 'plan.json')
            or selection.get('development_summary_sha256') != sha(a.dev + '.summary.json')):
        raise ValueError('Frozen selection or candidates differ')
    scored = []
    for path, partition in ((a.dev, 'dev'), (a.test, 'test')):
        summary = json.loads(Path(path + '.summary.json').read_text(encoding='utf-8'))
        if (summary.get('status') != 'complete' or summary.get('partition') != partition
                or summary.get('results_sha256') != sha(path)
                or (partition == 'test' and summary.get('selection_sha256') != sha(a.selection))):
            raise ValueError('Incomplete or mismatched evaluation')
        scored.extend(rows(path))
    text = {r['case_id']: r for r in rows(root / 'text-results.jsonl')}
    visual = {r['case_id']: r for r in rows(root / 'visual-results.jsonl')}
    candidates = rows(candidate_path)
    score = {r['case_id']: r for r in scored}
    if (len(text) != 210 or len(visual) != 210 or len(candidates) != 210 or len(score) != 210
            or {r['case_id'] for r in candidates} != set(text) or set(visual) != set(text) or set(score) != set(text)):
        raise ValueError('Case identities do not match')
    records = []
    for candidate in candidates:
        case = candidate['case_id']
        baseline, clip, result = text[case], visual[case]['clip'], score[case]
        pages = candidate['pages']
        if len(pages) != len(set(pages)) or len(pages) != result['candidate_count'] or clip['status'] != 'ok':
            raise ValueError('Invalid candidate set or CLIP ranking')
        allowed = set(pages)
        ranked = [page for page in clip['pages'] if page in allowed]
        if len(ranked) != len(pages):
            raise ValueError('CLIP full ranking omits a candidate')
        gold = baseline['gold_page']
        clip_rank = ranked.index(gold) + 1 if baseline['gold_status'] == 'consistent' and gold in allowed else None
        records.append({'case_id': case, 'group': baseline['group'], 'partition': baseline['partition'],
                        'gold_status': baseline['gold_status'], 'candidate_count': len(pages),
                        'clip_candidate_rank': clip_rank, 'colqwen_candidate_rank': result['candidate_rank']})
    metrics = {}
    for group in ('standard', 'visual'):
        metrics[group] = {}
        for partition in ('all', 'dev', 'test'):
            subset = [r for r in records if r['group'] == group and (partition == 'all' or r['partition'] == partition)]
            n = len(subset)
            values = {}
            for key in ('clip_candidate_rank', 'colqwen_candidate_rank'):
                ranks = [r[key] for r in subset]
                values[key] = {**{f'hit{k}': sum(x is not None and x <= k for x in ranks) / n for k in (1, 5, 20)},
                               'mrr': sum(1 / x if x else 0 for x in ranks) / n}
            metrics[group][partition] = {'n': n, **values}
    output = {'scope': 'both models ranked within the same frozen text+CLIP page union',
              'strict_blind_test': False, 'candidate_results_sha256': sha(candidate_path),
              'baseline_freeze_files': freeze['files'], 'selection_sha256': sha(a.selection),
              'metrics': metrics, 'cases': records}
    with Path(a.output).open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': a.output, 'sha256': sha(a.output), 'metrics': metrics}))


if __name__ == '__main__':
    main()
