"""Select a page-level RRF weight on completed development scores, then evaluate."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ranking(text, visual, weight, k=60):
    scores = {}
    for lane, w in ((text, 1), (visual, weight)):
        if not w:
            continue
        seen = set()
        for page in lane:
            if page in seen:
                continue
            seen.add(page)
            scores[page] = scores.get(page, 0) + w / (k + len(seen))
    return sorted(scores, key=lambda page: (-scores[page], page))


def evaluate(base, scored, weight):
    lookup = {x['case_id']: x for x in base}
    records = []
    for result in scored:
        original = lookup[result['case_id']]
        if result['partition'] != original['partition'] or result['gold_status'] != original['gold_status']:
            raise ValueError('Scored and baseline identities disagree')
        pages = ranking(original['hybrid']['pages'], result['top20'], weight)
        gold = original['gold_page']
        rank = pages.index(gold) + 1 if original['gold_status'] == 'consistent' and gold in pages else None
        records.append({'case_id': result['case_id'], 'group': result['group'], 'rank': rank,
                        'text_rank': result['text_rank'], 'candidate_rank': result['candidate_rank']})
    metrics = {}
    for group in ('standard', 'visual'):
        group_rows = [r for r in records if r['group'] == group]
        n = len(group_rows)
        if not n:
            raise ValueError('Both question groups are required')
        ranks = [r['rank'] for r in group_rows]
        metrics[group] = {'n': n, **{f'hit{k}': sum(r is not None and r <= k for r in ranks) / n for k in (1, 5, 20)},
                          'mrr': sum(1 / r if r else 0 for r in ranks) / n}
    return records, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['choose', 'evaluate'])
    p.add_argument('--baseline', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--plan')
    p.add_argument('--selection')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    summary = json.loads(Path(args.results + '.summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or summary.get('results_sha256') != sha(args.results):
        raise ValueError('A complete, hash-checked scoring result is required')
    scored = rows(args.results)
    base = rows(args.baseline)
    ids = {r['case_id'] for r in scored}
    expected = {r['case_id'] for r in base if r['partition'] == summary['partition']}
    if len(scored) != len(ids) or ids != expected:
        raise ValueError('Case coverage mismatch')
    if args.mode == 'choose':
        if summary['partition'] != 'dev':
            raise ValueError('Selection only uses development cases')
        options = []
        for weight in (0, .05, .1, .25, .5, 1):
            _, metrics = evaluate(base, scored, weight)
            options.append({'weight': weight, 'metrics': metrics})
        baseline = options[0]['metrics']
        eligible = [x for x in options if all(x['metrics'][g]['hit5'] >= baseline[g]['hit5'] for g in ('standard', 'visual'))]
        choice = max(eligible, key=lambda x: (sum(x['metrics'][g]['mrr'] for g in ('standard', 'visual')), -x['weight']))
        out = {'status': 'frozen_before_holdout_evaluation', 'weight': choice['weight'], 'k': 60,
               'visual_depth': 20, 'options': options, 'development_results_sha256': sha(args.results),
               'development_summary_sha256': sha(args.results + '.summary.json'),
               'baseline_sha256': sha(args.baseline), 'script_sha256': sha(__file__), 'strict_blind_test': False}
    else:
        if not args.plan:
            raise ValueError('Frozen development plan required')
        plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
        if (plan.get('status') != 'frozen_before_holdout_evaluation' or plan['baseline_sha256'] != sha(args.baseline)
                or plan['script_sha256'] != sha(__file__) or plan['k'] != 60 or plan['visual_depth'] != 20):
            raise ValueError('Fusion plan mismatch')
        if summary['partition'] == 'dev' and plan['development_results_sha256'] != sha(args.results):
            raise ValueError('Development results changed')
        if summary['partition'] == 'test':
            if not args.selection or summary.get('selection_sha256') != sha(args.selection):
                raise ValueError('Test scoring selection mismatch')
            selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
            if selection.get('development_summary_sha256') != plan['development_summary_sha256']:
                raise ValueError('Fusion and test scoring use different development results')
        records, metrics = evaluate(base, scored, plan['weight'])
        out = {'partition': summary['partition'], 'weight': plan['weight'], 'metrics': metrics, 'records': records,
               'plan_sha256': sha(args.plan), 'results_sha256': sha(args.results), 'strict_blind_test': False,
               'scope': 'RRF of full text ranking and top 20 candidate-only ColQwen pages'}
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': args.output, 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
