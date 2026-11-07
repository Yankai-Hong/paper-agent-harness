"""Export compact per-case ranks from completed, frozen evaluation files."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('baseline', 'visual', 'dev', 'test', 'fusion-dev', 'fusion-test', 'plan', 'selection', 'output'):
        p.add_argument('--' + option, required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
    selection = json.loads(Path(a.selection).read_text(encoding='utf-8'))
    scored, fused = [], []
    for path, fusion_path, partition in ((a.dev, a.fusion_dev, 'dev'), (a.test, a.fusion_test, 'test')):
        summary = json.loads(Path(path + '.summary.json').read_text(encoding='utf-8'))
        fusion = json.loads(Path(fusion_path).read_text(encoding='utf-8'))
        if (summary.get('status') != 'complete' or summary.get('partition') != partition
                or summary.get('results_sha256') != sha(path) or fusion.get('partition') != partition
                or fusion.get('results_sha256') != sha(path) or fusion.get('plan_sha256') != sha(a.plan)):
            raise ValueError('Evaluation outputs differ from their frozen summaries')
        if partition == 'test' and summary.get('selection_sha256') != sha(a.selection):
            raise ValueError('Holdout does not use the frozen selection')
        scored.extend(rows(path))
        fused.extend(fusion['records'])
    if (plan.get('development_results_sha256') != sha(a.dev) or plan.get('baseline_sha256') != sha(a.baseline)
            or selection.get('development_summary_sha256') != sha(a.dev + '.summary.json')):
        raise ValueError('Development plan or baseline differs')
    baseline, visual = rows(a.baseline), rows(a.visual)
    visual_by_id = {x['case_id']: x for x in visual}
    score_by_id = {x['case_id']: x for x in scored}
    fusion_by_id = {x['case_id']: x for x in fused}
    ids = {x['case_id'] for x in baseline}
    if (len(baseline) != 210 or len(ids) != 210 or len(visual) != 210 or len(scored) != 210 or len(fused) != 210
            or set(visual_by_id) != ids or set(score_by_id) != ids or set(fusion_by_id) != ids):
        raise ValueError('Expected 210 distinct matched questions')
    records = []
    for base in baseline:
        case = base['case_id']
        score, visual_case, fusion = score_by_id[case], visual_by_id[case], fusion_by_id[case]
        if (base['partition'] != score['partition'] or base['gold_status'] != score['gold_status']
                or base['group'] != score['group'] or fusion['text_rank'] != score['text_rank']):
            raise ValueError('Case metadata differs')
        records.append({'case_id': case, 'group': base['group'], 'partition': base['partition'],
                        'gold_status': base['gold_status'], 'text_target_in_index': base['hybrid']['gold_in_index'],
                        'candidate_count': score['candidate_count'], 'candidate_covered': score['candidate_hit'],
                        'bm25_rank': score['bm25_rank'], 'bge_rank': score['dense_rank'], 'text_rank': score['text_rank'],
                        'clip_rank': score['clip_rank'], 'text_clip_rank': score['equal_fusion_rank'],
                        'colqwen_candidate_rank': score['candidate_rank'], 'text_colqwen_rank': fusion['rank'],
                        'clip_status': visual_case['clip']['status'], 'text_status': base['hybrid']['status']})
    output = {'scope': 'page ranks; ColQwen limited to frozen candidate union', 'strict_blind_test': False,
              'baseline_sha256': sha(a.baseline), 'visual_sha256': sha(a.visual), 'dev_sha256': sha(a.dev),
              'test_sha256': sha(a.test), 'fusion_plan_sha256': sha(a.plan), 'selection_sha256': sha(a.selection),
              'cases': records}
    with Path(a.output).open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': a.output, 'cases': len(records), 'sha256': sha(a.output)}))


if __name__ == '__main__':
    main()
