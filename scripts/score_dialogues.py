"""Score actual predictions against separately reviewed labels; refuse draft labels."""
import argparse
import json
from pathlib import Path

from paper_agent.storage import read_jsonl, write_json, digest
from paper_agent.task_metrics import classification, boundary, pipeline_success, rate


def score(run_dir, judgments_path, view=None):
    root = Path(run_dir)
    rows = read_jsonl(judgments_path)
    if not rows or len({r['turn_id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique reviewer judgments')
    fields = ['rewrite_correct', 'behavior_correct', 'answer_correct', 'citations_correct', 'no_unrecovered_error']
    for row in rows:
        if row.get('review_status') != 'reviewed' or not row.get('reviewer'):
            raise ValueError('Draft or unidentified review cannot produce effectiveness scores')
        if any(not isinstance(row.get(key), bool) for key in fields):
            raise ValueError('Independent boolean judgments required')
        if row.get('intent') not in {'search', 'fact', 'explain', 'compare', 'out_of_scope'}:
            raise ValueError('Invalid intent gold')
        if row.get('decision') not in {'answer', 'refuse', 'clarify'}:
            raise ValueError('Invalid boundary gold')
    judgments = {r['turn_id']: r for r in rows}
    ids = [r['turn_id'] for r in read_jsonl(view)] if view else list(judgments)
    if len(ids) != len(set(ids)) or not ids or set(ids)-set(judgments):
        raise ValueError('View must reference unique reviewed IDs')
    predictions = read_jsonl(root/'predictions.jsonl')
    if len({r['turn_id'] for r in predictions}) != len(predictions):
        raise ValueError('Duplicate predictions')
    predicted_intents, predicted_decisions, success_judgments = {}, {}, {}
    completed = set()
    for row in predictions:
        identifier = row['turn_id']
        if identifier not in ids or row.get('error'):
            continue
        from paper_agent.storage import safe_id
        trace_path = root/'traces'/safe_id(row['session_id'])/f"{safe_id(row['trace_id'])}.json"
        trace = json.loads(trace_path.read_text(encoding='utf-8'))
        if trace['status'] != 'completed':
            continue
        completed.add(identifier)
        predicted_decisions[identifier] = row['answer']['decision']
        nlu = next((s for s in trace['stages'] if s['stage'] == 'NLU' and s['status'] == 'ok'), None)
        if nlu:
            predicted_intents[identifier] = nlu['output']['intent']
        success_judgments[identifier] = judgments[identifier]
    metrics = {'intent_accuracy': classification({i: judgments[i]['intent'] for i in ids}, predicted_intents),
        'boundary': boundary({i: judgments[i]['decision'] for i in ids}, predicted_decisions),
        'pipeline_success': pipeline_success(ids, success_judgments),
        'rewrite_accuracy': rate(sum(i in completed and judgments[i]['rewrite_correct'] for i in ids), len(ids)),
        'reviewed_citation_turn_accuracy': rate(sum(i in completed and judgments[i]['citations_correct'] for i in ids), len(ids)),
        'judgments_hash': digest(rows), 'predictions_hash': digest(predictions), 'view_ids': ids,
        'note': 'Citation score is turn-level reviewer judgment, not per-citation entailment precision.'}
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('judgments')
    parser.add_argument('--view')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_json(args.output, score(args.run_dir, args.judgments, args.view))
    print('Saved scores from independently reviewed judgments.')
