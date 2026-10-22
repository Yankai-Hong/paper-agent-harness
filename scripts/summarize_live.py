"""Objective live-run diagnostics; keep draft-label agreement separate from gold accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path

from paper_agent.storage import read_jsonl, write_json, write_jsonl, digest
from paper_agent.task_metrics import classification, boundary, rate
from paper_agent.evaluation import retrieval_metrics


def collect(root):
    predictions = {}
    for path in Path(root).rglob('predictions.jsonl'):
        for row in read_jsonl(path):
            if row['turn_id'] in predictions:
                raise ValueError('Duplicate live prediction')
            traces = path.parent/'traces'/row['session_id']
            trace = None
            if row.get('trace_id'):
                trace = json.loads((traces/f"{row['trace_id']}.json").read_text(encoding='utf-8'))
            predictions[row['turn_id']] = {**row, 'trace': trace, 'run_directory': str(path.parent)}
    # Failed turns also retain useful successful component spans.
    questions = {r['turn_id']: r['question'] for r in read_jsonl('data/dialogues-draft/dialogues.jsonl')}
    for identifier, row in predictions.items():
        if row['trace'] is None:
            for path in (Path(row['run_directory'])/'traces'/row['session_id']).glob('*.json'):
                trace = json.loads(path.read_text(encoding='utf-8'))
                if trace['question'] == questions.get(identifier):
                    row['trace'] = trace
                    break
    return predictions


def summarize(root, output, expected_rows=None):
    root, output = Path(root), Path(output)
    rows = expected_rows or read_jsonl('data/dialogues-draft/dialogues.jsonl')
    predictions = collect(root)
    expected = {r['turn_id']: r for r in rows}
    if set(predictions)-set(expected):
        raise ValueError('Unexpected prediction IDs')
    chunks = {c['chunk_id']: c for c in read_jsonl('indexes/bge/chunks.jsonl')}
    official = {q['query_id']: q for q in read_jsonl('data/qasper/retrieval.jsonl')}
    details, calls, intents, decisions, retrieval_pred, retrieval_gold = [], [], {}, {}, [], []
    for identifier, row in expected.items():
        pred = predictions.get(identifier, {})
        trace = pred.get('trace') or {}
        stages = trace.get('stages', [])
        calls.extend(trace.get('api_calls', []))
        nlu = next((s.get('output') for s in stages if s['stage'] == 'NLU' and s['status'] == 'ok'), None)
        if nlu:
            intents[identifier] = nlu['intent']
        answer = pred.get('answer', {})
        if answer:
            decisions[identifier] = answer['decision']
        claims = answer.get('claims', [])
        citations = [c for claim in claims for c in claim['citations']]
        citation_valid = all(c['chunk_id'] in chunks and chunks[c['chunk_id']]['paper_id'] == c['paper_id'] for c in citations)
        detail = {'turn_id': identifier, 'status': trace.get('status', 'missing'),
            'decision': answer.get('decision'), 'intent': intents.get(identifier),
            'error': pred.get('error'), 'citation_count': len(citations), 'citation_ids_valid': citation_valid,
            'retrieval_attempts': sum(s['stage'] == 'Retrieve' and s['status'] == 'ok' for s in stages)}
        details.append(detail)
        if row['turn'] == 1:
            q = official[row['source_query_ids'][0]]
            retrieval_gold.append({**q, 'query_id': identifier})
            evidence = next((s['output']['evidence_ids'] for s in reversed(stages)
                             if s['stage'] == 'Rerank' and s['status'] == 'ok'), [])
            retrieval_pred.append({'query_id': identifier, 'hits': [chunks[i] for i in evidence]})
    completed = sum(d['status'] == 'completed' for d in details)
    original_ids = {r['turn_id'] for r in rows}
    intent_view = [r for r in read_jsonl('data/dialogues-draft/intent-draft.jsonl') if r['turn_id'] in original_ids]
    boundary_view = [r for r in read_jsonl('data/dialogues-draft/boundary-draft.jsonl') if r['turn_id'] in original_ids]
    e2e_ids = {r['turn_id'] for r in read_jsonl('data/dialogues-draft/e2e-view.jsonl')} & original_ids
    fields = ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
    metrics = {'expected_turns': len(rows), 'predictions': len(predictions), 'completion': rate(completed, len(rows)),
        'decisions': dict(Counter(d['decision'] or 'error_or_missing' for d in details)),
        'e2e_view_execution_completion': rate(sum(d['turn_id'] in e2e_ids and d['status'] == 'completed' for d in details),len(e2e_ids)),
        'draft_intent_agreement_not_gold_accuracy': classification({r['turn_id']:r['provisional_intent'] for r in intent_view},
            {i:v for i,v in intents.items() if i in {r['turn_id'] for r in intent_view}}),
        'constructed_boundary_diagnostics_not_human_gold': boundary({r['turn_id']:r['provisional_decision'] or 'answer' for r in boundary_view},
            {i:v for i,v in decisions.items() if i in {r['turn_id'] for r in boundary_view}}),
        'first_turn_official_evidence_coverage_up_to_10_chunks': retrieval_metrics(retrieval_gold, retrieval_pred, k=10),
        'citation_structure': {'citations': sum(d['citation_count'] for d in details),
            'answer_turns': sum(d['decision']=='answer' for d in details),
            'answer_turns_valid_ids': sum(d['decision']=='answer' and d['citation_ids_valid'] for d in details)},
        'api_calls': len(calls), 'all_usage_complete': all(c.get('usage_complete',False) for c in calls),
        'tokens': {f:sum(c.get('usage',{}).get(f,0) for c in calls) for f in fields},
        'call_errors': dict(Counter(c.get('finish_reason') or c.get('error_type','unknown') for c in calls if c['status']!='ok')),
        'dataset_hash': digest(rows), 'scope': 'Template-derived exploratory multi-turn run; not independently human-reviewed evaluation.'}
    write_json(output/'metrics.json', metrics)
    write_jsonl(output/'per_turn.jsonl', details)
    write_json(output/'api_call_records.json', calls)
    return metrics


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sessions', type=int)
    args=parser.parse_args()
    rows=read_jsonl('data/dialogues-draft/dialogues.jsonl')
    if args.sessions:
        ids=list(dict.fromkeys(r['session_id'] for r in rows))[:args.sessions]
        rows=[r for r in rows if r['session_id'] in ids]
    m=summarize(args.root,args.output,rows)
    print({k:m[k] for k in ['expected_turns','completion','decisions','api_calls','tokens']})
