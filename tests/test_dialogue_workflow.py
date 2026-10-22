import importlib.util
from pathlib import Path

import pytest

from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.corpus import Chunker
from paper_agent.retrieval import build_index, Retriever
from paper_agent.storage import write_json, write_jsonl, read_jsonl


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_cap_is_shared_across_sessions_and_counts_failures(tmp_path, monkeypatch):
    import json
    module = script('run_dialogues')
    def transport(url, body, headers, timeout):
        if 'Resolve references' in body['messages'][0]['content']:
            result = {'intent': 'fact', 'standalone_query': 'retrieval', 'needs_clarification': False,
                      'paper_ids': None, 'resolved_entities': {}}
        else:
            result = {'decision': 'refuse', 'reason': 'test fixture'}
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 3}}
    monkeypatch.setattr(module, 'CompatibleReasoner', lambda config, sid: CompatibleReasoner(config, sid, transport))
    chunks = Chunker().split({'paper_id': 'p', 'title': 'fixture', 'paragraphs': [
        {'paragraph_id': 'a', 'section': 'Method', 'text': 'retrieval sparse algorithm'}]})
    write_jsonl(tmp_path/'chunks.jsonl', chunks)
    build_index(tmp_path/'chunks.jsonl', tmp_path/'index')
    rows = [{'turn_id': f'{i}-1', 'session_id': str(i), 'turn': 1, 'question': 'retrieval'} for i in range(3)]
    out = tmp_path/'run'
    module.run(rows, Retriever(tmp_path/'index'), APIConfig('test', 'fixture', max_calls=3), out, mode='bm25')
    predictions = read_jsonl(out/'predictions.jsonl')
    assert predictions[0]['answer']['decision'] == 'refuse'
    assert 'error' in predictions[1] and 'error' in predictions[2]
    assert json.loads((out/'usage.json').read_text())['calls'] == 3
    with pytest.raises(ValueError, match='new'):
        module.run(rows, Retriever(tmp_path/'index'), APIConfig('test', 'fixture'), out)


def test_scoring_rejects_drafts_and_counts_missing_predictions(tmp_path):
    module = script('score_dialogues')
    labels = [{'turn_id': str(i), 'review_status': 'reviewed', 'reviewer': 'fixture-reviewer',
        'intent': 'fact', 'decision': 'answer', 'rewrite_correct': True, 'behavior_correct': True,
        'answer_correct': True, 'citations_correct': True, 'no_unrecovered_error': True} for i in range(2)]
    write_jsonl(tmp_path/'labels.jsonl', labels)
    write_jsonl(tmp_path/'predictions.jsonl', [{'turn_id': '0', 'session_id': 's', 'trace_id': 't',
                                             'answer': {'decision': 'answer'}}])
    write_json(tmp_path/'traces/s/t.json', {'status': 'completed', 'stages': [
        {'stage': 'NLU', 'status': 'ok', 'output': {'intent': 'fact'}}]})
    metrics = module.score(tmp_path, tmp_path/'labels.jsonl')
    assert metrics['intent_accuracy']['rate'] == .5
    assert metrics['pipeline_success']['rate'] == .5
    labels[0]['review_status'] = 'draft_unreviewed'
    write_jsonl(tmp_path/'labels.jsonl', labels)
    with pytest.raises(ValueError, match='Draft'):
        module.score(tmp_path, tmp_path/'labels.jsonl')
