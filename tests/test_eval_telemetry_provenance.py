import json
from pathlib import Path
from types import SimpleNamespace

from paper_agent.eval_tasks import run_e2e_session
from paper_agent.observability import Exporter
from paper_agent.storage import digest
from test_reliability import pipeline


def test_evaluation_rule_projection_provenance_and_isolation(tmp_path):
    rows = [{'session_id':'s', 'turn_id':'t', 'turn':1, 'question':'alpha',
             'success_conditions':{'decision':'answer'}}]
    sent = []
    def make_factory(run):
        def factory(session_id):
            return pipeline(tmp_path / run, mode='bm25',
                            exporter=Exporter(True, sent.append, flush_seconds=.2))
        return factory
    config = {'mode':'bm25', 'reflect':True}
    provenance = {'dataset_hash':digest(rows), 'evaluation_config_hash':digest(config),
                  'evaluation_schema':'paper_agent_e2e_rule_v1', 'config':config}
    first = run_e2e_session(make_factory('a'), 's', rows, tmp_path/'run-a', provenance)[0]
    second = run_e2e_session(make_factory('b'), 's', rows, tmp_path/'run-b', provenance)[0]
    traces = [json.loads(Path(p['trace_path']).read_text(encoding='utf-8')) for p in (first,second)]
    assert traces[0]['observability_mapping']['langfuse_trace_id'] != traces[1]['observability_mapping']['langfuse_trace_id']
    assert first['assessment']['task_success'] is None
    projections = [p for p in sent if p['scores_only']]
    assert len(projections) == 2
    for prediction, trace, payload in zip((first, second), traces, projections):
        assert prediction['evaluation_provenance'] == provenance
        assert payload['trace_id'] == trace['observability_mapping']['langfuse_trace_id']
        execution = next(s for s in payload['scores'] if s['name']=='execution_success')
        assert execution['value'] == 1 and execution['metadata']['source'] == 'rule'
        assert execution['metadata']['dataset_hash'] == digest(rows)
        assert execution['metadata']['evaluation_config_hash'] == digest(config)
        assert execution['metadata']['evaluation_schema'] == 'paper_agent_e2e_rule_v1'
        assert 'case_id' in execution['metadata'] and 'run_id' in execution['metadata']
        assert not any(s['name']=='task_success' for s in payload['scores'])
        assert next(s for s in trace['scores'] if s['name']=='task_success')['value'] is None
        assert 'alpha evidence' not in json.dumps(payload)


def test_cli_e2e_saves_config_provenance(monkeypatch, tmp_path):
    from paper_agent import eval_cli
    from paper_agent.providers import APIConfig
    from paper_agent.storage import write_jsonl
    rows = [{'session_id':'s','turn_id':'t','turn':1,'question':'alpha',
             'success_conditions':{'decision':'answer'}}]
    data = tmp_path/'data.jsonl'
    write_jsonl(data, rows)
    probe = pipeline(tmp_path/'probe', mode='bm25')
    probe.retriever.chunks = list(probe.retriever.by_id.values())
    monkeypatch.setattr(eval_cli, '_config', lambda args: APIConfig('secret', 'test'))
    monkeypatch.setattr(eval_cli, '_build_retriever', lambda args: probe.retriever)
    monkeypatch.setattr(eval_cli, '_pipeline_factory', lambda args, retriever, config, output:
                        lambda session_id: pipeline(tmp_path/'pipeline', mode='bm25'))
    args = SimpleNamespace(data=str(data),output=str(tmp_path/'output'),mode='bm25',workers=1,judge=False)
    metrics, predictions = eval_cli.cmd_e2e(args)
    provenance = predictions[0]['evaluation_provenance']
    assert provenance['dataset_hash'] == metrics['dataset_hash'] == digest(rows)
    assert metrics['evaluation_config_hash'] == digest(provenance['config'])
    assert 'secret' not in json.dumps(provenance)
    assert json.loads((tmp_path/'output'/'predictions.jsonl').read_text(encoding='utf-8'))['evaluation_provenance'] == provenance


def test_synthetic_fixture_is_offline_and_refuses_existing_output(tmp_path):
    import importlib.util
    import pytest
    spec = importlib.util.spec_from_file_location('prepare_langfuse_fixture', Path('scripts/prepare_langfuse_fixture.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    out = tmp_path / 'fixture'
    info = fixture.prepare(out)
    assert info['cases'] == 1 and info['model_calls'] == info['external_uploads'] == 0
    assert info['dataset_hash'] == digest(fixture.ROWS)
    assert (out/'index'/'metadata.json').is_file()
    with pytest.raises(FileExistsError):
        fixture.prepare(out)
    assert json.loads((out/'manifest.json').read_text(encoding='utf-8')) == info
