import numpy as np
import pytest
from paper_agent.eval_tasks import bootstrap_ci, run_boundary, run_rewrite
from paper_agent.eval_protocol import compare_runs, assess
from paper_agent.storage import write_jsonl
from scripts.prepare_eval import session_history


def test_history_is_sorted_past_same_session():
    rows = [{"session_id": s, "turn": t, "turn_id": f"{s}-{t}", "question": f"{s}-{t}"}
            for s,t in [("b",1),("a",3),("a",1),("a",2)]]
    assert [m["content"] for m in session_history(rows, rows[1])] == ["a-1", "a-2"]
    assert session_history(rows, rows[0]) == []


def test_cluster_bootstrap_preserves_multiplicity():
    scores, clusters = [0, 0, 1, 0.5], ["a", "a", "b", "c"]
    rng = np.random.default_rng(7)
    means = []
    for _ in range(99):
        drawn = rng.choice(["a", "b", "c"], 3, replace=True)
        values = [v for c in drawn for v,k in zip(scores,clusters) if c == k]
        means.append(np.mean(values))
    actual = bootstrap_ci(scores, clusters, seed=7, samples=99)
    assert actual["bootstrap_95_percentile_interval"] == pytest.approx(np.quantile(means, [.025,.975]))
    assert bootstrap_ci([1,1], ["s","s"])["bootstrap_95_percentile_interval"] == [1,1]
    with pytest.raises(ValueError): bootstrap_ci([], [])


def test_boundary_replays_only_user_turns():
    calls = []
    class Pipeline:
        def run(self, q, state):
            calls.append((q, list(state.messages)))
            state.messages.append({"role":"user", "content":q})
            return {"answer":{"decision":"refuse"}}
    rows = [{"id":"x", "question":"target", "history":[{"role":"user","content":"past"}, {"role":"assistant","content":"old"}]}]
    run_boundary(lambda sid: Pipeline(), rows)
    assert [q for q,_ in calls] == ["past", "target"]
    calls.clear()
    run_boundary(lambda sid: Pipeline(), rows, "context-only")
    assert len(calls) == 1 and len(calls[0][1]) == 2


def test_compare_pairs_missing_and_failure(tmp_path):
    write_jsonl(tmp_path/"a"/"per_case.jsonl", [{"id":"1", "score":1}, {"id":"2", "score":0}])
    write_jsonl(tmp_path/"b"/"per_case.jsonl", [{"id":"1", "error":"timeout"}, {"id":"3", "score":1}])
    result = compare_runs(tmp_path/"a", tmp_path/"b", "score")
    assert result["paired"]["mean"] == -1
    assert len(result["missing_in_b"]) == len(result["missing_in_a"]) == 1
    assert len(result["regressed"]) == 1


def test_execution_does_not_imply_task_success():
    result = assess({}, {"answer":{"decision":"answer", "claims":[]}}, {"status":"completed"})
    assert result["execution_success"] is True
    assert result["task_success"] is None
    assert result["pipeline_success"] is False


def test_rewrite_rejects_legacy_and_cross_session():
    with pytest.raises(ValueError): run_rewrite(None, [{"id":"a", "question":"q"}])
    row = {"id":"a-t2", "session_id":"a", "turn":2, "question":"q",
           "history":[{"role":"user", "content":"old", "turn_id":"b-t1"}]}
    with pytest.raises(ValueError): run_rewrite(None, [row])


def test_ablation_executes_config_matrix(tmp_path, monkeypatch):
    import argparse
    from paper_agent import eval_cli
    from paper_agent.storage import write_json
    config = tmp_path/"config.json"
    write_json(config, {"ablations":[{"id":"first", "args":["e2e","fixture.jsonl","--index","index","--no-reflect"]},
                                     {"id":"second", "args":["e2e","fixture.jsonl","--index","index"]}]})
    calls = []
    def dispatch(args):
        calls.append(args)
        return {"n":1}
    monkeypatch.setattr(eval_cli, "dispatch", dispatch)
    result = eval_cli.cmd_ablation(argparse.Namespace(config=str(config),output=str(tmp_path/"out")))
    assert len(calls) == 2 and calls[0].no_reflect and not calls[1].no_reflect
    assert calls[0].output != calls[1].output
    assert all(r["status"] == "completed" for r in result)


def test_entity_metric_accepts_title_alias_and_reports_denominator():
    from paper_agent.eval_tasks import rewrite_metrics
    rows = [{"id":"a", "required_entities":["123.456"], "required_entity_aliases":{"123.456":["123.456", "Example Paper"]}},
            {"id":"b", "required_entities":["123.456"], "entity_preservation_applicable":False}]
    result = rewrite_metrics(rows, {"a":{"standalone_query":"Example Paper methods"}})
    assert result["entity_retention_rate"] == 1 and result["entity_applicable_cases"] == 1 and result["n"] == 2


def test_compare_legacy_metadata_preserves_session_cluster(tmp_path):
    from paper_agent.storage import write_json
    for name, labels in [("a",["fact","search"]),("b",["fact","fact"])]:
        write_jsonl(tmp_path/name/"per_case.jsonl", [{"id":str(i),"intent":v} for i,v in enumerate(labels)])
        write_json(tmp_path/name/"case_metadata.json",{str(i):{"session_id":"s","expected_label":"fact"} for i in range(2)})
    result=compare_runs(tmp_path/"a",tmp_path/"b","correct")
    assert result["paired"]["clusters"] == 1 and result["paired"]["mean"] == .5
