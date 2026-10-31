import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_shared_budget_is_single_use_and_persists_failed_attempt(tmp_path):
    runner = module("run_multimodal_acceptance")
    path = tmp_path / "task-budget.json"
    budget = runner.BudgetLedger(path, limit=1)
    record = {"request_id": "one", "stage": "reflect", "image_count": 1}
    budget.reserve("a01", record)
    reserved = json.loads(path.read_text())
    assert reserved["records"][0]["status"] == "reserved_unknown"
    budget.update({**record, "status": "error", "transport_error": "TimeoutError"})
    saved = json.loads(path.read_text())
    assert saved["records"][0]["transport_error"] == "TimeoutError"
    assert saved["records"][0]["status"] == "error"
    assert "usage_raw" not in saved["records"][0]
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.reserve("a02", {**record, "request_id": "two"})
    with pytest.raises(FileExistsError):
        runner.BudgetLedger(path, limit=60)
    assert json.loads(path.read_text()) == saved


def test_visual_matching_never_infers_unknown_number_or_text_from_page():
    scorer = module("multimodal_eval_record")
    target = {"paper_id": "p", "page": 3, "locator": "Figure 1"}
    assert scorer.score_visual(target, [{"paper_id": "p", "page": 3}], True) is None
    assert scorer.score_visual(target, [{"paper_id": "p", "page": 3, "figure_number": "Fig. 1"}], True) is True
    assert scorer.score_visual(target, [{"paper_id": "other", "page": 3, "figure_number": "Figure 1"}], True) is False
    assert scorer.score_visual(target, [], False) is None
    assert scorer.score_visual(target, None, False) is None
    assert scorer.score_visual({**target, "locator": "Abstract"}, [{"paper_id": "p", "page": 3}], True) is None


def test_failed_generation_still_preserves_completed_ranking(tmp_path):
    runner = module("run_multimodal_acceptance")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"status": "error", "stages": [
        {"stage": "Rerank", "status": "ok", "output": {"ranked_refs": [{"paper_id": "p", "chunk_id": "c"}]}},
        {"stage": "Generate", "status": "error"}]}))
    ok, hits = runner.retrieval_from_trace(trace, {("p", "c"): {"paper_id": "p", "chunk_id": "c", "page": 2}})
    assert ok and hits == [{"paper_id": "p", "chunk_id": "c", "page": 2}]


def test_followup_uses_existing_budget_and_cannot_run_twice(tmp_path):
    runner = module("run_multimodal_acceptance")
    path = tmp_path / "task-budget.json"
    first = runner.BudgetLedger(path, limit=2)
    first.reserve("c02", {"request_id":"one", "stage":"generate", "image_count":1})
    with pytest.raises(ValueError, match="unknown completion"):
        runner.BudgetLedger(path, limit=2, followup=True)
    first.update({"request_id":"one", "status":"error"})
    second = runner.BudgetLedger(path, limit=2, followup=True)
    second.reserve("c02", {"request_id":"two", "stage":"reflect", "image_count":1})
    assert len(json.loads(path.read_text())["records"]) == 2
    with pytest.raises(RuntimeError, match="budget exhausted"):
        second.reserve("m01", {"request_id":"three", "stage":"reflect", "image_count":1})
    second.update({"request_id":"two", "status":"ok"})
    with pytest.raises(FileExistsError):
        runner.BudgetLedger(path, limit=2, followup=True)


def test_review_followup_is_single_use_and_rejects_bad_ledger(tmp_path):
    runner = module('run_multimodal_acceptance')
    path = tmp_path / 'task.json'
    budget = runner.BudgetLedger(path, limit=2)
    budget.reserve('t02', {'request_id':'first', 'stage':'nlu', 'image_count':0})
    budget.update({'request_id':'first', 'status':'ok'})
    review = runner.BudgetLedger(path, limit=2, review_followup=True)
    review.reserve('u01', {'request_id':'second', 'stage':'reflect', 'image_count':0})
    review.update({'request_id':'second', 'status':'error'})
    with pytest.raises(FileExistsError):
        runner.BudgetLedger(path, limit=2, review_followup=True)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        review.reserve('m01', {'request_id':'third', 'stage':'nlu', 'image_count':0})


def test_linked_followup_preserves_cumulative_budget(tmp_path):
    runner = module('run_multimodal_acceptance')
    path = tmp_path / 'budget.json'
    first = runner.BudgetLedger(path, limit=2)
    first.reserve('t02', {'request_id':'one', 'stage':'nlu', 'image_count':0})
    first.update({'request_id':'one', 'status':'ok'})
    linked = runner.BudgetLedger(path, limit=2, linked_followup=True)
    linked.reserve('t02', {'request_id':'two', 'stage':'reflect', 'image_count':1})
    linked.update({'request_id':'two', 'status':'ok'})
    assert len(json.loads(path.read_text())['records']) == 2
    with pytest.raises(FileExistsError):
        runner.BudgetLedger(path, limit=2, linked_followup=True)
