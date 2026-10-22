import pytest
from paper_agent.task_metrics import boundary, classification, pipeline_success, memory_consistency, token_reduction


def test_all_refusal_cannot_be_full_accuracy():
    gold = {"a": "answer", "b": "refuse", "c": "clarify"}
    out = boundary(gold, dict.fromkeys(gold, "refuse"))
    assert out["decision_accuracy"]["rate"] == pytest.approx(1/3)
    assert out["refusal_recall"]["rate"] == 1
    assert out["refusal_precision"]["rate"] == pytest.approx(1/3)
    assert out["false_refusal_rate"]["rate"] == 1


def test_missing_judgments_and_wrong_memories_are_failures():
    assert classification({"a": "x", "b": "x"}, {"a": "x"})["rate"] == .5
    assert pipeline_success(["a"], {})["rate"] == 0
    with pytest.raises(ValueError):
        pipeline_success(["a"], {"a": {"answer_correct": True}})
    assert memory_consistency({"it": "correct"}, {"it": "same_wrong_answer"})["rate"] == 0


def test_token_comparison_accounts_for_cache_and_extra_work():
    out = token_reduction({"s": {"input_tokens": 100, "cache_read_input_tokens": 100}},
                          {"s": {"input_tokens": 250}})
    assert out["reduction"] == -.25
    with pytest.raises(ValueError):
        token_reduction({"a": {}}, {"b": {}})
