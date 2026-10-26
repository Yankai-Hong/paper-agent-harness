from paper_agent.claim_eval import evaluate_claim, summarize_claims


class Judge:
    calls = []
    def __init__(self, verdict='supported', fail=False):
        self.verdict, self.fail = verdict, fail
    def structured(self, stage, payload):
        if self.fail:
            raise TimeoutError()
        return {'verdict':self.verdict, 'reason':'fixture', 'flags':[], 'citations':[
            {'index':e['index'], 'verdict':self.verdict, 'reason':'fixture'} for e in payload['evidence']]}


EVIDENCE = [{'paper_id':'p', 'chunk_id':'c', 'text':'Accuracy is 70%.'}]


def test_missing_wrong_paper_and_nonprovided_citation():
    for refs in [[], [{'paper_id':'other', 'chunk_id':'c'}], [{'paper_id':'p', 'chunk_id':'absent'}]]:
        result = evaluate_claim({'text':'Accuracy is 70%.', 'citations':refs}, EVIDENCE, Judge(fail=True))
        assert result['verdict'] == 'insufficient'
        assert result['calls'] == []


def test_duplicate_citations_count_once_and_unknown_not_excluded():
    claim = {'text':'Accuracy is 70%.', 'citations':[{'paper_id':'p', 'chunk_id':'c'}]*2}
    good = evaluate_claim(claim, EVIDENCE, Judge())
    failed = evaluate_claim(claim, EVIDENCE, Judge(fail=True))
    summary = summarize_claims([good, failed])
    assert summary['citation_validity']['denominator'] == 2
    assert summary['claim_support']['rate'] == .5
    assert summary['unknown'] == 1


def test_semantic_categories_are_distinct_from_structural_validity():
    for verdict in ['supported', 'contradicted', 'insufficient']:
        result = evaluate_claim({'text':'Accuracy is 90%.', 'citations':[{'paper_id':'p','chunk_id':'c'}]}, EVIDENCE, Judge(verdict))
        assert result['verdict'] == verdict
        assert result['citations'][0]['valid']


def test_judge_missing_citation_is_failure():
    class Missing(Judge):
        def structured(self, stage, payload):
            return {'verdict':'supported', 'reason':'x', 'flags':[], 'citations':[]}
    result = evaluate_claim({'text':'x', 'citations':[{'paper_id':'p','chunk_id':'c'}]}, EVIDENCE, Missing())
    assert result['error'] == 'ValueError'
    assert result['verdict'] == 'unknown'


def test_rewrite_error_keeps_denominator():
    from paper_agent.eval_tasks import rewrite_metrics
    rows = [{'id':i, 'expected_query':'alpha', 'required_entities':['alpha']} for i in ['a','b']]
    scores = rewrite_metrics(rows, {'a':{'standalone_query':'alpha'}, 'b':{'error':'timeout'}})
    assert scores['entity_applicable_cases'] == 2
    assert scores['entity_retention_rate'] == .5
    assert scores['token_overlap_with_expected'] == .5


def test_fact_source_only_checkpoint_is_not_empty():
    from paper_agent.eval_tasks import memory_metrics
    scores = memory_metrics([{'id':'a','checkpoints':{'fact_source_papers':['p']}}], {'a':{'fact_source_papers':['p']}})
    assert scores['empty_checkpoints'] == 0
    assert scores['checkpoint_accuracy']['correct'] == 1


def test_audit_preserves_fact_intent_and_marks_ambiguous_rewrites(tmp_path):
    import importlib.util
    from pathlib import Path
    from paper_agent.storage import read_jsonl
    spec = importlib.util.spec_from_file_location('phase_a_audit', Path('scripts/phase_a_audit.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path/'audit'
    summary = module.audit(out=out)
    intent = {r['id']:r for r in read_jsonl(out/'intent.jsonl')}
    assert intent['d02-t1']['intent'] == 'fact'
    assert intent['d02-t1']['review_status'] == 'needs_human_review'
    assert summary['query_rewrite']['missing_expected_query_after'] == 0
    rewrite = {r['id']:r for r in read_jsonl(out/'query_rewrite.jsonl')}
    assert rewrite['d02-t4']['expected_action'] == 'clarify'
    assert rewrite['d02-t4']['review_status'] == 'needs_human_review'
    assert not read_jsonl(out/'human-gold.jsonl')
    import pytest
    with pytest.raises(FileExistsError): module.audit(out=out)
