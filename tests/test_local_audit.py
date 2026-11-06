import json

import pytest

from paper_agent.local_audit import LocalClaimAudit, LocalRequestAudit


def test_audit_preserves_truncated_response_and_final_status_without_request(tmp_path):
    audit = LocalRequestAudit(tmp_path / 'new', 'private-key')
    record = {'request_id':'one', 'stage':'generate', 'image_count':1}
    audit.reserve(record, 'case1')
    path = audit.root / 'one.json'
    assert json.loads(path.read_text())['status'] == 'reserved_unknown'
    body = {'messages':[{'content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,abcd'}}]}]}
    response = {'choices':[{'message':{'content':'{"claims":['}, 'finish_reason':'length'}]}
    audit.capture_response('one', response, body)
    audit.finalize({**record, 'status':'error', 'finish_reason':'length', 'error_type':'ValueError',
                    'usage_complete':True, 'usage_raw':{'prompt_tokens':8,'completion_tokens':9}})
    row = json.loads(path.read_text())
    assert row['response_text'] == '{"claims":[' and row['status'] == 'error'
    assert row['usage_raw'] == {'prompt_tokens':8,'completion_tokens':9}
    assert 'data:image' not in path.read_text() and 'abcd' not in path.read_text()
    with pytest.raises(FileExistsError):
        LocalRequestAudit(audit.root, 'private-key')


@pytest.mark.parametrize('content', ['data:image/png;base64,abcd', 'private-key', 'A'*300])
def test_audit_withholds_unsafe_model_output(tmp_path, content):
    audit = LocalRequestAudit(tmp_path / 'new', 'private-key')
    audit.reserve({'request_id':'one','stage':'generate','image_count':0}, 'case1')
    audit.capture_response('one', {'choices':[{'message':{'content':content}}]}, {'messages':[]})
    text = (audit.root / 'one.json').read_text()
    assert json.loads(text)['response_status'] == 'withheld_unsafe_content'
    assert content not in text


def test_audit_never_persists_other_stage_response(tmp_path):
    audit = LocalRequestAudit(tmp_path / 'new', 'private-key')
    audit.reserve({'request_id':'one','stage':'reflect','image_count':0}, 'case1')
    audit.capture_response('one', {'choices':[{'message':{'content':'private document excerpt'}}]}, {'messages':[]})
    assert 'private document excerpt' not in (audit.root / 'one.json').read_text()


def test_schema_diagnostics_retain_only_safe_field_names(tmp_path):
    audit = LocalRequestAudit(tmp_path/'requests', 'private-key')
    rec = {'request_id':'nlu1','stage':'nlu','image_count':0}
    audit.reserve(rec, 'case1')
    audit.finalize({**rec, 'status':'error', 'error_type':'ValidationError',
        'schema_error_path':['sub_questions',0,'private-key'], 'schema_validator':'required',
        'schema_missing_fields':['paper_ids','private-key'], 'reasoning':'hidden reasoning'})
    raw = (audit.root/'nlu1.json').read_text()
    row = json.loads(raw)
    assert row['schema_error_path']==['sub_questions',0,'[redacted]']
    assert row['schema_validator']=='required' and row['schema_missing_fields']==['paper_ids']
    assert 'private-key' not in raw and 'hidden reasoning' not in raw


def test_request_audit_preserves_repair_response_without_request_body(tmp_path):
    audit = LocalRequestAudit(tmp_path/'requests','private-key')
    audit.reserve({'request_id':'repair','stage':'repair_citations','image_count':0},'case')
    audit.capture_response('repair', {'choices':[{'message':{'content':'{"claims":[]}'}}]},
                           {'messages':[{'content':'private evidence text'}]})
    raw = (audit.root/'repair.json').read_text()
    assert json.loads(raw)['response_text'] == '{"claims":[]}' and 'private evidence text' not in raw


def test_claim_audit_preserves_original_and_cited_locations_on_refusal(tmp_path):
    audit = LocalClaimAudit(tmp_path/'claims', 'private-key')
    answer = {'decision':'answer', 'claims':[{'key':'k','text':'A claim',
        'citations':[{'paper_id':'p','chunk_id':'c'}]}]}
    evidence = [{'paper_id':'p','chunk_id':'c','page':3,'bbox':[1,2,3,4],
                 'pdf_sha256':'a'*64,'text':'private evidence text'}]
    key = audit.begin('case','turn',answer,evidence)
    answer['claims'].clear()
    audit.finalize(key, [{'claim_key':'k','status':'unsupported'}],
                   {'decision':'refuse','reason_code':'claim_self_check_failed'})
    raw = (audit.root/(key+'.json')).read_text()
    saved = json.loads(raw)
    assert saved['generated']['claims'][0]['text'] == 'A claim'
    assert saved['cited_regions'][0]['bbox'] == [1,2,3,4]
    assert saved['verdicts'][0]['status'] == 'unsupported'
    assert saved['final_decision'] == 'refuse' and saved['reason_code'] == 'claim_self_check_failed'
    assert 'private evidence text' not in raw


@pytest.mark.parametrize('content', ['data:image/png;base64,abcd', 'private-key', 'A'*300])
def test_claim_audit_withholds_unsafe_claims(tmp_path, content):
    audit = LocalClaimAudit(tmp_path/'claims', 'private-key')
    key = audit.begin('case','turn',{'claims':[{'key':'k','text':content,'citations':[]}]},[])
    assert key is None
    raw = next(audit.root.glob('*.json')).read_text()
    assert json.loads(raw)['status'] == 'withheld_unsafe_content' and content not in raw
