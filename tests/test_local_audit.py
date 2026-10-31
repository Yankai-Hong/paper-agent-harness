import json

import pytest

from paper_agent.local_audit import LocalRequestAudit


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
