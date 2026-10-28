"""Source-grounded diagnostic fixtures, explicitly NOT independent human gold."""
from dataclasses import replace
from pathlib import Path
from paper_agent.claim_eval import evaluate_claim, summarize_claims
from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
from paper_agent.storage import read_jsonl, write_jsonl, write_json


def recording_reasoner(config, session):
    raw = []
    def transport(url, body, headers, timeout):
        response = post_json(url, body, headers, timeout, connect_timeout=config.connect_timeout)
        raw.append({'model':response.get('model'), 'choices':response.get('choices'), 'usage':response.get('usage')})
        return response
    return CompatibleReasoner(config, session, transport), raw


def main():
    root = Path('reports/history/development/phase-a/claim-benchmark')
    root.mkdir(parents=True, exist_ok=False)
    chunk = next(r for r in read_jsonl('indexes/bge/chunks.jsonl') if r['chunk_id']=='c-89f9839c13d2f99e19b625a3')
    cite = {'paper_id':chunk['paper_id'], 'chunk_id':chunk['chunk_id']}
    examples = [
        ('source-size','HotelRec contains 50 million reviews.',[cite],'supported'),
        ('wrong-number','HotelRec contains 5 million reviews.',[cite],'contradicted'),
        ('source-platform','HotelRec is based on TripAdvisor.',[cite],'supported'),
        ('wrong-entity','HotelRec is based on Amazon product reviews.',[cite],'contradicted'),
        ('unsupported-inference','HotelRec improves recommendation accuracy by 20%.',[cite],'insufficient'),
        ('uncited','HotelRec contains 50 million reviews.',[],'insufficient'),
        ('missing-id','HotelRec contains 50 million reviews.',[dict(cite,chunk_id='absent')],'insufficient'),
        ('wrong-paper','HotelRec contains 50 million reviews.',[dict(cite,paper_id='1809.02208')],'insufficient')]
    config = replace(APIConfig.from_env(),max_tokens=4096,thinking='disabled')
    rows = []
    for identifier,text,refs,expected in examples:
        reasoner, raw = recording_reasoner(config,'fixture-'+identifier)
        result = evaluate_claim({'text':text,'citations':refs},[chunk],reasoner)
        rows.append({**result,'case_id':identifier,'session_id':chunk['paper_id'],'expected':expected,
            'label_source':'agent_source_based_diagnostic_not_human_gold','human_reviewed':False,
            'correct':result['verdict']==expected,'raw_outputs':raw})
        write_jsonl(root/'cases.jsonl',rows)
    write_json(root/'metrics.json', {**summarize_claims(rows),'fixture_agreement':sum(r['correct'] for r in rows)/len(rows),
        'human_gold_validation':None,'note':'Diagnostic fixture agreement does not validate judge accuracy against human gold.'})


if __name__=='__main__': main()
