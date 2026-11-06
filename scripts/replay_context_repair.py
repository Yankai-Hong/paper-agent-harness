"""Replay a saved public-paper claim through bounded context repair and verification."""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-audit', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    from paper_agent.local_audit import LocalClaimAudit, LocalRequestAudit
    from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
    from paper_agent.pipeline import _apply_context_repairs, _repair_collection_conditions
    from paper_agent.citation_guards import check_claim_conditions, check_claim_numbers, check_question_collection_conditions
    old=json.loads(Path(args.original_audit).read_text(encoding='utf8'))
    original=old['generated']
    refs={(r['paper_id'],r['chunk_id']) for c in original['claims'] for r in c['citations']}
    evidence=[json.loads(s) for s in Path(args.evidence).read_text(encoding='utf8').splitlines() if s.strip()]
    evidence=[e for e in evidence if (e['paper_id'],e['chunk_id']) in refs]
    if {(e['paper_id'],e['chunk_id']) for e in evidence} != refs:
        raise ValueError('Original cited text is unavailable')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    config=replace(APIConfig.from_env(),model='deepseek-v4.1-flash',max_tokens=1024,timeout=60,
                   max_calls=4,max_retries=0,repair_retries=0,thinking='disabled',circuit_enabled=False)
    requests=LocalRequestAudit(out/'requests',config.api_key)
    claims=LocalClaimAudit(out/'claim-audit',config.api_key)
    audit_id=claims.begin('h07-replay','h07-replay',original,evidence)
    reasoner=None
    def transport(url,body,headers,timeout):
        record=reasoner.calls[-1]; requests.reserve(record,'h07-replay')
        reply=post_json(url,body,headers,timeout,connect_timeout=config.connect_timeout)
        requests.capture_response(record['request_id'],reply,body)
        return reply
    class RecordedReasoner(CompatibleReasoner):
        def structured(self,stage,payload,images=None):
            before=len(self.calls)
            try: return super().structured(stage,payload,images=images)
            finally:
                for record in self.calls[before:]: requests.finalize(record)
    reasoner=RecordedReasoner(config,'h07-replay',transport=transport)
    question='What time comparison does the Sentence-BERT abstract report for finding the most similar pair among 10,000 sentences with BERT versus SBERT?'
    initial=check_claim_conditions(original['claims'],evidence)
    keys=[r['claim_key'] for r in initial if r['status']=='context_dependent']
    context={'status':'pending','claim_keys':keys}; verdicts=None; collection=None
    answer={'decision':'refuse','reason_code':'context_repair_unknown','claims':[]}
    try:
        selected=[c for c in original['claims'] if c['key'] in keys]
        selected_refs={(r['paper_id'],r['chunk_id']) for c in selected for r in c['citations']}
        cited=[e for e in evidence if (e['paper_id'],e['chunk_id']) in selected_refs]
        repair=reasoner.structured('repair_context',{'question':question,'claims':original['claims'],
                                      'repair_claim_keys':keys,'evidence':cited})
        context['response']=repair
        revised=_apply_context_repairs(original['claims'],evidence,repair['repairs'],keys)
        conditions=check_claim_conditions(revised,evidence); numbers=check_claim_numbers(revised,evidence)
        if any(r['status']!='ok' for r in conditions) or any(r['status'] in {'missing','invalid_reference'} for r in numbers):
            raise ValueError('Repaired claims failed local guards')
        if any(r['status']!='ok' for r in check_question_collection_conditions(question,revised)):
            revised,collection=_repair_collection_conditions(question,revised,evidence)
            if any(r['status']!='ok' for r in check_question_collection_conditions(question,revised)):
                raise ValueError('Repaired claims omit collection condition')
        checked=reasoner.structured('verify_claims',{'question':question,'claims':revised,'evidence':evidence})
        verdicts=checked.get('verdicts',[])
        if len(verdicts)!=len(revised) or {v['claim_key'] for v in verdicts}!={c['key'] for c in revised} or any(v['status']!='supported' for v in verdicts):
            raise ValueError('Self-check did not support every claim')
        answer={'decision':'answer','claims':revised}
        context['status']='accepted_after_self_check'
    except Exception as exc:
        context.update(status='rejected',error_type=type(exc).__name__)
    claims.finalize(audit_id,verdicts,answer,context_repair=context,condition_check=initial,collection_repair=collection)
    with (out/'result.json').open('x',encoding='utf8') as stream:
        json.dump({'answer':answer,'context_repair':context,'collection_repair':collection,'verdicts':verdicts},stream,ensure_ascii=False,indent=2)
    print(json.dumps({'decision':answer['decision'],'context_status':context['status'],'calls':len(reasoner.calls)}))


if __name__ == '__main__':
    main()
