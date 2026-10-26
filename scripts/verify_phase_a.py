"""Offline delivery assertions; no model calls, no promotion of labels."""
import json
from pathlib import Path
from paper_agent.storage import read_jsonl, write_json, write_jsonl
from paper_agent.providers import APIConfig


def main():
    root=Path('reports/phase-a')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    expected={'intent':100,'rewrite':90,'boundary':55,'memory':80,'e2e':120}
    for name,n in expected.items():
        assert summary[name]['total']==n
        assert summary[name]['valid']+summary[name]['failed']+summary[name]['excluded']==n
    for row in read_jsonl('data/eval/v0.6/query_rewrite.jsonl'):
        assert row['expected_query']
        assert all(h['session_id']==row['session_id'] and h['turn']<row['turn'] for h in row['history'])
    assert not read_jsonl('data/eval/v0.6/human-gold.jsonl')
    preds=read_jsonl(root/'full/e2e/predictions.jsonl')
    claims=read_jsonl(root/'full/e2e/claim-review.jsonl')
    metrics=json.loads((root/'full/e2e/claim-metrics.json').read_text(encoding='utf-8'))
    assert sum(len((r.get('answer') or {}).get('claims',[])) for r in preds)==len(claims)==metrics['claims']
    assert sum(len(c['citations']) for c in claims)==metrics['citation_validity']['denominator']
    assert sum(c['verdict']=='unknown' for c in claims)==metrics['unknown']
    queue=[{**r,'review_status':'needs_human_review','selection':'unsupported/unknown or fixed every-tenth sample'}
           for i,r in enumerate(claims) if r['verdict']!='supported' or i%10==0]
    write_jsonl(root/'claim-human-review-queue.jsonl',queue)
    write_jsonl(root/'claim-failures.jsonl',[r for r in claims if r['verdict']!='supported'])
    config=APIConfig.from_env()
    key=config.api_key.encode()
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.log','.xml'}]
    assert key and all(key not in p.read_bytes() for p in paths), 'Credential detected; redact affected artifact without printing secret'
    jobs=json.loads((root/'full/jobs.json').read_text(encoding='utf-8'))
    assert len(jobs)==9 and all(j['returncode']==0 for j in jobs)
    write_json(root/'delivery-verification.json',{'status':'passed','case_denominators':expected,
        'claims':len(claims),'claim_review_queue':len(queue),'checked_artifacts_for_configured_key':len(paths),
        'human_gold':'unavailable','formal_acceptance':'blocked_by_human_review_and_quality_scope',
        'note':'All experiment processes completed; this does not assert all model cases succeeded.'})
    print('Delivery checks passed; human review queue',len(queue))


if __name__=='__main__': main()
