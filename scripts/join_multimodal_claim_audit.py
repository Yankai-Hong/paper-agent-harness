"""Join local original and repaired claim IDs to independently located sources.

This is a read-only post-run analysis of evidence and predictions. It does not
change claims, call a model, perform semantic verification, or copy document
text or image bytes. Output must be a new local file.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', required=True)
p.add_argument('--evidence', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
root = Path(args.run)
rows = [json.loads(line) for line in Path(args.evidence).read_text(encoding='utf8').splitlines() if line.strip()]
catalog = {(r.get('paper_id'),r.get('chunk_id')):r for r in rows}
predictions = json.loads((root/'predictions.json').read_text(encoding='utf8'))['cases']
fields = ('paper_id','chunk_id','evidence_type','docling_label','page','bbox','page_size',
          'coordinate_system','page_rotation','pdf_sha256','figure_number','image_sha256',
          'caption_association_status','caption_source_chunk_ids','caption_geometry')

def resolve(claims):
    result = []
    for claim in claims or []:
        citations = []
        for ref in claim.get('citations',[]):
            source = catalog.get((ref.get('paper_id'),ref.get('chunk_id')))
            citations.append({k:source[k] for k in fields if k in source} if source
                             else {'paper_id':ref.get('paper_id'),'chunk_id':ref.get('chunk_id'),
                                   'resolution':'unknown'})
        result.append({'claim_key':claim.get('key'),'citations':citations})
    return result

result = []
for path in sorted((root/'claim-audit').glob('*.json')):
    audit = json.loads(path.read_text(encoding='utf8'))
    case = audit.get('case_id')
    final = (predictions.get(case) or {}).get('answer') or {}
    repair = audit.get('citation_repair') or {}
    result.append({'case_id':case, 'audit_id':audit.get('audit_id'), 'audit_status':audit.get('status'),
        'original':resolve((audit.get('generated') or {}).get('claims',[])),
        'repaired':resolve((repair.get('response') or {}).get('claims',[])),
        'final':resolve(final.get('claims',[])), 'final_decision':final.get('decision'),
        'final_reason_code':final.get('reason_code'), 'verdicts':audit.get('verdicts'),
        'numeric_check':audit.get('numeric_check'), 'condition_check':audit.get('condition_check'), 'recheck':audit.get('recheck'),
        'citation_repair_status':repair.get('status')})
with Path(args.output).open('x', encoding='utf8') as stream:
    json.dump({'semantic_support':'not independently determined by this join', 'cases':result},
              stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print('Joined',len(result),'claim audit records')
