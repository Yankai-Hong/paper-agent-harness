"""Inspect every official annotation, including disagreements hidden by alternative-gold selection."""
import json
import tarfile
from pathlib import Path
from paper_agent.storage import read_jsonl, write_jsonl, write_json


def main():
    with tarfile.open('data/raw/qasper-test-v0.3.tgz','r:gz') as archive:
        member=next(m for m in archive.getmembers() if Path(m.name).name=='qasper-test-v0.3.json')
        raw=json.load(archive.extractfile(member))
    source={qa['question_id']:(pid,qa) for pid,paper in raw.items() for qa in paper['qas']}
    rows=[]
    for row in read_jsonl('data/qasper/retrieval.jsonl'):
        pid,qa=source[row['query_id']]
        answers=[a['answer'] for a in qa['answers']]
        flags=set(a['unanswerable'] for a in answers)
        yesno={a['yes_no'] for a in answers if a.get('yes_no') is not None}
        conflict=len(flags)>1 or len(yesno)>1
        rows.append({'query_id':row['query_id'],'paper_id':pid,'question':qa['question'],
            'original_annotations':answers,'reference_units':row['gold_units'],
            'answerability_disagreement':len(flags)>1,'yes_no_disagreement':len(yesno)>1,
            'unanimously_unanswerable':flags=={True},'review_status':'needs_human_review' if conflict else 'official_annotations_retained',
            'evidence_warning':'All original annotations retained, including unmapped FLOAT/table evidence; mapped alternatives alone do not establish consensus answerability.',
            'label_source':'QASPER official test v0.3; no new local human review'})
    out=Path('reports/history/development/phase-a/source-annotation-audit')
    out.mkdir(parents=True,exist_ok=False)
    write_jsonl(out/'cases.jsonl',rows)
    write_jsonl(out/'needs_human_review.jsonl',[r for r in rows if r['review_status']=='needs_human_review'])
    write_json(out/'summary.json',{'total':len(rows),'answerability_disagreements':sum(r['answerability_disagreement'] for r in rows),
        'yes_no_disagreements':sum(r['yes_no_disagreement'] for r in rows),'unanimously_unanswerable':sum(r['unanimously_unanswerable'] for r in rows),
        'primary_retrieval_denominator':'All 100 fixed queries with at least one mapped answerable annotation; no cases deleted after observing model predictions.',
        'not_a_claim_gold':'Official QA answers are reference material, not human verification of generated claim entailment.'})
    print((out/'summary.json').read_text(encoding='utf-8'))


if __name__=='__main__': main()
