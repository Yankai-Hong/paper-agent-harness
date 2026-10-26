"""Versioned source/rule audit; never asserts that an automated review is human gold."""
import copy
import hashlib
from collections import Counter
from pathlib import Path
from paper_agent.storage import read_jsonl, write_json, write_jsonl


def audit(source=Path('data/eval/v0.4'), out=Path('data/eval/v0.6')):
    if out.exists():
        raise FileExistsError('Choose a new dataset version; do not overwrite audit history')
    dialogues = read_jsonl(source/'dialogs.jsonl')
    by_id = {r['turn_id']: r for r in dialogues}
    titles = {r['paper_id']: r['title'] for r in read_jsonl('data/qasper/papers.jsonl')}
    queries = {r['query_id']: r for r in read_jsonl('data/qasper/retrieval.jsonl')}
    chunks = read_jsonl('indexes/bge/chunks.jsonl')
    audits, summary = [], {}
    for name in ['intent', 'query_rewrite', 'boundary', 'dialogs', 'memory', 'retrieval']:
        rows = read_jsonl(source/(name+'.jsonl'))
        updated = []
        for row in rows:
            new = copy.deepcopy(row)
            identifier = row.get('id', row.get('turn_id', row.get('query_id')))
            original = by_id.get(identifier)
            status = 'needs_human_review'
            reason = 'Semantic behavior/quality requires independent human confirmation; original provisional label retained.'
            basis = {'source_file': str(source/(name+'.jsonl')), 'source_id': identifier}
            if original:
                basis.update(source_query_ids=original['source_query_ids'], source_paper_ids=original['source_paper_ids'])
                valid = all(q in queries for q in original['source_query_ids']) and all(p in titles for p in original['source_paper_ids'])
                if not valid:
                    raise ValueError('Missing source reference: '+identifier)
            if name == 'query_rewrite':
                if row['turn'] == 2:
                    q = queries[original['source_query_ids'][0]]['query']
                    title = titles[original['source_paper_ids'][0]]
                    new['expected_query'] = f'请解释论文《{title}》对于问题“{q}”的原文证据，保留引用；不能从常识补充结论。'
                    status = 'source_rule_verified'
                    reason = 'Unique previous paper/question substituted; instruction and constraints preserved.'
                elif row['turn'] == 3:
                    title = titles[original['source_paper_ids'][0]]
                    new['expected_query'] = f'关于论文《{title}》，'+row['question']
                    status = 'source_rule_verified'
                    reason = 'Restored omitted first comparison paper from same-session opening; original comparison preserved.'
                elif '任意一篇' not in row['question']:
                    new['expected_query'] = row['question']
                    status = 'source_rule_verified'
                    reason = 'Already standalone; no unsupported referent added.'
                else:
                    new['expected_query'] = row['question']
                    new['expected_action'] = 'clarify'
                    reason = 'Explicit unresolved referent and comparison criterion: identity rewrite is a reference, not a resolved query.'
                new['reference_is_unique'] = False
                new['scoring'] = 'token overlap diagnostic + entity aliases; semantic equivalence needs independent review'
            elif name == 'intent' and (original['turn'] in {2, 3} or row['question'].startswith('请检索')):
                expected = {1:'search', 2:'explain', 3:'compare'}[original['turn']]
                new['intent'] = expected
                status, reason = 'source_rule_verified', 'Explicit task verb in source template matches routing taxonomy.'
            elif name == 'boundary' and '私人银行账户余额' in row['question']:
                new['decision'] = 'refuse'
                status, reason = 'source_rule_verified', 'Private account lookup is explicitly outside paper QA scope.'
            elif name == 'boundary' and '未公开实验' in row['question']:
                new['decision'] = 'refuse'
                status, reason = 'source_rule_verified', 'Private unpublished run result absent from fixed public corpus.'
            elif name == 'retrieval':
                if not row.get('gold_units'):
                    raise ValueError('Unanswerable query cannot enter retrieval recall denominator')
                status, reason = 'official_source_annotation', 'Preserved QASPER official evidence annotation; not new human review.'
            elif name == 'memory':
                if not any(row.get('checkpoints', {}).values()):
                    raise ValueError('Empty memory checkpoints')
                reason = 'Nonempty paper/source checkpoints verified structurally; does not establish fact consistency/conflict recovery.'
                new['not_applicable'] = ['exact_fact_update', 'conflict_resolution', 'archive_precision']
            if original:
                units = set(original['reference_units'])
                basis['source_evidence_ids'] = [c['chunk_id'] for c in chunks
                    if units.intersection(c['paper_id']+'::'+s['paragraph_id'] for s in c.get('spans', []))]
                # Source evidence remains a retrieval aid, never gold for template-derived follow-up claims.
            new.update(dataset_version=out.name, review_status=status, human_reviewed=False,
                       label_source='source_rule_audit' if status == 'source_rule_verified' else status)
            audits.append({'dataset':name, 'id':identifier, 'original':row, 'proposed':new,
                           'reason':reason, 'basis':basis, 'review_status':status})
            updated.append(new)
        write_jsonl(out/(name+'.jsonl'), updated)
        write_jsonl(out/'source-verified'/(name+'.jsonl'), [r for r in updated if r['review_status'] != 'needs_human_review'])
        summary[name] = {'before':len(rows), 'after':len(updated),
            'statuses':dict(Counter(r['review_status'] for r in updated)),
            'missing_expected_query_before':sum(not r.get('expected_query') for r in rows) if name == 'query_rewrite' else None,
            'missing_expected_query_after':sum(not r.get('expected_query') for r in updated) if name == 'query_rewrite' else None,
            'label_distribution_before':dict(Counter(str(r.get('intent', r.get('decision'))) for r in rows)),
            'label_distribution_after':dict(Counter(str(r.get('intent', r.get('decision'))) for r in updated))}
    write_jsonl(out/'audit.jsonl', audits)
    write_jsonl(out/'needs_human_review.jsonl', [r for r in audits if r['review_status']=='needs_human_review'])
    write_jsonl(out/'human-gold.jsonl', [])
    write_json(out/'manifest.json', {'version':out.name, 'human_gold_count':0, 'counts':summary,
        'hash_algorithm':'SHA-256 file bytes', 'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*.jsonl')}})
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/eval/v0.4'))
    parser.add_argument('--output', type=Path, default=Path('data/eval/v0.6'))
    args = parser.parse_args()
    print(audit(args.source, args.output))
