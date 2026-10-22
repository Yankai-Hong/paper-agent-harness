"""Create source-linked DRAFT conversations for human review, never gold labels."""
import argparse
from pathlib import Path

from paper_agent.storage import read_jsonl, write_jsonl, write_json, digest


def prepare(source, output):
    source, output = Path(source), Path(output)
    papers = {p['paper_id']: p for p in read_jsonl(source/'papers.jsonl')}
    queries = read_jsonl(source/'retrieval.jsonl')
    selected = list({q['source_paper_id']: q for q in queries}.values())[:30]
    if len(selected) < 30:
        raise ValueError('Need 30 distinct papers with official evidence questions')
    rows = []
    for i, query in enumerate(selected):
        other = selected[(i+1) % len(selected)]
        title = papers[query['source_paper_id']]['title']
        other_title = papers[other['source_paper_id']]['title']
        opening = f'关于论文《{title}》，{query["query"]}'
        if i % 2 == 0:
            opening = '请检索以下问题的相关原文：' + opening
        final, intent, decision = [
            ('顺便查一下我明天的私人银行账户余额。', 'out_of_scope', 'refuse'),
            ('它更好吗？这里的“它”可能指刚才的任意一篇，比较指标我还没确定。', 'compare', 'clarify'),
            (f'请给出我今天私下运行的未公开实验 run-{i:02d} 的精确准确率；没有证据就说明无法确定。', 'fact', 'refuse')][i % 3]
        prompts = [opening,
                   '请解释刚才那篇论文对于这个问题的证据，保留引用；不能从常识补充结论。',
                   f'与《{other_title}》对比，分别说明两篇论文如何回答“{query["query"]}”与“{other["query"]}”，并说明是否具有可比性。', final]
        intents = ['search' if i % 2 == 0 else 'fact', 'explain', 'compare', intent]
        for turn, prompt in enumerate(prompts, 1):
            rows.append({'turn_id': f'd{i+1:02d}-t{turn}', 'session_id': f'd{i+1:02d}',
                'turn': turn, 'question': prompt, 'review_status': 'draft_unreviewed',
                'provisional_intent': intents[turn-1],
                'provisional_decision': decision if turn == 4 else None,
                'source_query_ids': [query['query_id']] + ([other['query_id']] if turn == 3 else []),
                'source_paper_ids': [query['source_paper_id']] + ([other['source_paper_id']] if turn == 3 else []),
                'reference_units': query['gold_units'] + (other['gold_units'] if turn == 3 else []),
                'annotation_note': 'Template-derived follow-ups; source evidence is NOT gold for new questions.'})
    write_jsonl(output/'dialogues.jsonl', rows)
    write_jsonl(output/'intent-draft.jsonl', rows[:100])
    boundary = [r for r in rows if r['turn'] == 4] + [r for r in rows if r['turn'] == 1][:25]
    write_jsonl(output/'boundary-draft.jsonl', boundary)
    # References only: execute all preceding turns even when scoring a subset.
    write_jsonl(output/'e2e-view.jsonl', [{'turn_id': r['turn_id']} for r in rows[:70]])
    write_jsonl(output/'judgments-template.jsonl', [{'turn_id': r['turn_id'], 'review_status': 'pending',
        'intent': None, 'decision': None, 'rewrite_correct': None, 'behavior_correct': None,
        'answer_correct': None, 'citations_correct': None, 'no_unrecovered_error': None,
        'reviewer': None, 'notes': ''} for r in rows])
    write_json(output/'manifest.json', {'sessions': 30, 'turns': 120, 'intent_view': 100,
        'boundary_view': 55, 'e2e_view': 70, 'source_queries_hash': digest(queries),
        'status': 'draft_unreviewed', 'valid_for_effectiveness_claims': False,
        'warning': 'Counts match target shapes only. Independent review and real model runs are required.'})
    print('Created 30 x 4 source-linked DRAFT dialogues; no model calls or gold-label claims.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/qasper')
    parser.add_argument('--output', default='data/dialogues-draft')
    args = parser.parse_args()
    prepare(args.source, args.output)
