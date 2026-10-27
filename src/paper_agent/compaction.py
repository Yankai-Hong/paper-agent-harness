"""Citation-preserving semantic selection and conversation overview; archive originals."""
from dataclasses import asdict
import json
import math

from .memory import ContextArchive


def semantic_compact_state(state, archive_dir, reasoner, max_facts=40, max_entities=20, max_messages=8):
    if (len(state.memory.facts)<=max_facts and len(state.memory.entities)<=max_entities
            and len(state.messages)<=max_messages):
        return {'compressed':False}
    original={'memory':asdict(state.memory),'messages':state.messages,'turn_count':state.turn_count, 'commit_protocol':1}
    paper_order=list(state.memory.summary.get('paper_order',[]))
    for message in state.messages:
        if isinstance(message.get('content'),dict):
            for claim in message['content'].get('claims',[]):
                for citation in claim['citations']:
                    if citation['paper_id'] not in paper_order:
                        paper_order.append(citation['paper_id'])
    # Keep earliest and recent topics, with order grounded in actual cited records.
    paper_order=paper_order if len(paper_order)<=20 else paper_order[:1]+paper_order[-19:]
    summary=reasoner.structured('memory_summary', {**original,'cited_paper_order':paper_order})
    if not isinstance(summary.get('overview'),str) or len(summary['overview'])>1200:
        raise ValueError('Summary overview must be a bounded string')
    archive_id=ContextArchive(archive_dir).store(original)
    state.archive_ids = list(dict.fromkeys(state.archive_ids + [archive_id]))
    # Preserve factual text and citations exactly; selection is deterministic.
    state.memory.facts=dict(list(state.memory.facts.items())[-max_facts:])
    state.memory.entities=dict(list(state.memory.entities.items())[-max_entities:])
    state.memory.summary={**state.memory.summary, 'overview':summary['overview'],'paper_order':paper_order,
                          'archive_id':archive_id}
    state.messages=state.messages[-max_messages:]
    before = len(json.dumps(original, ensure_ascii=False))
    after = len(json.dumps({'memory':asdict(state.memory), 'messages':state.messages}, ensure_ascii=False))
    return {'compressed':True,'method':'semantic_overview_and_cited_topic_order','archive_id':archive_id,
            'summary':state.memory.summary, 'summary_success':True,
            'context_chars_before':before, 'context_chars_after':after,
            'token_estimate_before':math.ceil(before/4), 'token_estimate_after':math.ceil(after/4),
            'token_measurement':'character estimate; not API token usage',
            'retained_facts':list(state.memory.facts), 'retained_entities':list(state.memory.entities),
            'retained_fact_consistency':all(original['memory']['facts'][k] == v for k,v in state.memory.facts.items())}
