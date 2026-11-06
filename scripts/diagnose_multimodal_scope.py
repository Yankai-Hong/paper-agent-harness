"""Reproduce prior NLU scope decisions with paid requests and a fresh audit."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from paper_agent.decomposition import scope_guard
from paper_agent.local_audit import LocalRequestAudit
from paper_agent.observability import Exporter
from paper_agent.pipeline import PaperAgentState, _corpus_language
from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
from paper_agent.storage import read_jsonl, write_json

out = Path('runs/multimodal-scope-diagnostic-20261031')
out.mkdir(parents=True, exist_ok=False)
config = replace(APIConfig.from_env(), model='deepseek-v4.1-flash', max_tokens=1024,
                 timeout=60, max_calls=3, max_retries=0, repair_retries=0,
                 thinking='disabled', circuit_enabled=False)
audit = LocalRequestAudit(out/'requests', config.api_key)
case_set = json.loads(Path('data/eval/multimodal-holdout-v1/cases.json').read_text(encoding='utf8'))
texts = read_jsonl('data/raw/multimodal-holdout-v1-docling/text.jsonl')
catalog = list({c['paper_id']:{'paper_id':c['paper_id'], 'title':c.get('title','')} for c in texts}.values())
current = {'case_id':''}
reasoner = None

def transport(url, body, headers, timeout):
    record = reasoner.calls[-1]
    audit.reserve(record, current['case_id'])
    response = post_json(url, body, headers, timeout, connect_timeout=config.connect_timeout)
    audit.capture_response(record['request_id'], response, body)
    return response

class RecordedReasoner(CompatibleReasoner):
    def structured(self, stage, payload, images=None):
        before = len(self.calls)
        try:
            return super().structured(stage, payload, images=images)
        finally:
            for record in self.calls[before:]:
                audit.finalize(record)

reasoner = RecordedReasoner(config, 'multimodal-scope-diagnostic', transport=transport)
result = {}
for case in case_set['cases']:
    if case['id'] not in ('h09','h10','h12'):
        continue
    current['case_id'] = case['id']
    state = PaperAgentState('scope-'+case['id'])
    q = case['question']
    payload = {'question':q, 'memory':asdict(state.memory), 'recent_messages':[],
               'paper_catalog':catalog, 'historical_context_not_evidence':[],
               'literal_single_paper_scope':None, 'retrieval_corpus_language':_corpus_language(texts),
               'query_language_instruction':'When retrieval_corpus_language is en or zh and the question is in another language, rewrite the retrieval query in that corpus language, preserving all meaning, constraints and paper scope. If unknown, do not assume a language. The user\'s original question remains authoritative for answering.',
               'decomposition_enabled':False,
               'retrieval_plan_instruction':'When enabled and multiple evidence targets are needed, optionally provide 2-3 sub_questions within parent paper_ids; otherwise omit.'}
    try:
        value = reasoner.structured('nlu', payload)
        encoded = json.dumps(value, ensure_ascii=False)
        if config.api_key in encoded or 'data:image/' in encoded:
            raise ValueError('Unsafe response')
        result[case['id']] = {'nlu':value, 'scope_guard':scope_guard(q,catalog,value.get('paper_ids'),state,
            value.get('standalone_query'),value.get('resolved_entities'))}
    except Exception as exc:
        result[case['id']] = {'failure':type(exc).__name__}
    write_json(out/'results.json', result)
print('Completed', len(result), 'case decisions; requests', len(reasoner.calls))
