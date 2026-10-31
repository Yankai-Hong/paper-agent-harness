"""Execute the frozen holdout once, with a new local per-request audit.

Build and freeze the indices and fixture separately before running. This script
never reads expected answers while forming queries and does not resume a run.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

from run_multimodal_acceptance import retrieval_from_trace, save

FIXTURE = '26f06df36b6b7a06f1c6456d2d96b1db1ab61ee2820f0526bb21e64bf9c56d76'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'index', 'semantic-index', 'visual-index', 'visual-root',
                 'visual-python', 'visual-image-model', 'visual-text-model'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--cases', default='data/eval/multimodal-holdout-v1/cases.json')
    p.add_argument('--encoder', default='models/bge-m3')
    p.add_argument('--reranker', default='models/bge-reranker-v2-m3')
    p.add_argument('--device', default='cpu')
    p.add_argument('--visual-device', default='cpu')
    args = p.parse_args()

    from paper_agent.local_audit import LocalRequestAudit
    from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
    from paper_agent.pipeline import PaperAgentState, PaperPipeline
    from paper_agent.memory_manager import MemoryManager
    from paper_agent.observability import Exporter
    from paper_agent.retrieval import Retriever, BGEEncoder, BGEReranker
    from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever
    from paper_agent.visual_text_retrieval import VisualTextRetriever

    fixture_bytes = Path(args.cases).read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != FIXTURE:
        raise ValueError('Frozen fixture hash mismatch')
    fixture = json.loads(fixture_bytes)
    if len(fixture['cases']) != 13 or len(fixture['image_queries']) != 3:
        raise ValueError('Unexpected frozen fixture size')
    config = replace(APIConfig.from_env(), model='deepseek-v4.1-flash', max_tokens=1024,
                     timeout=60, max_calls=10000, max_retries=0, repair_retries=0,
                     thinking='disabled', circuit_enabled=False)
    visual_encoder = SubprocessEncoder(args.visual_python, args.visual_image_model,
                                       args.visual_text_model, args.visual_device)
    visual = VisualRetriever(args.visual_index, args.visual_root, visual_encoder)
    encoder = BGEEncoder(args.encoder, args.device, 4, 768)
    reranker = BGEReranker(args.reranker, args.device, 4, 768)
    text = Retriever(args.index, encoder, reranker)
    semantic = VisualTextRetriever(args.semantic_index, visual, encoder)
    catalog = {(e['paper_id'], e['chunk_id']):e for e in list(text.by_id.values()) + visual.evidence}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    audit = LocalRequestAudit(out / 'requests', config.api_key)
    save(out / 'config.json', {'fixture_sha256':FIXTURE, 'model':config.model,
         'timeout_seconds':60, 'max_output_tokens':1024, 'automatic_retries':0,
         'telemetry':'disabled', 'query':'exact fixture question; no gold scope prefix',
         'visual_index_metadata':visual.metadata, 'text_index_metadata':text.metadata,
         'semantic_index':args.semantic_index, 'device':args.device,
         'visual_device':args.visual_device})
    current = {'case_id':None}
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

    reasoner = RecordedReasoner(config, 'holdout', transport=transport, image_root=args.visual_root)
    results = {'cases':{}}
    save(out / 'predictions.json', results)
    for case in fixture['cases']:
        cid = case['id']
        current['case_id'] = cid
        start = time.monotonic()
        first = len(reasoner.calls)
        row = {'retrieval_complete':False, 'top5':None, 'answer':None,
               'answer_correct':None, 'citation_location':None, 'semantic_support':None,
               'failure':None}
        session = 'holdout-' + cid
        try:
            manager = MemoryManager.local(out / 'memory' / session)
            pipe = PaperPipeline(text, reasoner, out / 'traces' / session, out / 'memory' / session,
                mode='rerank', archive_recall=False, decomposition=False, max_retrieval_attempts=2,
                memory_manager=manager, exporter=Exporter(enabled=False), visual_retriever=visual,
                visual_text_retriever=semantic, visual_root=args.visual_root, visual_claim_check=True)
            answer = pipe.run(case['question'], PaperAgentState(session), turn_id=session,
                              case_id=cid, run_id='multimodal-holdout-v1')
            row['answer'] = answer.get('answer')
            row['retrieval_complete'], row['top5'] = retrieval_from_trace(answer['trace_path'], catalog)
        except Exception as exc:
            row['failure'] = type(exc).__name__
            traces = list((out / 'traces' / session).rglob(session + '.json'))
            if len(traces) == 1:
                row['retrieval_complete'], row['top5'] = retrieval_from_trace(traces[0], catalog)
        row['elapsed_seconds'] = time.monotonic() - start
        row['request_ids'] = [r['request_id'] for r in reasoner.calls[first:]]
        results['cases'][cid] = row
        save(out / 'predictions.json', results)
        print(cid, 'complete', row['failure'], 'requests', len(row['request_ids']), flush=True)
    current['case_id'] = None
    for case in fixture['image_queries']:
        start = time.monotonic()
        row = {'retrieval_complete':False, 'top5':None, 'failure':None}
        try:
            found = visual.search_image(case['query_file'], top_k=5)
            fields = ('paper_id','chunk_id','evidence_type','page','bbox','figure_number','image_path')
            row['top5'] = [{k:h[k] for k in fields if k in h} for h in found['hits']]
            row['retrieval_complete'] = True
        except Exception as exc:
            row['failure'] = type(exc).__name__
        row['elapsed_seconds'] = time.monotonic() - start
        results['cases'][case['id']] = row
        save(out / 'predictions.json', results)
        print(case['id'], 'complete', row['failure'], flush=True)


if __name__ == '__main__':
    main()
