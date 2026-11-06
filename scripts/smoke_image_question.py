"""Execute real mixed/image-only questions with isolated local request ledgers."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--clip-text-index', required=True)
    p.add_argument('--visual-index', required=True)
    p.add_argument('--visual-root', required=True)
    p.add_argument('--visual-python', required=True)
    p.add_argument('--image-model', required=True)
    p.add_argument('--text-model', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--index', default='indexes/multimodal-acceptance-repair-text-20261102')
    p.add_argument('--query-image', default='data/raw/multimodal-holdout-v3-20261101/v3img01.png')
    p.add_argument('--negative-image', help='Optional off-corpus image for a pure-image abstention check')
    p.add_argument('--question', default='What are the stages shown in this image? Cite the relevant paper evidence.')
    args = p.parse_args()
    from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
    from paper_agent.local_audit import LocalRequestAudit
    from paper_agent.pipeline import PaperPipeline, PaperAgentState
    from paper_agent.observability import Exporter
    from paper_agent.retrieval import Retriever
    from paper_agent.clip_text_index import ClipTextRetriever
    from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    config = replace(APIConfig.from_env(), model='deepseek-v4.1-flash', max_tokens=1024, timeout=60,
                     max_calls=30, max_retries=0, repair_retries=0, thinking='disabled', circuit_enabled=False)
    enc = SubprocessEncoder(args.visual_python, args.image_model, args.text_model, args.device)
    visual = VisualRetriever(args.visual_index, args.visual_root, enc)
    retriever = Retriever(args.index)
    clip_text = ClipTextRetriever(args.clip_text_index, enc, retriever.metadata['corpus_hash'])
    audit = LocalRequestAudit(output/'requests', config.api_key)
    case = {'id':None}; reasoner = None
    def transport(url, body, headers, timeout):
        record = reasoner.calls[-1]
        audit.reserve(record, case['id'])
        response = post_json(url, body, headers, timeout, connect_timeout=config.connect_timeout)
        audit.capture_response(record['request_id'], response, body)
        return response
    class Recorded(CompatibleReasoner):
        def structured(self, stage, payload, images=None):
            before = len(self.calls)
            try: return super().structured(stage, payload, images=images)
            finally:
                for record in self.calls[before:]: audit.finalize(record)
    reasoner = Recorded(config, 'image-smoke', transport=transport, image_root=args.visual_root)
    rows = []
    cases = [('mixed', args.question, args.query_image), ('image_only', '', args.query_image)]
    if args.negative_image:
        cases.append(('image_negative', '', args.negative_image))
    for cid, question, query_image in cases:
        case['id'] = cid
        pipe = PaperPipeline(retriever, reasoner, output/'traces'/cid, output/'memory'/cid, mode='bm25',
                             archive_recall=False, visual_retriever=visual, visual_root=args.visual_root,
                             clip_text_retriever=clip_text, exporter=Exporter(enabled=False),
                             budget_limits={'seconds':300,'model_requests':8,'subqueries':20,'candidates':320,'rerank':120})
        start = time.monotonic(); first = len(reasoner.calls)
        try:
            result = pipe.run(question, PaperAgentState(cid), turn_id=cid, images=[query_image])
            row = {'id':cid,'result':result,'error':None}
        except Exception as exc:
            row = {'id':cid,'error':type(exc).__name__}
        row['elapsed_seconds'] = time.monotonic()-start
        row['requests'] = reasoner.calls[first:]
        rows.append(row)
        (output/'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),encoding='utf-8')
        print(cid, row['error'], 'requests',len(row['requests']),flush=True)


if __name__ == '__main__': main()
