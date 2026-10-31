import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .corpus import Chunker, ingest_jsonl, pdf_paper
from .retrieval import Retriever, BGEEncoder, BGEReranker, build_index
from .storage import read_jsonl, write_jsonl
from .evaluation import evaluate_retriever
from .memory import SessionMemory


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def model_options(parser):
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)


def visual_options(parser):
    parser.add_argument("--visual-index", required=True, help="Independent CLIP index directory")
    parser.add_argument("--visual-root", required=True, help="Controlled crop directory")
    parser.add_argument("--visual-python", required=True, help="Python in the isolated visual environment")
    parser.add_argument("--visual-image-model", required=True, help="Pinned local CLIP image model")
    parser.add_argument("--visual-text-model", required=True, help="Pinned local multilingual CLIP text model")
    parser.add_argument("--visual-device", default="cpu")


def visual_encoder(args):
    from .visual_retrieval import SubprocessEncoder
    return SubprocessEncoder(args.visual_python, args.visual_image_model, args.visual_text_model,
                             device=args.visual_device)


def main():
    parser = argparse.ArgumentParser(description="论文检索与问答；仅 ask/chat 使用配置的模型 API")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("source")
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--size", type=int, default=500)
    ingest.add_argument("--overlap", type=int, default=80)
    ingest.add_argument("--tokenizer", help="Local BGE model path; omit for lexical smoke-test chunks")
    pdf = sub.add_parser("pdf")
    pdf.add_argument("source")
    pdf.add_argument("--output", required=True)
    index = sub.add_parser("index")
    index.add_argument("chunks")
    index.add_argument("--output", required=True)
    index.add_argument("--dense", action="store_true")
    model_options(index)
    visual_index = sub.add_parser("visual-index", help="Build a separate CLIP figure/table index")
    visual_options(visual_index)
    visual_index.add_argument("evidence", help="Docling evidence JSONL")
    semantic_index = sub.add_parser("visual-semantic-index", help="Build a separate BGE caption/table index")
    semantic_index.add_argument("evidence")
    semantic_index.add_argument("--visual-root", required=True)
    semantic_index.add_argument("--output", required=True)
    model_options(semantic_index)
    collect = sub.add_parser("visual-collect", help="Combine Docling evidence and prepare a text-only corpus")
    collect.add_argument("sources", nargs="+", help="Per-paper Docling JSONL files")
    collect.add_argument("--evidence-output", required=True)
    collect.add_argument("--text-output", required=True)
    for name in ("visual-search-text", "visual-search-image"):
        p = sub.add_parser(name, help="Search the visual index")
        visual_options(p)
        p.add_argument("query", help="Text query or local query image")
        p.add_argument("--paper-id", action="append")
        p.add_argument("--figure-number")
        p.add_argument("--top-k", type=int, default=5)
    for command in ("search", "shell", "ask", "chat"):
        p = sub.add_parser(command)
        p.add_argument("--index", required=True)
        p.add_argument("--mode", choices=["bm25", "dense", "hybrid", "rerank"], default="bm25")
        model_options(p)
        if command in {"ask", "chat"}:
            p.add_argument("--env-file", default=".env")
            p.add_argument("--session", default="default")
            p.add_argument("--state-dir", default="runs")
            p.add_argument("--no-rewrite", action="store_true")
            p.add_argument("--no-reflect", action="store_true")
            p.add_argument("--no-compact", action="store_true")
            p.add_argument("--memory-strategy", choices=["bounded", "semantic"], default="bounded")
            p.add_argument("--visual-index", help="Enable visual retrieval using this CLIP index")
            p.add_argument("--visual-semantic-index", help="Optional BGE caption/table index for multimodal fusion")
            p.add_argument("--visual-root")
            p.add_argument("--visual-python")
            p.add_argument("--visual-image-model")
            p.add_argument("--visual-text-model")
            p.add_argument("--visual-device", default="cpu")
            p.add_argument("--visual-claim-check", action="store_true", help="Use a model self-check of each cited claim (extra API call)")
            if command == "ask":
                p.add_argument("query")
        elif command == "search":
            p.add_argument("query")
            p.add_argument("--paper-id", action="append")
    from .eval_cli import add_eval_parser, dispatch as dispatch_eval
    add_eval_parser(sub)
    args = parser.parse_args()
    if args.command == "pdf":
        paper = pdf_paper(args.source)
        write_jsonl(args.output, [paper])
        emit({"paper_id": paper["paper_id"], "paragraphs": len(paper["paragraphs"])})
    elif args.command == "ingest":
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, local_files_only=True)
        chunks = ingest_jsonl(args.source, args.output, Chunker(args.size, args.overlap, tokenizer,
                             args.tokenizer or "lexical-v1"))
        emit({"chunks": len(chunks), "tokenizer": args.tokenizer or "lexical-v1"})
    elif args.command == "index":
        encoder = BGEEncoder(args.encoder, args.device, args.batch_size, args.max_length) if args.dense else None
        emit(build_index(args.chunks, args.output, encoder))
    elif args.command == "eval":
        emit(dispatch_eval(args))
    elif args.command == "visual-collect":
        evidence_output, text_output = Path(args.evidence_output), Path(args.text_output)
        if evidence_output.resolve() == text_output.resolve() or evidence_output.exists() or text_output.exists():
            raise ValueError("Choose two distinct new output files; existing data is never overwritten")
        rows = [row for source in args.sources for row in read_jsonl(source)]
        keys = [(r.get("paper_id"), r.get("chunk_id")) for r in rows]
        if not rows or any(not all(k) for k in keys) or len(keys) != len(set(keys)):
            raise ValueError("Evidence must have unique paper and chunk identities")
        texts = [r for r in rows if r.get("evidence_type") == "text" and r.get("text", "").strip()]
        if not texts or len({r["chunk_id"] for r in texts}) != len(texts):
            raise ValueError("Text corpus must have globally unique chunk IDs")
        write_jsonl(evidence_output, rows)
        write_jsonl(text_output, texts)
        emit({"evidence": len(rows), "text_chunks": len(texts), "visual_items": len(rows) - len(texts)})
    elif args.command == "visual-index":
        from .visual_retrieval import build_visual_index
        emit(build_visual_index(args.evidence, args.visual_index, args.visual_root, visual_encoder(args)))
    elif args.command == "visual-semantic-index":
        from .visual_text_retrieval import build_visual_text_index
        encoder = BGEEncoder(args.encoder, args.device, args.batch_size, args.max_length)
        emit(build_visual_text_index(args.evidence, args.output, args.visual_root, encoder))
    elif args.command in {"visual-search-text", "visual-search-image"}:
        from .visual_retrieval import VisualRetriever
        visual = VisualRetriever(args.visual_index, args.visual_root, visual_encoder(args))
        method = visual.search_text if args.command == "visual-search-text" else visual.search_image
        emit(method(args.query, top_k=args.top_k, paper_ids=args.paper_id, figure_number=args.figure_number))
    else:
        visual = None
        if args.command in {"ask", "chat"}:
            from .providers import APIConfig, CompatibleReasoner
            from .pipeline import PaperAgentState, PaperPipeline
            config = APIConfig.from_env(args.env_file)
            options = (args.visual_root, args.visual_python, args.visual_image_model, args.visual_text_model)
            if args.visual_index:
                if not all(options):
                    parser.error("visual mode requires --visual-root, --visual-python and both model paths")
                from .visual_retrieval import VisualRetriever
                visual = VisualRetriever(args.visual_index, args.visual_root, visual_encoder(args))
            elif any(options):
                parser.error("visual options require --visual-index")
            if args.visual_claim_check and not visual:
                parser.error("--visual-claim-check requires --visual-index")
            reasoner = CompatibleReasoner(config, args.session, image_root=args.visual_root if visual else None)
        encoder = BGEEncoder(args.encoder, args.device, args.batch_size, args.max_length) if args.mode != "bm25" else None
        reranker = BGEReranker(args.reranker, args.device, args.batch_size, args.max_length) if args.mode == "rerank" else None
        retriever = Retriever(args.index, encoder, reranker)
        if args.command in {"ask", "chat"}:
            semantic = None
            if args.visual_semantic_index:
                if not visual or encoder is None:
                    parser.error("--visual-semantic-index requires visual options and a dense or hybrid/rerank text mode")
                from .visual_text_retrieval import VisualTextRetriever
                semantic = VisualTextRetriever(args.visual_semantic_index, visual, encoder)
            root = Path(args.state_dir)
            pipeline = PaperPipeline(retriever, reasoner, root/"traces"/args.session,
                root/"memory", args.mode, not args.no_rewrite, not args.no_reflect, not args.no_compact,
                args.memory_strategy, visual_retriever=visual, visual_root=args.visual_root, visual_text_retriever=semantic,
                visual_claim_check=args.visual_claim_check)
            state = PaperAgentState.load(pipeline.memory_manager.root, args.session)
            if args.command == "ask":
                emit(pipeline.run(args.query, state))
                return
            print("模型问答模式（使用 API 配额）。/corpus /trace /memory /quit")
            last = None
            while True:
                try:
                    query = input("paper> ").strip()
                    if query in {"/quit", "/exit"}:
                        break
                    if query == "/corpus":
                        emit(retriever.metadata)
                    elif query == "/memory":
                        emit(asdict(state.memory))
                    elif query == "/trace":
                        emit(json.loads((pipeline.trace_dir/f"{last['trace_id']}.json").read_text(encoding="utf-8"))
                             if last else {"status": "no completed turn in this process", "directory": str(pipeline.trace_dir)})
                    elif query.startswith("/"):
                        print("未知命令")
                    elif query:
                        last = pipeline.run(query, state)
                        emit(last)
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as exc:
                    print(f"本轮失败：{exc}；详情见 trace 文件。")
        elif args.command == "search":
            emit(retriever.search(args.query, args.mode, paper_ids=args.paper_id))
        else:
            print("离线证据检索模式（不生成模型答案）。命令：/corpus /trace /memory /quit")
            last = None
            memory = SessionMemory("offline")
            while True:
                try:
                    query = input("paper> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if query in {"/quit", "/exit"}:
                    break
                if query == "/corpus":
                    emit(retriever.metadata)
                elif query == "/trace":
                    emit(last or {"status": "no retrieval yet"})
                elif query == "/memory":
                    emit({"status": "offline retrieval does not infer memory facts", "session_id": memory.session_id})
                elif query.startswith("/"):
                    print("未知命令")
                elif query:
                    last = retriever.search(query, args.mode)
                    emit(last)


if __name__ == "__main__":
    main()
