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
    else:
        if args.command in {"ask", "chat"}:
            from .providers import APIConfig, CompatibleReasoner
            from .pipeline import PaperAgentState, PaperPipeline
            config = APIConfig.from_env(args.env_file)
            reasoner = CompatibleReasoner(config, args.session)
        encoder = BGEEncoder(args.encoder, args.device, args.batch_size, args.max_length) if args.mode != "bm25" else None
        reranker = BGEReranker(args.reranker, args.device, args.batch_size, args.max_length) if args.mode == "rerank" else None
        retriever = Retriever(args.index, encoder, reranker)
        if args.command in {"ask", "chat"}:
            root = Path(args.state_dir)
            pipeline = PaperPipeline(retriever, reasoner, root/"traces"/args.session,
                root/"memory", args.mode, not args.no_rewrite, not args.no_reflect, not args.no_compact,
                args.memory_strategy)
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
