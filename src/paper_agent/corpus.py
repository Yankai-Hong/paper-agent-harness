"""Section/paragraph-aware chunks with exact character provenance.

QASPER schema: https://huggingface.co/datasets/allenai/qasper
PDF extraction: https://github.com/py-pdf/pypdf (direct dependency).
"""
import re
from pathlib import Path

from .storage import digest, read_jsonl, write_jsonl


def lexical_spans(text):
    # Reproducible fallback for offline smoke tests, NOT the BGE tokenizer.
    return [(m.start(), m.end()) for m in re.finditer(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)]


def lexical_tokens(text):
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text.casefold())


class Chunker:
    def __init__(self, size=500, overlap=80, tokenizer=None, tokenizer_id="lexical-v1"):
        if size < 2 or not 0 <= overlap < size:
            raise ValueError("Require size >= 2 and 0 <= overlap < size")
        self.size, self.overlap = size, overlap
        self.tokenizer, self.tokenizer_id = tokenizer, tokenizer_id

    def spans(self, text):
        if self.tokenizer is None:
            return lexical_spans(text)
        output = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(a, b) for a, b in output["offset_mapping"] if b > a]

    def split(self, paper):
        if not paper.get("paper_id") or not paper.get("title"):
            raise ValueError("Paper requires paper_id and title")
        chunks, pending, count = [], [], 0

        def emit(parts):
            text = "\n\n".join(p["text"] for p in parts)
            identity = {"paper_id": paper["paper_id"], "parts": parts,
                        "chunker": [self.tokenizer_id, self.size, self.overlap]}
            chunks.append({"chunk_id": "c-" + digest(identity)[:24], "paper_id": paper["paper_id"],
                           "title": paper["title"], "section": parts[0]["section"],
                           "text": text, "text_hash": digest(text),
                           "chunker": {"tokenizer": self.tokenizer_id, "size": self.size, "overlap": self.overlap},
                           "source_url": paper.get("source_url"), "spans": parts})

        for index, paragraph in enumerate(paper["paragraphs"]):
            text = paragraph["text"]
            offsets = self.spans(text)
            if not offsets:
                continue
            section = paragraph.get("section", "")
            if pending and (pending[0]["section"] != section or count + len(offsets) > self.size):
                emit(pending)
                pending, count = [], 0
            windows = [(0, len(text))]
            if len(offsets) > self.size:
                windows = []
                start = 0
                while start < len(offsets):
                    end = min(len(offsets), start + self.size)
                    windows.append((offsets[start][0], offsets[end - 1][1]))
                    if end == len(offsets):
                        break
                    start = end - self.overlap
            for a, b in windows:
                part = {"paragraph_id": paragraph.get("paragraph_id", f"p{index}"),
                        "section": section, "start": a, "end": b, "text": text[a:b],
                        "page": paragraph.get("page")}
                if len(offsets) > self.size:
                    emit([part])
                else:
                    pending.append(part)
                    count += len(offsets)
        if pending:
            emit(pending)
        return chunks


def ingest_jsonl(source, output, chunker):
    papers = read_jsonl(source)
    ids = [p["paper_id"] for p in papers]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate paper_id")
    chunks = [c for p in papers for c in chunker.split(p)]
    if not chunks:
        raise ValueError("No usable text in corpus")
    write_jsonl(output, chunks)
    return chunks


def pdf_paper(path):
    from pypdf import PdfReader
    path = Path(path)
    reader = PdfReader(str(path))
    paragraphs = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text(extraction_mode="layout") or ""
        for index, block in enumerate(re.split(r"\n\s*\n", text)):
            if block.strip():
                paragraphs.append({"paragraph_id": f"page-{number}-p{index}", "page": number,
                                   "section": "", "text": block.strip()})
    if not paragraphs:
        raise ValueError("PDF contains no extractable text; OCR is not supported")
    import hashlib
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"paper_id": "pdf-" + file_hash[:20], "title": str((reader.metadata or {}).get("/Title") or path.stem),
            "source_url": None, "file_hash": file_hash, "paragraphs": paragraphs,
            "parse_note": "Text-only extraction; sections and reading order require review."}
