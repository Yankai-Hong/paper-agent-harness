"""BGE retrieval of visual evidence descriptions, separate from CLIP images."""
import hashlib
import json
import tempfile
from pathlib import Path

from .retrieval import Retriever, build_index
from .storage import digest, read_jsonl, write_json, write_jsonl
from .visual_evidence import validate_image
from .visual_retrieval import _number


def _representations(item):
    caption = item.get('caption')
    if isinstance(caption, str) and caption.strip():
        yield 'caption', caption.strip()
    # Structure extraction is a parser output, not independently verified fact.
    table = item.get('table_text')
    if (item.get('evidence_type') == 'table' and isinstance(table, str) and table.strip()
            and item.get('table_text_status') == 'docling_extracted_unverified'):
        yield 'table', table.strip()


def build_visual_text_index(evidence_path, directory, image_root, encoder):
    """Create a new BGE-only index; never overwrite an existing index."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Refusing to overwrite existing visual text index')
    evidence = [e for e in read_jsonl(evidence_path) if e.get('evidence_type') in {'figure', 'table'}]
    keys = [(e.get('paper_id'), e.get('chunk_id')) for e in evidence]
    if not evidence or len(keys) != len(set(keys)):
        raise ValueError('Visual corpus must have unique evidence keys')
    rows = []
    for entry in evidence:
        validate_image(entry, image_root)
        for kind, content in _representations(entry):
            identity = json.dumps([entry['paper_id'], entry['chunk_id'], kind], ensure_ascii=False, separators=(',', ':'))
            rows.append({'paper_id':entry['paper_id'], 'chunk_id':'rep-'+hashlib.sha256(identity.encode()).hexdigest()[:24],
                         'text':content, 'source_chunk_id':entry['chunk_id'], 'representation':kind})
    if not rows:
        raise ValueError('No valid caption or structured table representations')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='visual-text-index-', dir=directory.parent) as tmp:
        tmp = Path(tmp)
        src, dest = tmp/'representations.jsonl', tmp/'index'
        write_jsonl(src, rows)
        meta = build_index(src, dest, encoder)
        meta.update(visual_text_schema=1, visual_corpus_hash=digest(evidence),
                    representations={'caption':sum(r['representation']=='caption' for r in rows),
                                     'table':sum(r['representation']=='table' for r in rows)})
        write_json(dest/'metadata.json', meta)
        dest.rename(directory)
    return meta


class VisualTextRetriever:
    """Search caption/table text with BGE; return only trusted visual entities."""
    def __init__(self, directory, visual_retriever, encoder):
        self.visual = visual_retriever
        self.index = Retriever(directory, encoder=encoder)
        meta = self.index.metadata
        if meta.get('visual_text_schema') != 1 or meta.get('visual_corpus_hash') != visual_retriever.metadata.get('corpus_hash'):
            raise ValueError('Visual text index does not match visual evidence')
        if self.index.index.d != meta.get('dimension'):
            raise ValueError('Visual text index dimension mismatch')
        self.by_key = {(e['paper_id'], e['chunk_id']):e for e in visual_retriever.evidence}
        for row in self.index.chunks:
            source = self.by_key.get((row.get('paper_id'), row.get('source_chunk_id')))
            if source is None or (row.get('representation'), row.get('text')) not in list(_representations(source)):
                raise ValueError('Visual text representation does not match source evidence')

    def search(self, query, kind='caption', top_k=20, paper_ids=None, figure_number=None):
        if kind not in {'caption', 'table'} or not isinstance(top_k, int) or top_k < 1:
            raise ValueError('Invalid visual text search')
        count = len(self.index.chunks)
        found = self.index.search(query, mode='dense', top_k=count, candidate_k=count, paper_ids=paper_ids)
        hits, seen = [], set()
        for row in found['hits']:
            if row.get('representation') != kind:
                continue
            key = (row['paper_id'], row['source_chunk_id'])
            source = self.by_key[key]
            if figure_number is not None and _number(source.get('figure_number')) != _number(figure_number):
                continue
            if key in seen:
                continue
            validate_image(source, self.visual.image_root)
            hits.append({**source, 'rank':len(hits)+1, 'visual_text_representation':kind})
            seen.add(key)
            if len(hits) == top_k:
                break
        return {'hits':hits, 'timing_ms':found['timing_ms'], 'corpus_hash':self.index.metadata['corpus_hash']}
