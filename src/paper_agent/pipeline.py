"""Seven-stage domain orchestration; new code, not an upstream feature.

reasoner.structured(stage, payload) must return schema-compatible dictionaries.
Offline tests inject a scripted reasoner. No API invocation exists by default.
"""
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from pathlib import Path
import time
import uuid
import math
import copy
import json
import os
import re

from .memory import SessionMemory, ContextArchive, validate_citations
from .storage import write_json, safe_id, digest
from .observability import invocation, Exporter
from .budgets import TurnBudget, current_budget, BudgetExceeded
from .decomposition import plan, scope_guard, exclusion_kind, exclusive_kind, explicit_single_paper_prefix, explicit_natural_metadata_scope, unknown_literal_metadata_id, catalog_with_literal_aliases, current_ordered_references

STAGES = ("Guardrails", "NLU", "Retrieve", "Rerank", "Reflect", "Generate", "Memory")
_TRACE_ROUTES = {'text_original', 'text_rewrite', 'text_dense_rescue', 'frontmatter_located'} | {
    f'{name}_{variant}' for name in ('text_bm25', 'text_dense', 'rerank', 'clip',
        'caption_bm25', 'caption_bge', 'table_bge') for variant in ('original', 'rewrite')}
_TRACE_ROUTES |= {'image_to_text', 'image_to_image'}


@dataclass(frozen=True)
class PaperQuery:
    text: str = ''
    images: tuple = ()


def _interleave_evidence(*lanes):
    """Select independent evidence by rank without cross-entity fusion votes."""
    merged = _merge_evidence_entities(*lanes)
    seen, output = set(), []
    for i in range(max((len(lane) for lane in lanes), default=0)):
        for lane in lanes:
            if i < len(lane):
                key = (lane[i]['paper_id'], lane[i]['chunk_id'])
                if key not in seen:
                    seen.add(key); output.append(merged[key])
    return output


def _paper_artifact_digest_question(question):
    """Recognize a narrow factual artifact-digest request, not evidence of absence."""
    return bool(re.search(r'(?i)\bsha[- ]?256\b|\bchecksum\b|校验和|哈希', question)
                and re.search(r'(?i)模型|权重|数据集|论文|文件|\bmodel\b|\bweights?\b|\bcheckpoint\b|\bdataset\b|\bpaper\b|\bfile\b', question)
                and re.search(r'(?i)什么|多少|是否|有没有|\bwhat\b|\bwhich\b|\bdoes\b|\bis\b|[?？]', question))


def _corpus_language(items, paper_id=None):
    """Conservatively identify English or Chinese without exposing corpus text."""
    latin = cjk = 0
    common = set()
    english_markers = {'the','and','of','to','in','for','is','are','with','that','we','our','this','from','as','by'}
    for item in items:
        if paper_id is not None and item.get('paper_id') != paper_id:
            continue
        if item.get('evidence_type') not in (None, 'text') or item.get('docling_label') == 'caption':
            continue
        value = item.get('text')
        if not isinstance(value, str):
            continue
        value = value[:4000]
        latin += len(re.findall(r'[A-Za-z]', value))
        cjk += len(re.findall(r'[\u3400-\u9fff\uf900-\ufaff]', value))
        common.update(set(re.findall(r'[A-Za-z]+', value.lower())) & english_markers)
    total = latin + cjk
    if cjk >= 20 and total and cjk / total >= .8:
        return 'zh'
    if latin >= 100 and total and latin / total >= .95 and len(common) >= 3:
        return 'en'
    return 'unknown'


def _answer_language(question):
    """Identify language of the actual question, excluding an explicit scope prefix."""
    question = re.sub(r'^\s*only\s+paper\s+[A-Za-z0-9_.-]+\s*[:：]\s*', '', question, flags=re.IGNORECASE)
    cjk = len(re.findall(r'[\u3400-\u9fff\uf900-\ufaff]', question))
    latin = len(re.findall(r'[A-Za-z]', question))
    if cjk >= 2 and cjk >= latin / 4:
        return 'zh'
    if latin >= 8 and cjk == 0:
        return 'en'
    return 'unknown'


def _requested_figure(question):
    match = re.search(r'(?i)(?:\b(fig(?:ure)?|table)\s*\.?|([图表]))\s*(\d+[a-z]?)', question)
    if not match:
        return None
    kind = match.group(1) or ('Figure' if match.group(2) == '图' else 'Table')
    return kind + ' ' + match.group(3)


def _figure_refs(question):
    from .visual_retrieval import _number
    return {_number((m.group(1) or ('Figure' if m.group(2) == '图' else 'Table')) + ' ' + m.group(3))
            for m in re.finditer(r'(?i)(?:\b(fig(?:ure)?|table)\s*\.?|([图表]))\s*(\d+[a-z]?)', question)}


def _safe_query_variants(question, rewritten):
    """Keep original; do not use a rewrite that adds/changes figure IDs."""
    original = question.strip()
    variants = [original]
    original_figures, rewritten_figures = _figure_refs(original), _figure_refs(rewritten)
    if rewritten.strip() and rewritten.strip() != original and rewritten_figures <= original_figures:
        variants.append(rewritten.strip())
    return variants


def _ambiguous_bare_figure(question, paper_ids, visual_evidence):
    """A bare shared figure number is not a unique cross-paper target."""
    if paper_ids is not None or len(_figure_refs(question)) != 1:
        return False
    stripped = re.sub(r'(?i)(?:\b(?:fig(?:ure)?|table)\s*\.?|[图表])\s*\d+[a-z]?', ' ', question)
    stripped = re.sub(r'(?i)\b(?:what|which|is|the|does|show|shows|describe|explain|please|a|an|of|in|this|that|paper|figure|table)\b', ' ', stripped)
    stripped = re.sub(r'请问|请|这个|那个|这张|该|是|什么|内容|显示|展示|描述|介绍|说明|图|表|论文|的|中|吗|呢', '', stripped)
    if re.search(r'[A-Za-z0-9\u3400-\u9fff]', stripped):
        return False
    wanted = next(iter(_figure_refs(question)))
    from .visual_retrieval import _number
    papers = {e.get('paper_id') for e in visual_evidence if _number(e.get('figure_number')) == wanted}
    return len(papers) > 1


def _fuse_ranked_lanes(lanes):
    from .retrieval import rrf
    merged = _merge_evidence_entities(*lanes)
    ranking = rrf([[(e['paper_id'],e['chunk_id']) for e in lane] for lane in lanes])
    return [merged[key] for key, _ in ranking]


def _merge_evidence_entities(*lanes):
    """Duplicate keys must describe the same source; rank scores are not identity."""
    fields = ('evidence_type', 'text', 'page', 'bbox', 'page_size', 'coordinate_system',
              'page_rotation', 'figure_number', 'image_path', 'image_sha256')
    merged = {}
    for lane in lanes:
        for item in lane:
            key = (item['paper_id'], item['chunk_id'])
            if key in merged:
                previous = merged[key]
                if any(previous.get(field) != item.get(field) for field in fields):
                    raise ValueError('Conflicting evidence metadata for the same citation key')
            merged[key] = item
    return merged


def _map_exact_caption_entities(candidates, visual_evidence):
    """Map only a unique exact Docling caption from the same PDF and page."""
    lookup = {}
    for item in visual_evidence:
        caption, pdf = item.get('caption'), item.get('pdf_sha256')
        if (item.get('evidence_type') in {'figure', 'table'} and isinstance(caption, str)
                and caption.strip() and isinstance(pdf, str) and re.fullmatch(r'[0-9a-f]{64}', pdf)):
            key = (item.get('paper_id'), item.get('page'), pdf, caption.strip())
            lookup.setdefault(key, []).append(item)
    text, captions, mapping = [], [], {}
    for item in candidates:
        pdf, content = item.get('pdf_sha256'), item.get('text')
        key = (item.get('paper_id'), item.get('page'), pdf, content.strip() if isinstance(content, str) else None)
        matches = lookup.get(key, []) if item.get('docling_label') == 'caption' else []
        if len(matches) != 1:
            text.append(item)
            continue
        source = matches[0]
        captions.append(source)
        mapping[(item['paper_id'], item['chunk_id'])] = (source['paper_id'], source['chunk_id'])
    return text, captions, mapping


def _fuse_text_visual_candidates(text, captions, visuals):
    """Fuse caption/image rankings first, then give the visual lane one vote."""
    from .retrieval import rrf
    indexed = _merge_evidence_entities(text, captions, visuals)
    def keys(items):
        return [(item['paper_id'], item['chunk_id']) for item in items]
    visual_rank = [key for key, _ in rrf([keys(captions), keys(visuals)])]
    return [indexed[key] for key, _ in rrf([keys(text), visual_rank])]


def _expand_verified_associations(evidence, by_id, limit=5):
    """Promote explicit linked prose only after checking the trusted text index.

    Nested parser links are never sent as hidden evidence: a verified link is
    surfaced as its own citation key, otherwise it is discarded.
    """
    from .visual_retrieval import _number
    clean, links = [], []
    seen = {(item.get('paper_id'), item.get('chunk_id')) for item in evidence}
    for item in evidence:
        if item.get('evidence_type') not in {'figure', 'table'}:
            clean.append(item)
            continue
        # Caption text and serialized table text are not inside the image's
        # bounding box.  Never smuggle them under the visual citation key.
        clean.append({**{k: v for k, v in item.items() if k not in {
            'associated_text', 'caption', 'table_text', 'table_structure'}}, 'text':''})
        if not item.get('figure_number') or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('pdf_sha256', ''))):
            continue
        associations = item.get('associated_text')
        if not isinstance(associations, list):
            continue
        for ref in associations:
            if len(links) >= limit or not isinstance(ref, dict):
                continue
            cid = ref.get('chunk_id')
            source = by_id.get(cid) if isinstance(cid, str) else None
            if not isinstance(source, dict):
                continue
            key = (source.get('paper_id'), source.get('chunk_id'))
            text, span, label = source.get('text'), ref.get('reference_span'), ref.get('reference_text')
            if (key in seen or key != (item.get('paper_id'), cid) or ref.get('paper_id') != key[0]
                    or source.get('evidence_type') not in (None, 'text') or source.get('pdf_sha256') != item['pdf_sha256']
                    or source.get('page') != ref.get('page') or not isinstance(text, str) or text != ref.get('text')
                    or ref.get('method') != 'explicit_figure_or_table_reference'
                    or not isinstance(span, list) or len(span) != 2 or any(type(i) is not int for i in span)
                    or not 0 <= span[0] < span[1] <= len(text) or text[span[0]:span[1]] != label
                    or not isinstance(label, str) or _number(label) != _number(item['figure_number'])):
                continue
            links.append(source)
            seen.add(key)
    return clean + links, [item['chunk_id'] for item in links]


def _valid_located_text(item):
    """Accept only bounded, page-located text suitable for an independent citation."""
    if not isinstance(item, dict) or item.get('evidence_type') not in (None, 'text'):
        return False
    box, size = item.get('bbox'), item.get('page_size')
    if (not isinstance(item.get('page'), int) or isinstance(item.get('page'), bool)
            or item['page'] < 1 or not isinstance(box, (list, tuple)) or len(box) != 4
            or not isinstance(size, (list, tuple)) or len(size) != 2):
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (*box, *size)):
        return False
    x0, y0, x1, y1 = box
    return (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]
            and item.get('coordinate_system') == 'top-left-points'
            and item.get('page_rotation') in (0, 90, 180, 270)
            and isinstance(item.get('text'), str) and bool(item['text'].strip()))


def _linked_table_caption(table, by_id):
    """Resolve only an explicitly linked independent caption on the same PDF page."""
    status = table.get('caption_association_status')
    if table.get('evidence_type') != 'table' or status not in {
            'agent_page_reviewed_not_human', 'geometry_unique_unverified', 'docling_link_unverified'}:
        return None
    ids = table.get('caption_source_chunk_ids')
    if status == 'docling_link_unverified':
        refs = table.get('caption_source_refs')
        if not isinstance(refs, list) or len(refs) != 1:
            return None
        found = [row for row in by_id.values() if row.get('docling_ref') == refs[0]
                 and row.get('paper_id') == table.get('paper_id')]
        if len(found) != 1:
            return None
        cap = found[0]
    else:
        if not isinstance(ids, list) or len(ids) != 1:
            return None
        cap = by_id.get(ids[0])
    if (not _valid_located_text(cap) or any(cap.get(k) != table.get(k) for k in
            ('paper_id', 'page', 'page_size', 'page_rotation', 'coordinate_system', 'pdf_sha256'))
            or not re.fullmatch(r'[0-9a-f]{64}', str(table.get('pdf_sha256') or ''))
            or cap.get('text', '').strip() != (table.get('caption') or '').strip()):
        return None
    geom = table.get('caption_geometry') or {}
    if status != 'docling_link_unverified' and (geom.get('caption_bbox') != cap.get('bbox') or geom.get('visual_bbox') != table.get('bbox')):
        return None
    number = table.get('figure_number') or ''
    if not re.fullmatch(r'Table\s+\d+[a-z]?', number, re.I) or not re.match(
            r'^\s*' + re.escape(number) + r'\s*[:.\-–—]', cap['text'], re.I):
        return None
    return cap


def _attach_table_number_citations(claims, evidence, visual_sources=()):
    """Add an already retrieved, explicitly linked caption or report unknown."""
    by_id = {e['chunk_id']:e for e in evidence}
    trusted_visual = {(e['paper_id'],e['chunk_id']):e for e in visual_sources}
    added, unknown = [], []
    for claim in claims:
        labels = re.findall(r'(?:\bTable|表)\s*\d+[a-z]?', claim.get('text', ''), re.I)
        for label in labels:
            number = re.search(r'\d+[a-z]?', label, re.I).group(0)
            tables = [by_id.get(ref.get('chunk_id')) for ref in claim.get('citations', [])]
            tables = [t for t in tables if t and t.get('evidence_type') == 'table'
                      and re.fullmatch(r'Table\s*' + re.escape(number), t.get('figure_number') or '', re.I)]
            if not tables:
                continue  # A text-only claim may itself cite the printed table label.
            originals = []
            for table in tables:
                source = trusted_visual.get((table['paper_id'],table['chunk_id']), table)
                fields = ('paper_id','chunk_id','evidence_type','page','bbox','page_size','coordinate_system',
                          'page_rotation','pdf_sha256','image_sha256','figure_number','caption_association_status')
                originals.append(source if all(source.get(k) == table.get(k) for k in fields) else None)
            captions = [_linked_table_caption(t, by_id) if t is not None else None for t in originals]
            if any(c is None for c in captions):
                unknown.append(claim.get('key'))
                continue
            for cap in captions:
                ref = {'paper_id':cap['paper_id'], 'chunk_id':cap['chunk_id']}
                if ref not in claim['citations']:
                    claim['citations'].append(ref)
                    added.append({'claim_key':claim['key'], **ref})
    return {'added':added, 'unknown':unknown}


def _apply_context_repairs(claims, evidence, repairs, required):
    """Replace only 'this effort' with a phrase grounded in its cited text."""
    by_key = {(e['paper_id'], e['chunk_id']):e for e in evidence}
    if len(repairs) != len(required) or {r.get('claim_key') for r in repairs} != set(required):
        raise ValueError('Incomplete context repair')
    updated = copy.deepcopy(claims)
    by_claim = {c['key']:c for c in updated}
    for repair in repairs:
        claim = by_claim[repair['claim_key']]
        refs = {(r['paper_id'],r['chunk_id']) for r in claim['citations']}
        ref = (repair['paper_id'],repair['chunk_id'])
        source = by_key.get(ref)
        replacement, quote = repair['replacement'], repair['source_quote']
        if (ref not in refs or not _valid_located_text(source) or len(quote) < 12 or len(replacement) > 250
                or quote not in source['text'] or quote not in replacement
                or len(re.findall(r'\bthis\s+effort\b', claim['text'], re.I)) != 1):
            raise ValueError('Ungrounded context repair')
        claim['text'] = re.sub(r'\bthis\s+effort\b', lambda _:replacement, claim['text'], count=1, flags=re.I)
    return updated


def _repair_collection_conditions(question, claims, evidence):
    """Append a source-quoted collection qualifier only with a unique located match."""
    from .citation_guards import check_question_collection_conditions, check_claim_numbers
    required = {x['claim_key'] for x in check_question_collection_conditions(question, claims)
                if x['status'] == 'collection_condition_missing'}
    match = re.search(r'(?i)\b(?:among|within|from\s+a\s+collection\s+of|in\s+a\s+collection\s+of)\s+'
                      r'(\d[\d,]*)\s+([a-z]+)\b', question)
    if not required or not match:
        return copy.deepcopy(claims), {'added':[], 'unknown':list(required)}
    count, noun = match.group(1).replace(',', ''), match.group(2).lower()
    by_key = {(e['paper_id'],e['chunk_id']):e for e in evidence}
    updated, added, unknown = copy.deepcopy(claims), [], []
    stops = {'about','after','before','being','from','have','into','more','than','that','their','there',
             'these','this','those','time','with','without','which','would','sentence','sentences',
             'takes','seconds','hours','requires','results','using','finding','effort','reduced','reduces',
             'model','task','collection','reported','find','text','number'}
    for claim in updated:
        if claim.get('key') not in required:
            continue
        refs = {(r['paper_id'],r['chunk_id']) for r in claim['citations']}
        papers = {p for p,_ in refs}
        pdfs = {by_key[k].get('pdf_sha256') for k in refs if k in by_key}
        terms = {w.lower() for w in re.findall(r'[A-Za-z]{4,}',claim['text']) if w.lower() not in stops}
        candidates = []
        if len(papers) == 1 and len(pdfs) == 1 and None not in pdfs:
            for source in evidence:
                if not _valid_located_text(source) or source.get('paper_id') not in papers or source.get('pdf_sha256') not in pdfs:
                    continue
                for found in re.finditer(r'(?i)\bin\s+a\s+collection\s+of\s+(\d[\d,]*)\s+([a-z]+)\b',source['text']):
                    if found.group(1).replace(',', '') != count or found.group(2).lower() != noun:
                        continue
                    context = source['text'][max(0,found.start()-180):min(len(source['text']),found.end()+180)]
                    overlap = terms & {w.lower() for w in re.findall(r'[A-Za-z]{4,}',context)}
                    if len(overlap) >= 2:
                        candidates.append((len(overlap),source,found.group(0)))
        candidates.sort(key=lambda x:(-x[0],x[1]['chunk_id']))
        if not candidates or (len(candidates)>1 and candidates[0][0] == candidates[1][0]):
            unknown.append(claim['key']); continue
        _, source, quote = candidates[0]
        before = copy.deepcopy(claim)
        punctuation = '.' if claim['text'].rstrip().endswith('.') else ''
        claim['text'] = claim['text'].rstrip().removesuffix('.') + ' (' + quote + ')' + punctuation
        ref = {'paper_id':source['paper_id'],'chunk_id':source['chunk_id']}
        if ref not in claim['citations']:
            claim['citations'].append(ref)
        numeric = check_claim_numbers([claim],evidence)
        if any(r['status'] in {'missing','invalid_reference'} for r in numeric):
            claim.clear(); claim.update(before); unknown.append(claim['key']); continue
        added.append({'claim_key':claim['key'], **ref, 'source_quote':quote})
    return updated, {'added':added,'unknown':unknown}


def _expand_located_context(evidence, by_id, limit=8):
    """Add uniquely matched captions and nearby text as separately cited items.

    Expansion is a candidate aid, not a claim of semantic support. Caption
    metadata is surfaced only when a single exact independently located text
    item matches the same paper, PDF and page. Neighbors must be contiguous in
    the source index and geometrically adjacent in the same column and page.
    """
    corpus = list(by_id.values())
    selected = {(e.get('paper_id'), e.get('chunk_id')) for e in evidence}
    captions = {}
    for item in corpus:
        if item.get('docling_label') == 'caption' and _valid_located_text(item):
            key = (item.get('paper_id'), item.get('pdf_sha256'), item.get('page'), item['text'].strip())
            captions.setdefault(key, []).append(item)
    additions = []
    for item in evidence:
        if len(additions) >= limit or item.get('evidence_type') not in {'figure', 'table'}:
            continue
        caption, pdf = item.get('caption'), item.get('pdf_sha256')
        if not isinstance(caption, str) or not re.fullmatch(r'[0-9a-f]{64}', str(pdf or '')):
            continue
        matches = captions.get((item.get('paper_id'), pdf, item.get('page'), caption.strip()), [])
        linked = _linked_table_caption(item, by_id)
        if linked is not None:
            matches = [linked]
        if len(matches) == 1:
            source = matches[0]
            key = (source['paper_id'], source['chunk_id'])
            if key not in selected:
                additions.append(source)
                selected.add(key)
    positions = {item.get('chunk_id'): i for i, item in enumerate(corpus)}
    for item in evidence:
        if len(additions) >= limit or not _valid_located_text(item) or item.get('docling_label') == 'caption':
            continue
        idx = positions.get(item.get('chunk_id'))
        trusted = by_id.get(item.get('chunk_id'))
        if idx is None or not isinstance(trusted, dict) or any(trusted.get(k) != item.get(k) for k in (
                'paper_id', 'page', 'bbox', 'page_size', 'pdf_sha256', 'text')):
            continue
        if not re.fullmatch(r'[0-9a-f]{64}', str(item.get('pdf_sha256') or '')):
            continue
        for neighbor_idx in (idx - 1, idx + 1):
            if len(additions) >= limit or not 0 <= neighbor_idx < len(corpus):
                continue
            other = corpus[neighbor_idx]
            key = (other.get('paper_id'), other.get('chunk_id'))
            if (key in selected or not _valid_located_text(other) or other.get('docling_label') == 'caption'
                    or any(other.get(k) != item.get(k) for k in ('paper_id', 'page', 'page_size', 'pdf_sha256',
                                                                    'coordinate_system', 'page_rotation'))):
                continue
            a, b = item['bbox'], other['bbox']
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            gap = max(a[1], b[1]) - min(a[3], b[3])
            if overlap >= .8 * min(a[2]-a[0], b[2]-b[0]) and 0 <= gap <= 12:
                additions.append(other)
                selected.add(key)
    return evidence + additions, [item['chunk_id'] for item in additions]


def _frontmatter_candidates(question, paper_id, by_id):
    """Surface independently located front matter for explicit title/author requests."""
    if (not paper_id or not re.search(r'标题|题目|title', question, re.I)
            or not re.search(r'作者|署名|author', question, re.I)):
        return []
    rows = [e for e in by_id.values() if e.get('paper_id') == paper_id and e.get('page') == 1
            and _valid_located_text(e) and re.fullmatch(r'[0-9a-f]{64}', str(e.get('pdf_sha256') or ''))]
    titles = [e for e in rows if isinstance(e.get('title'), str) and len(e['title']) >= 12
              and e.get('text', '').strip().casefold() == e['title'].strip().casefold()]
    if len(titles) != 1:
        return []
    title = titles[0]
    below = [e for e in rows if e is not title and e.get('pdf_sha256') == title['pdf_sha256']
             and e.get('page_size') == title.get('page_size')
             and e.get('coordinate_system') == title.get('coordinate_system')
             and e.get('page_rotation') == title.get('page_rotation')
             and 0 <= e['bbox'][1] - title['bbox'][3] <= 80
             and min(e['bbox'][2], title['bbox'][2]) > max(e['bbox'][0], title['bbox'][0])]
    below.sort(key=lambda e:e['bbox'][1])
    # A candidate is not asserted to be an author list; the model must verify it.
    if not below or (len(below) > 1 and below[1]['bbox'][1] - below[0]['bbox'][1] < 3):
        return []
    return [title, below[0]]


def _validate_repaired_claims(original, repaired, evidence):
    """Only add available same-paper text citations; never rewrite claims."""
    if not isinstance(repaired, list) or len(original) != len(repaired):
        raise ValueError('Citation repair must preserve claim count')
    if len({c.get('key') for c in original}) != len(original):
        raise ValueError('Citation repair requires unique claim keys')
    by_key = {(e['paper_id'], e['chunk_id']):e for e in evidence}
    changed = False
    for before, after in zip(original, repaired):
        if before['key'] != after.get('key') or before['text'] != after.get('text'):
            raise ValueError('Citation repair changed claim identity or text')
        old = {(r['paper_id'], r['chunk_id']) for r in before['citations']}
        new = {(r['paper_id'], r['chunk_id']) for r in after.get('citations', [])}
        if len(new) != len(after.get('citations', [])) or not old <= new:
            raise ValueError('Citation repair removed or duplicated a citation')
        for key in new - old:
            source = by_key.get(key)
            if (source is None or source.get('evidence_type') not in (None, 'text')
                    or key[0] not in {paper for paper,_ in old}
                    or (source.get('pdf_sha256') and not _valid_located_text(source))):
                raise ValueError('Citation repair added unavailable or unlocated evidence')
        changed |= bool(new - old)
    validate_citations(repaired, evidence)
    if not changed:
        raise ValueError('Citation repair did not add evidence')


def _private_visual_trace(trace):
    """Project a multimodal trace onto safe metadata only; never persist content."""
    import hashlib
    def hashed(value):
        return hashlib.sha256(str(value).encode('utf-8')).hexdigest()
    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None
    def fields(obj, keys):
        return {k: obj[k] for k in keys if k in obj}
    out = fields(trace, ('status', 'commit_status', 'started_at', 'ended_at', 'mode', 'rewrite',
                         'reflect', 'compact', 'memory_strategy', 'memory_backend', 'degraded', 'query_mode'))
    if trace.get('budget_exhausted') in {'turn_deadline_exhausted', 'turn_model_requests_exhausted',
            'turn_subqueries_exhausted', 'turn_candidates_exhausted', 'turn_rerank_exhausted',
            'turn_retry_budget_exhausted'}:
        out['budget_exhausted'] = trace['budget_exhausted']
    for key in ('session_id', 'turn_id', 'case_id', 'run_id'):
        if trace.get(key) is not None:
            out[key] = hashed(trace[key])
    for key in ('corpus_hash',):
        value = trace.get(key)
        if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value):
            out[key] = value
    out['privacy'] = 'visual_metadata_only_v1'
    out['stages'] = []
    for stage in trace.get('stages', []):
        if not isinstance(stage, dict) or stage.get('stage') not in STAGES:
            continue
        safe = fields(stage, ('stage', 'attempt', 'status', 'started_at', 'ended_at', 'elapsed_ms', 'degraded'))
        if stage.get('invocation_id'):
            safe['invocation_id'] = hashed(stage['invocation_id'])
        if stage.get('stage') in {'Retrieve', 'Rerank'}:
            source = stage.get('output') or {}
            allowed = ('candidate_ids',) if stage['stage'] == 'Retrieve' else ('evidence_ids', 'associated_ids', 'context_ids')
            ids = {}
            for key in allowed:
                value = source.get(key)
                if isinstance(value, list):
                    ids[key] = [v for v in value if isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}', v)]
            if stage['stage'] == 'Rerank' and isinstance(source.get('ranked_refs'), list):
                ids['ranked_refs'] = [ref for ref in source['ranked_refs'] if isinstance(ref, dict)
                    and set(ref) == {'paper_id', 'chunk_id'} and all(isinstance(ref[k], str)
                    and re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}', ref[k]) for k in ref)]
            if stage['stage'] == 'Rerank' and isinstance(source.get('fused_refs'), list):
                ids['fused_refs'] = [ref for ref in source['fused_refs'] if isinstance(ref, dict)
                    and set(ref) == {'paper_id', 'chunk_id'} and all(isinstance(ref[k], str)
                    and re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}', ref[k]) for k in ref)]
            if isinstance(source.get('routes'), list):
                routes = []
                for route in source['routes'][:24]:
                    if not isinstance(route, dict) or route.get('route') not in _TRACE_ROUTES:
                        continue
                    refs = route.get('refs')
                    if not isinstance(refs, list):
                        continue
                    refs = [ref for ref in refs[:60] if isinstance(ref, dict) and set(ref) == {'paper_id', 'chunk_id'}
                            and all(isinstance(ref[k], str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}', ref[k]) for k in ref)]
                    row = {'route':route['route'], 'refs':refs}
                    if route.get('status') in {'ok', 'empty', 'failed'}:
                        row['status'] = route['status']
                    if route.get('status') == 'failed' and re.fullmatch(r'[A-Za-z0-9_]{1,80}', str(route.get('error_type', ''))):
                        row['error_type'] = route['error_type']
                    routes.append(row)
                ids['routes'] = routes
            safe['output'] = ids
        if type(stage.get('visual_image_count')) is int:
            safe['visual_image_count'] = stage['visual_image_count']
        visual = stage.get('visual_retrieval')
        if isinstance(visual, dict):
            safe['visual_retrieval'] = {k: number(visual[k]) for k in ('candidate_count', 'caption_candidate_count') if k in visual and number(visual[k]) is not None}
            timing = visual.get('timing_ms')
            if isinstance(timing, dict):
                safe['visual_retrieval']['timing_ms'] = {k:number(v) for k,v in timing.items() if k in {'visual_retrieve'} and number(v) is not None}
            elif number(timing) is not None:
                safe['visual_retrieval']['timing_ms'] = timing
        citation = stage.get('citation_validation')
        if isinstance(citation, dict):
            safe['citation_validation'] = {k: citation[k] for k in ('id_valid', 'source_consistent') if type(citation.get(k)) is bool}
            if citation.get('semantic_support') == 'not_evaluated':
                safe['citation_validation']['semantic_support'] = 'not_evaluated'
        numeric = stage.get('numeric_validation')
        if isinstance(numeric, dict):
            safe['numeric_validation'] = {k:numeric[k] for k in ('missing_count','visual_unknown_count')
                                          if type(numeric.get(k)) is int and numeric[k] >= 0}
        audit = stage.get('claim_audit')
        if isinstance(audit, dict):
            safe['claim_audit'] = {k: audit[k] for k in ('status','claim_count','image_count')
                                   if k in audit and ((k == 'status' and audit[k] in {'supported','rejected','unavailable','disagreement_unknown','unsupported','unknown'})
                                   or (k != 'status' and type(audit[k]) is int and audit[k] >= 0))}
        out['stages'].append(safe)
    out['api_calls'] = []
    for call in trace.get('api_calls', []):
        if not isinstance(call, dict):
            continue
        safe = fields(call, ('stage', 'status', 'attempt', 'max_tokens', 'image_count', 'repair', 'started_at',
                             'ended_at', 'elapsed_ms', 'http_status', 'usage_complete', 'finish_reason'))
        for key in ('invocation_id', 'request_id'):
            if call.get(key):
                safe[key] = hashed(call[key])
        if re.fullmatch(r'[0-9a-f]{64}', str(call.get('request_hash', ''))):
            safe['request_hash'] = call['request_hash']
        for key in ('model_requested', 'model_returned'):
            value = call.get(key)
            if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9._:/+@-]{1,120}', value):
                safe[key] = value
        for key in ('usage', 'usage_raw'):
            value = call.get(key)
            if isinstance(value, dict):
                safe[key] = {k:v for k,v in value.items() if k in {'input_tokens','output_tokens','cache_read_input_tokens','cache_creation_input_tokens','prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens'} and type(v) is int and v >= 0}
                for detail in ('prompt_tokens_details','completion_tokens_details'):
                    if isinstance(value.get(detail), dict):
                        safe[key][detail] = {k:v for k,v in value[detail].items() if k in {'cached_tokens','reasoning_tokens'} and type(v) is int and v >= 0}
        out['api_calls'].append(safe)
    budget = trace.get('budget')
    if isinstance(budget, dict):
        out['budget'] = {}
        for key in ('used', 'limits'):
            if isinstance(budget.get(key), dict):
                out['budget'][key] = {k:number(v) for k,v in budget[key].items() if k in {'seconds','model_requests','subqueries','candidates','rerank'} and number(v) is not None}
        if number(budget.get('elapsed_seconds')) is not None:
            out['budget']['elapsed_seconds'] = budget['elapsed_seconds']
    return out


@dataclass
class PaperAgentState:
    session_id: str
    messages: list = field(default_factory=list)
    turn_count: int = 0
    memory: SessionMemory = None
    committed_turns: dict = field(default_factory=dict)
    archive_ids: list = field(default_factory=list)
    memory_namespace: str | None = None

    def __post_init__(self):
        safe_id(self.session_id)
        if self.memory is None:
            self.memory = SessionMemory(self.session_id)

    @classmethod
    def load(cls, root, session_id):
        import json
        snapshot = Path(root)/f"{safe_id(session_id)}.state.json"
        if snapshot.exists():
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            if data["session_id"] != session_id:
                raise ValueError("Session identity mismatch")
            data["memory"] = SessionMemory.from_snapshot(data["memory"])
            return cls(**data)
        path = Path(root)/f"{safe_id(session_id)}.history.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(session_id, history.get("messages", []), history.get("turn_count", 0),
                   SessionMemory.load(root, session_id))

    def save(self, root):
        # A single atomic snapshot is authoritative, avoiding split history/memory commits.
        write_json(Path(root)/f"{self.session_id}.state.json", asdict(self))


class TurnTrace:
    def __init__(self, session_id, question, private_visual=False):
        self.private_visual = private_visual
        self.value = {"turn_id": uuid.uuid4().hex, "session_id": session_id, "question": question,
                      "status": "running", "stages": [], "started_at": time.time()}

    @contextmanager
    def stage(self, name, inputs=None, attempt=1):
        if current_budget.get():
            current_budget.get().remaining()
        entry = {"stage": name, "attempt": attempt, "input": inputs, "status": "running",
                 "invocation_id": uuid.uuid4().hex, "started_at": time.time()}
        token = invocation.set({k:self.value[k] for k in ('turn_id','session_id','case_id','run_id') if k in self.value} | {'invocation_id':entry['invocation_id']})
        self.value["stages"].append(entry)
        started = time.perf_counter()
        try:
            yield entry
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            entry["elapsed_ms"] = (time.perf_counter()-started)*1000
            entry['ended_at'] = time.time()
            invocation.reset(token)

    def finish(self, root):
        self.value['ended_at'] = time.time()
        visited = {s["stage"] for s in self.value["stages"]}
        for name in STAGES:
            if name not in visited:
                self.value["stages"].append({"stage": name, "status": "skipped",
                                              "reason": self.value["status"], "elapsed_ms": 0})
        from .observability import local_trace_path
        path=local_trace_path(root,self.value)
        if self.private_visual:
            self.value = _private_visual_trace(self.value)
        try:
            from .observability import trace_mapping
            self.value['observability_mapping'] = trace_mapping(self.value)
        except Exception as exc:
            self.value['observability_mapping'] = {'status':'unavailable','error_type':type(exc).__name__}
        write_json(path, self.value)
        return path


class PaperPipeline:
    def __init__(self, retriever, reasoner, trace_dir, memory_dir, mode="rerank", rewrite=True, reflect=True,
                 compact=True, memory_strategy="bounded", archive_recall=True, max_retrieval_attempts=2,
                 allow_degraded=True, memory_manager=None, exporter=None, decomposition=None, budget_limits=None,
                 visual_retriever=None, visual_root=None, visual_claim_check=False, visual_text_retriever=None,
                 clip_text_retriever=None,
                 claim_audit_sink=None):
        self.retriever, self.reasoner = retriever, reasoner
        self.visual_retriever = visual_retriever
        self.visual_text_retriever = visual_text_retriever
        self.clip_text_retriever = clip_text_retriever
        if clip_text_retriever is not None and visual_retriever is None:
            raise ValueError('Image-to-text retrieval requires a visual retriever')
        if clip_text_retriever is not None and clip_text_retriever.metadata.get('source_hash') != retriever.metadata.get('corpus_hash'):
            raise ValueError('CLIP and paper text corpus versions differ')
        if clip_text_retriever is not None and hasattr(clip_text_retriever, 'encoder'):
            if clip_text_retriever.encoder.provenance != visual_retriever.encoder.provenance:
                raise ValueError('CLIP visual and text model versions differ')
            text_pdfs = {(x.get('paper_id'), x.get('pdf_sha256')) for x in clip_text_retriever.sources}
            if any((x.get('paper_id'), x.get('pdf_sha256')) not in text_pdfs
                   or not re.fullmatch(r'[0-9a-f]{64}', str(x.get('pdf_sha256') or ''))
                   for x in visual_retriever.evidence):
                raise ValueError('Visual and CLIP text PDF provenance differs')
        if visual_text_retriever is not None and visual_retriever is None:
            raise ValueError('BGE visual text retrieval requires the visual evidence index')
        self.visual_root = Path(visual_root or visual_retriever.image_root).resolve() if visual_retriever is not None else None
        self.visual_claim_check = bool(visual_claim_check)
        self.claim_audit_sink = claim_audit_sink
        if self.visual_claim_check and visual_retriever is None:
            raise ValueError("Visual claim check requires a visual retriever")
        if visual_retriever is not None and not reflect:
            raise ValueError("Visual evidence requires reflection")
        self.trace_dir, self.memory_dir = Path(trace_dir), Path(memory_dir)
        self.mode, self.rewrite, self.reflect = mode, rewrite, reflect
        self.compact = compact
        if memory_strategy not in {"bounded", "semantic"}:
            raise ValueError("Unknown memory strategy")
        self.memory_strategy = memory_strategy
        self.archive_recall = archive_recall
        if not isinstance(max_retrieval_attempts, int) or not 1 <= max_retrieval_attempts <= 3:
            raise ValueError('Retrieval attempts must be between 1 and 3')
        self.max_retrieval_attempts = max_retrieval_attempts
        self.allow_degraded = allow_degraded
        self.memory_manager = memory_manager
        self.exporter = exporter or Exporter.from_env()
        self.decomposition = (os.getenv('PAPER_DECOMPOSITION','false').lower() == 'true') if decomposition is None else bool(decomposition)
        defaults = ({'seconds':240,'model_requests':8,'subqueries':18,'candidates':320,'rerank':180}
                    if visual_text_retriever is not None or clip_text_retriever is not None else
                    {'seconds':180,'model_requests':8,'subqueries':6,'candidates':90,'rerank':60})
        self.budget_limits = budget_limits if budget_limits is not None else {
            name:(float if name=='seconds' else int)(os.getenv('PAPER_TURN_'+name.upper(),str(default)))
            for name,default in defaults.items()}
        if memory_manager is None and os.getenv("MEMORY_BACKEND", "legacy") == "three_layer":
            from .memory_manager import MemoryManager
            self.memory_manager = MemoryManager.from_env(self.memory_dir, getattr(retriever, "encoder", None))
        elif memory_manager is None:
            from .memory_manager import MemoryManager
            self.memory_manager = MemoryManager.local(self.memory_dir)

    def run_query(self, query, state, **kwargs):
        if not isinstance(query, PaperQuery):
            raise TypeError('Expected PaperQuery')
        return self.run(query.text, state, images=query.images, **kwargs)

    def run(self, question, state, turn_id=None, case_id=None, run_id=None, images=()):
        from .query_images import validate_query_image
        if not isinstance(question, str) or not isinstance(images, (list, tuple)) or len(images) > 2:
            raise ValueError('Provide text and at most two local query images')
        query_images = [validate_query_image(path) for path in images]
        if query_images and (self.visual_retriever is None or self.clip_text_retriever is None):
            raise ValueError('Image queries require both CLIP visual and CLIP text indices')
        pure_image = bool(query_images and not question.strip())
        if pure_image:
            question = 'Find paper figures and text related to the provided image and describe the matching evidence.'
        image_identity = digest([item['sha256'] for item in query_images]) if query_images else None
        budget = TurnBudget(**self.budget_limits)
        turn_id = safe_id(turn_id or uuid.uuid4().hex)
        if self.memory_manager:
            self.memory_manager.events = []
            try:
                self.memory_manager.load_working_memory(state, bounded=self.compact)
            except Exception as exc:
                failed = TurnTrace(state.session_id, question, private_visual=self.visual_retriever is not None)
                failed.value.update(turn_id=turn_id, case_id=case_id, run_id=run_id, status="error", commit_status="uncommitted",
                                    memory_events=self.memory_manager.events, error_type=type(exc).__name__)
                failed.finish(self.trace_dir)
                try:
                    self.exporter.export(failed.value)
                    self.exporter.flush()
                except Exception:
                    pass  # Telemetry cannot replace the original memory error.
                raise
        if turn_id in state.committed_turns:
            committed = state.committed_turns[turn_id]
            if committed["question"] != question or committed.get('image_identity') != image_identity:
                raise ValueError("turn_id already committed with a different question")
            return {**committed["result"], "replayed_commit": True}
        original = copy.deepcopy(state)
        if self.reasoner is None:
            raise RuntimeError("No reasoner configured; use offline search/eval commands")
        trace = TurnTrace(state.session_id, question, private_visual=self.visual_retriever is not None)
        trace.value.update(case_id=case_id, run_id=run_id)
        trace.value['query_mode'] = 'image' if pure_image else 'text_image' if query_images else 'text'
        if turn_id is not None:
            trace.value["turn_id"] = safe_id(turn_id)
        trace.value.update(corpus_hash=self.retriever.metadata.get("corpus_hash"), mode=self.mode,
                           rewrite=self.rewrite, reflect=self.reflect, compact=self.compact,
                           memory_strategy=self.memory_strategy,
                           memory_backend="three_layer" if self.memory_manager.episodic else "sqlite_local")
        call_start = len(getattr(self.reasoner, "calls", []))
        trace.value.update(degraded=False, degradation_reasons=[], commit_status="uncommitted")
        state.turn_count += 1
        answer = {"decision": "error", "claims": []}
        result = {}
        budget_token = current_budget.set(budget)
        try:
            with trace.stage("Guardrails", {"question": question}) as event:
                if not question.strip() or len(question) > 20000:
                    raise ValueError("Question must contain 1..20000 characters")
                if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in question):
                    raise ValueError("Question contains unsupported control characters")
                event["output"] = {"accepted": True}
            with trace.stage("NLU", {"question": question}) as event:
                recalled = self.memory_manager.retrieve_memories(state, question) if self.memory_manager and self.archive_recall else ContextArchive(self.memory_dir/"archives").search(
                    state.session_id, question, current_memory=state.memory) if self.archive_recall else []
                trace.value["archive_recall"] = recalled
                catalog = catalog_with_literal_aliases(self.retriever.by_id.values())
                explicit_scope = explicit_single_paper_prefix(question, catalog)
                corpus_language = _corpus_language(self.retriever.by_id.values(), explicit_scope)
                nlu = self.reasoner.structured("nlu", {"question": question,
                    "memory": asdict(state.memory), "recent_messages": state.messages[-8:] if self.compact else state.messages,
                    "paper_catalog": catalog, "historical_context_not_evidence": recalled,
                    "literal_single_paper_scope": explicit_scope,
                    "retrieval_corpus_language": corpus_language,
                    "query_language_instruction": "When retrieval_corpus_language is en or zh and the question is in another language, rewrite the retrieval query in that corpus language, preserving all meaning, constraints and paper scope. If unknown, do not assume a language. The user's original question remains authoritative for answering.",
                    "decomposition_enabled": self.decomposition,
                    "user_image_present": bool(query_images),
                    "user_image_scope_instruction": "The user supplied the image being asked about. Search the full corpus without assuming its paper identity; do not ask which paper solely because the image is not identified. Do not invent image contents." if query_images else None,
                    "retrieval_plan_instruction": 'When enabled and multiple evidence targets are needed, optionally provide 2-3 sub_questions within parent paper_ids; otherwise omit.'}) if not pure_image else {'intent':'search', 'standalone_query':question, 'paper_ids':None, 'needs_clarification':False, 'resolved_entities':{}}
                if nlu.get("intent") not in {"search", "fact", "explain", "compare", "out_of_scope"}:
                    raise ValueError("Invalid intent")
                if not isinstance(nlu.get("standalone_query"), str) or not nlu["standalone_query"].strip():
                    raise ValueError("Missing standalone query")
                if nlu.get("paper_ids") is not None:
                    known = {p["paper_id"] for p in catalog}
                    if not isinstance(nlu["paper_ids"], list):
                        raise ValueError("NLU returned invalid paper IDs")
                    if any(p not in known for p in nlu["paper_ids"]):
                        nlu = {**nlu, 'paper_ids':None, 'needs_clarification':True, 'clarification_reason':'other'}
                        event['scope_guard'] = 'unknown_paper_id'
                # Resolve only an image-referent ambiguity. Explicit paper or
                # historical scope remains subject to the normal guard.
                if (query_images and nlu.get('needs_clarification') and nlu.get('clarification_reason') == 'paper_reference'
                        and re.search(r'(?i)\b(?:this|the provided|the supplied) (?:image|picture|figure)\b|(?:这|该|此)(?:张|幅)?(?:图片|图像)', question)
                        and not _figure_refs(question)
                        and scope_guard(question, catalog, None, state, question, {}) is None):
                    nlu = {**nlu, 'paper_ids':None, 'standalone_query':question,
                           'resolved_entities':{}, 'needs_clarification':False}
                    event['image_reference_resolved'] = True
                digest_probe = (_paper_artifact_digest_question(question) and explicit_scope is not None
                    and self.reflect and nlu.get('intent') == 'out_of_scope'
                    and nlu.get('out_of_scope_reason') not in {'not_paper_question', 'other'}
                    and not nlu.get('needs_clarification'))
                if digest_probe and nlu.get('paper_ids') is None:
                    nlu = {**nlu, 'paper_ids':[explicit_scope]}
                scope_issue = None if pure_image else scope_guard(question, catalog, nlu.get('paper_ids'), state,
                                          nlu.get('standalone_query'),nlu.get('resolved_entities'))
                if scope_issue:
                    nlu = {**nlu, 'needs_clarification':True}
                    event['scope_guard'] = scope_issue
                elif explicit_scope and nlu.get('paper_ids') == [explicit_scope]:
                    if nlu.get('needs_clarification') and nlu.get('clarification_reason') == 'paper_reference':
                        nlu = {**nlu, 'needs_clarification':False}
                        event['explicit_scope_resolved'] = True
                    if digest_probe:
                        # Existence must be checked from retrieved evidence; no
                        # hash scan or out-of-scope classification proves absence.
                        nlu = {**nlu, 'intent':'fact'}
                        event['artifact_evidence_route'] = True
                elif (nlu.get('needs_clarification') and nlu.get('clarification_reason') == 'paper_reference'
                      and (local_order := current_ordered_references(question, catalog))
                      and set(nlu.get('paper_ids') or []) == set(local_order.values())):
                    nlu = {**nlu, 'needs_clarification':False}
                    event['explicit_order_resolved'] = True
                if unknown_literal_metadata_id(question, catalog):
                    nlu = {**nlu, 'needs_clarification':True, 'clarification_reason':'other'}
                    event['scope_guard'] = 'unknown_paper_id'
                event["output"] = nlu
                if self.archive_recall:
                    extra = self.memory_manager.retrieve_memories(state, nlu["standalone_query"], nlu.get("paper_ids")) if self.memory_manager else ContextArchive(self.memory_dir/"archives").search(
                        state.session_id, nlu["standalone_query"], current_memory=state.memory,
                        paper_ids=nlu.get("paper_ids"))
                    merged, seen, used = [], set(), 0
                    # The manager has ranked the resolved-query candidates across all sources.
                    # Pre-NLU scores belong to a different query and must not displace them.
                    for record in (extra if self.memory_manager else recalled + extra):
                        key = (record["kind"], record.get("turn_id"), digest(record["content"]))
                        size = len(json.dumps(record, ensure_ascii=False))
                        if key not in seen and len(merged) < 6 and used + size <= 5000:
                            merged.append(record)
                            seen.add(key)
                            used += size
                    recalled = merged
                    trace.value["archive_recall"] = recalled
            evidence = []
            raw_evidence = []
            decision = "refuse" if nlu["intent"] == "out_of_scope" else "clarify" if nlu.get("needs_clarification") else "search"
            if (decision == 'search' and self.visual_text_retriever is not None
                    and _ambiguous_bare_figure(question, nlu.get('paper_ids'), self.visual_retriever.evidence)):
                decision = 'clarify'
            constraint_reason = None
            has_data_condition = (exclusion_kind(question, catalog) == 'non_paper' or
                                  exclusive_kind(question, catalog) == 'non_paper')
            if decision == 'search' and has_data_condition and not self.reflect:
                # A keyword filter cannot prove that a paper's results satisfy
                # temporal/data conditions. Never bypass evidence reflection.
                decision = 'clarify'
                constraint_reason = 'constraint_verification_unavailable'
            query = nlu["standalone_query"] if self.rewrite else question
            # The NLU's rewritten query may omit a time/data exclusion. Preserve
            # the user's original constraint for both retrieval modes.
            if has_data_condition and question not in query:
                query = question + '\n' + query
            if self.visual_text_retriever is not None:
                query_variants = _safe_query_variants(question, query)
                if len(query_variants) == 1 and query.strip() != question.strip():
                    # Do not search a model rewrite that invents another figure.
                    query = question
            else:
                query_variants = [query]
            goals = plan(question, nlu, self.decomposition and self.rewrite)
            coverage = {g['id']:[] for g in goals}
            trace.value['retrieval_plan'] = goals
            image_vectors = None
            if query_images and decision == 'search':
                # One encoding per user image, shared by both CLIP indices and retries.
                image_vectors = self.visual_retriever.encoder.encode_images([item['path'] for item in query_images])
                for item in query_images:
                    validate_query_image(item['path'], item['sha256'])
            if decision == "search":
                for attempt in range(1, self.max_retrieval_attempts + 1):
                    with trace.stage("Retrieve", {"query": query}, attempt) as event:
                        # Separate spans reflect real component work, not seven LLM calls.
                        retrieval_mode = "hybrid" if self.mode == "rerank" else self.mode
                        targets = goals if attempt == 1 else [g for g in goals if g['id'] in verdict.get('missing_goal_ids',[])]
                        if not targets:
                            targets = [{'id':'repair','query':query,'paper_ids':nlu.get('paper_ids')}]
                        combined = {}
                        text_lanes = []
                        route_trace = []
                        dense_rescue_refs = []
                        allowance = min(30, budget.available('candidates'))
                        if allowance < len(targets):
                            raise BudgetExceeded('turn_candidates_exhausted')
                        for target in targets:
                            target_query = target['query'] if attempt == 1 and len(goals)>1 else query
                            if attempt > 1:
                                target_query = question+'\n'+nlu['standalone_query']+'\n'+query
                            if self.visual_text_retriever is not None and not _figure_refs(target_query) <= _figure_refs(question):
                                target_query = question
                            cap = max(1, allowance//len(targets))
                            searches = ([target_query] if self.visual_text_retriever is None else
                                        list(dict.fromkeys(([question] + query_variants + [target_query]))))
                            if pure_image:
                                searches = []
                            # A decomposition may contain additional targets, but every
                            # query remains constrained to the target paper scope.
                            for q_index, current_query in enumerate(searches):
                                result = None
                                route_failed = False
                                for route in ([retrieval_mode,'bm25','dense'] if self.allow_degraded and retrieval_mode=='hybrid' else [retrieval_mode]):
                                    budget.take('subqueries')
                                    try:
                                        result = self.retriever.search(current_query, mode=route, top_k=cap, candidate_k=cap, paper_ids=target['paper_ids'])
                                        budget.remaining()
                                        budget.take('candidates',len(result['hits'][:cap]))
                                        if route != retrieval_mode:
                                            event.update(degraded=True, reason='retrieval_route_failed', fallback=route)
                                        break
                                    except BudgetExceeded:
                                        raise
                                    except Exception as exc:
                                        if route == 'dense' or retrieval_mode != 'hybrid' or not self.allow_degraded:
                                            if not query_images:
                                                raise
                                            result = {'hits':[]}
                                            route_failed = True
                                            event.update(degraded=True, reason='retrieval_route_failed', error_type=type(exc).__name__)
                                            route_trace.append({'route':'text_original' if q_index == 0 else 'text_rewrite',
                                                                'status':'failed', 'error_type':type(exc).__name__, 'refs':[]})
                                            break
                                lane = result['hits'][:cap]
                                text_lanes.append(lane)
                                if not route_failed:
                                    route_trace.append({'route':'text_original' if q_index == 0 and (self.visual_text_retriever is not None or query_images) else 'text_rewrite',
                                                        'status':'ok' if lane else 'empty',
                                                        'refs':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in lane]})
                                if self.visual_text_retriever is not None:
                                    suffix = '_original' if q_index == 0 else '_rewrite'
                                    for channel in ('bm25','dense'):
                                        refs = result.get('channel_refs', {}).get(channel, [])
                                        if refs:
                                            route_trace.append({'route':'text_'+channel+suffix, 'refs':refs})
                                            if channel == 'dense':
                                                dense_rescue_refs.append((target['id'],target['paper_ids'],refs))
                                for c in lane:
                                    if target['paper_ids'] is not None and c['paper_id'] not in target['paper_ids']:
                                        raise ValueError('Retriever returned out-of-scope evidence')
                                    combined[(c['paper_id'],c['chunk_id'])] = c
                                    ref = {'paper_id':c['paper_id'],'chunk_id':c['chunk_id']}
                                    if target['id'] in coverage and ref not in coverage[target['id']]:
                                        coverage[target['id']].append(ref)
                        merged_text = (_fuse_ranked_lanes(text_lanes) if self.visual_text_retriever is not None else list(combined.values()))
                        # Preserve a small number of novel dense candidates that
                        # hybrid truncation discarded; rerank them with the same
                        # model and final five-item limit. This is not an extra
                        # fusion vote or a claim of improved final recall.
                        if self.visual_text_retriever is not None:
                            seen = {(c['paper_id'],c['chunk_id']) for c in merged_text}
                            for goal_id, scope, refs in dense_rescue_refs:
                                added = []
                                for ref in refs:
                                    key = (ref.get('paper_id'),ref.get('chunk_id'))
                                    if key in seen:
                                        continue
                                    source = self.retriever.by_id.get(key[1])
                                    if source is None or source.get('paper_id') != key[0] or (scope is not None and key[0] not in scope):
                                        raise ValueError('Untrusted dense candidate')
                                    budget.take('candidates')
                                    merged_text.append(source)
                                    seen.add(key)
                                    added.append({'paper_id':key[0],'chunk_id':key[1]})
                                    if goal_id in coverage and added[-1] not in coverage[goal_id]:
                                        coverage[goal_id].append(added[-1])
                                    if len(added) == 3:
                                        break
                                route_trace.append({'route':'text_dense_rescue', 'refs':added})
                        result = {'hits':merged_text,'candidate_ids':[c['chunk_id'] for c in merged_text]}
                        event['candidate_coverage_not_semantic_support'] = copy.deepcopy(coverage)
                        event["output"] = {"candidate_ids": result["candidate_ids"],
                                           'candidate_refs':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in merged_text],
                                           'routes':route_trace if self.visual_text_retriever is not None or query_images else []}
                    with trace.stage("Rerank", {"candidate_ids": result["candidate_ids"]}, attempt) as event:
                        candidates = result["hits"]
                        figure_number = _requested_figure(question)
                        if self.visual_text_retriever is not None and len(_figure_refs(question)) != 1:
                            figure_number = None
                        # Text indexes may also contain captions from visual entities.
                        # Never let those bypass the visual index, image checks or figure scope.
                        filtered = []
                        for item in candidates:
                            if item.get('evidence_type') in {'figure', 'table'}:
                                if self.visual_retriever is None:
                                    continue
                                from .visual_evidence import validate_image
                                from .visual_retrieval import _number
                                if figure_number is not None and _number(item.get('figure_number')) != _number(figure_number):
                                    continue
                                if hasattr(self.visual_retriever, 'evidence'):
                                    source = next((e for e in self.visual_retriever.evidence if (e['paper_id'], e['chunk_id']) == (item['paper_id'], item['chunk_id'])), None)
                                    if source is None:
                                        continue
                                    _merge_evidence_entities([item], [source])
                                validate_image(item, self.visual_root)
                            filtered.append(item)
                        candidates = filtered
                        rerank_routes = []
                        if self.mode == "rerank" and candidates:
                            try:
                                rank_queries = query_variants if self.visual_text_retriever is not None else [query]
                                budget.take('rerank',len(candidates)*len(rank_queries))
                                if self.retriever.reranker is None:
                                    raise ValueError("Cross-encoder required")
                                # A visual entity may have structured table cells in
                                # text; a text reranker is allowed to score its caption
                                # only, never those cells as a visual signal.
                                text_inputs = [(c.get('caption') or '') if c.get('evidence_type') in {'figure','table'}
                                               else c['text'] for c in candidates]
                                ranked = []
                                for i, rank_query in enumerate(rank_queries):
                                    scores = self.retriever.reranker.score(rank_query, text_inputs)
                                    if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
                                        raise ValueError("Invalid reranker scores")
                                    lane = [c for _, c in sorted(zip(scores, candidates), key=lambda v: (-v[0], v[1]["chunk_id"]))]
                                    ranked.append(lane)
                                    rerank_routes.append({'route':'rerank_original' if i == 0 and self.visual_text_retriever is not None else 'rerank_rewrite',
                                                          'refs':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in lane]})
                                candidates = _fuse_ranked_lanes(ranked) if self.visual_text_retriever is not None else ranked[0]
                            except BudgetExceeded:
                                raise
                            except Exception as exc:
                                if not self.allow_degraded:
                                    raise
                                event.update(degraded=True, reason="reranker_unavailable", error_type=type(exc).__name__)
                        # Text, figure captions and images retain independent ranks.
                        if self.visual_retriever is not None and not pure_image:
                            from .visual_evidence import validate_image
                            from .visual_retrieval import _number
                            if query_images:
                                # Keep independently located captions as text citations.
                                mapped_captions, caption_mapping = [], {}
                            else:
                                candidates, mapped_captions, caption_mapping = _map_exact_caption_entities(
                                    candidates, getattr(self.visual_retriever, 'evidence', []))
                            # A mixed legacy text index may contain image entities
                            # directly. They belong to the caption lane, not both
                            # the text and caption lanes.
                            mapped_captions += [c for c in candidates if c.get('evidence_type') in {'figure','table'}]
                            candidates = [c for c in candidates if c.get('evidence_type') not in {'figure','table'}]
                            if figure_number is not None:
                                mapped_captions = [c for c in mapped_captions if _number(c.get('figure_number')) == _number(figure_number)]
                            allowed_mapped = {(c['paper_id'], c['chunk_id']) for c in mapped_captions}
                            for goal, refs in coverage.items():
                                unique = []
                                for ref in refs:
                                    source_key = (ref['paper_id'], ref['chunk_id'])
                                    target_key = caption_mapping.get(source_key, source_key)
                                    if source_key in caption_mapping and target_key not in allowed_mapped:
                                        continue
                                    target = {'paper_id':target_key[0], 'chunk_id':target_key[1]}
                                    if target not in unique:
                                        unique.append(target)
                                coverage[goal] = unique
                            if self.visual_text_retriever is None:
                                budget.take('subqueries', 2)
                                def visual_branch(name, operation):
                                    try:
                                        response = operation()
                                        status = 'ok' if response['hits'] else 'empty'
                                    except Exception as exc:
                                        if not query_images or isinstance(exc, BudgetExceeded):
                                            raise
                                        response, status = {'hits':[]}, 'failed'
                                        event.update(degraded=True, reason='image_retrieval_branch_failed')
                                    if query_images:
                                        rerank_routes.append({'route':name, 'status':status,
                                            'refs':[{'paper_id':h['paper_id'],'chunk_id':h['chunk_id']} for h in response['hits']]})
                                    return response
                                visual_result = visual_branch('clip_original', lambda: self.visual_retriever.search_text(query, top_k=5, paper_ids=nlu.get('paper_ids'), figure_number=figure_number))
                                caption_result = visual_branch('caption_bm25_original', lambda: self.visual_retriever.search_caption(query, top_k=5, paper_ids=nlu.get('paper_ids'), figure_number=figure_number))
                                visual_candidates = visual_result['hits']
                                caption_candidates = list(_merge_evidence_entities(mapped_captions, caption_result['hits']).values())
                                budget.take('candidates', len(visual_candidates) + len(caption_result['hits']))
                                extra_visual_lanes = []
                            else:
                                budget.take('subqueries', 4 * len(query_variants))
                                # Each route keeps 20 before fusion; final evidence
                                # remains capped separately at five and three images.
                                clips = self.visual_retriever.search_texts(query_variants, top_k=20, paper_ids=nlu.get('paper_ids'), figure_number=figure_number)
                                lexical = [self.visual_retriever.search_caption(q, top_k=20, paper_ids=nlu.get('paper_ids'), figure_number=figure_number) for q in query_variants]
                                sem_caps = [self.visual_text_retriever.search(q, kind='caption', top_k=20, paper_ids=nlu.get('paper_ids'), figure_number=figure_number) for q in query_variants]
                                sem_tables = [self.visual_text_retriever.search(q, kind='table', top_k=20, paper_ids=nlu.get('paper_ids'), figure_number=figure_number) for q in query_variants]
                                all_routes = [('clip', clips), ('caption_bm25', lexical), ('caption_bge', sem_caps), ('table_bge', sem_tables)]
                                for route_name, responses in all_routes:
                                    for variant, response in enumerate(responses):
                                        budget.take('candidates', len(response['hits']))
                                        rerank_routes.append({'route':route_name + ('_original' if variant == 0 else '_rewrite'),
                                                              'refs':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in response['hits']]})
                                visual_result = clips[0]
                                visual_candidates = _fuse_ranked_lanes([r['hits'] for r in clips])
                                caption_candidates = _fuse_ranked_lanes([mapped_captions] + [r['hits'] for r in lexical])
                                extra_visual_lanes = [_fuse_ranked_lanes([r['hits'] for r in sem_caps]),
                                                      _fuse_ranked_lanes([r['hits'] for r in sem_tables])]
                            for item in visual_candidates + caption_candidates + [c for lane in extra_visual_lanes for c in lane]:
                                if nlu.get('paper_ids') is not None and item['paper_id'] not in nlu['paper_ids']:
                                    raise ValueError('Visual retriever returned out-of-scope evidence')
                                if figure_number is not None and _number(item.get('figure_number')) != _number(figure_number):
                                    raise ValueError('Visual retriever returned evidence outside the requested figure')
                                validate_image(item, self.visual_root)
                            # Caption and image searches describe the same class
                            # of evidence. Fuse them first so two visual routes
                            # do not automatically outweigh the single text lane.
                            if self.visual_text_retriever is None:
                                candidates = (_interleave_evidence(candidates, caption_candidates, visual_candidates)
                                              if query_images else _fuse_text_visual_candidates(candidates, caption_candidates, visual_candidates))
                            else:
                                text_visual = _fuse_ranked_lanes([caption_candidates] + extra_visual_lanes)
                                combined_visual = _fuse_ranked_lanes([text_visual, visual_candidates])
                                candidates = (_interleave_evidence(candidates, combined_visual) if query_images
                                              else _fuse_ranked_lanes([candidates, combined_visual]))
                            event['visual_retrieval'] = {'candidate_count':len(visual_candidates), 'caption_candidate_count':len(caption_candidates), 'timing_ms':visual_result.get('timing_ms')}
                        if image_vectors is not None:
                            image_text_lanes, image_visual_lanes = [], []
                            for vector in image_vectors:
                                for route, dest, call in (
                                    ('image_to_text', image_text_lanes, lambda: self.clip_text_retriever.search_vector(vector.reshape(1, -1), 20, nlu.get('paper_ids'))),
                                    ('image_to_image', image_visual_lanes, lambda: self.visual_retriever._search(vector.reshape(1, -1), 20, nlu.get('paper_ids'), figure_number, time.perf_counter()))):
                                    try:
                                        response = call(); hits = response['hits']
                                        budget.take('candidates', len(hits))
                                        # Restore trusted full source; representations never become citations.
                                        if route == 'image_to_text':
                                            hits = [self.retriever.by_id[h['chunk_id']] for h in hits
                                                    if h['chunk_id'] in self.retriever.by_id and self.retriever.by_id[h['chunk_id']]['paper_id'] == h['paper_id']]
                                        dest.append(hits)
                                        rerank_routes.append({'route':route, 'status':'ok' if hits else 'empty',
                                            'refs':[{'paper_id':h['paper_id'],'chunk_id':h['chunk_id']} for h in hits]})
                                    except BudgetExceeded:
                                        raise
                                    except Exception as exc:
                                        event.update(degraded=True, reason='image_retrieval_branch_failed')
                                        rerank_routes.append({'route':route, 'status':'failed', 'error_type':type(exc).__name__, 'refs':[]})
                            candidates = _interleave_evidence(candidates, *image_text_lanes, *image_visual_lanes)
                        # Preserve the actual fused ranking separately from the
                        # bounded set sent to the model for evaluation.
                        metadata_scope = explicit_scope or (explicit_natural_metadata_scope(question, catalog, nlu.get('paper_ids')) if not scope_issue else None)
                        frontmatter = _frontmatter_candidates(question, metadata_scope, self.retriever.by_id)
                        if frontmatter:
                            candidates = list(_merge_evidence_entities(frontmatter, candidates).values())
                            rerank_routes.append({'route':'frontmatter_located', 'refs':[
                                {'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in frontmatter]})
                        ranked_refs = [{'paper_id': c['paper_id'], 'chunk_id': c['chunk_id']} for c in candidates[:5]]
                        # Retain first retrieval evidence when filling gaps on retry.
                        selected = candidates[:5]
                        if (self.visual_text_retriever is not None or query_images) and len(goals) == 1:
                            # Backfill beyond the first five when the model-image
                            # cap would otherwise leave fewer than five entities.
                            selected, image_count = [], 0
                            for candidate in candidates:
                                is_image = candidate.get('evidence_type') in {'figure','table'}
                                if is_image and image_count >= 3 - len(query_images):
                                    continue
                                selected.append(candidate)
                                image_count += int(is_image)
                                if len(selected) == 5:
                                    break
                        if len(goals)>1:
                            selected = []
                            for goal in goals:
                                selected += next(([c] for c in candidates if {'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} in coverage[goal['id']]), [])
                            selected = list({(c['paper_id'],c['chunk_id']):c for c in selected+candidates}.values())[:5]
                        evidence = list(_merge_evidence_entities(raw_evidence, selected).values())
                        # Never send unseen visual candidates to the reasoning model.
                        visuals_seen = 0
                        bounded = []
                        for item in evidence:
                            if item.get('evidence_type') in {'figure', 'table'}:
                                visuals_seen += 1
                                if visuals_seen > 3 - len(query_images):
                                    continue
                            bounded.append(item)
                        raw_evidence = bounded
                        evidence = bounded
                        associated_ids = []
                        context_ids = []
                        if self.visual_retriever is not None:
                            evidence, context_ids = _expand_located_context(evidence, self.retriever.by_id)
                            evidence, associated_ids = _expand_verified_associations(evidence, self.retriever.by_id)
                            budget.take('candidates', len(associated_ids) + len(context_ids))
                        event["output"] = {"evidence_ids": [c["chunk_id"] for c in evidence], 'associated_ids': associated_ids, 'context_ids':context_ids, 'ranked_refs': ranked_refs,
                                           'fused_refs':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in candidates[:20]],
                                           'routes':rerank_routes if self.visual_text_retriever is not None or query_images else []}
                    with trace.stage("Reflect", {"evidence_ids": [c["chunk_id"] for c in evidence]}, attempt) as event:
                        visual_images = query_images + [c for c in evidence if c.get('evidence_type') in {'figure', 'table'}]
                        reflect_payload = {"question": question, "evidence": evidence,
                            "resolved_question": nlu["standalone_query"],
                            "attempt": attempt, "intent": nlu["intent"],
                            "subgoals": goals, "candidate_coverage_not_semantic_support": {g:[ref for ref in refs if (ref['paper_id'],ref['chunk_id']) in {(c['paper_id'],c['chunk_id']) for c in evidence}] for g,refs in coverage.items()},
                            "historical_context_not_evidence": recalled,
                            "user_image_instruction": "User images are query inputs, not paper evidence or valid citations. Focus on evidence related to the image and question; ignore unrelated retrieved candidates. Cite only independently located paper evidence." if query_images else None}
                        if self.reflect:
                            verdict = (self.reasoner.structured("reflect", reflect_payload, images=visual_images)
                                       if visual_images else self.reasoner.structured("reflect", reflect_payload))
                        else:
                            verdict = {"decision": "answer"}
                        event['visual_image_count'] = len(visual_images)
                        decision = verdict.get("decision")
                        if decision not in {"answer", "retrieve_again", "clarify", "refuse"}:
                            raise ValueError("Invalid reflection decision")
                        if any(g not in coverage for g in verdict.get('missing_goal_ids',[])):
                            raise ValueError('Unknown reflection goal')
                        uncovered = any(not any((ref['paper_id'],ref['chunk_id']) in {(c['paper_id'],c['chunk_id']) for c in evidence} for ref in refs) for refs in coverage.values())
                        if decision == 'answer' and len(goals)>1 and (uncovered or verdict.get('missing_goal_ids') or not self.reflect):
                            decision='refuse'
                            verdict={**verdict,'decision':decision,'reason_code':'missing_target_evidence'}
                        if decision == "retrieve_again" and attempt == self.max_retrieval_attempts:
                            decision = "refuse"
                            verdict = {**verdict, "decision": decision, "reason_code": "retrieval_budget_exhausted"}
                        event["output"] = verdict
                    if decision != "retrieve_again":
                        break
                    query = verdict.get("next_query", "")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("Reflection must specify next_query")
            with trace.stage("Generate", {"decision": decision}) as event:
                local_audit_id, local_verdicts, local_recheck = None, None, None
                local_numeric, local_repair, local_conditions, local_caption = None, None, None, None
                local_context, local_completeness, local_question_conditions, local_collection = None, None, None, None
                if decision == "answer":
                    if not evidence:
                        decision = "refuse"
                        answer = {"decision": decision, "text": "语料中没有足够证据。", "claims": []}
                    else:
                        generate_payload = {"question": question,
                            "answer_language": _answer_language(question),
                            "resolved_question": nlu["standalone_query"], "evidence": evidence,
                            "subgoals": goals, "candidate_coverage_not_semantic_support": coverage,
                            "historical_context_not_evidence": recalled}
                        if query_images:
                            generate_payload['user_image_instruction'] = 'User images are query inputs, not paper evidence or valid citations. Answer the question using evidence relevant to the image; omit unrelated retrieved candidates. Cite only independently located paper evidence.'
                        visual_images = query_images + [c for c in evidence if c.get('evidence_type') in {'figure', 'table'}]
                        answer = (self.reasoner.structured("generate", generate_payload, images=visual_images)
                                  if visual_images else self.reasoner.structured("generate", generate_payload))
                        if self.claim_audit_sink is not None:
                            local_audit_id = self.claim_audit_sink.begin(case_id, turn_id, answer, evidence)
                        event['visual_image_count'] = len(visual_images)
                        if answer.get('decision') == 'refuse' and not answer.get('claims'):
                            decision = 'refuse'
                            answer = {'decision':'refuse', 'reason_code':'visual_or_text_evidence_insufficient',
                                      'text':'当前证据不足或图表无法辨认，无法可靠作答。', 'claims':[]}
                        elif not answer.get("claims"):
                            raise ValueError("Answer must include claims")
                        try:
                            validate_citations(answer["claims"], evidence)
                            trusted = {(e['paper_id'], e['chunk_id']):e for e in evidence}
                            for claim in answer['claims']:
                                for ref in claim['citations']:
                                    src = trusted[(ref['paper_id'], ref['chunk_id'])]
                                    if (src.get('pdf_sha256') and src.get('evidence_type') in (None, 'text')
                                            and not _valid_located_text(src)):
                                        raise ValueError('Cited PDF text has no valid location')
                        except ValueError:
                            event.update(degraded=True, reason="citation_validation_failed")
                            # Controlled refusal: invalid citations are never silently repaired.
                            decision = "refuse"
                            answer = {"decision": decision, "reason_code": "citation_validation_failed",
                                      "text": "当前证据无法支撑可靠引用，因此不给出结论。", "claims": []}
                        else:
                            local_caption = _attach_table_number_citations(answer['claims'], evidence,
                                getattr(self.visual_retriever, 'evidence', []) if self.visual_retriever else [])
                            if local_caption['unknown']:
                                decision = 'refuse'
                                answer = {'decision':'refuse', 'reason_code':'table_caption_unknown',
                                          'text':'表号的独立来源无法可靠核实。', 'claims':[]}
                            from .citation_guards import check_claim_numbers
                            local_numeric = check_claim_numbers(answer['claims'], evidence)
                            missing_keys = {row['claim_key'] for row in local_numeric if row['status'] == 'missing'}
                            if missing_keys and local_audit_id is not None and self.visual_claim_check:
                                repair_original = [copy.deepcopy(c) for c in answer['claims'] if c.get('key') in missing_keys]
                                local_repair = {'reason':'uncited_explicit_numeric_condition', 'status':'pending'}
                                try:
                                    # One explicit repair, only among already selected text evidence;
                                    # this is not an automatic transport retry.
                                    text_evidence = [e for e in evidence if e.get('evidence_type') in (None, 'text')]
                                    repair = self.reasoner.structured('repair_citations',
                                        {'question':question, 'claims':repair_original, 'evidence':text_evidence,
                                         'missing_numeric_conditions':[r for r in local_numeric if r['claim_key'] in missing_keys]})
                                    local_repair['response'] = copy.deepcopy(repair)
                                    repaired = repair.get('claims', [])
                                    _validate_repaired_claims(repair_original, repaired, evidence)
                                    replaced = {c['key']:c for c in repaired}
                                    answer['claims'] = [replaced.get(c['key'],c) for c in answer['claims']]
                                    local_numeric = check_claim_numbers(answer['claims'], evidence)
                                    if any(row['status'] in {'missing','invalid_reference'} for row in local_numeric):
                                        raise ValueError('Citation repair did not cover numeric conditions')
                                    local_repair['status'] = 'validated_pending_self_check'
                                except BudgetExceeded:
                                    raise
                                except Exception:
                                    local_repair['status'] = 'rejected'
                            if any(row['status'] in {'missing','invalid_reference'} for row in local_numeric):
                                decision = 'refuse'
                                answer = {'decision':'refuse', 'reason_code':'numeric_citation_incomplete',
                                          'text':'引用文本缺少结论中的数值或单位，无法可靠作答。', 'claims':[]}
                            if decision == 'answer':
                                from .citation_guards import check_claim_conditions
                                local_conditions = check_claim_conditions(answer['claims'], evidence)
                                context_keys = [r['claim_key'] for r in local_conditions if r['status'] == 'context_dependent']
                                eligible = [c for c in answer['claims'] if c.get('key') in context_keys]
                                if (context_keys and local_audit_id is not None and self.visual_claim_check
                                        and all(re.search(r'\bthis\s+effort\b', c['text'], re.I) for c in eligible)):
                                    local_context = {'status':'pending', 'claim_keys':context_keys}
                                    try:
                                        refs = {(r['paper_id'],r['chunk_id']) for c in eligible for r in c['citations']}
                                        cited = [e for e in evidence if (e['paper_id'],e['chunk_id']) in refs and _valid_located_text(e)]
                                        repair = self.reasoner.structured('repair_context', {'question':question,
                                            'claims':answer['claims'], 'repair_claim_keys':context_keys, 'evidence':cited})
                                        local_context['response'] = copy.deepcopy(repair)
                                        revised = _apply_context_repairs(answer['claims'], evidence, repair['repairs'], context_keys)
                                        conditions = check_claim_conditions(revised, evidence)
                                        numeric = check_claim_numbers(revised, evidence)
                                        if any(r['status'] != 'ok' for r in conditions) or any(r['status'] in {'missing','invalid_reference'} for r in numeric):
                                            raise ValueError('Context repair not independently checkable')
                                        answer['claims'], local_conditions, local_numeric = revised, conditions, numeric
                                        local_context['status'] = 'validated_pending_self_check'
                                    except BudgetExceeded:
                                        raise
                                    except Exception:
                                        local_context['status'] = 'rejected'
                                if any(row['status'] != 'ok' for row in local_conditions):
                                    decision = 'refuse'
                                    answer = {'decision':'refuse','reason_code':'claim_condition_unknown',
                                              'text':'声明的指代或数据条件无法由所引区域单独核实。','claims':[]}
                            if decision == 'answer':
                                from .citation_guards import check_question_collection_conditions
                                local_question_conditions = check_question_collection_conditions(question, answer['claims'])
                                if any(r['status'] != 'ok' for r in local_question_conditions) and local_audit_id is not None and self.visual_claim_check:
                                    answer['claims'], local_collection = _repair_collection_conditions(question, answer['claims'], evidence)
                                    local_question_conditions = check_question_collection_conditions(question, answer['claims'])
                                    local_numeric = check_claim_numbers(answer['claims'], evidence)
                                if any(r['status'] != 'ok' for r in local_question_conditions):
                                    decision = 'refuse'
                                    answer = {'decision':'refuse','reason_code':'question_condition_missing',
                                              'text':'答案缺少问题中明确的集合规模条件。','claims':[]}
                            event['numeric_validation'] = {
                                'missing_count':sum(row['status'] in {'missing','invalid_reference'} for row in local_numeric),
                                'visual_unknown_count':sum(row['status'] == 'unknown_visual' for row in local_numeric)}
                            # Location information comes from trusted retrieved evidence, never model output.
                            evidence_by_key = {(e['paper_id'], e['chunk_id']):e for e in evidence}
                            for claim in answer['claims']:
                                for cite in claim['citations']:
                                    source = evidence_by_key[(cite['paper_id'], cite['chunk_id'])]
                                    for field_name in ('evidence_type', 'docling_label', 'page', 'bbox', 'page_size', 'pdf_sha256', 'figure_number', 'coordinate_system', 'page_rotation', 'image_sha256',
                                                       'caption_association_status', 'caption_source_chunk_ids', 'caption_geometry'):
                                        if field_name in source:
                                            cite[field_name] = source[field_name]
                                    if isinstance(source.get('title'),str) and source['title'].strip():
                                        cite['source_metadata'] = {'title':source['title'], 'source':'local_index',
                                            'status':'unverified', 'scope':'paper_identity_not_cited_region'}
                                    from .visual_evidence import composite_provenance
                                    cite.update(composite_provenance(source))
                                    if source.get('image_path'):
                                        from .visual_evidence import validate_image
                                        cite['image_path'] = str(validate_image(source, self.visual_root))
                            event["citation_validation"] = {"id_valid": True, "source_consistent": True, "semantic_support": "not_evaluated"} if answer['claims'] else {'status':'no_claims'}
                            answer["decision"] = decision
                            if self.visual_claim_check and decision == 'answer' and answer['claims']:
                                claims = answer['claims']
                                keys = [claim.get('key') for claim in claims]
                                cited = {(ref['paper_id'], ref['chunk_id']) for claim in claims for ref in claim['citations']}
                                audit_evidence = [item for item in evidence if (item['paper_id'], item['chunk_id']) in cited]
                                audit_images = [item for item in audit_evidence if item.get('evidence_type') in {'figure','table'}]
                                audit_status = 'rejected'
                                if len(keys) == len(set(keys)) and all(isinstance(key, str) and key for key in keys):
                                    try:
                                        audit_claims = [{'key':claim['key'], 'text':claim['text'],
                                                         'citations':[{'paper_id':ref['paper_id'], 'chunk_id':ref['chunk_id']}
                                                                      for ref in claim['citations']]} for claim in claims]
                                        audit_payload = {'question': question, 'claims': audit_claims, 'evidence': audit_evidence}
                                        audit = (self.reasoner.structured('verify_claims', audit_payload, images=audit_images)
                                                 if audit_images else self.reasoner.structured('verify_claims', audit_payload))
                                        verdicts = audit.get('verdicts', [])
                                        local_verdicts = copy.deepcopy(verdicts)
                                        statuses = {v['claim_key']: v['status'] for v in verdicts}
                                        complete_verdicts = (len(verdicts) == len(keys) and len(statuses) == len(keys)
                                                             and set(statuses) == set(keys))
                                        if complete_verdicts and all(value == 'supported' for value in statuses.values()):
                                            audit_status = 'supported'
                                        elif (complete_verdicts and local_audit_id is not None
                                              and any(value == 'supported' for value in statuses.values())):
                                            # A disputed factual claim may be essential. Recheck it
                                            # once against its own sources; never silently drop it.
                                            disputed = [c for c in audit_claims if statuses[c['key']] != 'supported']
                                            refs = {(r['paper_id'],r['chunk_id']) for c in disputed for r in c['citations']}
                                            cited_disputed = [e for e in evidence if (e['paper_id'],e['chunk_id']) in refs]
                                            imgs = [e for e in cited_disputed if e.get('evidence_type') in {'figure','table'}]
                                            local_recheck = {'status':'pending', 'disputed_claim_keys':[c['key'] for c in disputed]}
                                            payload = {'question':question, 'claims':disputed, 'evidence':cited_disputed}
                                            checked = (self.reasoner.structured('verify_claims', payload, images=imgs)
                                                       if imgs else self.reasoner.structured('verify_claims', payload))
                                            repeated = checked.get('verdicts', [])
                                            local_recheck['verdicts'] = copy.deepcopy(repeated)
                                            repeated_map = {v['claim_key']:v['status'] for v in repeated}
                                            if (len(repeated) == len(disputed) and len(repeated_map) == len(disputed)
                                                    and set(repeated_map) == {c['key'] for c in disputed}):
                                                disagrees = any(repeated_map[k] != statuses[k] for k in repeated_map)
                                                audit_status = ('disagreement_unknown' if disagrees else
                                                    'unknown' if 'unknown' in repeated_map.values() else 'unsupported')
                                                local_recheck['status'] = audit_status
                                            else:
                                                audit_status = local_recheck['status'] = 'unavailable'
                                    except BudgetExceeded:
                                        raise
                                    except Exception:
                                        audit_status = 'unavailable'
                                        if local_recheck is not None:
                                            local_recheck['status'] = 'unavailable'
                                event['claim_audit'] = {'status':audit_status, 'claim_count':len(claims), 'image_count':len(audit_images)}
                                if audit_status not in {'supported', 'supported_after_recheck'}:
                                    decision = 'refuse'
                                    answer = {'decision':'refuse', 'reason_code':'claim_self_check_failed',
                                              'text':'引用证据无法充分核实全部陈述，暂不给出结论。', 'claims':[]}
                            if (decision == 'answer' and self.visual_claim_check and answer.get('claims')
                                    and re.search(r'阶段|步骤|\bstages?\b|\bsteps?\b', question, re.I)
                                    and re.search(r'三|各|分别|\b(?:three|each|multiple)\b|\d+', question, re.I)):
                                local_completeness = {'status':'pending'}
                                try:
                                    payload = {'question':question,
                                        'candidate_claims':answer['claims'], 'evidence':evidence,
                                        'instruction':'Judge whether these exact claims include all requested stage distinctions; do not fill omissions from evidence.'}
                                    completeness = (self.reasoner.structured('reflect', payload, images=visual_images)
                                                    if visual_images else self.reasoner.structured('reflect', payload))
                                    local_completeness = {'status':completeness.get('decision','unknown')}
                                    if completeness.get('decision') != 'answer':
                                        decision = 'refuse'
                                        answer = {'decision':'refuse','reason_code':'answer_incomplete',
                                                  'text':'现有陈述未完整覆盖所问阶段及其条件。','claims':[]}
                                except Exception:
                                    local_completeness = {'status':'unavailable'}
                                    decision = 'refuse'
                                    answer = {'decision':'refuse','reason_code':'answer_completeness_unknown',
                                              'text':'无法核实答案是否完整。','claims':[]}
                else:
                    answer = {"decision": decision, "claims": [], "text":
                        "请明确所指论文或比较对象。" if decision == "clarify" else "当前语料证据不足或请求超出论文库范围。"}
                    if constraint_reason:
                        answer.update(reason_code=constraint_reason,
                                      text='当前无法核实时间或数据限制，请提供可验证的条件或证据。')
                if local_audit_id is not None:
                    if local_context and local_context.get('status') == 'validated_pending_self_check':
                        local_context['status'] = 'accepted_after_self_check' if answer.get('decision') == 'answer' else 'rejected_after_self_check'
                    if local_repair and local_repair.get('status') == 'validated_pending_self_check':
                        local_repair['status'] = ('accepted_after_self_check' if answer.get('decision') == 'answer'
                                                  else 'rejected_after_self_check')
                    self.claim_audit_sink.finalize(local_audit_id, local_verdicts, answer, local_recheck,
                                                   local_repair, local_numeric, local_conditions, local_caption, local_context, local_completeness,
                                                   local_question_conditions, local_collection)
                event["output"] = answer
            with trace.stage("Memory", {"decision": answer["decision"]}) as event:
                if answer["decision"] == "answer":
                    state.memory.update(nlu.get("resolved_entities", {}), [])
                    state.memory.summary["discussed_papers"] = list(dict.fromkeys(
                        state.memory.summary.get("discussed_papers", []) + [c["paper_id"] for c in evidence]))
                else:
                    state.memory.unresolved = [question]
                state.messages.extend([{"role": "user", "content": question, "turn_id": turn_id},
                                       {"role": "assistant", "content": answer, "turn_id": turn_id}])
                if self.compact and not self.memory_manager:
                    if self.memory_strategy == "semantic":
                        from .compaction import semantic_compact_state
                        try:
                            event["compaction"] = semantic_compact_state(state, self.memory_dir/"archives", self.reasoner)
                        except Exception as exc:
                            event.update(degraded=True, reason="semantic_compaction_failed")
                            # Graceful fallback: semantic summary failure must not lose the turn.
                            event["compaction_fallback"] = {"from": "semantic", "to": "bounded",
                                                            "error": f"{type(exc).__name__}: {exc}"}
                            event["compaction"] = compact_state(state, self.memory_dir/"archives")
                    else:
                        event["compaction"] = compact_state(state, self.memory_dir/"archives")
                from .observability import local_trace_path
                trace_path = str(local_trace_path(self.trace_dir,trace.value))
                state.committed_turns[turn_id] = {"question": question, "image_identity":image_identity, "result": {"answer": answer, "trace_id": turn_id,
                                                                                   "trace_path": trace_path}}
                if self.memory_manager:
                    event["compaction"] = self.memory_manager.commit(state, question, answer, evidence, turn_id,
                        original.turn_count, self.compact, self.memory_strategy, self.reasoner)
                    event["memory_events"] = self.memory_manager.events
                    if any(e.get("degraded") for e in self.memory_manager.events):
                        event.update(degraded=True, reason="memory_projection_or_compaction_degraded")
                else:
                    state.save(self.memory_dir)
                trace.value["commit_status"] = "committed"
                event["output"] = asdict(state.memory)
            trace.value.update(status="completed", decision=answer["decision"], answer=answer)
            result.update(answer=answer, trace_id=trace.value["turn_id"], trace_path=trace_path)
            return result
        except Exception as exc:
            state.__dict__.update(original.__dict__)
            trace.value["status"] = "error"
            trace.value['error_type'] = type(exc).__name__
            if isinstance(exc, BudgetExceeded) and str(exc) in {
                    'turn_deadline_exhausted', 'turn_model_requests_exhausted', 'turn_subqueries_exhausted',
                    'turn_candidates_exhausted', 'turn_rerank_exhausted', 'turn_retry_budget_exhausted'}:
                trace.value['budget_exhausted'] = str(exc)
            raise
        finally:
            if self.memory_manager:
                trace.value["memory_events"] = self.memory_manager.events
            degraded = [s.get("reason") for s in trace.value["stages"] if s.get("degraded")]
            trace.value.update(degraded=bool(degraded), degradation_reasons=degraded)
            trace.value["api_calls"] = getattr(self.reasoner, "calls", [])[call_start:]
            trace.value['budget'] = {'used':budget.used,'elapsed_seconds':budget.clock()-budget.started,'limits':self.budget_limits}
            current_budget.reset(budget_token)
            try:
                trace.finish(self.trace_dir)
            except Exception as exc:
                # The SQLite transaction may already be durable. Report the
                # missing audit projection instead of masking a committed answer.
                if trace.value['commit_status'] == 'committed':
                    result.update(trace_status='write_failed', degraded=True)
                trace.value['trace_write_error_type'] = type(exc).__name__
            try:
                self.exporter.export(trace.value)
                self.exporter.flush()
            except Exception:
                pass  # Optional export cannot change committed business state.


def compact_state(state, archive_dir, max_facts=40, max_entities=20, max_messages=8):
    """Deterministic bounded working memory; archive evicted state, no LLM summary claim."""
    if len(state.memory.entities) <= max_entities and len(state.messages) <= max_messages:
        return {"compressed": False}
    before = len(json.dumps({"memory": asdict(state.memory), "messages": state.messages}, ensure_ascii=False))
    archive_id = ContextArchive(archive_dir).store({"memory": asdict(state.memory), "messages": state.messages,
                                                  "turn_count": state.turn_count, "commit_protocol": 1})
    state.archive_ids = list(dict.fromkeys(state.archive_ids + [archive_id]))
    state.memory.entities = dict(list(state.memory.entities.items())[-max_entities:]) if max_entities else {}
    state.messages = state.messages[-max_messages:] if max_messages else []
    after = len(json.dumps({"memory": asdict(state.memory), "messages": state.messages}, ensure_ascii=False))
    return {"compressed": True, "archive_id": archive_id, "method": "bounded_recent_state",
            "context_chars_before": before, "context_chars_after": after,
            "token_estimate_before": math.ceil(before/4), "token_estimate_after": math.ceil(after/4),
            "token_measurement": "character estimate; not API token usage",
            "retained_entities": list(state.memory.entities)}
