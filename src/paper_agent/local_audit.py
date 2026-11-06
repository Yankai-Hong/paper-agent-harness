"""Local-only request audit. Never persist request bodies, headers, or images."""
import json
import math
import os
import re
import tempfile
import time
import uuid
import copy
from pathlib import Path

from .storage import safe_id


def _atomic_private(path, value):
    fd, name = tempfile.mkstemp(prefix='.audit-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class LocalRequestAudit:
    """One new run directory, with a durable reservation before every request.

    Response text is retained only for Generate and Verify and only after
    fail-closed checks against secrets and image/base64 data. No request body,
    headers, document evidence, or input images are written by this class.
    """
    def __init__(self, root, secret):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.secret = secret
        self.records = {}

    def reserve(self, record, case_id):
        request_id = safe_id(record['request_id'])
        if request_id in self.records:
            raise ValueError('Duplicate request ID')
        stage = record.get('stage')
        if stage not in {'nlu', 'reflect', 'generate', 'verify_claims', 'repair_citations', 'repair_context', 'memory_summary'}:
            raise ValueError('Unexpected model stage')
        row = {'request_id':request_id, 'case_id':safe_id(case_id), 'stage':stage,
               'status':'reserved_unknown', 'started_at':time.time(),
               'image_count':record.get('image_count', 0), 'response_status':'not_received'}
        path = self.root / (request_id + '.json')
        if path.exists():
            raise ValueError('Request record already exists')
        _atomic_private(path, row)
        self.records[request_id] = row

    def capture_response(self, request_id, response, body):
        row = self.records[request_id]
        row['response_status'] = 'received'
        if row['stage'] in {'generate', 'verify_claims', 'repair_citations', 'repair_context'}:
            try:
                content = response['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                content = None
            if isinstance(content, str):
                image_strings = []
                for message in body.get('messages', []):
                    blocks = message.get('content')
                    if isinstance(blocks, list):
                        for block in blocks:
                            url = block.get('image_url', {}).get('url', '') if isinstance(block, dict) else ''
                            if isinstance(url, str) and url.startswith('data:image/') and ';base64,' in url:
                                image_strings.append(url.split(';base64,', 1)[1])
                unsafe = (len(content) > 100000 or (self.secret and self.secret in content)
                          or re.search(r'data:image/[^;]+;base64,', content, re.I)
                          or re.search(r'[A-Za-z0-9+/]{256,}={0,2}', content)
                          or any(value and value in content for value in image_strings))
                if unsafe:
                    row['response_status'] = 'withheld_unsafe_content'
                else:
                    row['response_text'] = content
            else:
                row['response_status'] = 'missing_response_text'
        _atomic_private(self.root / (request_id + '.json'), row)

    def finalize(self, record):
        request_id = record['request_id']
        row = self.records.get(request_id)
        if row is None:
            return
        for key in ('status', 'elapsed_ms', 'http_status', 'transport_error', 'error_type',
                    'finish_reason', 'usage_complete', 'model_requested', 'model_returned'):
            value = record.get(key)
            if isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value)):
                row[key] = value
        # Keep only schema field names and validators, never raw response text,
        # validation messages, provider reasoning, or arbitrary model fields.
        allowed_paths = set('intent standalone_query needs_clarification paper_ids resolved_entities sub_questions id query target decision missing_goal_ids next_query reason claims key text citations paper_id chunk_id overview clarification_reason out_of_scope_reason'.split())
        path = record.get('schema_error_path')
        if isinstance(path, list):
            row['schema_error_path'] = [part if type(part) is int or part in allowed_paths else '[redacted]'
                                        for part in path if isinstance(part, (str, int)) and not isinstance(part, bool)]
        missing = record.get('schema_missing_fields')
        if isinstance(missing, list):
            row['schema_missing_fields'] = [field for field in missing if isinstance(field, str) and field in allowed_paths]
        validators = {'type','required','enum','additionalProperties','minLength','maxLength',
                      'minItems','maxItems','uniqueItems','items','properties','anyOf','oneOf'}
        if record.get('schema_validator') in validators:
            row['schema_validator'] = record['schema_validator']
        usage = record.get('usage_raw')
        if isinstance(usage, dict):
            row['usage_raw'] = {k:v for k,v in usage.items() if k in {
                'prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens'} and type(v) is int and v >= 0}
            for key in ('prompt_tokens_details','completion_tokens_details'):
                if isinstance(usage.get(key), dict):
                    row['usage_raw'][key] = {k:v for k,v in usage[key].items() if k in {'cached_tokens','reasoning_tokens'} and type(v) is int and v >= 0}
        _atomic_private(self.root / (request_id + '.json'), row)


class LocalClaimAudit:
    """Optional local-only ledger of generated claims and their cited locations.

    It never persists evidence text, questions, request bodies, headers or
    images. A generated snapshot is durable before self-check, so refusal or
    interruption cannot silently erase the original claims.
    """
    def __init__(self, root, secret):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.secret = secret
        self.records = {}

    def _safe(self, value):
        raw = json.dumps(value, ensure_ascii=False)
        return not (len(raw) > 200000 or (self.secret and self.secret in raw)
                    or re.search(r'data:image/[^;]+;base64,', raw, re.I)
                    or re.search(r'[A-Za-z0-9+/]{256,}={0,2}', raw))

    def begin(self, case_id, turn_id, answer, evidence):
        key = uuid.uuid4().hex
        refs = {(cite.get('paper_id'), cite.get('chunk_id'))
                for claim in answer.get('claims', []) if isinstance(claim, dict)
                for cite in claim.get('citations', []) if isinstance(cite, dict)}
        fields = ('paper_id', 'chunk_id', 'evidence_type', 'docling_label', 'page',
                  'bbox', 'page_size', 'coordinate_system', 'page_rotation',
                  'pdf_sha256', 'figure_number', 'image_sha256', 'caption_association_status',
                  'caption_source_chunk_ids', 'caption_geometry')
        sources = [{k:item[k] for k in fields if k in item} for item in evidence
                   if (item.get('paper_id'), item.get('chunk_id')) in refs]
        from .visual_evidence import composite_provenance
        for located in sources:
            source = next(item for item in evidence if (item.get('paper_id'),item.get('chunk_id')) == (located.get('paper_id'),located.get('chunk_id')))
            located.update(composite_provenance(source))
            if isinstance(source.get('title'),str) and source['title'].strip():
                located['source_metadata'] = {'title':source['title'], 'source':'local_index',
                    'status':'unverified', 'scope':'paper_identity_not_cited_region'}
        record = {'audit_id':key, 'case_id':case_id, 'turn_id':turn_id,
                  'status':'generated_pending', 'generated':copy.deepcopy(answer), 'cited_regions':sources}
        if not self._safe(record):
            record = {'audit_id':key, 'status':'withheld_unsafe_content'}
        _atomic_private(self.root / (key + '.json'), record)
        self.records[key] = record
        return None if record['status'] == 'withheld_unsafe_content' else key

    def finalize(self, key, verdicts, final_answer, recheck=None, citation_repair=None, numeric_check=None, condition_check=None, caption_repair=None, context_repair=None, completeness=None, question_conditions=None, collection_repair=None):
        record = self.records[key]
        if record['status'] == 'withheld_unsafe_content':
            return
        update = {'status':'complete', 'verdicts':verdicts, 'recheck':recheck,
                  'citation_repair':citation_repair, 'numeric_check':numeric_check, 'condition_check':condition_check,
                  'caption_repair':caption_repair, 'final_answer':copy.deepcopy(final_answer),
                  'context_repair':context_repair, 'completeness':completeness,
                  'question_conditions':question_conditions,
                  'collection_repair':collection_repair,
                  'final_decision':final_answer.get('decision'),
                  'reason_code':final_answer.get('reason_code')}
        if not self._safe(update):
            update = {'status':'withheld_unsafe_verdict'}
        record.update(update)
        _atomic_private(self.root / (key + '.json'), record)
