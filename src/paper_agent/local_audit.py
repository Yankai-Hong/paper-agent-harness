"""Local-only request audit. Never persist request bodies, headers, or images."""
import json
import math
import os
import re
import tempfile
import time
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
        if stage not in {'nlu', 'reflect', 'generate', 'verify_claims', 'memory_summary'}:
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
        if row['stage'] in {'generate', 'verify_claims'}:
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
        usage = record.get('usage_raw')
        if isinstance(usage, dict):
            row['usage_raw'] = {k:v for k,v in usage.items() if k in {
                'prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens'} and type(v) is int and v >= 0}
            for key in ('prompt_tokens_details','completion_tokens_details'):
                if isinstance(usage.get(key), dict):
                    row['usage_raw'][key] = {k:v for k,v in usage[key].items() if k in {'cached_tokens','reasoning_tokens'} and type(v) is int and v >= 0}
        _atomic_private(self.root / (request_id + '.json'), row)
