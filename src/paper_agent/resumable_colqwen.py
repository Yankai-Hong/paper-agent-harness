"""Crash-resumable, separately persisted ColQwen page-token collection."""
import json
import re
import hashlib
import os
import uuid
from pathlib import Path

import numpy as np

from .colqwen_store import _digest, _matrix, maxsim
from .storage import digest


class ResumableColQwenStore:
    """Commit individual pages only after every token has been read back.

    Incomplete per-page intent records are retained. Replaying a page requires
    exactly the same paper, source image and vector fingerprint; orphan writes
    never count as completed pages or permit a seal.
    """

    def __init__(self, directory, contract, *, create=False):
        import chromadb
        from chromadb.config import Settings

        self.root = Path(directory)
        self.contract = dict(contract)
        if (not self.contract.get('model_revision') or not self.contract.get('source_hash')
                or type(self.contract.get('dimension')) is not int or self.contract['dimension'] < 1):
            raise ValueError('Incomplete ColQwen model/source contract')
        self.pending = self.root / 'pending'
        self.committed = self.root / 'pages'
        self.references = self.root / 'references'
        self.final = self.root / 'sealed.json'
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            self.pending.mkdir()
            self.committed.mkdir()
            self.references.mkdir()
            self._write_new(self.root / 'contract.json', {'schema': 1, 'contract': self.contract, 'signature': digest(self.contract)})
        else:
            data = json.loads((self.root / 'contract.json').read_text(encoding='utf-8'))
            if data != {'schema': 1, 'contract': self.contract, 'signature': digest(self.contract)}:
                raise ValueError('ColQwen collection contract mismatch')
            if not self.references.exists() and not self.final.exists():
                self.references.mkdir()
        self.client = chromadb.PersistentClient(path=str(self.root / 'chroma'),
                                                 settings=Settings(anonymized_telemetry=False, allow_reset=False))
        if create:
            self.collection = self.client.create_collection('paper_colqwen_resumable', embedding_function=None,
                metadata={'hnsw:space': 'cosine', 'contract_hash': digest(self.contract)})
        else:
            self.collection = self.client.get_collection('paper_colqwen_resumable', embedding_function=None)
        if self.collection.metadata != {'hnsw:space': 'cosine', 'contract_hash': digest(self.contract)}:
            raise ValueError('Chroma collection contract mismatch')
        if self.final.exists():
            sealed = json.loads(self.final.read_text(encoding='utf-8'))
            records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.committed.glob('*.json'))]
            if (sealed.get('schema') != 1 or sealed.get('contract_hash') != digest(self.contract)
                    or sealed.get('pages') != len(records) or sealed.get('tokens') != self.collection.count()
                    or sealed.get('page_records_hash') != digest(records)):
                raise ValueError('Sealed ColQwen page manifest mismatch')
            if sealed.get('patch_provenance_sha256'):
                provenance = self.root / 'patch-provenance.jsonl'
                if not provenance.is_file() or self._file_hash(provenance) != sealed['patch_provenance_sha256']:
                    raise ValueError('Sealed patch provenance mismatch')

    @staticmethod
    def _write_new(path, data):
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename refuses an existing destination. On POSIX, use an
        # exclusive hard link because rename can replace an existing file.
        # If publication fails, preserve the unique temporary file.
        if os.name == 'nt':
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()

    @staticmethod
    def _file_hash(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    def _reference(self, record):
        marker = self.references / (record['page_id'] + '.json')
        if not marker.is_file():
            return None
        meta = json.loads(marker.read_text(encoding='utf-8'))
        filename = meta.get('file', '')
        if (Path(filename).name != filename or not filename.startswith(record['page_id'] + '-')
                or not filename.endswith('.npy')):
            raise ValueError('Invalid page reference path')
        path = self.references / filename
        if not path.is_file() or self._file_hash(path) != meta.get('file_sha256'):
            raise ValueError('Page reference file checksum mismatch')
        array = _matrix(np.load(path, allow_pickle=False), self.contract['dimension'])
        if len(array) != record['count'] or _digest(array) != record['vectors_sha256']:
            raise ValueError('Page reference vectors mismatch')
        return array

    def _ensure_reference(self, record, array):
        if self._reference(record) is not None:
            return
        filename = record['page_id'] + '-' + uuid.uuid4().hex + '.npy'
        path = self.references / filename
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        self._write_new(self.references / (record['page_id'] + '.json'),
                        {'file': filename, 'file_sha256': self._file_hash(path)})

    def add_page(self, page_id, paper_id, image_sha256, vectors):
        if self.final.exists():
            raise ValueError('Sealed collection is immutable')
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+', page_id) or not paper_id
                or not re.fullmatch(r'[0-9a-f]{64}', image_sha256)):
            raise ValueError('Invalid page identity or source image hash')
        array = _matrix(vectors, self.contract['dimension'])
        record = {'page_id': page_id, 'paper_id': paper_id, 'image_sha256': image_sha256,
                  'count': len(array), 'vectors_sha256': _digest(array)}
        pending = self.pending / (page_id + '.json')
        committed = self.committed / (page_id + '.json')
        if pending.exists():
            if json.loads(pending.read_text(encoding='utf-8')) != record:
                raise ValueError('Incomplete page cannot be resumed with different source or vectors')
        else:
            self._write_new(pending, record)
        self._ensure_reference(record, array)
        if committed.exists():
            existing = json.loads(committed.read_text(encoding='utf-8'))
            if any(existing.get(key) != value for key, value in record.items()):
                raise ValueError('Committed page conflicts with resumed input')
            self.get_page(page_id)
            return
        for start in range(0, len(array), 512):
            end = min(start + 512, len(array))
            ids = [f'{page_id}:{i:06d}' for i in range(start, end)]
            meta = [{'page_id': page_id, 'paper_id': paper_id, 'image_sha256': image_sha256, 'token_index': i}
                    for i in range(start, end)]
            self.collection.upsert(ids=ids, embeddings=array[start:end].tolist(), metadatas=meta)
        stored = self._read_record(record, expected=array)
        # The backend can return tiny float32 changes during index updates.
        # A hashed input reference bounds the numerical difference on each read.
        self._write_new(committed, {**record, 'stored_vectors_sha256': _digest(stored)})

    def _read_record(self, record, expected=None):
        vectors = []
        page_id = record['page_id']
        for start in range(0, record['count'], 512):
            ids = [f'{page_id}:{i:06d}' for i in range(start, min(start + 512, record['count']))]
            result = self.collection.get(ids=ids, include=['embeddings', 'metadatas'])
            lookup = dict(zip(result['ids'], zip(result['embeddings'], result['metadatas'])))
            if set(lookup) != set(ids):
                raise ValueError('Missing page token vectors')
            for token_id in ids:
                vector, meta = lookup[token_id]
                expected_metadata = {'page_id': page_id, 'paper_id': record['paper_id'], 'image_sha256': record['image_sha256'],
                                     'token_index': int(token_id.rsplit(':', 1)[1])}
                if meta != expected_metadata:
                    raise ValueError('Page token provenance mismatch')
                vectors.append(vector)
        array = _matrix(vectors, self.contract['dimension'])
        stored_hash = record.get('stored_vectors_sha256')
        reference = self._reference(record)
        if reference is not None and not np.allclose(array, reference, atol=1e-6, rtol=0):
            raise ValueError('Page token vector checksum mismatch')
        if expected is not None and not np.allclose(array, expected, atol=1e-6, rtol=0):
            raise ValueError('Chroma changed page token vectors beyond tolerance')
        if reference is None and expected is None and _digest(array) != (stored_hash or record['vectors_sha256']):
            raise ValueError('Page token vector checksum mismatch')
        return array

    def get_page(self, page_id):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', page_id):
            raise ValueError('Invalid page ID')
        path = self.committed / (page_id + '.json')
        if not path.is_file():
            raise ValueError('Page has not been committed')
        return self._read_record(json.loads(path.read_text(encoding='utf-8')))

    def seal(self, *, patch_provenance_sha256=None):
        if self.final.exists():
            raise ValueError('Collection is already sealed')
        pending = {p.stem for p in self.pending.glob('*.json')}
        committed = {p.stem for p in self.committed.glob('*.json')}
        if not committed or pending != committed:
            raise ValueError('Cannot seal with missing or unfinished pages')
        records = [json.loads((self.committed / (page + '.json')).read_text(encoding='utf-8')) for page in sorted(committed)]
        if self.collection.count() != sum(r['count'] for r in records):
            raise ValueError('Unexpected token count or orphan writes')
        for record in records:
            self._read_record(record)
        final = {'schema': 1, 'contract_hash': digest(self.contract), 'pages': len(records),
                 'tokens': self.collection.count(), 'page_records_hash': digest(records)}
        if patch_provenance_sha256 is not None:
            path = self.root / 'patch-provenance.jsonl'
            if not path.is_file() or self._file_hash(path) != patch_provenance_sha256:
                raise ValueError('Patch provenance file hash mismatch')
            final['patch_provenance_sha256'] = patch_provenance_sha256
        self._write_new(self.final, final)

    def rerank(self, query, page_ids):
        if not self.final.is_file():
            raise ValueError('Cannot rerank an unsealed collection')
        if len(page_ids) != len(set(page_ids)):
            raise ValueError('Duplicate candidate page')
        query = _matrix(query, self.contract['dimension'])
        return sorted(((page, maxsim(query, self.get_page(page))) for page in page_ids), key=lambda x: (-x[1], x[0]))
