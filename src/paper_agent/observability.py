"""Optional metadata-only export. Local trace remains the authoritative record."""
from contextvars import ContextVar
import hashlib
import os
import queue
import re
import threading
import math
import json

invocation = ContextVar('paper_agent_invocation', default={})

def trace_mapping(trace):
    """Deterministic OTel 128-bit ID; length-delimited identity avoids collisions.

    Missing run_id denotes ordinary interaction, scoped by session and stable
    turn_id. Explicit run IDs distinguish even identical session/case/turns.
    """
    run_id = trace.get('run_id')
    if run_id is not None and not isinstance(run_id,str):
        run_id = str(run_id)
    identity=['paper-agent-trace-v2', 'run' if run_id is not None else 'interactive',
              run_id, trace['session_id'], trace['turn_id']]
    encoded=json.dumps(identity,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    trace_id=hashlib.sha256(encoded).hexdigest()[:32]
    if trace_id == '0'*32: trace_id='0'*31+'1'
    return {'scheme':'paper-agent-trace-v2','langfuse_trace_id':trace_id,
            'scope':'run' if run_id is not None else 'interactive'}

def score_id(trace_id, source, name):
    identity=['paper-agent-score-v2',trace_id,source,name]
    return hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()[:32]

def local_trace_path(root, trace):
    """Preserve interactive path; isolate experiment runs without exposing run IDs."""
    from pathlib import Path
    from .storage import safe_id
    root=Path(root)
    run_id=trace.get('run_id')
    if run_id is None:
        return root/(safe_id(trace['turn_id'])+'.json')
    scope=hashlib.sha256(json.dumps(['paper-agent-run-v1',str(run_id)],ensure_ascii=False,
                                   separators=(',',':')).encode()).hexdigest()[:32]
    return root/'runs'/scope/(safe_id(trace['turn_id'])+'.json')

def _identifier_hash(items):
    # Identifiers are pseudonymized; no queries, paper text or preferences leave.
    if not isinstance(items,list): return None
    normalized=sorted(str(x) for x in items if isinstance(x,str))
    if not normalized: return None
    return hashlib.sha256(json.dumps(normalized,separators=(',',':')).encode()).hexdigest()

def trace_summary(trace):
    events=trace.get('memory_events') or []
    recall=trace.get('archive_recall') or []
    reasons={'retrieval_route_failed','reranker_unavailable','citation_validation_failed',
             'memory_projection_or_compaction_degraded','semantic_compaction_failed'}
    memory_stages=[s for s in trace.get('stages',[]) if isinstance(s,dict) and s.get('stage')=='Memory']
    compactions=[s.get('compaction') for s in memory_stages if isinstance(s.get('compaction'),dict)]
    operations={'redis_read','redis_write','chroma_read','chroma_outbox','chroma_consistency','chroma_similarity',
                'file_projection','projection_recovery','semantic_compaction'}
    return {'memory_recall_count':len(recall),
            'memory_long_term_recall_count':sum(isinstance(r,dict) and r.get('retrieval_source')=='sqlite' for r in recall),
            'memory_episode_recall_count':sum(isinstance(r,dict) and r.get('kind')=='episode' for r in recall),
            'memory_archive_recall_count':sum(isinstance(r,dict) and r.get('retrieval_source')=='archive' for r in recall),
            'memory_repair_count':sum(len(e.get('episode_ids',[])) for e in events if isinstance(e,dict) and e.get('operation')=='chroma_repair_queued' and isinstance(e.get('episode_ids',[]),list)),
            'memory_compaction_count':sum(c.get('compressed') is True for c in compactions),
            'memory_degraded_count':sum(isinstance(e,dict) and bool(e.get('degraded')) for e in events),
            'memory_degradation_operations':sorted({e['operation'] for e in events if isinstance(e,dict) and e.get('degraded') and e.get('operation') in operations}),
            'degradation_reasons':[r for r in trace.get('degradation_reasons',[]) if r in reasons][:8]}

def stage_summary(stage):
    name=stage.get('stage'); inp=stage.get('input') or {}; out=stage.get('output') or {}
    result={}
    if name=='Retrieve':
        ids=out.get('candidate_ids',[])
        result['candidate_count']=len(ids) if isinstance(ids,list) else 0
        result['candidate_ids_hash']=_identifier_hash(ids)
        coverage=stage.get('candidate_coverage_not_semantic_support') or {}
        if isinstance(coverage,dict):
            result['subgoal_count']=len(coverage)
            result['subgoals_with_candidates']=sum(bool(v) for v in coverage.values())
    if name=='Rerank':
        candidates=inp.get('candidate_ids',[]); evidence=out.get('evidence_ids',[])
        result.update(rerank_candidate_count=len(candidates) if isinstance(candidates,list) else 0,
                      evidence_count=len(evidence) if isinstance(evidence,list) else 0,
                      evidence_ids_hash=_identifier_hash(evidence))
    if stage.get('reason') in {'retrieval_route_failed','reranker_unavailable','citation_validation_failed',
                              'memory_projection_or_compaction_degraded','semantic_compaction_failed'}:
        result['degradation_reason']=stage['reason']
    return {k:v for k,v in result.items() if v is not None}

def attach_scores(path, scores, source, exporter=None):
    """Attach rule/Judge results locally, retaining unknown rather than assigning zero."""
    import json
    from pathlib import Path
    from .storage import write_json
    if source not in {'rule','same_model_judge'}:
        raise ValueError('Score source must be explicit')
    path=Path(path)
    if not path.exists():
        return
    trace=json.loads(path.read_text(encoding='utf-8'))
    mapping=trace.get('observability_mapping')
    mapping_valid = isinstance(mapping,dict) and mapping == trace_mapping(trace)
    if not mapping_valid:
        # Old traces may already exist under v1 on the server; guessing would
        # attach their score to a different or colliding trace.
        trace['score_export_status']='withheld_missing_or_mismatched_mapping'
    existing={(s['source'],s['name']):s for s in trace.get('scores',[])}
    for name,value in scores.items():
        if value is None or isinstance(value,(bool,int)) or (isinstance(value,float) and math.isfinite(value)):
            previous=existing.get((source,name),{})
            record={'source':source,'name':name,'value':int(value) if isinstance(value,bool) else value,
                    'status':'unknown' if value is None else 'known'}
            if value is None and (previous.get('value') is not None or previous.get('remote_status')=='unknown_not_retracted'):
                record['remote_status']='unknown_not_retracted'
            existing[(source,name)]=record
    trace['scores']=list(existing.values())
    write_json(path,trace)
    if exporter and mapping_valid:
        try:
            exporter.export_scores(trace)
            exporter.flush()  # Bounded wait; queued scores can still be lost on process exit.
        except Exception:
            pass  # Score export is optional and never changes the evaluation result.

FIELDS = frozenset('stage attempt status started_at ended_at elapsed_ms repair retry_reason retry_delay_seconds error_type error_category circuit_state circuit_after http_status usage_complete model_requested model_returned commit_status degraded decision source value'.split())
SCORE_NAMES = frozenset('execution_success task_success pipeline_success behavior_match schema_valid stage_flow_valid citation_structure_valid memory_committed nlu_correct reflect_correct retrieval_effective generate_valid degraded relevance factual_correctness completeness evidence_support citation_correctness appropriate_abstention overall'.split())
USAGE_NAMES = frozenset('input_tokens output_tokens cache_read_input_tokens cache_creation_input_tokens'.split())

def metadata(value):
    result = {}
    for key in FIELDS:
        if key not in value:
            continue
        item = value.get(key)
        if isinstance(item, (bool, int)) or (isinstance(item,float) and math.isfinite(item)) or item is None:
            result[key] = item
        elif isinstance(item, str):
            # Names are identifiers, not free-form text. Any unexpected field is
            # withheld entirely rather than relying on a best-effort redactor.
            if re.fullmatch(r'[A-Za-z0-9._:/+@-]{1,120}',item):
                result[key] = re.sub(r'(?i)(bearer\S*|sk-\S+|[\w.+-]+@[\w.-]+)', '[redacted]', item)
    for key in ('session_id', 'turn_id', 'case_id', 'run_id', 'invocation_id', 'request_id'):
        if value.get(key):
            result[key] = hashlib.sha256(str(value[key]).encode()).hexdigest()
    for key in ('dataset_hash', 'evaluation_config_hash'):
        item=value.get(key)
        if isinstance(item,str) and re.fullmatch(r'[0-9a-f]{64}',item):
            result[key]=item
    if value.get('evaluation_schema') == 'paper_agent_e2e_rule_v1':
        result['evaluation_schema']='paper_agent_e2e_rule_v1'
    return result

def _level(meta):
    if meta.get('status') == 'error':
        return 'ERROR'
    if meta.get('degraded') or meta.get('status') == 'skipped':
        return 'WARNING'
    return 'DEFAULT'

def _sdk_usage(usage):
    # Langfuse treats each usage key as an exclusive bucket. Provider records
    # already subtract cache hits from input; do not send inclusive totals.
    names={'input_tokens':'input', 'output_tokens':'output',
           'cache_read_input_tokens':'cache_read_input_tokens',
           'cache_creation_input_tokens':'cache_creation_input_tokens'}
    return {names[k]:v for k,v in usage.items() if k in names} if usage is not None else None

def _new_private_provider():
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    return TracerProvider(resource=Resource(attributes={}),shutdown_on_exit=False)

_client_registry = {}
_client_lock = threading.Lock()

class LangfuseSink:
    """Lazy Python SDK v4 integration; no environment changes or automatic uploads."""
    def __init__(self, dedup_dir=None):
        from pathlib import Path
        self.client = None
        self._allowed_spans = None
        self.dedup_dir = Path(dedup_dir or os.getenv('PAPER_TELEMETRY_DEDUP_DIR', '.cache/paper-agent-telemetry'))

    def _claim_tree(self, trace_id):
        # The SDK allocates observation IDs itself. Claim BEFORE sending any
        # observation: at-most-once attempt is safer than a duplicate subtree
        # after an ambiguous network error or process crash.
        if not re.fullmatch(r'[0-9a-f]{32}',trace_id) or trace_id == '0'*32:
            raise ValueError('Invalid trace ID')
        self.dedup_dir.mkdir(parents=True,exist_ok=True)
        # Separate destinations/projects even when they happen to use the same
        # trace ID. The file name contains neither endpoint nor public key.
        target = hashlib.sha256(json.dumps([os.getenv('LANGFUSE_BASE_URL',''),
                                            os.getenv('LANGFUSE_PUBLIC_KEY','')],
                                           separators=(',',':')).encode()).hexdigest()[:16]
        try:
            fd=os.open(self.dedup_dir/(target+'-'+trace_id+'.claimed'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError:
            return False
        os.close(fd)
        return True

    def __call__(self, trace):
        if self.client is None:
            import atexit
            from importlib.metadata import version
            if version('langfuse') != '4.8.1':
                raise RuntimeError('Isolated adapter requires langfuse 4.8.1')
            from langfuse import Langfuse
            # A private provider prevents unrelated globally instrumented spans
            # from bypassing this adapter's metadata-only policy. Disabling
            # automatic shutdown avoids an unbounded SDK/OTel atexit flush.
            public_key=os.environ['LANGFUSE_PUBLIC_KEY']
            secret_key=os.environ['LANGFUSE_SECRET_KEY']
            base_url=os.environ['LANGFUSE_BASE_URL']
            key=hashlib.sha256(public_key.encode()).hexdigest()
            target=hashlib.sha256(json.dumps([base_url,secret_key],separators=(',',':')).encode()).hexdigest()
            with _client_lock:
                cached=_client_registry.get(key)
                if cached:
                    if cached[0] != target:
                        raise RuntimeError('Langfuse public key reused with a different destination or secret')
                    self.client,self._allowed_spans=cached[1:]
                else:
                    private_provider=_new_private_provider()
                    allowed=set()
                    # Only spans explicitly created by this adapter may pass.
                    # Even another SDK client reusing this manager is filtered.
                    def allowed_span(span):
                        context=getattr(span,'context',None)
                        return context is not None and f'{context.span_id:016x}' in allowed
                    client = Langfuse(public_key=public_key, secret_key=secret_key,
                                      base_url=base_url, timeout=2, tracer_provider=private_provider,
                                      environment='paper-agent',release='paper-agent',
                                      should_export_span=allowed_span)
                    resources=getattr(client,'_resources',None)
                    # SDK 4.8.1 caches by public key; fail closed on reuse of
                    # resources initialized by unrelated application code.
                    if (resources is None or getattr(resources,'tracer_provider',None) is not private_provider
                            or getattr(resources,'base_url',None) != base_url):
                        raise RuntimeError('Langfuse resources are already shared or destination mismatched')
                    # SDK 4.8.1 registers an unbounded shutdown at exit.
                    atexit.unregister(resources.shutdown)
                    self.client,self._allowed_spans=client,allowed
                    _client_registry[key]=(target,client,allowed)
        client = self.client
        scores_only = bool(trace.get('scores_only'))
        claimed = not scores_only and self._claim_tree(trace['trace_id'])
        if claimed:
            root = client.start_observation(name='Turn', trace_context={'trace_id': trace['trace_id']}, metadata=trace['metadata'],level=_level(trace['metadata']))
            self._allow(root)
            for stage in trace['stages']:
                stage_name=stage['metadata'].get('stage','unknown')
                attempt=stage['metadata'].get('attempt')
                if type(attempt) is int and attempt>1:
                    stage_name=f'{stage_name} #{attempt}'
                span = root.start_observation(name=stage_name, metadata=stage['metadata'],level=_level(stage['metadata']))
                self._allow(span)
                for call in stage['calls']:
                    generation = span.start_observation(name='LLM request', as_type='generation',
                        metadata=call['metadata'], model=call['metadata'].get('model_requested'),
                        level=_level(call['metadata']),usage_details=_sdk_usage(call.get('usage')))
                    self._allow(generation)
                    self._end(generation)
                self._end(span)
            self._end(root)
        for score in trace['scores']:
            client.create_score(trace_id=trace['trace_id'], name=score['name'],
                                value=score['value'], data_type='NUMERIC', metadata=score['metadata'],
                                score_id=score_id(trace['trace_id'],score['metadata']['source'],score['name']))
        client.flush()
        # SDK flush initiates delivery but does not confirm a server-side readback.
        if scores_only:
            return 'scores_sent_unverified' if trace['scores'] else 'no_scores'
        return 'tree_sent_unverified' if claimed else 'tree_already_claimed_unverified'

    def _allow(self, observation):
        if self._allowed_spans is not None:
            self._allowed_spans.add(observation.id)

    def _end(self, observation):
        try:
            observation.end()
        finally:
            if self._allowed_spans is not None:
                self._allowed_spans.discard(observation.id)

class Exporter:
    def __init__(self, enabled=False, sink=None, sample_rate=1., flush_seconds=.5):
        if not 0 <= sample_rate <= 1 or not 0 <= flush_seconds <= 2:
            raise ValueError('Invalid telemetry limits')
        self.enabled, self.sink = enabled, sink or LangfuseSink()
        self.sample_rate, self.flush_seconds = sample_rate, flush_seconds
        self.queue = queue.Queue(maxsize=1)
        self.worker = None
        self._worker_lock = threading.Lock()
        self.status = 'disabled' if not enabled else 'ready'

    @classmethod
    def from_env(cls):
        enabled=os.getenv('PAPER_TELEMETRY_ENABLED', 'false').lower() == 'true'
        if not enabled:
            return cls()
        try:
            return cls(enabled=True,sample_rate=float(os.getenv('PAPER_TELEMETRY_SAMPLE_RATE', '1')),
                       flush_seconds=float(os.getenv('PAPER_TELEMETRY_FLUSH_SECONDS', '.5')))
        except (ValueError, OverflowError):
            exporter=cls()
            exporter.status='disabled_invalid_configuration'
            return exporter

    def export(self, trace, scores_only=False):
        if not self.enabled:
            return
        trace_id = trace_mapping(trace)['langfuse_trace_id']
        if int(trace_id[:8],16)/0x100000000 >= self.sample_rate:
            return
        stages = []
        for stage in trace['stages']:
            calls = []
            for call in trace.get('api_calls', []):
                if call.get('invocation_id') == stage.get('invocation_id') and stage.get('invocation_id'):
                    item = {'metadata': metadata(call)}
                    if call.get('usage_complete'):
                        item['usage'] = {k:v for k,v in call.get('usage', {}).items() if k in USAGE_NAMES and type(v) is int and v>=0}
                    calls.append(item)
            stages.append({'metadata': {**metadata(stage), **stage_summary(stage)}, 'calls': calls})
        score_context={k:v for k,v in metadata(trace).items() if k in {
            'session_id','turn_id','case_id','run_id','dataset_hash','evaluation_config_hash','evaluation_schema'}}
        scores = [{'name': s['name'], 'value': s['value'], 'metadata': {**score_context,**metadata(s)}}
                  for s in trace.get('scores', []) if s.get('source') in {'rule','same_model_judge'}
                  and s.get('name') in SCORE_NAMES and isinstance(s.get('value'), (int,float))
                  and math.isfinite(s['value'])]
        payload = {'trace_id':trace_id, 'metadata':{**metadata(trace), **trace_summary(trace)}, 'stages':[] if scores_only else stages,
                   'scores':scores, 'scores_only':scores_only}
        if self.worker is None:
            with self._worker_lock:
                if self.worker is None:
                    self.worker = threading.Thread(target=self._run, daemon=True)
                    self.worker.start()
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.status = 'dropped_queue_full'

    def export_scores(self, trace):
        self.export(trace,scores_only=True)

    def _run(self):
        while True:
            payload = self.queue.get()
            try:
                status = self.sink(payload)
                self.status = status if isinstance(status,str) and status in {
                    'scores_sent_unverified', 'no_scores',
                    'tree_sent_unverified', 'tree_already_claimed_unverified'
                } else 'sent_unverified'
            except Exception:
                self.status = 'export_failed'
            finally:
                self.queue.task_done()

    def flush(self):
        # A stuck external SDK occupies only this one daemon; no unbounded threads.
        import time
        end = time.monotonic() + self.flush_seconds
        while self.queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(.001)
        return self.queue.unfinished_tasks == 0
