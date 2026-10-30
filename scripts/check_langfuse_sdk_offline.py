"""Exercise the actual optional SDK with an in-memory exporter, never a server.

Run in a fresh process with the pinned SDK on PYTHONPATH. No HTTP traffic is
permitted, and the SDK score queue is intercepted before it can perform I/O.
This checks the adapter/SDK contract, not remote ingestion or readback.
"""
import json
import os
import socket
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    try:
        installed = version('langfuse')
        if installed != '4.8.1':
            print(json.dumps({'status': 'skipped', 'reason': 'requires_langfuse_4.8.1'}))
            return
        import langfuse
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except (PackageNotFoundError, ImportError):
        print(json.dumps({'status': 'skipped', 'reason': 'sdk_or_dependency_unavailable'}))
        return

    from paper_agent.observability import Exporter, LangfuseSink, score_id, trace_mapping

    def reject_network(*args, **kwargs):
        raise RuntimeError('network disabled for offline SDK check')

    socket.socket.connect = reject_network
    socket.socket.connect_ex = reject_network
    os.environ.update(LANGFUSE_PUBLIC_KEY='pk-lf-synthetic-offline',
                      LANGFUSE_SECRET_KEY='sk-lf-synthetic-offline',
                      LANGFUSE_BASE_URL='http://127.0.0.1:1',
                      LANGFUSE_MEDIA_UPLOAD_ENABLED='false')
    os.environ.pop('LANGFUSE_TRACING_ENABLED', None)
    memory = InMemorySpanExporter()
    score_events = []
    real_client = langfuse.Langfuse

    def offline_client(**kwargs):
        client = real_client(**kwargs, span_exporter=memory)
        client._resources.add_score_task = lambda event, **_kw: score_events.append(event)
        return client

    langfuse.Langfuse = offline_client
    base = {'session_id': 'synthetic-session', 'turn_id': 'synthetic-turn',
            'case_id': 'synthetic-case', 'status': 'completed', 'commit_status': 'committed',
            'question': 'private question must never appear',
            'stages': [{'stage': 'NLU', 'status': 'ok', 'invocation_id': 'synthetic-invocation'}],
            'api_calls': [{'invocation_id': 'synthetic-invocation', 'model_requested': 'synthetic-model',
                           'status': 'ok', 'usage_complete': True,
                           'usage': {'input_tokens': 3, 'output_tokens': 2, 'cache_read_input_tokens': 1}}],
            'scores': [{'source': 'rule', 'name': 'behavior_match', 'value': 1}]}
    payloads = []
    projector = Exporter(True, payloads.append, flush_seconds=2)
    for run in ('run-a', 'run-b'):
        projector.export(dict(base, run_id=run))
        assert projector.flush(), 'in-memory projection timed out'
    assert len(payloads) == 2

    folder = Path('.cache/paper-agent-sdk-offline') / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    sink = LangfuseSink(Path(folder))
    assert sink(payloads[0]) == 'tree_sent_unverified'
    assert sink(payloads[0]) == 'tree_already_claimed_unverified'
    assert sink(payloads[1]) == 'tree_sent_unverified'
    second_sink = LangfuseSink(Path(folder))
    assert second_sink(payloads[0]) == 'tree_already_claimed_unverified'
    assert second_sink.client is sink.client
    # A second consumer of the same SDK client must not bypass our allowlist.
    unrelated = sink.client.start_observation(name='external-private', input='private external input')
    unrelated.end()
    sink.client.flush()
    spans = list(memory.get_finished_spans())
    assert len(spans) == 6
    assert len(score_events) == 4
    traces = {}
    for span in spans:
        traces.setdefault(format(span.context.trace_id, '032x'), []).append(span)
    expected = [trace_mapping(dict(base, run_id=run))['langfuse_trace_id'] for run in ('run-a', 'run-b')]
    assert set(traces) == set(expected) and expected[0] != expected[1]
    for trace_id, group in traces.items():
        by_name = {span.name: span for span in group}
        assert set(by_name) == {'Turn', 'NLU', 'LLM request'}
        assert by_name['Turn'].attributes['langfuse.internal.as_root'] is True
        assert by_name['NLU'].parent.span_id == by_name['Turn'].context.span_id
        assert by_name['LLM request'].parent.span_id == by_name['NLU'].context.span_id
        assert json.loads(by_name['LLM request'].attributes['langfuse.observation.usage_details']) == {
            'input': 3, 'output': 2, 'cache_read_input_tokens': 1}
    # SDK 4.8.1 builds ScoreBody with camelCase extra fields. Its consumer
    # serializes model_dump(exclude_none=True), so inspect that actual payload.
    score_bodies = [event['body'].model_dump(exclude_none=True) for event in score_events]
    assert [body['traceId'] for body in score_bodies] == [expected[0], expected[0], expected[1], expected[0]]
    assert [event['body'].id for event in score_events] == [
        score_id(expected[0], 'rule', 'behavior_match'),
        score_id(expected[0], 'rule', 'behavior_match'),
        score_id(expected[1], 'rule', 'behavior_match'),
        score_id(expected[0], 'rule', 'behavior_match')]
    assert sink.client._resources.tracer_provider is not None
    serialized = repr([dict(span.attributes) for span in spans])
    assert 'private question' not in serialized and 'private external input' not in serialized
    print(json.dumps({'status': 'offline_sdk_passed', 'sdk_version': installed,
                      'runs': 2, 'traces': len(traces), 'spans': len(spans),
                      'score_events_intercepted': len(score_events),
                      'network': 'blocked', 'provider_requests': 0, 'actual_tokens': 'not_applicable',
                      'server_readback': 'not_tested'}, sort_keys=True))


if __name__ == '__main__':
    main()
