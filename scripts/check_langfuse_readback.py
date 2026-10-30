"""Read-only, bounded verification of two existing local Trace projections.

Never sends telemetry or calls a model. Output is aggregate checks only: no
server response, identifiers, exception text, credentials or local content.
Requires Langfuse 4.8.1 and the v4 observations/scores APIs.
"""
import argparse
import json
import logging
import os
import queue
import threading
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from paper_agent.observability import _level, _sdk_usage, metadata, score_id, stage_summary, trace_mapping, trace_summary


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _check(expected, actual):
    if actual is None:
        return 'unknown'
    return 'passed' if actual == expected else 'failed'


def _merge(*values):
    if 'failed' in values:
        return 'failed'
    return 'unknown' if 'unknown' in values or not values else 'passed'


def _valid_endpoint(url):
    parts = urlsplit(url)
    return (bool(parts.hostname) and not (parts.username or parts.password or parts.query or parts.fragment) and
            (parts.scheme == 'https' or (parts.scheme == 'http' and parts.hostname in ('localhost','127.0.0.1','::1'))))


def _pages(getter, *, deadline, kwargs):
    cursor = None
    seen = set()
    result = []
    for _ in range(10):
        if time.monotonic() >= deadline:
            raise TimeoutError()
        response = getter(**kwargs, cursor=cursor, limit=100,
                          request_options={'timeout_in_seconds': 3, 'max_retries': 0})
        data = _get(response, 'data')
        meta = _get(response, 'meta')
        if not isinstance(data, list) or meta is None:
            raise ValueError()
        result.extend(data)
        cursor = _get(meta, 'cursor')
        if not cursor:
            return result
        if cursor in seen:
            raise ValueError()
        seen.add(cursor)
    raise ValueError()  # Refuse to treat truncated pagination as complete.


def verify_trace(trace, observations, scores):
    """Compare response objects without emitting or preserving response values."""
    trace_id = trace_mapping(trace)['langfuse_trace_id']
    result = {}
    if not observations:
        return {k:'unknown' for k in ('tree', 'stage_status', 'request', 'usage', 'retrieval', 'memory', 'rule_score')}
    ids = [_get(o, 'id') for o in observations]
    roots = [o for o in observations if _get(o, 'name') == 'Turn']
    by_id = {_get(o, 'id'): o for o in observations}
    stages = [s for s in trace.get('stages', []) if isinstance(s, dict)]
    calls = trace.get('api_calls', [])
    if len(roots) != 1 or len(by_id) != len(ids) or None in by_id:
        result['tree'] = 'failed'
        return {**{k:'unknown' for k in ('stage_status', 'request', 'usage', 'retrieval', 'memory', 'rule_score')}, **result}
    root = roots[0]
    root_id = _get(root, 'id')
    children = [o for o in observations if _get(o, 'parent_observation_id') == root_id]
    generations = [o for o in observations if _get(o, 'name') == 'LLM request']
    stage_ids = {_get(o, 'id') for o in children}
    expected_names = Counter(s.get('stage', 'unknown') + (f" #{s['attempt']}" if type(s.get('attempt')) is int and s['attempt'] > 1 else '') for s in stages)
    correct_tree = (not _get(root, 'parent_observation_id') and
                    all(_get(o, 'trace_id') == trace_id for o in observations) and
                    Counter(_get(o, 'name') for o in children) == expected_names and
                    len(generations) == len(calls) and
                    all(_get(o, 'parent_observation_id') in stage_ids for o in generations) and
                    len(observations) == 1 + len(children) + len(generations))
    result['tree'] = 'passed' if correct_tree else 'failed'
    stage_checks = []
    retrieve_checks = []
    usage_checks = []
    request_checks = []
    for stage in stages:
        wanted = metadata(stage)
        matches = [o for o in children if _get(o, 'name') == stage.get('stage', 'unknown') + (f" #{stage['attempt']}" if type(stage.get('attempt')) is int and stage['attempt'] > 1 else '')]
        if wanted.get('invocation_id'):
            matches = [o for o in matches if (_get(o, 'metadata') or {}).get('invocation_id') == wanted['invocation_id']]
        if len(matches) != 1:
            stage_checks.append('unknown' if not matches else 'failed')
            continue
        remote = matches[0]
        remeta = _get(remote, 'metadata')
        if not isinstance(remeta, dict):
            stage_checks.append('unknown')
            continue
        stage_checks += [_check(wanted.get('status'), remeta.get('status')),
                         _check(_level(wanted), _get(remote, 'level')),
                         _check(wanted.get('elapsed_ms'), remeta.get('elapsed_ms'))]
        if stage.get('stage') in ('Retrieve', 'Rerank'):
            retrieve_checks.extend(_check(v, remeta.get(k)) for k,v in stage_summary(stage).items())
        for call in calls:
            if not stage.get('invocation_id') or call.get('invocation_id') != stage.get('invocation_id'):
                continue
            callmeta = metadata(call)
            matches = [o for o in generations if _get(o, 'parent_observation_id') == _get(remote, 'id') and
                       isinstance(_get(o, 'metadata'), dict) and _get(o, 'metadata').get('request_id') == callmeta.get('request_id')]
            if len(matches) != 1 or not call.get('usage_complete'):
                usage_checks.append('unknown' if len(matches) <= 1 else 'failed')
            if len(matches) != 1:
                request_checks.append('unknown' if not matches else 'failed')
                continue
            call_remote = matches[0]
            call_remote_meta = _get(call_remote, 'metadata')
            request_checks.extend([_check(callmeta.get('status'), call_remote_meta.get('status')),
                                   _check(callmeta.get('elapsed_ms'), call_remote_meta.get('elapsed_ms')),
                                   _check(_level(callmeta), _get(call_remote, 'level')),
                                   _check(callmeta.get('model_requested'), _get(call_remote, 'provided_model_name'))])
            for key in ('repair','retry_reason'):
                if callmeta.get(key) is not None:
                    request_checks.append(_check(callmeta[key], call_remote_meta.get(key)))
            if not call.get('usage_complete'):
                continue
            expected_usage = _sdk_usage(call.get('usage'))
            actual_usage = _get(matches[0], 'usage_details')
            if not expected_usage or not isinstance(actual_usage, dict):
                usage_checks.append('unknown')
            else:
                usage_checks.extend(_check(v, actual_usage.get(k)) for k,v in expected_usage.items())
    result['stage_status'] = _merge(*stage_checks)
    result['request'] = _merge(*request_checks)
    result['retrieval'] = _merge(*retrieve_checks)
    result['usage'] = _merge(*usage_checks)
    remeta = _get(root, 'metadata')
    result['memory'] = (_merge(*[_check(v, remeta.get(k)) for k,v in trace_summary(trace).items()])
                        if isinstance(remeta, dict) else 'unknown')
    rule = next((s for s in trace.get('scores', []) if s.get('name') == 'execution_success' and
                 s.get('source') == 'rule' and s.get('value') is not None), None)
    if not rule or not scores:
        result['rule_score'] = 'unknown'
    else:
        wanted_score = score_id(trace_id, 'rule', 'execution_success')
        found = [s for s in scores if _get(s, 'id') == wanted_score]
        if len(found) != 1:
            result['rule_score'] = 'unknown' if not found else 'failed'
        else:
            score = found[0]
            subject = _get(score, 'subject')
            scoremeta = _get(score, 'metadata')
            result['rule_score'] = _merge(_check('execution_success', _get(score, 'name')),
                                          _check(rule['value'], _get(score, 'value')),
                                          _check('trace', _get(subject, 'kind')),
                                          _check(trace_id, _get(subject, 'id')),
                                          _check('rule', _get(scoremeta, 'source')))
    return result


def readback(api, traces, deadline):
    ids = [trace_mapping(t)['langfuse_trace_id'] for t in traces]
    if (len(traces) != 2 or len(set(ids)) != 2 or
        any(traces[0].get(k) != traces[1].get(k) for k in ('session_id','turn_id','case_id')) or
        any(t.get('run_id') is None for t in traces) or traces[0]['run_id'] == traces[1]['run_id']):
        return {'status':'not_verified', 'reason':'trace_identity_not_distinct'}
    checks = []
    isolation = []
    for trace, trace_id in zip(traces, ids):
        obs = _pages(api.observations.get_many, deadline=deadline,
                     kwargs={'trace_id':trace_id, 'fields':'core,basic,metadata,model,usage'})
        scores = _pages(api.scores_v3.get_many_v3, deadline=deadline,
                        kwargs={'trace_id':trace_id, 'fields':'details,subject'})
        checks.append(verify_trace(trace, obs, scores))
        isolation.append(_merge(*[_check(trace_id, _get(o, 'trace_id')) for o in obs]))
    # Readback is never a full E2E acceptance: replay and UI require separate evidence.
    return {'status':'readback_checked' if all(v == 'passed' for c in checks for v in c.values()) else 'not_verified',
            'trace_isolation':_merge(*isolation), 'runs':checks, 'replay':'not_checked', 'ui':'not_checked',
            'complete_end_to_end_acceptance':'not_verified'}


def bounded_readback(api, traces, timeout=30):
    """Return by deadline even if DNS or an SDK call does not honor its timeout."""
    output = queue.Queue(maxsize=1)
    def task():
        try:
            output.put_nowait(readback(api, traces, time.monotonic()+timeout))
        except Exception:
            output.put_nowait({'status':'not_verified','reason':'readback_unavailable'})
    worker = threading.Thread(target=task, daemon=True)
    worker.start()
    try:
        return output.get(timeout=timeout)
    except queue.Empty:
        return {'status':'not_verified','reason':'readback_timeout'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_a', help='Local JSON Trace for first run')
    parser.add_argument('trace_b', help='Local JSON Trace for second run')
    args = parser.parse_args()
    for name in ('httpx','httpcore','langfuse','dotenv'):
        logging.getLogger(name).disabled = True
    try:
        from dotenv import load_dotenv
        load_dotenv('.env', override=False)
    except ImportError:
        pass
    except Exception:
        print(json.dumps({'status':'not_verified', 'reason':'local_configuration_unavailable'}))
        return 2
    try:
        traces = [json.loads(Path(path).read_text(encoding='utf-8')) for path in (args.trace_a, args.trace_b)]
        if any(t.get('observability_mapping') != trace_mapping(t) for t in traces):
            raise ValueError()
    except Exception:
        print(json.dumps({'status':'not_verified', 'reason':'invalid_local_trace'}))
        return 2
    if not all(os.getenv(k) for k in ('LANGFUSE_BASE_URL','LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY')):
        print(json.dumps({'status':'not_verified', 'reason':'missing_configuration'}))
        return 2
    try:
        if version('langfuse') != '4.8.1':
            raise ValueError()
        url = os.environ['LANGFUSE_BASE_URL'].rstrip('/')
        if not _valid_endpoint(url):
            raise ValueError()
        from langfuse.api import LangfuseAPI
        api = LangfuseAPI(base_url=url+'/api/public', username=os.environ['LANGFUSE_PUBLIC_KEY'],
                          password=os.environ['LANGFUSE_SECRET_KEY'], timeout=3, follow_redirects=False)
        result = bounded_readback(api, traces)
    except Exception:
        # Never serialize arbitrary exception details or server responses.
        result = {'status':'not_verified', 'reason':'readback_unavailable'}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'readback_checked' else 2


if __name__ == '__main__':
    raise SystemExit(main())
