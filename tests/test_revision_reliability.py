import pytest
from paper_agent.budgets import TurnBudget,current_budget,BudgetExceeded
from paper_agent.circuit import Circuit,CircuitOpen,classify
from paper_agent.providers import APIConfig,CompatibleReasoner,ProviderTransportError
from test_reliability import response

@pytest.mark.parametrize('status,kind,category',[(401,None,'authentication'),(403,None,'authentication'),(429,None,'rate_limit'),(503,None,'server'),(None,'TimeoutError','timeout'),(None,'URLError','connection'),(400,None,'request')])
def test_error_classification(status,kind,category):
    assert classify(status,kind)==category

def test_half_open_single_probe_and_recovery():
    now=[0.]
    c=Circuit(2,10,lambda:now[0])
    c.enter();c.finish('server'); assert c.state=='CLOSED'
    c.enter();c.finish('timeout'); assert c.state=='OPEN'
    with pytest.raises(CircuitOpen): c.enter()
    now[0]=10
    assert c.enter()=='HALF_OPEN'
    with pytest.raises(CircuitOpen): c.enter()
    c.finish('server'); assert c.state=='OPEN'
    now[0]=20;c.enter();c.finish(); assert c.state=='CLOSED'
    c.enter();c.finish('schema'); assert c.state=='CLOSED'

def test_circuit_provider_no_extra_requests():
    r=CompatibleReasoner(APIConfig('secret','circuit-test',max_retries=5,retry_base_seconds=0,circuit_enabled=True,circuit_threshold=2),'s',lambda *a: (_ for _ in ()).throw(ProviderTransportError(503)))
    with pytest.raises(CircuitOpen): r.structured('reflect',{})
    assert len(r.calls)==2 and r.calls[-1]['circuit_after']=='OPEN'
    with pytest.raises(CircuitOpen): r.structured('reflect',{})
    assert len(r.calls)==2

def test_budget_counts_schema_repair_and_transport_attempts():
    for transport in (lambda *a:response(None),lambda *a: (_ for _ in ()).throw(ProviderTransportError(429))):
        r=CompatibleReasoner(APIConfig('secret','test',retry_base_seconds=0),'s',transport)
        budget=TurnBudget(model_requests=1)
        token=current_budget.set(budget)
        try:
            with pytest.raises(BudgetExceeded): r.structured('reflect',{})
            assert len(r.calls)==1 and budget.used['model_requests']==1
        finally: current_budget.reset(token)

def test_timeout_is_remaining_and_late_response_rejected():
    now=[0.]
    b=TurnBudget(seconds=1,clock=lambda:now[0])
    def transport(url,body,headers,timeout):
        assert timeout<=1
        now[0]=2
        return response({'decision':'refuse','reason':'no evidence'})
    r=CompatibleReasoner(APIConfig('secret','test'),'s',transport)
    token=current_budget.set(b)
    try:
        with pytest.raises(BudgetExceeded):r.structured('reflect',{})
        assert r.calls[0]['error_category']=='budget'
    finally:current_budget.reset(token)

def test_concurrent_old_success_cannot_close_open_or_release_probe():
    now=[0.]
    c=Circuit(1,1,lambda:now[0])
    _, stale = c.enter_ticket()
    _, failure = c.enter_ticket()
    c.finish('server',failure)
    assert c.state == 'OPEN'
    c.finish(None,stale)
    assert c.state == 'OPEN'
    now[0]=1
    _, probe = c.enter_ticket()
    c.finish(None,stale)
    with pytest.raises(CircuitOpen): c.enter_ticket()
    c.finish(None,probe)
    assert c.state == 'CLOSED'

def test_half_open_budget_failure_does_not_falsely_recover():
    now=[0.]
    c=Circuit(1,1,lambda:now[0])
    _, ticket=c.enter_ticket(); c.finish('server',ticket)
    now[0]=1
    _, ticket=c.enter_ticket(); c.finish('budget',ticket)
    assert c.state=='OPEN'

def test_circuit_failure_streak_resets_on_reachable_schema_error():
    c=Circuit(2,1)
    _, ticket=c.enter_ticket(); c.finish('server',ticket)
    _, ticket=c.enter_ticket(); c.finish('schema',ticket)
    _, ticket=c.enter_ticket(); c.finish('server',ticket)
    assert c.state=='CLOSED' and c.failures==1

def test_provider_does_not_copy_untrusted_usage_or_model_text():
    def transport(*_):
        result=response({'decision':'refuse','reason':'no evidence'})
        result.update(model='secret@private.example',usage={'prompt_tokens':4,'completion_tokens':1,
            'private':'secret user text','completion_tokens_details':{'reasoning_tokens':1,'raw':'secret'}})
        return result
    r=CompatibleReasoner(APIConfig('secret','test'), 's',transport)
    r.structured('reflect',{})
    assert 'secret' not in str(r.calls)
    assert r.calls[0]['usage_raw']=={'prompt_tokens':4,'completion_tokens':1,'completion_tokens_details':{'reasoning_tokens':1}}

def test_circuit_isolated_by_account_without_exposing_key():
    from uuid import uuid4
    model='test-'+uuid4().hex
    first=CompatibleReasoner(APIConfig('bad-secret',model,circuit_enabled=True), 's',lambda *_: (_ for _ in ()).throw(ProviderTransportError(401)))
    second=CompatibleReasoner(APIConfig('good-secret',model,circuit_enabled=True), 's',lambda *_:response({'decision':'refuse','reason':'insufficient'}))
    with pytest.raises(RuntimeError): first.structured('reflect',{})
    assert first.circuit.state=='OPEN'
    assert second.structured('reflect',{})['decision']=='refuse'
    assert second.circuit.state=='CLOSED'
