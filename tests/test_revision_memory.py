import json
import sqlite3
from dataclasses import asdict
import pytest
from paper_agent.long_term import MemoryStore
from paper_agent.memory_manager import MemoryManager
from paper_agent.pipeline import PaperAgentState
from paper_agent.migration import migrate, migrate_file_snapshots, rollback_copy
from test_reliability import pipeline

def test_scope_sharing_update_delete_and_no_paper_claims(tmp_path):
    def manager(user='alice', project='project', shared=False):
        return MemoryManager(tmp_path/'memory',None,None,user_id=user,project_id=project,share_project=shared)
    m=manager(shared=True)
    p=pipeline(tmp_path,memory_manager=m,mode='bm25')
    state=PaperAgentState('one')
    p.run('偏好:输出语言=中文',state,'t1')
    p.run('alpha',state,'t2')
    assert len(m.store.memories('new',shared=True))==1
    assert manager(shared=True).retrieve_memories(PaperAgentState('new'),'language',['p'])
    assert not manager().retrieve_memories(PaperAgentState('new'),'language',['p'])
    assert not manager('bob',shared=True).retrieve_memories(PaperAgentState('new'),'language')
    assert not manager(project='other',shared=True).retrieve_memories(PaperAgentState('new'),'language')
    with pytest.raises(ValueError): manager('bob').load_working_memory(state)
    p.run('更新记忆:输出语言=英文',state,'t3')
    active=m.store.memories('new','active',shared=True)
    assert active[0]['text']=='英文' and active[0]['version']==2
    m.store.delete('new',active[0]['memory_id'],shared=False)
    assert m.store.memories('new','active',shared=True)
    m.store.delete('new',active[0]['memory_id'],shared=True)
    assert not m.store.memories('new','active',shared=True)
    with m.store.transaction() as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='facts'").fetchall()

def legacy_database(path):
    state=PaperAgentState('old')
    state.turn_count=1
    state.committed_turns={'t':{'question':'偏好:语言=中文','result':{'answer':{'decision':'refuse','claims':[]}}}}
    raw=asdict(state); raw['memory']['facts']={'old':{'text':'paper claim','evidence_ids':['c']}}
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE facts(fact_id,session_id,source_type,turn_id,text,status,canonical_key); CREATE TABLE sessions(session_id,version,snapshot); CREATE TABLE episodes(episode_id,session_id,payload,indexed);')
        db.execute('INSERT INTO sessions VALUES (?,?,?)',('old',1,json.dumps(raw)))
        db.executemany('INSERT INTO facts VALUES (?,?,?,?,?,?,?)',[
            ('u','old','user_explicit','t','中文','active',json.dumps(['user_explicit',None,'语言'],ensure_ascii=False)),
            ('p','old','paper_evidence','t','paper claim','active','[]'),
            ('x','old','conversation','unknown','guess','active','[]')])

def test_migration_repeat_backup_old_restore_rollback(tmp_path):
    old=tmp_path/'facts.sqlite3'; legacy_database(old)
    original=old.read_bytes()
    store=MemoryStore(tmp_path/'memory.sqlite3')
    audit=migrate(old,store,owner_confirmed=True)
    assert audit['long_term']==1 and sum(audit['omitted'].values())==2
    assert migrate(old,store,owner_confirmed=True)==audit
    assert old.read_bytes()==original
    assert len(store.memories('old'))==1
    assert 'facts' not in store.snapshot('old')['memory']
    assert store.snapshot('old')['committed_turns']['t']['question']=='偏好:语言=中文'
    from pathlib import Path
    backup=Path(audit['backup'])
    with sqlite3.connect(backup.resolve().as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('SELECT count(*) FROM facts').fetchone()[0]==3
        with pytest.raises(sqlite3.OperationalError): db.execute('DELETE FROM facts')
        restored=tmp_path/'rollback.sqlite3'
        with sqlite3.connect(restored) as target: db.backup(target)
    with sqlite3.connect(restored) as db:
        assert db.execute('SELECT snapshot FROM sessions').fetchone()[0]

def test_unconfirmed_owner_never_promoted(tmp_path):
    old=tmp_path/'facts.sqlite3'; legacy_database(old)
    store=MemoryStore(tmp_path/'memory.sqlite3')
    assert migrate(old,store)['long_term']==0
    assert not store.memories('old')
    assert store.snapshot('old') is None
    with pytest.raises(ValueError,match='new empty target'):
        migrate(old,store,owner_confirmed=True)

def test_important_event_shared_only_after_opt_in(tmp_path):
    m=MemoryManager(tmp_path/'m',None,None,share_project=True)
    p=pipeline(tmp_path,memory_manager=m,mode='bm25')
    p.run('实验结果: alpha下降10%',PaperAgentState('one'),'t')
    assert m.retrieve_memories(PaperAgentState('two'),'alpha')
    m.share_project=False
    assert not m.retrieve_memories(PaperAgentState('two'),'alpha')

def test_legacy_files_backup_and_explicit_restore(tmp_path):
    source=tmp_path/'legacy'; source.mkdir()
    (source/'s.history.json').write_text(json.dumps({'messages':[{'role':'user','content':'early'}], 'turn_count':1}),encoding='utf-8')
    (source/'s.json').write_text(json.dumps({'session_id':'s','entities':{'it':'p'},'facts':{'f':{'text':'old'}},'unresolved':[]}),encoding='utf-8')
    old_bytes={p.name:p.read_bytes() for p in source.glob('*.json')}
    unconfirmed=MemoryStore(tmp_path/'no'/'memory.sqlite3')
    audit=migrate_file_snapshots(source,unconfirmed)
    assert audit['sessions_omitted']==1 and unconfirmed.snapshot('s') is None
    assert all((source/'migration-backups'/'v2-files'/name).read_bytes()==data for name,data in old_bytes.items())
    with pytest.raises(ValueError,match='new empty target'):
        migrate_file_snapshots(source,unconfirmed,True)
    target=MemoryStore(tmp_path/'yes'/'memory.sqlite3')
    audit=migrate_file_snapshots(source,target,True)
    assert audit['sessions']==1 and audit['old_fact_entries_promoted']==0
    assert target.snapshot('s')['messages'][0]['content']=='early'
    assert target.snapshot('s')['memory']['summary']['legacy_context_unverified']
    assert not target.memories('s')
    assert migrate_file_snapshots(source,target,True)==audit
    restored=PaperAgentState('s')
    MemoryManager(tmp_path/'yes',None,None).load_working_memory(restored)
    assert restored.messages[0]['content']=='early' and restored.turn_count==1

def test_unconfirmed_file_session_not_loaded_into_pipeline(tmp_path):
    source=tmp_path/'memory'; source.mkdir()
    (source/'s.history.json').write_text(json.dumps({'messages':[{'role':'user','content':'private'}], 'turn_count':1}),encoding='utf-8')
    (source/'s.json').write_text(json.dumps({'session_id':'s','entities':{},'facts':{},'unresolved':[]}),encoding='utf-8')
    state=PaperAgentState.load(source,'s')
    m=MemoryManager(source,None,None)
    with pytest.raises(ValueError,match='ownership-confirmed'):
        m.load_working_memory(state)
    new=PaperAgentState('s')
    m.load_working_memory(new)
    assert new.messages==[]

def test_migration_rejects_value_only_match(tmp_path):
    old=tmp_path/'facts.sqlite3'; legacy_database(old)
    with sqlite3.connect(old) as db:
        db.execute('UPDATE facts SET canonical_key=? WHERE fact_id=?',(json.dumps(['user_explicit',None,'other']),'u'))
    store=MemoryStore(tmp_path/'memory.sqlite3')
    audit=migrate(old,store,owner_confirmed=True)
    assert audit['long_term']==0 and not store.memories('old')

def test_project_roots_and_projection_objects_are_isolated(tmp_path):
    from paper_agent.memory_stores import RedisWorkingStore
    from test_memory_manager import Cache
    cache=Cache()
    first=RedisWorkingStore('unused',client=cache)
    a=MemoryManager(tmp_path/'a',first,None,user_id='alice',project_id='same')
    b=MemoryManager(tmp_path/'b',RedisWorkingStore('unused',client=cache),None,user_id='alice',project_id='same')
    assert a.identity!=b.identity and a.working.key('s')!=b.working.key('s')
    with pytest.raises(ValueError,match='another memory namespace'):
        MemoryManager(tmp_path/'b',first,None,user_id='alice',project_id='same')

def test_archived_declaration_not_resurrected_by_episode(tmp_path):
    m=MemoryManager(tmp_path/'memory',None,None)
    p=pipeline(tmp_path,memory_manager=m,mode='bm25')
    p.run('偏好:语言=中文',PaperAgentState('s'),'t')
    item=m.store.memories('s','active')[0]
    m.store.delete('s',item['memory_id'])
    assert not m.retrieve_memories(PaperAgentState('s'),'语言')
    assert any(r['kind']=='episode' for r in m.retrieve_memories(PaperAgentState('s'),'语言历史'))

def test_rollback_copy_keeps_old_ledger(tmp_path):
    old=tmp_path/'facts.sqlite3'; legacy_database(old)
    store=MemoryStore(tmp_path/'memory.sqlite3')
    audit=migrate(old,store,owner_confirmed=True)
    dest=rollback_copy(audit['backup'],tmp_path/'rollback')
    assert PaperAgentState.load(dest,'old').committed_turns['t']['question']=='偏好:语言=中文'
    with pytest.raises(FileExistsError): rollback_copy(audit['backup'],dest)

def test_same_chunk_id_in_two_papers_remains_valid():
    from paper_agent.memory import validate_citations
    evidence=[{'paper_id':'a','chunk_id':'same'}, {'paper_id':'b','chunk_id':'same'}]
    for paper in ('a','b'):
        assert validate_citations([{'text':'x','citations':[{'paper_id':paper,'chunk_id':'same'}]}],evidence)
