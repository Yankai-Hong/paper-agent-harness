"""Versioned non-destructive migration. Legacy SQLite is never opened writable."""
import json
import os
from pathlib import Path
import sqlite3
from .storage import digest, safe_id, write_json

VERSION=1

def clean_memory(memory):
    value={k:v for k,v in memory.items() if k in {'session_id','entities','unresolved','summary'}}
    if memory.get('facts'):
        value['summary']={**value.get('summary',{}),'legacy_context_unverified':True}
    return value

def migrate(source, target, owner_confirmed=False):
    source=Path(source)
    if not source.exists(): return {'version':VERSION,'status':'no_legacy_database'}
    with target.transaction() as db:
        row=db.execute('SELECT audit FROM migrations WHERE version=?',(VERSION,)).fetchone()
        if row:
            audit=json.loads(row[0])
            if audit.get('source') and audit['source'] != str(source.resolve()):
                raise ValueError('Migration target already belongs to another source')
            if owner_confirmed and not audit.get('owner_confirmed'):
                raise ValueError('Ownership was not confirmed for this migration; use a new empty target')
            write_json(Path(target.path).parent/'migration-v1-audit.json',audit)
            return audit
    backup=source.parent/'migration-backups'/('v1-'+source.name)
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as old, sqlite3.connect(backup) as copy:
            old.backup(copy)
        os.chmod(backup,0o444)
    import hashlib
    audit={'version':VERSION,'source':str(source.resolve()),'backup':str(backup.resolve()),
           'backup_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'owner_confirmed':owner_confirmed,
           'long_term':0,'episodes':0,'sessions':0,'omitted':{},'records':[],
           'sessions_omitted':0,'episodes_omitted':0}
    from .memory_manager import extract_declarations
    with sqlite3.connect(backup.resolve().as_uri()+'?mode=ro',uri=True) as old, target.transaction() as db:
        old.row_factory=sqlite3.Row
        snapshots={r['session_id']:json.loads(r['snapshot']) for r in old.execute('SELECT * FROM sessions')}
        for sid,state in snapshots.items():
            if not owner_confirmed:
                audit['sessions_omitted']+=1
                continue
            safe_id(sid)
            if state.get('session_id') != sid or state.get('memory',{}).get('session_id') != sid:
                raise ValueError('Legacy snapshot session identity mismatch')
            state['memory']=clean_memory(state['memory'])
            state['memory_namespace']=None
            inserted=db.execute('INSERT OR IGNORE INTO sessions VALUES (?,?,?)',(sid,state['turn_count'],json.dumps(state,ensure_ascii=False))).rowcount
            audit['sessions']+=inserted
            audit['sessions_omitted']+=1-inserted
        for row in old.execute('SELECT * FROM facts'):
            reason='owner_unconfirmed'
            if row['source_type']=='paper_evidence': reason='paper_evidence_not_long_term'
            elif owner_confirmed and row['source_type']=='user_explicit':
                ledger=snapshots.get(row['session_id'],{}).get('committed_turns',{})
                question=ledger.get(row['turn_id'],{}).get('question','')
                candidates,_=extract_declarations(question,{},row['turn_id'])
                # Verify both attribute and value against the committed user
                # message; a coincidentally matching value is not provenance.
                from .memory_stores import canonical_key
                legacy_key=None
                if 'canonical_key' in row.keys():
                    try:
                        identity=json.loads(row['canonical_key'])
                        if isinstance(identity,list) and len(identity)==3 and identity[0]=='user_explicit' and identity[1] is None:
                            legacy_key=identity[2]
                    except (ValueError,TypeError):
                        pass
                matched=next((m for m in candidates if m['text']==row['text'] and canonical_key(m['key'])==legacy_key),None)
                if matched and row['status'] in {'active','conflict'}:
                    matched['status']=row['status']
                    matched['provenance_refs']=[{'legacy_id':row['fact_id'],'source_turn_id':row['turn_id']}]
                    result=target.put(db,row['session_id'],row['turn_id'],matched)
                    reason='long_term' if result['action']!='duplicate' else 'duplicate_declaration'
                    audit['long_term']+=reason=='long_term'
                else: reason='unverified_declaration_or_historical_version'
            if reason!='long_term': audit['omitted'][reason]=audit['omitted'].get(reason,0)+1
            audit['records'].append({'legacy_id':row['fact_id'],'outcome':reason})
        for row in old.execute('SELECT * FROM episodes'):
            if not owner_confirmed:
                audit['episodes_omitted']+=1
                continue
            event=json.loads(row['payload'])
            if event.get('session_id')!=row['session_id'] or event.get('episode_id')!=row['episode_id']:
                raise ValueError('Legacy event identity mismatch')
            # Old events remain private to their original session.
            event.pop('share_project',None)
            inserted=db.execute('INSERT OR IGNORE INTO episodes VALUES (?,?,?,0)',(row['episode_id'],row['session_id'],json.dumps(event,ensure_ascii=False))).rowcount
            audit['episodes']+=inserted
            audit['episodes_omitted']+=1-inserted
        db.execute('INSERT INTO migrations VALUES (?,?)',(VERSION,json.dumps(audit)))
    write_json(Path(target.path).parent/'migration-v1-audit.json',audit)
    return audit

def migrate_file_snapshots(source_dir, target, owner_confirmed=False):
    """Back up old standalone snapshots; import only after ownership review.

    The old files had no turn ledger in some versions. Their recent context is
    recoverable, but invented IDs are never used to authorize old archives.
    """
    from dataclasses import asdict
    import hashlib
    from .pipeline import PaperAgentState
    source_dir=Path(source_dir)
    with target.transaction() as db:
        row=db.execute('SELECT audit FROM migrations WHERE version=2').fetchone()
        if row:
            audit=json.loads(row[0])
            if audit['source'] != str(source_dir.resolve()):
                raise ValueError('File migration target already belongs to another source')
            if owner_confirmed and not audit['owner_confirmed']:
                raise ValueError('Ownership was not confirmed for this migration; use a new empty target')
            write_json(Path(target.path).parent/'migration-v2-audit.json',audit)
            return audit
    sessions={}
    for path in sorted(source_dir.glob('*.state.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        sid=safe_id(path.name[:-len('.state.json')])
        if data.get('session_id')!=sid or data.get('memory',{}).get('session_id')!=sid:
            raise ValueError('Legacy state session identity mismatch')
        if data.get('memory_namespace') is None:
            sessions[sid]=[path]
    for path in sorted(source_dir.glob('*.history.json')):
        sid=safe_id(path.name[:-len('.history.json')])
        memory=source_dir/(sid+'.json')
        group=sessions.setdefault(sid,[])
        group.extend([path]+([memory] if memory.exists() else []))
    # Standalone session-memory files have an explicit session_id. Ignore
    # unrelated metadata JSON rather than guessing its ownership or format.
    for path in sorted(source_dir.glob('*.json')):
        if path.name.endswith(('.history.json','.state.json')):
            continue
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,OSError):
            continue
        if isinstance(data,dict) and data.get('session_id') == path.stem and 'facts' in data:
            sid=safe_id(path.stem)
            if path not in sessions.setdefault(sid,[]):
                sessions[sid].append(path)
    if not sessions:
        return {'version':2,'status':'no_legacy_files','sessions_found':[]}
    backup_dir=source_dir/'migration-backups'/'v2-files'
    backup_dir.mkdir(parents=True,exist_ok=True)
    files=[]
    for path in sorted({p for group in sessions.values() for p in group}):
        raw=path.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        backup=backup_dir/path.name
        if backup.exists():
            if hashlib.sha256(backup.read_bytes()).hexdigest()!=sha:
                raise ValueError('Existing legacy file backup differs; choose a new migration source')
        else:
            with backup.open('xb') as out:
                out.write(raw)
            os.chmod(backup,0o444)
        files.append({'path':str(path.resolve()),'backup':str(backup.resolve()),'sha256':sha})
    audit={'version':2,'source':str(source_dir.resolve()),'owner_confirmed':owner_confirmed,
           'sessions_found':sorted(sessions),'sessions':0,'sessions_omitted':0,
           'old_fact_entries_promoted':0,'long_term':0,'episodes':0,'files':files}
    with target.transaction() as db:
        for sid in sorted(sessions):
            if not owner_confirmed:
                audit['sessions_omitted']+=1
                continue
            state=PaperAgentState.load(source_dir,sid)
            state.memory_namespace=None
            inserted=db.execute('INSERT OR IGNORE INTO sessions VALUES (?,?,?)',
                (sid,state.turn_count,json.dumps(asdict(state),ensure_ascii=False))).rowcount
            audit['sessions']+=inserted
            audit['sessions_omitted']+=1-inserted
            if not inserted:
                continue
            from .memory_manager import extract_declarations
            # Promote only explicit messages verified against a committed ledger.
            # Split legacy history without a ledger remains working context only.
            for turn_id,entry in state.committed_turns.items():
                question=entry.get('question')
                if not isinstance(question,str):
                    continue
                if not any(m.get('role')=='user' and m.get('turn_id')==turn_id and m.get('content')==question for m in state.messages):
                    continue
                answer=entry.get('result',{}).get('answer',{})
                candidates,episode=extract_declarations(question,answer,turn_id)
                for candidate in candidates:
                    candidate['provenance_refs']=[{'source':'legacy_committed_user_message','session_id':sid,'turn_id':turn_id}]
                    if target.put(db,sid,turn_id,candidate)['action']!='duplicate':
                        audit['long_term']+=1
                if episode:
                    episode={**episode,'session_id':sid,'share_project':False}
                    episode['episode_id']=digest([sid,turn_id,episode['event_type']])
                    audit['episodes']+=db.execute('INSERT OR IGNORE INTO episodes VALUES (?,?,?,0)',
                        (episode['episode_id'],sid,json.dumps(episode,ensure_ascii=False))).rowcount
        db.execute('INSERT INTO migrations VALUES (?,?)',(2,json.dumps(audit)))
    write_json(Path(target.path).parent/'migration-v2-audit.json',audit)
    return audit

def rollback_copy(backup, destination):
    """Prepare an isolated old-code workspace data copy; never overwrite live data."""
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with sqlite3.connect(Path(backup).resolve().as_uri()+'?mode=ro',uri=True) as old, sqlite3.connect(destination/'facts.sqlite3') as restored:
        old.backup(restored)
        for sid,snapshot in old.execute('SELECT session_id,snapshot FROM sessions'):
            from .storage import safe_id
            write_json(destination/(safe_id(sid)+'.state.json'),json.loads(snapshot))
    return destination
