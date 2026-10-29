"""Confirmed local memories, scoped and versioned; never a paper evidence store."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from .storage import digest

class MemoryStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.transaction() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS long_term (
              memory_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, scope TEXT NOT NULL,
              kind TEXT NOT NULL, subject TEXT NOT NULL, attribute TEXT NOT NULL,
              text TEXT NOT NULL, status TEXT NOT NULL, source_type TEXT NOT NULL,
              turn_id TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
              replaces TEXT, version INTEGER NOT NULL, provenance_refs TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS current_memory ON long_term(scope,subject,attribute)
              WHERE status='active';
            CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY,version INTEGER NOT NULL,snapshot TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(episode_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,payload TEXT NOT NULL,indexed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS episode_audit_cursor(session_id TEXT PRIMARY KEY,last_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS migrations(version INTEGER PRIMARY KEY, audit TEXT NOT NULL);
            ''')

    @contextmanager
    def transaction(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def snapshot(self, session_id):
        with self.transaction() as db:
            row=db.execute('SELECT snapshot FROM sessions WHERE session_id=?',(session_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def memories(self, session_id, status=None, shared=False):
        with self.transaction() as db:
            rows=db.execute('SELECT * FROM long_term WHERE scope=? OR (? AND scope=?) ORDER BY created_at,memory_id',
                            ('session:'+session_id,int(shared),'project')).fetchall()
            return [dict(r,provenance_refs=json.loads(r['provenance_refs'])) for r in rows if status is None or r['status']==status]

    def put(self, db, session_id, turn_id, memory, shared=False):
        from .memory_stores import canonical_key
        if memory.get('source_type')!='user_explicit':
            raise ValueError('Only explicit user declarations may enter long-term memory')
        kind=memory.get('kind','preference')
        if kind not in {'preference','constraint','decision'}:
            raise ValueError('Events belong in episodic memory')
        scope=memory.get('scope','session:'+session_id)
        if scope != 'session:'+session_id and not (shared and scope=='project'):
            raise ValueError('Memory scope is not authorized')
        subject=canonical_key(memory.get('subject','user'))
        attribute=canonical_key(memory.get('attribute',memory['key']))
        text=memory['text'].strip()
        if not text: raise ValueError('Empty memory')
        status=memory.get('status','candidate')
        if status not in {'candidate','active','conflict'}: raise ValueError('Invalid new status')
        mid=digest([scope,subject,attribute,turn_id,text])
        exists=db.execute('SELECT memory_id FROM long_term WHERE memory_id=?',(mid,)).fetchone()
        if exists: return {'action':'duplicate','memory_id':mid}
        old=db.execute("SELECT * FROM long_term WHERE scope=? AND subject=? AND attribute=? AND status='active'",(scope,subject,attribute)).fetchone()
        replaces=None
        if old and old['text']==text: return {'action':'duplicate','memory_id':old['memory_id']}
        if old:
            if status=='active' and (memory.get('explicit_update') or memory.get('replaces')==old['memory_id']):
                replaces=old['memory_id']
                db.execute("UPDATE long_term SET status='superseded',updated_at=? WHERE memory_id=?",(time.time(),replaces))
            else: status='conflict'
        version=db.execute('SELECT coalesce(max(version),0)+1 FROM long_term WHERE scope=? AND subject=? AND attribute=?',(scope,subject,attribute)).fetchone()[0]
        now=time.time()
        refs=memory.get('provenance_refs',[{'session_id':session_id,'turn_id':turn_id,'source':'user_message'}])
        db.execute('INSERT INTO long_term VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (mid,session_id,scope,kind,subject,attribute,text,status,'user_explicit',turn_id,now,now,replaces,version,json.dumps(refs)))
        return {'action':'update' if replaces else status,'memory_id':mid}

    def delete(self, session_id, memory_id, shared=False):
        with self.transaction() as db:
            db.execute("UPDATE long_term SET status='archived',updated_at=? WHERE memory_id=? AND (scope=? OR (? AND scope='project'))",(time.time(),memory_id,'session:'+session_id,int(shared)))

    def episodes(self,session_id,pending=False,limit=None):
        with self.transaction() as db:
            sql='SELECT payload FROM episodes WHERE session_id=?'+(' AND indexed=0' if pending else '')+' ORDER BY episode_id'
            args=[session_id]
            if limit is not None: sql+=' LIMIT ?'; args.append(limit)
            return [json.loads(r[0]) for r in db.execute(sql,args)]

    def revise_episode(self, session_id, episode_id, content=None, archived=False, shared=False):
        """Logical deletion or revision; original versions retained in the SQLite payload."""
        with self.transaction() as db:
            row=db.execute('SELECT payload FROM episodes WHERE episode_id=?',(episode_id,)).fetchone()
            if not row: raise ValueError('Unknown episode')
            event=json.loads(row[0])
            if event['session_id']!=session_id and not (shared and event.get('share_project')):
                raise ValueError('Episode scope not authorized')
            old={k:v for k,v in event.items() if k!='versions'}
            event['versions']=event.get('versions',[])+[old]
            event['status']='archived' if archived else 'active'
            if content is not None: event['content']=content
            event['updated_at']=time.time()
            db.execute('UPDATE episodes SET payload=?,indexed=0 WHERE episode_id=?',(json.dumps(event,ensure_ascii=False),episode_id))

    def audit_batch(self,session_id,limit=32):
        with self.transaction() as db:
            cursor=db.execute('SELECT last_id FROM episode_audit_cursor WHERE session_id=?',(session_id,)).fetchone()
            after=cursor[0] if cursor else ''
            sql='SELECT payload FROM episodes WHERE session_id=? AND episode_id>? ORDER BY episode_id LIMIT ?'
            rows=db.execute(sql,(session_id,after,limit)).fetchall()
            if not rows and after: rows=db.execute(sql,(session_id,'',limit)).fetchall()
            return [json.loads(r[0]) for r in rows]
