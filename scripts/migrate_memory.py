"""Migrate one explicitly selected local memory directory, without deleting old data."""
import argparse
import json
from pathlib import Path
from paper_agent.long_term import MemoryStore
from paper_agent.migration import migrate, migrate_file_snapshots

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--confirm-local-owner',action='store_true',help='Only after reviewing that this directory belongs to the current local user/project')
    args=parser.parse_args()
    target=MemoryStore((args.output or args.directory)/'memory.sqlite3')
    audits=[migrate(args.directory/'facts.sqlite3',target,args.confirm_local_owner),
            migrate_file_snapshots(args.directory,target,args.confirm_local_owner)]
    print(json.dumps([{k:v for k,v in audit.items() if k not in {'records','files'}} for audit in audits],ensure_ascii=False))

if __name__=='__main__': main()
