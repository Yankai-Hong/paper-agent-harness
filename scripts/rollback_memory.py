"""Prepare an isolated rollback copy; run old Git version against the new directory."""
import argparse
from pathlib import Path
from paper_agent.migration import rollback_copy

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('backup',type=Path)
    p.add_argument('new_directory',type=Path)
    a=p.parse_args()
    print(rollback_copy(a.backup,a.new_directory))
