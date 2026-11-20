#!/usr/bin/env python3
"""Run with disposable scratch and automatic verified private evidence offload."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from infra.storage import storage_lifecycle as storage


def install():
    if sys.platform != 'darwin':
        raise RuntimeError('Automatic local retry installation currently supports macOS')
    label = 'ch.robot-learning-lab.storage'
    path = Path.home() / 'Library/LaunchAgents' / (label + '.plist')
    log = ROOT / '.cache/storage-sync.log'
    log.parent.mkdir(exist_ok=True)
    value = dict(Label=label, ProgramArguments=[str(ROOT / '.venv/bin/python'), '-B',
                 str(ROOT / 'infra/storage/storage_cli.py'), 'sync'], WorkingDirectory=str(ROOT),
                 StartInterval=3600, RunAtLoad=True, ProcessType='Background',
                 StandardOutPath=str(log), StandardErrorPath=str(log))
    encoded = plistlib.dumps(value)
    if path.exists() and path.read_bytes() == encoded:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    domain = 'gui/' + str(os.getuid())
    subprocess.run(['launchctl', 'bootout', domain, str(path)], capture_output=True)
    subprocess.run(['launchctl', 'bootstrap', domain, str(path)], check=True)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    runner = sub.add_parser('run', help='Run producer; {work} is cheap scratch, {evidence} is retained native evidence')
    runner.add_argument('name')
    runner.add_argument('command', nargs=argparse.REMAINDER)
    close = sub.add_parser('finish', help='Declare one existing artifacts run closed and offload its bulk')
    close.add_argument('evidence', type=Path)
    for name in ('sync', 'clean', 'status', 'install'):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.action == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('Supply a command after the run name')
        return storage.run(args.name, command)
    if args.action == 'finish':
        result = storage.finish(args.evidence)
        print(json.dumps(result, indent=2))
        return 0 if result['state'] == 'cold' else 2
    if args.action == 'sync':
        result = storage.sync()
        print(json.dumps([dict(id=r['id'], state=r['state'], error=r.get('last_error')) for r in result], indent=2))
        return int(any(r['state'] == 'retry' for r in result))
    if args.action == 'clean':
        print(json.dumps(dict(removed=storage.clean()), indent=2))
    elif args.action == 'install':
        print(install())
    else:
        rows = [json.loads(p.read_text()) for p in sorted(storage.RECORDS.glob('*.json'))]
        print(json.dumps([dict(id=r['id'], state=r['state'], work=r.get('work'),
                               evidence=r['evidence'], error=r.get('last_error')) for r in rows], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
