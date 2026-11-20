#!/usr/bin/env python3
"""Refuse staged new paths that version by file name or put bookkeeping in Git.

See docs/operations/git-and-bookkeeping.md. Checks only paths that are added or
renamed in the index, so existing names are grandfathered until next renamed.

Usage: infra/tools/check_repo_names.py            # check the staged index (pre-commit)
       infra/tools/check_repo_names.py PATH ...   # check the given paths
"""
import re
import subprocess
import sys

# A version token is a separate word: _v3, -r2, .v12, rev4, round2, pass3, attempt1.
VERSION = re.compile(r'(?:^|[_.\-])(?:v|r|rev|round|pass|attempt|iter)\d+[a-z]?(?=$|[_.\-])', re.I)
SUFFIX_WORDS = re.compile(r'(?:^|[_.\-])(?:final|new|old|copy|bak|tmp|draft|latest|fixed)(?=$|[_.\-])', re.I)
DATE = re.compile(r'(?:19|20)\d{2}-?(?:0[1-9]|1[0-2])-?(?:0[1-9]|[12]\d|3[01])')
BOOKKEEPING = re.compile(
    r'(?:^|[_.\-/])(?:reviews?|authori[sz]ation|verification|receipts?|attestation|bindings?|preimages?|claims|'
    r'dispatch|provider-status|watchdog|teardown|preflight|lifecycle|handoff|publication-verification|'
    r'regeneration|input-snapshot|executed-inputs|frozen-inputs|closeout|custody)(?=$|[_.\-/])', re.I)
RECORD_EXT = re.compile(r'\.(?:json|jsonl|md|txt|csv|log|stdout|stderr|lock|pid)(?:\.gz)?$', re.I)
ALWAYS_BOOKKEEPING = re.compile(r'\.(?:log|stdout|stderr|lock|pid)$', re.I)
# Fixed product and hardware names that look like version tokens.
ALLOWED_WORDS = {'v4l2'}
RECIPE_NAMES = {'scratch-regeneration.json'}


def staged_new_paths():
    out = subprocess.run(['git', 'diff', '--cached', '--name-status', '-M', '--diff-filter=AR', '-z'],
                         capture_output=True, text=True, check=True).stdout.split('\0')
    paths, i = [], 0
    while i < len(out) - 1:
        status = out[i]
        if status.startswith('R'):
            paths.append(out[i + 2]); i += 3
        else:
            paths.append(out[i + 1]); i += 2
    return paths


def problems(path):
    found = []
    if path.startswith(('.bookkeeping/', 'artifacts/', '.cache/')):
        return ['ignored folder; do not commit it']
    for segment in path.split('/'):
        stem = re.sub(r'\.[A-Za-z0-9]+(?:\.gz)?$', '', segment)
        token = VERSION.search(stem)
        if token and token.group(0).strip('_.-').lower() not in ALLOWED_WORDS:
            found.append(f'version token "{token.group(0).strip("_.-")}" in "{segment}": keep one name, let git hold versions')
        word = SUFFIX_WORDS.search(stem)
        if word:
            found.append(f'"{word.group(0).strip("_.-")}" in "{segment}": edit the file and commit instead of a copy')
        if DATE.search(stem) and not path.startswith('data/'):
            found.append(f'date stamp in "{segment}": name by content; dates go in commits and records')
    # Pinned environments and storage regeneration recipes are sources, not process records.
    if path.startswith('infra/requirements/') or path.rsplit('/', 1)[-1] in RECIPE_NAMES:
        pass
    elif ALWAYS_BOOKKEEPING.search(path) or (RECORD_EXT.search(path) and BOOKKEEPING.search(path)):
        found.append('process record: write it under .bookkeeping/ (Git-ignored)')
    if path.startswith('docs/research/') and path.count('/') > 2:
        found.append('research is one document per topic directly in docs/research/')
    return found


def main(argv):
    paths = argv or staged_new_paths()
    bad = {p: problems(p) for p in paths}
    bad = {p: v for p, v in bad.items() if v}
    if not bad:
        return 0
    print('Refused by docs/operations/git-and-bookkeeping.md:', file=sys.stderr)
    for path, reasons in sorted(bad.items()):
        for reason in dict.fromkeys(reasons):
            print(f'  {path}: {reason}', file=sys.stderr)
    print('Rename by content, move process records to .bookkeeping/, or ask the owner before using --no-verify.',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
