"""GPU spend reflection: before a GPU job starts, write down why it is worth the owner's money, and look at it.

Not a gate. The owner (2026-11-09): the spend guard "should not be a strict guard but rather just an incentive to
reflect if this training is useful and valuable". reflect() never refuses a job. It prints the reflection AGENTS.md
asks for before spending (the question, the approved target it serves, the decision it could change, its expected
cost, the cheaper check that came first, and why the owner would care), with notes on anything that deserves a
second thought, and appends it to data/spend-ledger.jsonl. Acting on a note is the agent's judgment in line with
AGENTS.md: spending beyond the owner's limits, or toward a new target, still means asking the owner first.

Inputs it reflects against:
  infra/config/spend-limits.json  per-job GPU-hour limit, and a weekly limit when the owner sets one (null means none)
  infra/config/targets.json       approved targets, their motions, training variants and held-out validation motions

Each reflection reserves the job's provider execution timeout, the most it can bill, so the ledger's weekly sum is an
upper bound. The provider helper calls consume() before every 'run' request; a run with no reflection in this
process is recorded as such and goes ahead.
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMITS = ROOT / 'infra/config/spend-limits.json'
TARGETS = ROOT / 'infra/config/targets.json'
LEDGER = ROOT / 'data/spend-ledger.jsonl'
_open = []  # reflections of this process not yet used by a provider 'run' call


def _load(path, notes):
    rel = path.relative_to(ROOT).as_posix()
    if not path.exists():
        notes.append(f'{rel} is missing')
        return {}
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        notes.append(f'{rel} has uncommitted changes: the owner has not seen them as a commit')
    return json.loads(path.read_text())


def _reserved_last_7_days(now):
    if not LEDGER.exists():
        return 0.0
    since = now - datetime.timedelta(days=7)
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return sum(r.get('reserved_gpu_hours', 0) for r in rows if datetime.datetime.fromisoformat(r['utc']) >= since)


def _on_target(name, listed):
    """A listed motion, its mirror image (NAME-mirrored) or a listed leg move with listed arm moves (LEGS-with-ARMS)."""
    if name in listed:
        return True
    if name.endswith('-mirrored') and name[:-len('-mirrored')] in listed:
        return True
    return any(name[:i] in listed and name[i + len('-with-'):] in listed for i in range(len(name)) if name.startswith('-with-', i))


def reflect(job, reserved_gpu_hours, target, question, decision, motions=(), cheaper_check=None, value=None):
    """Print and record why this job is worth running, with notes worth a second look. Never refuses."""
    notes = []
    limits, targets = _load(LIMITS, notes), _load(TARGETS, notes)
    approved = {t['id']: t for t in targets.get('targets', [])}
    if target not in approved:
        notes.append(f'target {target!r} is not an approved target ({sorted(approved)}): a new target is the owner\'s call')
    else:
        listed, held = set(approved[target]['motions']), set(approved[target].get('held_out', []))
        shown = sorted(set(motions) & held)
        if shown:
            notes.append(f'{len(shown)} held-out validation motions are on the training tape and stop measuring unseen '
                f'agility: {shown[:12]}')
        off = sorted(m for m in set(motions) - held if not _on_target(m, listed))
        if off:
            notes.append(f'{len(off)} training motions are neither target motions nor variants of them: {off[:12]}')
    for name, text in (('question', question), ('decision', decision), ('cheaper check that came first', cheaper_check),
                       ('value to the owner', value)):
        if not isinstance(text, str) or not text.strip():
            notes.append(f'no {name} written down')
    per_job = limits.get('per_job_gpu_hours')
    if per_job is not None and reserved_gpu_hours > per_job:
        notes.append(f'reserves {reserved_gpu_hours:.2f} GPU-h, above the owner\'s per-job limit of {per_job}: ask the owner first')
    now = datetime.datetime.now(datetime.timezone.utc)
    week = _reserved_last_7_days(now)
    weekly = limits.get('weekly_gpu_hours')
    if weekly is not None and week + reserved_gpu_hours > weekly:
        notes.append(f'{week:.2f} GPU-h reserved in the last 7 days + {reserved_gpu_hours:.2f} is above the weekly limit {weekly}: ask the owner first')
    row = dict(utc=now.isoformat(), job=job, reserved_gpu_hours=round(reserved_gpu_hours, 3), target=target,
        question=question, decision=decision, cheaper_check=cheaper_check, value=value, week_before=round(week, 3),
        notes=notes)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a') as out:
        out.write(json.dumps(row) + '\n')
    _open.append(dict(row, reserved_gpu_hours=reserved_gpu_hours))  # unrounded: consume() compares the exact timeout
    lines = [f'GPU job {job}: up to {reserved_gpu_hours:.2f} GPU-h ({week:.2f} reserved in the last 7 days)',
        f'  target:        {target}', f'  question:      {question}', f'  decision:      {decision}',
        f'  cheaper check: {cheaper_check}', f'  value:         {value}']
    lines += [f'  note: {n}' for n in notes] or ['  notes: none']
    print('\n'.join(lines), file=sys.stderr)
    return row


def consume(execution_timeout_ms):
    """Called by the provider helper before every 'run' request. Records a run that has no reflection; never refuses."""
    hours = execution_timeout_ms / 3.6e6
    for row in _open:
        if row['reserved_gpu_hours'] + 1e-6 >= hours:
            _open.remove(row)
            return row
    now = datetime.datetime.now(datetime.timezone.utc)
    row = dict(utc=now.isoformat(), job=None, reserved_gpu_hours=round(hours, 3), week_before=round(_reserved_last_7_days(now), 3),
        notes=['a GPU run started without a written reflection (lab.spend_guard.reflect)'])
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a') as out:
        out.write(json.dumps(row) + '\n')
    print(f'GPU run of up to {hours:.2f} GPU-h has no written reflection; recorded in {LEDGER.name}', file=sys.stderr)
    return row


def summary():
    now = datetime.datetime.now(datetime.timezone.utc)
    limits = json.loads(LIMITS.read_text()) if LIMITS.exists() else {}
    return dict(reserved_last_7_days=round(_reserved_last_7_days(now), 2), limits=limits)


if __name__ == '__main__':
    print(json.dumps(summary(), indent=2))
