"""Deterministic check that rejects yes/no results: every result is a measurement or a graded score.

check(record) raises BinaryResult when a result contains:
  - a boolean value outside a configuration block (plan, config, args, options, settings, policy);
  - a key that names a verdict (pass, fail, ok, feasible, keep, exclude, covered, go, verdict, ...);
  - a string value that is a verdict ("pass", "fail", "yes", "no", "go", "no-go", "accepted", ...);
  - a grade outside 0-5 or not a number.
grade() turns a native-unit measurement into a continuous 0-5 grade against a stated reference scale. The reference is
the scale of the curve, not a threshold: nothing in a grade jumps.

  python -m infra.storage.score_guard REPORT.json [...]
"""
import json
import math
import re
import sys
from pathlib import Path

CONFIG_KEYS = {'plan', 'config', 'configuration', 'args', 'arguments', 'options', 'settings', 'policy', 'provider_policy',
    'capabilities', 'flags', 'recipe', 'env', 'environment', 'viewer', 'view', 'display', 'widget', 'ui', 'render', 'camera'}
DATA_FLAG = re.compile(r'^(has|show|hide|fit|use|is_visible|visible|enabled|auto)(_|$)|mask')  # describe data, not results
VERDICT_WORDS = {'pass', 'passed', 'passes', 'fail', 'failed', 'fails', 'ok', 'success', 'succeeded', 'go', 'nogo',
    'feasible', 'infeasible', 'keep', 'kept', 'exclude', 'excluded', 'covered', 'qualified', 'verdict', 'gate', 'gates',
    'meets', 'acceptable', 'viable', 'eligible'}  # 'accepted' is left out: servos report an accepted target
VERDICT_VALUES = {'pass', 'passed', 'fail', 'failed', 'yes', 'no', 'go', 'no-go', 'nogo', 'no go', 'ok', 'not ok',
    'accepted', 'rejected', 'feasible', 'infeasible', 'true', 'false', 'kept', 'excluded'}
GRADE_KEYS = {'grade', 'grades', 'score', 'scores'}


class BinaryResult(ValueError):
    pass


def _words(key):
    return set(re.split(r'[^a-z0-9]+', re.sub(r'([a-z])([A-Z])', r'\1_\2', key).lower())) - {''}


def problems(value, path='$', in_config=False):
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            here, cfg = f'{path}.{k}', in_config or k.lower() in CONFIG_KEYS
            if not cfg and _words(str(k)) & VERDICT_WORDS and _verdict_shaped(v):
                out.append(f'{here}: key names a verdict; report the measurement or a 0-5 grade instead')
            if not cfg and k.lower() in GRADE_KEYS:
                out += _grade_problems(v, here)
            out += problems(v, here, cfg or bool(DATA_FLAG.search(str(k).lower())) and _flags(v))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out += problems(v, f'{path}[{i}]', in_config)
    elif isinstance(value, bool):
        if not in_config:
            out.append(f'{path}: boolean result; report the measurement or a 0-5 grade instead')
    elif isinstance(value, str):
        if not in_config and value.strip().lower() in VERDICT_VALUES:
            out.append(f'{path}: verdict string {value!r}; report the measurement or a 0-5 grade instead')
    return out


def _verdict_shaped(v):
    """A verdict-named key holding a count or a measurement is fine; a flag, verdict string or name list is not."""
    if isinstance(v, (bool, str)):
        return True
    if isinstance(v, list):
        return any(isinstance(x, (bool, str, dict)) for x in v)
    return isinstance(v, dict)


def _flags(v):
    return isinstance(v, bool) or isinstance(v, list) and all(isinstance(x, bool) for x in v)


def _grade_problems(v, path):
    vals = v.values() if isinstance(v, dict) else v if isinstance(v, list) else [v]
    bad = [x for x in vals if isinstance(x, bool) or isinstance(x, (int, float)) and not 0 <= x <= 5]
    return [f'{path}: grade {x!r} is not a number from 0 to 5' for x in bad]


def check(record):
    found = problems(record)
    if found:
        raise BinaryResult('yes/no result rejected:\n  ' + '\n  '.join(found[:40]) + (f'\n  ... {len(found) - 40} more' if len(found) > 40 else ''))
    return record


def grade(value, reference, better='lower'):
    """Continuous 0-5 grade. better='lower': 5 at zero, 2.5 at the reference, toward 0 far above it.
    better='higher': 0 at zero, 2.5 at the reference, toward 5 far above it."""
    x = max(float(value), 0.0) / float(reference)
    g = 5 * math.exp(-x * math.log(2)) if better == 'lower' else 5 * (1 - math.exp(-x * math.log(2)))
    return round(g, 2)


if __name__ == '__main__':
    status = 0
    for name in sys.argv[1:]:
        try:
            check(json.loads(Path(name).read_text()))
            print(f'{name}: scores only')
        except BinaryResult as error:
            print(f'{name}: {error}')
            status = 1
    sys.exit(status)
