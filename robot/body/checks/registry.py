"""A check is one module in modules/ that registers a function run(ctx) -> result under a name.
Result shape (every check, no pass/fail):
  per_part      {part: {...measurements in native units...}}
  worst_cases   [{part, mating_part, pose, location_mm, value, unit, what}]   worst first
  bad_events    {name: count}
  evidence      [repo-relative paths]
  notes         free text (method, resolution)
"""
import importlib
import pkgutil

CHECKS = {}


def register(name, applies=None):
    """applies(ctx) -> bool: does the check apply to the scoped parts (default: always)."""
    def deco(fn):
        CHECKS[name] = dict(run=fn, applies=applies or (lambda ctx: True), doc=(fn.__doc__ or "").strip())
        return fn
    return deco


def load_all():
    import checks.modules as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"checks.modules.{m.name}")
    return CHECKS
