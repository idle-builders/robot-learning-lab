"""Apply the declared edits of a composed body manifest (the `compose` entries of the body definitions, schema.json).

A composing script (today robot/body/leg/roll_flip_checks.py `manifest`, stage `roll-flip`) calls

    import compose; log += compose.apply("roll-flip", man)

and every definition's entry `into: roll-flip` is applied in the order their `after` gives: `takes: parts` calls
function(man["parts"]) and stores the list it returns; `takes: manifest` calls function(man), which edits it in place and
returns log lines. The functions are the builders' own (e.g. robot/body/torso/torso_screws.py swap_parts); a section that
needs to edit the composed manifest adds an entry to its own definition, never an import to the composing script
(owner 2026-11-22). Like the BOM tool this module knows no part. Its scripts are imported by module name from their own
folder (as the composing scripts always did), and a call into the running script itself uses that script's module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import definitions as DEF  # noqa: E402


def _function(call):
    script, fn = call.split(":")
    path = (ROOT / script).resolve()
    main = sys.modules.get("__main__")
    if main is not None and getattr(main, "__file__", None) and Path(main.__file__).resolve() == path:
        return getattr(main, fn)
    d = str(path.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    return getattr(importlib.import_module(path.stem), fn)


def apply(stage, man, D=None):
    """Apply every entry into stage to manifest man (in place); returns the log lines. Definition findings about the
    composition (unknown `after`, cycles, a missing script) are printed as warnings and logged; nothing is skipped silently."""
    D = D or DEF.load()
    order = D.composition(stage)
    log = []
    for f in D.findings:
        if f["kind"].startswith("compose"):
            msg = f"WARNING compose: {f['kind']} {f['file']} {f['entry']}: {f['what']}"
            print(msg); log.append(msg)
    for c in order:
        fn = _function(c["call"])
        if c["takes"] == "parts":
            man["parts"] = fn(man["parts"])
        else:
            log += list(fn(man) or [])
    man.setdefault("composed", {})[stage] = [f"{c['id']} ({c['file']})" for c in order]
    return log
