"""Scope: which parts of the manifest a run looks at. Selectors combine with AND; none given = every part;
hardware is never scoped (printed parts, servo cases, shells... are). A scoped part is later checked against its neighbours."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess

from checks import bodylib as BL
from checks.core import DATA, ROOT, R, kit, sha, WORK

LAST = WORK / "checks" / "last-run"


def _definition_sections(body):
    """{section: [part names]} as the body definitions declare them (`section` on their part entries)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    out = {}
    for n, e in DEF.load().claims(list(body.parts)).items():
        if e.get("section"):
            out.setdefault(e["section"], []).append(n)
    return out


def section_links(section, body):
    """(links, part names) of a robot section: the parts whose body definition declares it (no link rule)."""
    sec = _definition_sections(body)
    if section not in sec:
        raise SystemExit(f"unknown section {section!r}; declared in the definitions: {sorted(sec)}")
    return set(), sorted(sec[section])


def part_files(p):
    return [f for f in (p.get("native_step"), p.get("mesh")) if f]


def part_key(p):
    """Content key of a part: sha256 of its source files + placement. A source file that is not on disk gives the key
    'missing:<file>' (never raises: a hardware STEP of an unrelated part must not stop a run)."""
    h = hashlib.sha256()
    for f in part_files(p):
        try:
            h.update(sha(BL.src(f)).encode())
        except FileNotFoundError:
            return f"missing:{f}"
    h.update(json.dumps([p.get("placement_offset_mm"), p.get("native_step_offset_mm"), p.get("roll_flip")]).encode())
    return h.hexdigest()


def changed_since_rev(body, rev, names=None):
    """Parts (of names, default all) one of whose source files differs from (or is absent in) Git revision rev (git blob compare).
    Files git does not track (artifacts/ is ignored) count as changed: there is no revision to compare with."""
    out = set()
    for n in (names if names is not None else body.parts):
        p = body.parts[n]
        for f in part_files(p):
            path = BL.src(f)
            try:
                rel = str(path.resolve().relative_to(ROOT))
                old = subprocess.run(["git", "rev-parse", f"{rev}:{rel}"], cwd=ROOT, capture_output=True, text=True)
                new = subprocess.run(["git", "hash-object", str(path)], cwd=ROOT, capture_output=True, text=True)
                if old.returncode or old.stdout.strip() != new.stdout.strip():
                    out.add(n)
            except ValueError:
                out.add(n)
    return out


def changed_since_last_run(body, manifest, cands, neighbours_of):
    """(why, missing): candidates selected since the last run on this manifest, each with its reason: source_missing (own source
    file not on disk), never_recorded (no key from an earlier run), changed (own key differs), neighbour_changed: <name> (a neighbour's key
    differs from the recorded one), neighbour_never_recorded: <name> (a neighbour has no recorded key; only when none changed). Only candidates and their neighbours are hashed. missing = number of hashed parts whose
    source file is not on disk."""
    f = LAST / (manifest.stem + ".json")
    prev = json.loads(f.read_text()) if f.exists() else {}
    keys = {}

    def key(n):
        if n not in keys:
            keys[n] = part_key(body.parts[n])
        return keys[n]
    why = {}
    for n in cands:
        if key(n).startswith("missing:"):
            why[n] = "source_missing"
        elif n not in prev:
            why[n] = "never_recorded"
        elif prev[n] != key(n):
            why[n] = "changed"
        else:
            nb = list(neighbours_of([n]))
            ch = [m for m in nb if m in prev and prev[m] != key(m)]
            new = [m for m in nb if m not in prev]
            if ch:        # a changed neighbour is a reason; never-recorded neighbours (no earlier key to differ from) only when none changed
                why[n] = f"neighbour_changed: {ch[0]}" + (f" (+{len(ch) - 1} more)" if len(ch) > 1 else "")
            elif new:
                why[n] = f"neighbour_never_recorded: {new[0]}" + (f" (+{len(new) - 1} more)" if len(new) > 1 else "")
    return why, sum(1 for k in keys.values() if k.startswith("missing:"))


def save_last_run(body, manifest, names):
    """Record the keys of names (scope and neighbours)."""
    f = LAST / (manifest.stem + ".json")
    prev = json.loads(f.read_text()) if f.exists() else {}
    prev.update({n: part_key(body.parts[n]) for n in names})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(prev))


def select(body, manifest, parts=(), section=None, link=(), role=(), changed_since=None, neighbours_of=None):
    names = list(body.parts)
    desc = {}
    if parts:
        names = [n for n in names if any(fnmatch.fnmatchcase(n, g) for g in parts)]; desc["parts"] = list(parts)
    if section:
        links, globs = section_links(section, body)
        names = [n for n in names if body.parts[n]["owning_link"] in links or any(fnmatch.fnmatchcase(n, g) for g in globs)]
        desc["section"] = section
    if link:
        names = [n for n in names if any(fnmatch.fnmatchcase(body.parts[n]["owning_link"], g) for g in link)]; desc["link"] = list(link)
    if role:
        names = [n for n in names if body.parts[n].get("role") in role]; desc["role"] = list(role)
    # hardware (fasteners, proxies, nominal hardware) and sleeves are never scoped: sleeves are checked through their parent part
    names = [n for n in names if not BL.is_fastener(n, body.parts[n]) and body.parts[n].get("role") not in
             ("compression_limit_sleeves", "nominal_hardware", "fastener_proxy", "standard_fastener_reference")]
    if changed_since:                  # last: only the other selectors' survivors (and their neighbours) are hashed
        if changed_since == "last-run":
            why, missing = changed_since_last_run(body, manifest, names, neighbours_of)
            ch = set(why)
            desc["source_files_missing"] = missing
            desc["selected_because"] = why
        else:
            ch = changed_since_rev(body, changed_since, names)
        names = [n for n in names if n in ch]; desc["changed_since"] = changed_since
    if not desc:
        desc["default"] = "every part that is not hardware"
    return names, desc
