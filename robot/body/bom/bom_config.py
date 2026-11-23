"""Configured values of the body, read from the body definitions (robot/**/<builder>.bom.yaml): the single source of joint
ranges (`joints:`) and of every other configured value (`config:`) for the builders, the design checks and the design
dashboard (owner 2026-11-23: "a single source of truth ... and that is the bom.yaml"). No value lives here; a value that
is not declared raises (or, for the tolerant readers, is returned as missing) so it is reported, never filled in.

    from robot.body.bom import bom_config as config          # or sys.path insert of robot/body/bom, then `import bom_config`
    config.joint_ranges()                        # {joint: [lo, hi]}
    config.joint_range("R_knee")                 # [0.0, 95.0]; KeyError naming where to declare it
    config.value("thigh-shell", "GAP_MM")        # one configured value of a definition
    config.apply_joint_ranges(manifest)          # writes joints[]/controls[] range_deg from the declarations

Guide: robot/body/bom/README.md, 'Configured values'."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import definitions as DEF  # noqa: E402


class Undeclared(KeyError):
    pass


def _defs(defs=None, key="values:joints"):
    """The definitions. Under make.py (the generator cache) the *.bom.yaml reads are recorded as `key` (the declared
    values asked for), not as file reads, so an edit of an unrelated definition does not rerun a builder."""
    if defs is not None:
        return defs
    site = sys.modules.get("sitecustomize")
    hook = getattr(site, "gencache_values", None)
    return hook(key, DEF.load) if hook else DEF.load()


def joints(defs=None):
    """{joint: dict(range_deg, doc, decision, file, definition, entry)} as declared."""
    return _defs(defs).joints


def joint_ranges(defs=None):
    return {n: list(d["range_deg"]) for n, d in joints(defs).items()}


def joint_range(name, defs=None):
    d = joints(defs).get(name)
    if d is None:
        raise Undeclared(f"joint {name!r}: no range declared; add it to `joints:` of the definition that owns it")
    return list(d["range_deg"])


def value(definition, name, defs=None):
    """Configured value NAME of definition id `definition` (its `config:` block)."""
    c = _defs(defs, f"values:config:{definition}").config.get(definition, {})
    if name not in c:
        raise Undeclared(f"{definition}: config {name!r} is not declared in its <builder>.bom.yaml")
    return c[name]["value"]


def values(definition, defs=None):
    return {k: v["value"] for k, v in _defs(defs, f"values:config:{definition}").config.get(definition, {}).items()}


def apply_joint_ranges(manifest, defs=None, strict=True):
    """Set manifest joints[].range_deg and controls[].range_deg from the declarations (driven joints untouched).
    Returns log lines. strict: an undriven manifest joint without a declaration raises."""
    J, log = joints(defs), []
    missing = []
    for rec in list(manifest.get("joints", [])) + [c for c in manifest.get("controls", []) if isinstance(c, dict)]:
        n = rec.get("name")
        if n in J:
            cur = rec.get("range_deg")
            if cur is None or [float(x) for x in cur] != J[n]["range_deg"]:
                log.append(f"{n}: range_deg {cur} -> {J[n]['range_deg']} ({J[n]['file']} {J[n]['entry']})")
            rec["range_deg"] = list(J[n]["range_deg"])      # always the declared value (floats), key position kept
        elif not (rec.get("driven_by") or rec.get("coupled_to")):
            missing.append(n)
    if missing and strict:
        raise Undeclared(f"joints without a declared range: {sorted(set(missing))}")
    return log


# ------------------------------------------------------------------ per-part values (the part's definition entry)
def _materials():
    import json
    return json.loads((Path(__file__).resolve().parents[1] / "checks/data/materials.json").read_text())


def part_entry(name, defs=None):
    """The definition entry (parts[i], with its file) that claims manifest part `name`, matched by its `match` globs in
    discovery order (the BOM's first-claim rule). Raises when no definition claims it."""
    import fnmatch
    D = _defs(defs, f"values:part:{name}")
    for e in D.entries:
        if any(fnmatch.fnmatchcase(name, p) for p in DEF.names_of(e["match"])):
            return e
    raise Undeclared(f"part {name!r}: no definition claims it (parts: match in the owning <builder>.bom.yaml)")


def material(name, defs=None):
    """The material declared for part `name` (its entry's process.material)."""
    e = part_entry(name, defs)
    m = (e.get("process") or {}).get("material")
    if not m:
        raise Undeclared(f"part {name!r}: {e['file']} parts[{e['index']}] declares no process.material")
    return m


def density_kg_mm3(name, defs=None):
    """Density of part `name` from its declared material (robot/body/checks/data/materials.json, datasheet values)."""
    m = material(name, defs)
    row = _materials().get(m)
    if not row or not row.get("density_g_cm3"):
        raise Undeclared(f"part {name!r}: material {m!r} has no density_g_cm3 in robot/body/checks/data/materials.json")
    return float(row["density_g_cm3"]) * 1e-6


def mass_kg(name, volume_mm3, digits=6, defs=None):
    """Mass of part `name` from its volume and its declared material."""
    return round(float(volume_mm3) * density_kg_mm3(name, defs), digits)
