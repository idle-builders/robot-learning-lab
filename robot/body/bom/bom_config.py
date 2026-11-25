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


def body(defs=None):
    """The robot body as declared (`body:` in one definition, robot/body/complete_body.bom.yaml): dict(manifest, study, doc,
    file). The default body of every tool (make.py body, the checks, the BOM, publish_body.py)."""
    b = _defs(defs, "values:body").body
    if not b:
        raise Undeclared("no definition declares `body:` (manifest, study); declare it once, in robot/body/complete_body.bom.yaml")
    return b


def body_manifest(defs=None):
    """Repository path of the body manifest (`body.manifest`)."""
    return body(defs)["manifest"]


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


def _frozen(v):
    if isinstance(v, list):
        return tuple(_frozen(x) for x in v)
    if isinstance(v, dict):
        return {k: _frozen(x) for k, x in v.items()}
    return v


def unfrozen(v):
    """A configured value with its lists as lists (for constants that were lists of lists / dicts of lists)."""
    if isinstance(v, (list, tuple)):
        return [unfrozen(x) for x in v]
    if isinstance(v, dict):
        return {k: unfrozen(x) for k, x in v.items()}
    return v


class Section(dict):
    """A definition's `config:` values, lists as tuples (the builders' constants were tuples); a missing NAME raises
    Undeclared naming the file to declare it in."""

    def __init__(self, definition, vals):
        super().__init__({k: _frozen(v) for k, v in vals.items()})
        self.definition = definition

    def __missing__(self, name):
        raise Undeclared(f"{self.definition}: config {name!r} is not declared in its <builder>.bom.yaml")


def section(definition, defs=None):
    """All configured values of definition id `definition`, for a builder's module constants:
    `CFG = BOM_CONFIG.section("thigh-shell")`, then `WINDOW = CFG["WINDOW"]`."""
    return Section(definition, values(definition, defs))


def joint_origin(name, defs=None):
    """Declared origin (mm, assembled body frame) of joint `name`; raises when its declaration gives none."""
    d = joints(defs).get(name)
    if d is None or "origin_mm" not in d:
        raise Undeclared(f"joint {name!r}: no origin_mm declared; add it to its `joints:` entry")
    return list(d["origin_mm"])


def joint_axis(name, defs=None):
    """Declared neutral axis of joint `name`; raises when its declaration gives none."""
    d = joints(defs).get(name)
    if d is None or "axis" not in d:
        raise Undeclared(f"joint {name!r}: no axis declared; add it to its `joints:` entry")
    return list(d["axis"])


def apply_joint_frames(joint_records, defs=None):
    """Set origin_world_mm / axis_world_neutral of the joint records whose declaration gives them. Returns the names set."""
    J, done = joints(defs), []
    for rec in joint_records:
        d = J.get(rec.get("name"))
        if not d:
            continue
        if "origin_mm" in d:
            rec["origin_world_mm"] = list(d["origin_mm"])
        if "axis" in d:
            rec["axis_world_neutral"] = list(d["axis"])
        if "origin_mm" in d or "axis" in d:
            done.append(rec["name"])
    return done


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


def material_row(material):
    """The materials.json row of `material` (its entry name), flat: {name, density_g_cm3, <fem keys>}. Raises when absent."""
    row = _materials().get(material)
    if not isinstance(row, dict) or "density_g_cm3" not in row:
        raise Undeclared(f"material {material!r} is not in robot/body/checks/data/materials.json")
    return dict(name=material, density_g_cm3=float(row["density_g_cm3"]), **(row.get("fem") or {}))


def material_table(keys):
    """{key: material_row(entry)} for a builder's {key: materials.json entry} mapping (its config MATERIALS)."""
    return {k: material_row(v) for k, v in keys.items()}


def density_g_mm3(material):
    """Density of a material named by its materials.json entry, g/mm3."""
    return material_row(material)["density_g_cm3"] * 1e-3


def part_entry(name, defs=None):
    """The definition entry (parts[i], with its file) that claims manifest part `name`, matched by its `match` globs in
    discovery order (the BOM's first-claim rule). Raises when no definition claims it."""
    import fnmatch
    D = _defs(defs, f"values:part:{name}")
    for e in D.entries:
        if any(fnmatch.fnmatchcase(name, p) for p in DEF.names_of(e["match"])):
            return e
    raise Undeclared(f"part {name!r}: no definition claims it (parts: match in the owning <builder>.bom.yaml)")


def _material_of_text(text):
    """The materials.json entry a declared text names: the entry itself, else the entry one of whose `matches` words it
    contains (longest word wins; the checks' rule, checks/fem_core.py material_of_text)."""
    if not text:
        return None
    mats = _materials()
    if text in mats:
        return text
    t = str(text).lower()
    hits = [(len(w), k) for k, v in mats.items() if isinstance(v, dict) for w in v.get("matches", []) if w.lower() in t]
    return max(hits)[1] if hits else None


def _declared_material(e):
    """(material, where) an entry declares: process.material; else its spec or its bought contents' specs (spec.material,
    standard, label) when they all name the same materials.json entry."""
    m = (e.get("process") or {}).get("material")
    if m:
        return m, "process.material"
    for src, specs in (("spec", [e.get("spec") or {}]), ("contains", [it.get("spec") or {} for it in e.get("contains") or []])):
        found = set()
        for sp in specs:
            for k in ("material", "standard", "label"):
                hit = _material_of_text(sp.get(k))
                if hit:
                    found.add(hit)
                    break
            else:
                found.add(None)
        if specs and len(found) == 1 and None not in found:
            return found.pop(), src
    return None, None


def material(name, builder=None, defs=None, fallback="PRINTED_MATERIAL"):
    """The material declared for part `name`: its entry's process.material, else what its spec / bought contents name;
    for a part no definition claims, or whose entry names none, the builder definition's config `fallback`
    (PRINTED_MATERIAL, HARDWARE_MATERIAL) when `builder` (a definition id) is given. Raises when nothing declares one."""
    return _material_where(name, builder, defs, fallback)[0]


def _material_where(name, builder, defs, fallback):
    try:
        e = part_entry(name, defs)
    except Undeclared:
        if builder is None:
            raise
        return value(builder, fallback, defs), f"{builder} config {fallback}; no definition claims the part"
    m, src = _declared_material(e)
    if m:
        return m, f"{e['file']} parts[{e['index']}] {src}"
    if builder is None:
        raise Undeclared(f"part {name!r}: {e['file']} parts[{e['index']}] declares no material")
    return value(builder, fallback, defs), f"{builder} config {fallback}; {e['file']} parts[{e['index']}] declares none"


def density_kg_mm3(name, builder=None, defs=None, fallback="PRINTED_MATERIAL"):
    """Density of part `name` from its declared material (robot/body/checks/data/materials.json, datasheet values)."""
    return material_row(material(name, builder, defs, fallback))["density_g_cm3"] * 1e-6


def mass_kg(name, volume_mm3, builder=None, digits=6, defs=None, fallback="PRINTED_MATERIAL"):
    """Mass of part `name` from its volume and its declared material."""
    return round(float(volume_mm3) * density_kg_mm3(name, builder, defs, fallback), digits)


def mass_record(name, volume_mm3, builder=None, digits=6, defs=None, fallback="PRINTED_MATERIAL", basis="Exact native STEP volume"):
    """{mass_kg, mass_basis} of a part from its declared material, naming where the material is declared."""
    m, where = _material_where(name, builder, defs, fallback)
    rho = material_row(m)["density_g_cm3"]
    return dict(mass_kg=round(float(volume_mm3) * rho * 1e-6, digits),
                mass_basis=f"{basis} * {rho:g} g/cm3 ({m}, declared in {where})")
