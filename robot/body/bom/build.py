"""The robot BOM and build sequence, assembled from the body definitions (robot/**/<builder>.bom.yaml, schema
robot/body/bom/schema.json). This script has no knowledge of any part: the agents who own a builder declare in its
definition what it generates, which manifest parts it owns and what each is (line, make / buy, spec, supplier or
unknown), and how they go together. build.py resolves those declarations against the selected manifest, adds what it can
read from files (whether each STEP / mesh resolves and how, sha256, volume and mass from the manifest, the print setup
through checks/process.py), and writes the BOM and the per-subassembly sequence. What it cannot resolve is a finding that
names the definition file to fix (findings.json, and the bom check); nothing is filled in.

Output (derived, default .cache/work/bom/): bom.json, bom.csv, BOM.md (the builder's page), assembly.json and ASSEMBLY.md
(build sequence), findings.json, dashboard-bom.json.

  .venv/bin/python -B robot/body/bom/build.py [--manifest PATH] [--out DIR] [--assembly-report PATH] [--publish]

--publish writes the design-dashboard BOM of the selected complete body (data/results.sqlite design_boms) and points the
robot planning BOM at it (robot_bom).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "robot/body")); sys.path.insert(0, str(HERE))
import definitions as DEF  # noqa: E402
import files as FILES  # noqa: E402   (generic file resolution: disk, regenerable, blob index, path moves, bucket records)

DATA = ROOT / "robot/body/checks/data"


class _Ctx:
    """The little of checks.ctx that checks.process.effective reads."""

    def __init__(self, parts, settings):
        self.parts, self._settings, self.process_override = parts, settings, {}

    def settings(self, n):
        out = json.loads(json.dumps(self._settings.get("_defaults", {})))
        for k, v in self._settings.get(n, {}).items():
            out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
        return out


def print_setup(ctx, n):
    from checks.process import effective
    e = effective(ctx, n)
    keep = ("printer", "material", "nozzle_mm", "layer_height_mm", "line_width_mm", "up", "plate_yaw_deg")
    return dict({k: e.get(k) for k in keep}, source={k: e["source"].get(k) for k in keep if k in e["source"]},
                material_alternatives=e.get("material_alternatives") or [])


ASSEMBLY_REPORT = ROOT / ".cache/work/cad-speed/checks/assembly/report.json"

def assembly_index(names, order):
    """{part name: [{assembly, step (id), number (1-based within the assembly), as: base | part | screw | with}]}."""
    from checks.modules import assembly as A
    out = defaultdict(list)
    for an in A.build_order(order):
        a = order["assemblies"][an]
        for n in A.resolve(names, a.get("base", [])):
            out[n].append(dict(assembly=an, step="base", number=0, **{"as": "base"}))
        for i, st in enumerate(a.get("steps", [])):
            for key in ("part", "screw", "with"):
                for n in A.resolve(names, st.get(key)):
                    out[n].append(dict(assembly=an, step=st.get("id"), number=i + 1, **{"as": key}))
    return dict(out)


def tool_of(spec):
    """Driver for a screw spec (checks/data/tools.json via the head standard): 'hex key 2.0 mm', 'Phillips PH0', or None."""
    if not spec or spec.get("kind") != "screw":
        return None
    from checks import screwlib as SL
    std = spec.get("standard") or spec.get("head_standard")
    ht = next((k for k, v in SL.tools()["head_types"].items() if std in v.get("standards", [])), None)   # checks/data/tools.json
    if ht is None:
        return None
    th = (spec.get("thread") or "")
    if spec.get("family") == "PA":
        th = "PA" + th.lstrip("M")
    try:
        tl = SL.tool_for(ht, th)
    except Exception:  # noqa: BLE001
        return None
    if tl is None:
        return None
    size, _, kind, _ = tl
    return f"hex key {size:g} mm" if kind == "hex_key" else f"Phillips {size}"


def assembly_report(manifest_sha, path=None):
    """The assembly check's per-step measurements {(assembly, step id): row} when the report is complete and on this manifest."""
    f = Path(path) if path else ASSEMBLY_REPORT
    if not f.exists():
        return {}, None
    try:
        r = json.loads(f.read_text())
    except json.JSONDecodeError:
        return {}, None
    if not r.get("complete") or r.get("manifest_sha256") != manifest_sha or "assembly" not in r.get("checks", {}):
        return {}, f"{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f} not used (incomplete or another manifest)"
    out = {}
    for an, w in r["checks"]["assembly"].get("assemblies", {}).items():
        for row in w["steps"]:
            out[(an, row.get("id"))] = row
    return out, str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f)


def measured(row):
    """One-line measurement of a step from the assembly check, or None."""
    if not row:
        return None
    if row.get("error"):
        return row["error"]
    if row["kind"] == "part":
        if row.get("first_contact_mm") is None:
            return f"insertion path clear over {row.get('path_mm')} mm" + (f" (seat overlap {row['seat_overlap_mm3']} mm3)" if row.get("seat_overlap_mm3") else "")
        return (f"first contact {row['first_contact_mm']} mm out with {row.get('first_part')}, max overlap {row.get('max_overlap_mm3')} mm3"
                + (f", still {row['overlap_at_max_mm3']} mm3 at {row.get('path_mm')} mm" if row.get("overlap_at_max_mm3") else ""))
    s = f"driver path {row.get('obstruction_mm3', 0)} mm3 obstructed" + (f" (first {row.get('first_part')} at {row.get('first_distance_mm')} mm)" if row.get("obstruction_mm3") else "")
    if row.get("closed_in_by"):
        s += f"; closed in later by {', '.join(row['closed_in_by'][:3])}"
    if row.get("covered_later_by"):
        s += f"; covered later by {', '.join(row['covered_later_by'][:3])}"
    if row.get("screw_errors"):
        s += f"; {len(row['screw_errors'])} screw(s) not measured"
    return s


def assembly_sequence(bom, parts, order, report_path=None):
    """Per-subassembly build sequence from the declared order: per step its parts and hardware with their BOM line, the tool,
    the direction, the declared status (designer / seeded), the definition file, and the assembly check's measurement (a
    step the check measured clear is check-supported); per assembly the parts and hardware it consumes and the tools."""
    from checks.modules import assembly as A
    line_of = {p["part"]: l for l in bom["lines"] for p in l["parts"]}
    meas, rep_src = assembly_report(bom["manifest_sha256"], report_path)
    names = list(parts)
    out = []
    for no, an in enumerate(A.build_order(order), 1):
        a = order["assemblies"][an]
        consumed, tools, steps = defaultdict(int), defaultdict(int), []

        def item(n):
            l = line_of.get(n)
            consumed[l["id"] if l else "(no BOM line)"] += 1
            for c in (l or {}).get("contains_per_part", {}).get(n, []):
                consumed[c["line"]] += c["quantity"]
            t = tool_of((l or {}).get("spec"))
            if t:
                tools[t] += 1
            return dict(part=n, line=l and l["id"], line_name=l and l["name"], role=parts[n].get("role"), tool=t)
        base = [item(n) for n in A.resolve(names, a.get("base", []))]
        for i, st in enumerate(a.get("steps", [])):
            kind = "part" if "part" in st else "screw"
            row = meas.get((an, st.get("id")))
            m = measured(row)
            supported = bool(row) and not row.get("error") and (
                (kind == "part" and row.get("path_first_contact_mm") is None and (row.get("seat_parts_released_at_mm") or 0) < (row.get("path_mm") or 60))
                or (kind == "screw" and not row.get("screw_errors") and row.get("obstruction_mm3", 1) == 0))
            steps.append(dict(number=i + 1, id=st.get("id"), kind=kind, file=st.get("file"),
                              items=[item(n) for n in A.resolve(names, st.get(kind))],
                              with_items=[item(n) for n in A.resolve(names, st.get("with"))],
                              direction=st.get("direction"),
                              status="designer" if st.get("designer") else "seeded" if st.get("seeded") else "declared",
                              check_supported=supported if row else None, derived=st.get("derived"), measured=m))
        k = dict(steps=len(steps), check_supported=sum(1 for s in steps if s["check_supported"]),
                 not_supported=sum(1 for s in steps if s["check_supported"] is False), not_measured=sum(1 for s in steps if s["check_supported"] is None),
                 designer=sum(1 for s in steps if s["status"] == "designer"), seeded=sum(1 for s in steps if s["status"] == "seeded"),
                 parts=sum(len(s["items"]) for s in steps if s["kind"] == "part"), screws=sum(len(s["items"]) for s in steps if s["kind"] == "screw"),
                 with_items=sum(len(s["with_items"]) for s in steps))
        out.append(dict(number=no, name=an, file=a.get("file"), doc=a.get("doc"), after=a.get("after", []), base=base, steps=steps, counts=k,
                        consumes=[dict(line=x, name=(next((l["name"] for l in bom["lines"] if l["id"] == x), x)), quantity=v) for x, v in sorted(consumed.items(), key=lambda kv: str(kv[0]))],
                        tools=[dict(tool=t, screws=v) for t, v in sorted(tools.items())]))
    idx = assembly_index(names, order)
    missing = sorted(n for n in names if n not in idx)
    tot = lambda key: sum(x["counts"][key] for x in out)  # noqa: E731
    return dict(source="the assembly sections of the body definitions (*.bom.yaml)", check_report=rep_src, manifest=bom["manifest"],
                manifest_sha256=bom["manifest_sha256"], assemblies=out, parts_in_no_step=missing,
                counts=dict(assemblies=len(out), steps=tot("steps"), check_supported=tot("check_supported"), not_supported=tot("not_supported"),
                            not_measured=tot("not_measured"), designer=tot("designer"), seeded=tot("seeded"), parts_in_no_step=len(missing), parts=len(names)))


FILE_FLAGS = ("file_missing", "file_in_bucket_not_indexed", "mesh_only", "no_geometry", "mesh_npz_not_stl")


def build(manifest_path, hardware_json=None, assembly_report_path=None, definitions=None):
    """The BOM of a manifest from the definitions. Returns the bom dict (lines, counts, findings, assembly_sequence, ...)."""
    man = json.loads((ROOT / manifest_path).read_text())
    if isinstance(man.get("parts"), dict):
        man["parts"] = [{"name": k, **v} for k, v in man["parts"].items()]
    parts = {p["name"]: p for p in man["parts"]}
    settings = json.loads((DATA / "part_settings.json").read_text())
    rows = []
    for p in man["parts"]:
        r = FILES.part_files(p)
        r["section"] = None             # sections are declared on the definitions' part entries
        rows.append(r)
    names = list(parts)
    rowmap = {r["name"]: r for r in rows}
    D = definitions or DEF.load()
    claims = D.claims(names)
    order = D.order(names)
    gen_order = D.generator_order()
    aidx = assembly_index(names, order)
    ctx = _Ctx(parts, settings)
    lines, contains_per_part = {}, defaultdict(dict)
    AGREE = ("name", "make_buy", "kind", "spec", "supplier", "part_number", "unit_cost", "currency")

    def add_line(lid, e, file, qty_declared, kinds):
        l = lines.get(lid)
        if l is None:
            l = lines[lid] = dict(id=lid, name=e.get("name", lid), kinds=set(), make_buy=e.get("make_buy", "unknown"), quantity=0, quantity_declared=0,
                                  unit=e.get("unit", "ea"), spec=e.get("spec"), supplier=e.get("supplier"), part_number=e.get("part_number"),
                                  unit_cost=e.get("unit_cost"), currency=e.get("currency"), url=e.get("url"), notes=[], definitions=[], parts=[], material=e.get("material"))
        else:
            for k in AGREE:
                if e.get(k) is not None and l.get(k if k != "kind" else "kind_first") not in (None, e.get(k)) and k != "kind":
                    D.find("line_disagrees", file, f"line {lid}", f"{k} {e.get(k)!r} differs from {l.get(k)!r} declared in {', '.join(l['definitions'])}")
                if k != "kind" and l.get(k) is None and e.get(k) is not None:
                    l[k] = e.get(k)
        l["kinds"] |= set(kinds)
        if e.get("notes") and e["notes"] not in l["notes"]:
            l["notes"].append(e["notes"])
        if file not in l["definitions"]:
            l["definitions"].append(file)
        l["quantity_declared"] += qty_declared or 0
        return l
    for e in D.entries:
        hit = [n for n in DEF.match(names, e["match"]) if claims.get(n) is e]
        g = D.generator(e.get("generator")) if e.get("generator") else None
        if e.get("generator") and g is None:
            D.find("generator_unknown", e["file"], f"parts[{e['index']}] line {e['line']}", f"generator {e['generator']!r} is declared nowhere")
        l = add_line(e["line"], e, e["file"], e.get("quantity"), [e["kind"]])
        l["quantity"] += len(hit)
        for n in hit:
            r = rowmap[n]
            f = dict(part=n, definition=e["file"], owning_link=r["owning_link"], section=e.get("section") or r["section"], role=r["role"],
                     step=r["step_resolved"] or r["native_step"], step_status=r["step_status"], step_sha256=parts[n].get("native_step_sha256"),
                     mesh=r["mesh_resolved"] or r["mesh"], mesh_status=r["mesh_status"], mesh_sha256=parts[n].get("mesh_sha256"),
                     step_cold_record=r.get("step_cold_record"), mesh_cold_record=r.get("mesh_cold_record"),
                     volume_mm3=parts[n].get("volume_mm3"), mass_kg=parts[n].get("mass_kg"),
                     generator=g and dict(id=g["id"], file=g["file"], command=" ".join(g["command"]), rebuild=g.get("rebuild", True)),
                     assembly_steps=[f"{x['assembly']}:{x['step']}" for x in aidx.get(n, [])],
                     flags=[x for x in r["flags"] if x in FILE_FLAGS] + ([] if aidx.get(n) else ["no_assembly_step"]))
            if e["kind"] == "printed":
                f["print"] = dict(e["process"], source="definition") if e.get("process") else print_setup(ctx, n)
                f["print_settings_entry"] = bool(e.get("process")) or n in settings
            l["parts"].append(f)
            if e["kind"] == "printed" and not f["print_settings_entry"]:
                D.find("printed_without_process", e["file"], f"parts[{e['index']}] line {e['line']}", f"{n}: no print setup (definition process or part_settings.json)", part=n)
            if "file_missing" in f["flags"]:
                D.find("declared_file_missing", e["file"], f"parts[{e['index']}] line {e['line']}", f"{n}: {r['native_step'] if r['step_status'] == 'missing' else r['mesh']} resolves nowhere", part=n)
        for c in e.get("contains", []):
            for n in hit:
                if c.get("part") and c["part"] != n:
                    continue
                cl = add_line(c["line"], c, e["file"], c["quantity"], ["hardware"])
                cl["quantity"] += c["quantity"]
                cl.setdefault("in_parts", []).append(n)
                contains_per_part[n].setdefault(c["line"], 0)
                contains_per_part[n][c["line"]] += c["quantity"]
        if e["kind"] == "hardware" and e.get("make_buy") == "buy" and not e.get("spec"):
            D.find("hardware_without_spec", e["file"], f"parts[{e['index']}] line {e['line']}", f"{len(hit)} hardware part(s) with no declared spec")
    blocks = ("process", "expected_hardware", "fit_classes", "covers", "exact_fit", "loadpath_slices", "geometry_thresholds", "fem", "edges")
    for n, e in claims.items():
        tag = f"parts[{e['index']}] line {e['line']}"
        for k in blocks:
            if k in e and k in settings.get(n, {}):
                D.find("setting_shadowed", e["file"], tag, f"{n}: part_settings.json {k} is not read (the definition's wins): delete it there", part=n)
        if e["kind"] == "printed" and parts[n].get("role") in ("printed_structural_connector", "printed_structural_carrier") \
                and "fem" not in e and "fem" not in settings.get(n, {}):
            D.find("connector_without_fem", e["file"], tag, f"{n}: printed connector / carrier without an FEM setup (fem block)", part=n)
    mats = json.loads((DATA / "materials.json").read_text())
    for l in lines.values():        # manifest mass vs volume x the declared material's density (printed parts)
        for f in l["parts"]:
            mat = (f.get("print") or {}).get("material")
            rho = (mats.get(mat) or {}).get("density_g_cm3") if mat else None
            v, mkg = f.get("volume_mm3"), f.get("mass_kg")
            if rho and v and mkg and abs(1000 * mkg - v * rho / 1000) > 0.05 * v * rho / 1000:
                D.find("mass_differs_from_material", f["definition"], f"line {l['id']}",
                       f"{f['part']}: manifest {1000 * mkg:.2f} g, volume x {mat} density {v * rho / 1000:.2f} g", part=f["part"])
    for n in settings:
        if not n.startswith("_") and n not in parts and not settings[n].get("_kept"):
            D.find("setting_for_absent_part", "robot/body/checks/data/part_settings.json", n, f"{n}: settings for a part the manifest does not hold")
    for n, r in rowmap.items():         # the manifest as a consistent snapshot: its files still hash as it recorded them
        for field, path in FILES.hash_mismatch(parts[n], r):
            e = claims.get(n)
            D.find("file_changed_since_manifest", e and e["file"], e and f"parts[{e['index']}] line {e['line']}",
                   f"{n}: {field} {path} no longer matches the sha256 the manifest recorded (rerun its generator)", part=n)
    FILES.save_sha_cache()
    if not any(g.get("manifest_out") == manifest_path for g in D.generators):
        D.find("manifest_without_generator", None, None, f"{manifest_path}: no definition declares a generator with this manifest_out (the rebuild cannot recreate it)")
    for g, inp, exists in D.external_inputs():
        if not exists:
            D.find("input_missing", g["file"], g["id"], f"input {inp} is written by no generator and not on disk")
    for n in names:
        if n not in aidx:
            e = claims.get(n)
            D.find("no_assembly_step", e and e["file"], e and f"parts[{e['index']}] line {e['line']}", f"{n}: no assembly step names it", part=n)
    out_lines = []
    for l in lines.values():
        l["kinds"] = sorted(l["kinds"])
        l["notes"] = " | ".join(l["notes"]) or None
        l["sections"] = sorted({p["section"] or "unassigned" for p in l["parts"]}) or sorted({p["section"] or "unassigned" for n in l.get("in_parts", []) for p in [rowmap[n]]})
        masses = [parts[p["part"]].get("mass_kg") for p in l["parts"]]
        l["unit_mass_g"] = round(1000 * masses[0], 3) if masses and None not in masses and len(set(masses)) == 1 else None
        l["total_mass_g"] = round(1000 * sum(masses), 2) if masses and None not in masses else None
        if l["parts"] and not l.get("material"):
            l["material"] = rowmap[l["parts"][0]["part"]]["material"] or (l["parts"][0].get("print") or {}).get("material")
        l["assembly"] = line_assembly([p["part"] for p in l["parts"]] + l.get("in_parts", []), aidx)
        l["flags"] = sorted({x for p in l["parts"] for x in p["flags"]})
        l["origin"] = ", ".join(l["definitions"])
        out_lines.append(l)
    rank = {"printed": 0, "purchased": 1, "included": 2, "hardware": 3, "made": 4, "other": 5}
    out_lines.sort(key=lambda l: (min(rank.get(k, 9) for k in l["kinds"]), l["sections"][0] if l["sections"] else "", l["id"]))
    for l in out_lines:
        if l.get("in_parts"):
            for n in l["in_parts"]:
                line_of_part = next((x for x in out_lines for p in x["parts"] if p["part"] == n), None)
                if line_of_part is not None:
                    line_of_part.setdefault("contains_per_part", {}).setdefault(n, []).append(dict(line=l["id"], quantity=contains_per_part[n][l["id"]]))
    sha = hashlib.sha256((ROOT / manifest_path).read_bytes()).hexdigest()
    gens = [dict(id=i, file=D.generator(i)["file"], command=" ".join(D.generator(i)["command"]), manifest_out=D.generator(i).get("manifest_out")) for i in gen_order]
    defects = [dict(d, file=f["file"], matched=DEF.match(names, d["parts"])) for f in D.files for d in f["doc"].get("defects", [])]
    for d in defects:
        if not d["matched"]:
            D.find("defect_matches_nothing", d["file"], d["id"], "defect names no part of the manifest (fixed or renamed?)")
    for f in D.joint_findings(man):       # declared joint ranges (`joints:`) against the manifest's joints / controls
        D.findings.append(f)
    flagged = sorted(n for n, r in rowmap.items() if {"file_missing", "file_in_bucket_not_indexed"} & set(r["flags"]))
    explained = {n for d in defects for n in d["matched"]}
    bom = dict(manifest=manifest_path, manifest_sha256=sha, definitions=[f["file"] for f in D.files], generators=gens,
               defects=dict(listed=defects, file_flagged=len(flagged), file_flagged_unexplained=[n for n in flagged if n not in explained]),
               lines=out_lines, findings=D.findings,
               joints={n: dict(range_deg=d["range_deg"], file=d["file"], entry=d["entry"], doc=d.get("doc"), decision=d.get("decision"))
                       for n, d in sorted(D.joints.items())},
               config={k: {n: dict(v) for n, v in c.items()} for k, c in sorted(D.config.items())})
    bom["counts"] = counts(out_lines, rows, D.findings, claims, names, aidx)
    bom["assembly_sequence"] = assembly_sequence(bom, parts, order, assembly_report_path)
    return bom


def line_assembly(names, aidx):
    """[{assembly, steps: [ids]}] of a line's parts, in build order."""
    by = {}
    for n in names:
        for x in aidx.get(n, []):
            by.setdefault(x["assembly"], [])
            if x["step"] not in by[x["assembly"]]:
                by[x["assembly"]].append(x["step"])
    return [dict(assembly=k, steps=v) for k, v in by.items()]


def assembly_text(l, limit=6):
    return "; ".join(f"{a['assembly']} {', '.join(a['steps'][:limit])}{' ...' if len(a['steps']) > limit else ''}" for a in l.get("assembly") or []) or "-"


def counts(lines, rows, findings, claims, names, aidx):
    by, fk = defaultdict(int), defaultdict(int)
    for l in lines:
        by[l["make_buy"]] += 1
    for f in findings:
        fk[f["kind"]] += 1
    buy = [l for l in lines if l["make_buy"] == "buy"]
    pr = [p for l in lines if "printed" in l["kinds"] for p in l["parts"]]
    return dict(lines=len(lines), lines_by_make_buy=dict(by), parts=len(names), parts_claimed=len(claims), parts_with_assembly_step=sum(1 for n in names if aidx.get(n)),
                printed_parts=len(pr), printed_with_native_step=sum(1 for p in pr if p["step_status"] not in ("none", "missing")),
                printed_with_print_setup=sum(1 for p in pr if p.get("print_settings_entry")),
                buy_lines=len(buy), buy_lines_supplier_known=sum(1 for l in buy if (l.get("supplier") or "unknown") != "unknown"),
                buy_lines_part_number_known=sum(1 for l in buy if (l.get("part_number") or "unknown") != "unknown"),
                buy_lines_supplier_empty=sum(1 for l in buy if not l.get("supplier")),
                lines_quantity_differs_from_declared=sum(1 for l in lines if l["quantity_declared"] and l["quantity"] != l["quantity_declared"]),
                parts_mesh_only=sum(1 for r in rows if "mesh_only" in r["flags"]), parts_file_missing=sum(1 for r in rows if "file_missing" in r["flags"]),
                parts_file_in_bucket_not_indexed=sum(1 for r in rows if "file_in_bucket_not_indexed" in r["flags"]),
                parts_without_geometry=sum(1 for r in rows if "no_geometry" in r["flags"]),
                hardware_kinds_to_buy=sum(1 for l in lines if "hardware" in l["kinds"] and l["make_buy"] == "buy"),
                findings=len(findings), findings_by_kind=dict(fk))


def write_csv(bom, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "make_buy", "quantity", "quantity_declared", "unit", "sections", "material", "printer", "nozzle_mm", "layer_height_mm", "up",
                    "supplier", "part_number", "unit_cost", "currency", "parts", "step_files", "generator", "definition", "assembly_steps", "flags"])
        for l in bom["lines"]:
            pr = next((p["print"] for p in l["parts"] if p.get("print")), {}) or {}
            w.writerow([l["id"], l["name"], l["make_buy"], l["quantity"], l["quantity_declared"], l.get("unit", "ea"), "; ".join(l.get("sections") or []),
                        l.get("material") or "", pr.get("printer") or "", pr.get("nozzle_mm") or "", pr.get("layer_height_mm") or "", pr.get("up") or "",
                        l.get("supplier") or "", l.get("part_number") or "", l.get("unit_cost") or "", l.get("currency") or "",
                        "; ".join(p["part"] for p in l["parts"]) or "; ".join(l.get("in_parts", [])),
                        "; ".join(p.get("step") or "" for p in l["parts"]),
                        "; ".join(sorted({(p.get("generator") or {}).get("command") or "" for p in l["parts"]})), l["origin"],
                        "; ".join(sorted({s for p in l["parts"] for s in p["assembly_steps"]})), "; ".join(l["flags"])])


def findings_table(findings, limit=None):
    L = ["| kind | definition file | entry | what |", "|---|---|---|---|"]
    for f in findings[:limit]:
        L.append(f"| {f['kind']} | {f['file'] or '(none claims it)'} | {f['entry'] or ''} | {f['what']} |")
    if limit and len(findings) > limit:
        L.append(f"| ... | | | {len(findings) - limit} more in findings.json |")
    return L


def markdown(bom):
    c = bom["counts"]
    L = ["# APE0 robot BOM", "",
         f"Generated by `robot/body/bom/build.py` from `{bom['manifest']}` (sha256 {bom['manifest_sha256'][:12]}) and the "
         f"{len(bom['definitions'])} body definitions (`*.bom.yaml` next to each builder). Do not edit this page: edit the definition "
         "file a line or finding names and rerun.", "",
         f"{c['lines']} lines over {c['parts']} CAD parts ({c['parts_claimed']} claimed by a definition): "
         + ", ".join(f"{v} {k}" for k, v in c["lines_by_make_buy"].items()) + ". "
         f"Printed parts {c['printed_parts']}: with a native STEP {c['printed_with_native_step']}, with a print setup {c['printed_with_print_setup']}. "
         f"Parts with an assembly step: {c['parts_with_assembly_step']}. Buy lines {c['buy_lines']}: supplier known {c['buy_lines_supplier_known']}, "
         f"part number known {c['buy_lines_part_number_known']}, supplier empty {c['buy_lines_supplier_empty']}. Mesh only {c['parts_mesh_only']}; "
         f"file missing {c['parts_file_missing']}, in the bucket but not in the local blob index {c['parts_file_in_bucket_not_indexed']}; "
         f"no geometry {c['parts_without_geometry']}. Hardware kinds to buy: {c['hardware_kinds_to_buy']}. Findings: {c['findings']}.", ""]
    L += [f"## Findings ({c['findings']}): what the definitions do not say yet", "",
          "Each names the definition file to fix; " + ", ".join(f"{k} {v}" for k, v in sorted(c["findings_by_kind"].items())) + ".", ""]
    L += findings_table(bom["findings"], 60) + [""]
    L += ["## How to rebuild the body files", "", "Generator order derived from the declared inputs and outputs (`robot/body/bom/rebuild_chain.py`):", ""]
    for i, g in enumerate(bom["generators"]):
        L.append(f"{i + 1}. `{g['command']}` ({g['id']}, {g['file']})" + (f" -> `{g['manifest_out']}`" if g.get("manifest_out") else ""))
    L += ["", "Rerun in an isolated copy and compare with the selected body: `.venv/bin/python -B robot/body/bom/rebuild_chain.py setup`, `run`, `compare`; "
          "it ends with the assembly check, this BOM and the bom check.", ""]
    d = bom["defects"]
    L += [f"## Open defects (declared in the definitions): {len(d['listed'])} groups; files flagged missing or only in the bucket {d['file_flagged']}, "
          f"not explained by a defect {len(d['file_flagged_unexplained'])}", "", "| defect | definition | parts | why / state | fix | task |", "|---|---|---|---|---|---|"]
    for g in d["listed"]:
        L.append(f"| {g['id']} | {g['file']} | {len(g['matched'])}: {', '.join(g['matched'][:5])}{' ...' if len(g['matched']) > 5 else ''} | "
                 f"{g.get('why') or g.get('state') or ''} | {g.get('fix') or ''} | {g.get('task') or ''} |")
    if d["file_flagged_unexplained"]:
        L.append(f"| (not explained) | | {', '.join(d['file_flagged_unexplained'])} | | | |")
    L += ["", "## Print list", "", "| line | qty | section | material | printer / nozzle / layer | up | STEP | generator | assembly | definition |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for l in bom["lines"]:
        if "printed" not in l["kinds"] or not l["parts"]:
            continue
        p = l["parts"][0]; pr = p.get("print") or {}
        L.append(f"| {l['name']} | {l['quantity']} | {', '.join(l['sections'])} | {pr.get('material') or '-'} | {pr.get('printer') or '-'} / "
                 f"{pr.get('nozzle_mm') or '-'} / {pr.get('layer_height_mm') or '-'} | {pr.get('up') or '-'} | {p['step_status']} | "
                 f"{(p.get('generator') or {}).get('id') or '-'} | {assembly_text(l)} | {l['origin']} |")
    L += ["", "## Buy list (servos, electronics, cells, other bought parts)", "", "| line | qty | make / buy | sections | supplier / part no. | assembly | notes |", "|---|---|---|---|---|---|---|"]
    for l in bom["lines"]:
        if not ({"purchased", "included", "made", "other"} & set(l["kinds"])) or "printed" in l["kinds"]:
            continue
        L.append(f"| {l['name']} | {l['quantity']} | {l['make_buy']} | {', '.join(l.get('sections') or [])} | "
                 f"{' '.join(x for x in (l.get('supplier'), l.get('part_number')) if x) or '-'} | {assembly_text(l)} | {l.get('notes') or ''} |")
    L += ["", "## Hardware", "", "| spec | qty | make / buy | sections | tool | assembly |", "|---|---|---|---|---|---|"]
    for l in bom["lines"]:
        if l["kinds"] == ["hardware"]:
            L.append(f"| {l['name']} | {l['quantity']} | {l['make_buy']} | {', '.join(l['sections'])} | {tool_of(l.get('spec')) or '-'} | {assembly_text(l, 3)} |")
    seq = bom.get("assembly_sequence")
    L += ["", "## Assembly order", "", "Declared in the definitions' `assembly` sections; the build sequence per subassembly is `ASSEMBLY.md` / `assembly.json` "
          f"next to this page. Parts with a step: {c['parts_with_assembly_step']} of {c['parts']}.", ""]
    if seq:
        L += ["| # | assembly | steps | check-supported | not supported | not measured | designer | seeded | parts | screws | carried (with) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in seq["assemblies"]:
            k = a["counts"]
            L.append(f"| {a['number']} | {a['name']} | {k['steps']} | {k['check_supported']} | {k['not_supported']} | {k['not_measured']} | {k['designer']} | {k['seeded']} | {k['parts']} | {k['screws']} | {k['with_items']} |")
    return "\n".join(L) + "\n"


def assembly_markdown(seq):
    """ASSEMBLY.md: the build sequence per subassembly (generated from the definitions, never hand-written)."""
    k = seq["counts"]
    L = ["# APE0 assembly sequence", "",
         f"Generated by `robot/body/bom/build.py` from {seq['source']} and the BOM of `{seq['manifest']}` (sha256 {seq['manifest_sha256'][:12]}). "
         "Do not edit this page: edit the `assembly` section of the definition file each step names (draft new steps with "
         "`robot/body/checks/assembly_seed.py propose`) and rerun the BOM.", "",
         f"{k['assemblies']} subassemblies, {k['steps']} steps: {k['check_supported']} supported by the assembly check, {k['not_supported']} not, "
         f"{k['not_measured']} not measured; declared: {k['designer']} by a designer, {k['seeded']} seeded drafts. "
         f"Manifest parts in no step: {k['parts_in_no_step']} of {k['parts']}"
         + (f" ({', '.join(seq['parts_in_no_step'][:20])}{' ...' if len(seq['parts_in_no_step']) > 20 else ''})" if seq["parts_in_no_step"] else "") + ".",
         f"Measurements: {seq['check_report'] or 'no assembly check report on this manifest; run `robot/body/checks/run.py --checks assembly --label assembly`'}. "
         "Direction = the manifest-frame vector the part travels as it goes in (docs/operations/working-rules.md: frame).", "",
         "Check-supported: the insertion path touches nothing over its range (parts already overlapping at the seat let go), or the driver path is free. "
         "Status is what the definition declares: **designer** (the owner's order) or **seeded** (a draft).", ""]
    for a in seq["assemblies"]:
        c = a["counts"]
        L += [f"## {a['number']}. {a['name']} ({a.get('file')})", "", a.get("doc") or "", "",
              f"After: {', '.join(a['after']) or 'nothing (first)'}. Present at the start: {', '.join(x['part'] for x in a['base']) or 'only what the earlier assemblies built'}. "
              f"{c['steps']} steps ({c['check_supported']} check-supported, {c['designer']} designer, {c['seeded']} seeded), {c['parts']} parts, {c['screws']} screws, {c['with_items']} carried items.", "",
              "Consumes (per BOM line): " + "; ".join(f"{x['name']} x{x['quantity']}" for x in a["consumes"]) + ".", "",
              "Tools: " + ("; ".join(f"{t['tool']} ({t['screws']} screws)" for t in a["tools"]) or "none recorded") + ".", "",
              "| step | kind | parts / screws (BOM line) | with | direction | tool | status | check | measured | definition |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in a["steps"]:
            its = s["items"]
            if s["kind"] == "screw" and len(its) > 1:
                what = f"{len(its)} x {', '.join(sorted({x['line_name'] or '-' for x in its}))}: {', '.join(x['part'] for x in its)}"
            else:
                what = "; ".join(f"{x['part']} ({x['line_name'] or '-'})" for x in its) or "-"
            w = ", ".join(x["part"] for x in s["with_items"]) or ""
            d = s["direction"] if isinstance(s["direction"], str) else (str([round(v, 3) for v in s["direction"]]) if s["direction"] else "-")
            tool = ", ".join(sorted({x["tool"] for x in its + s["with_items"] if x["tool"]})) or ""
            chk = {True: "supported", False: "not supported", None: "-"}[s["check_supported"]]
            L.append(f"| {s['id']} | {s['kind']} | {what} | {w} | {d} | {tool} | {s['status']} | {chk} | {s['measured'] or '-'} | {s['file']} |")
        L.append("")
    return "\n".join(L) + "\n"


def dashboard_rows(bom):
    cat = {"make": "produced", "buy": "purchased", "included": "context", "context": "context"}
    rows = []
    for l in bom["lines"]:
        pr = next((p["print"] for p in l["parts"] if p.get("print")), None)
        rows.append(dict(id=l["id"], name=l["name"], category=cat.get(l["make_buy"], "unresolved"), quantity=l["quantity"],
                         part_names=[p["part"] for p in l["parts"]] or l.get("in_parts", []), material=l.get("material"), manufacturer=l.get("supplier"),
                         part_number=l.get("part_number"), unit_mass_g=l.get("unit_mass_g"), definition=l["origin"],
                         process=pr and {k: pr[k] for k in ("printer", "nozzle_mm", "layer_height_mm", "up") if pr.get(k) is not None},
                         notes=[x for x in [l.get("notes"), ("flags: " + ", ".join(l["flags"])) if l["flags"] else None] if x]))
    return rows


def design_bom_payload(bom):
    cat = {"make": "Produced parts", "buy": "Purchased parts", "included": "Included with purchased parts", "context": "Context"}
    items = []
    for l in bom["lines"]:
        pr = next((p["print"] for p in l["parts"] if p.get("print")), None)
        names = [p["part"] for p in l["parts"]] or l.get("in_parts", [])
        items.append(dict(id=l["id"], name=l["name"], category=cat.get(l["make_buy"], "To specify"), quantity=l["quantity"],
                          unit=l.get("unit", "ea"), material=str(l.get("material") or ""), supplier=l.get("supplier") or "",
                          part_number=l.get("part_number") or "", unit_mass_g=l.get("unit_mass_g"),
                          unit_cost=l.get("unit_cost"), currency=l.get("currency") or "",
                          notes=" | ".join(x for x in [l.get("notes") or "", f"print: {pr.get('printer')} {pr.get('material')} {pr.get('nozzle_mm')} mm, up {pr.get('up')}" if pr else "",
                                                       f"definition: {l['origin']}",
                                                       f"CAD parts: {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}"] if x)))
    return dict(scope=f"Generated from {bom['manifest']} by robot/body/bom/build.py from the body definitions (*.bom.yaml; sha256 {bom['manifest_sha256'][:12]}).",
                completeness="partial", notes="Suppliers and part numbers as the definitions declare them ('unknown' where no repo source names one); prices are not recorded.",
                exclusions=[], items=items)


def design_bom_for(manifest_path=None):
    """The design-dashboard BOM payload of a manifest (default: the selected body), for study publishers."""
    mp = manifest_path or json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
    return design_bom_payload(build(mp))


def publish(bom):
    import sqlite3
    sys.path.insert(0, str(ROOT / "infra/dashboard"))
    import design_bom as DB
    design = json.loads((ROOT / "robot/design.json").read_text())
    sel = next((v for v in design["section_selections"].values() if v.get("folder") == design.get("body")), None)
    rid = sel and (sel.get("result_id") or sel.get("study_id"))
    with sqlite3.connect(ROOT / "data/results.sqlite") as db:
        db.row_factory = sqlite3.Row
        DB.init_schema(db)
        if not db.execute("SELECT 1 FROM results WHERE id=?", (rid,)).fetchone():
            raise SystemExit(f"selected body result {rid!r} is not in data/results.sqlite; publish the study first")
        cur = DB.get_bom(db, rid)
        DB.put_bom(db, rid, design_bom_payload(bom), cur.get("revision") if cur else None)
        plan = DB.get_plan(db)
        DB.put_plan(db, dict(title="Robot BOM (selected body)",
                             notes=f"The selected complete body {rid}; its BOM is generated by robot/body/bom/build.py from the body definitions.",
                             exclusions=[], assemblies=[dict(result_id=rid, count=1)]), plan.get("revision") if plan else None)
        db.commit()
    return rid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=json.loads((ROOT / "robot/design.json").read_text())["body_manifest"])
    ap.add_argument("--out", default=str(ROOT / ".cache/work/bom"))
    ap.add_argument("--assembly-report", help=f"assembly check report to show per step (default {ASSEMBLY_REPORT.relative_to(ROOT)}, used when complete and on this manifest)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    bom = build(a.manifest, assembly_report_path=a.assembly_report)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "assembly.json").write_text(json.dumps(bom["assembly_sequence"], indent=1, default=str) + "\n")
    (out / "ASSEMBLY.md").write_text(assembly_markdown(bom["assembly_sequence"]))
    (out / "bom.json").write_text(json.dumps(bom, indent=1, default=str) + "\n")
    write_csv(bom, out / "bom.csv")
    (out / "BOM.md").write_text(markdown(bom))
    (out / "findings.json").write_text(json.dumps(bom["findings"], indent=1, default=str) + "\n")
    (out / "dashboard-bom.json").write_text(json.dumps(dashboard_rows(bom), indent=1, default=str) + "\n")
    print(json.dumps(dict(bom["counts"], assembly=bom["assembly_sequence"]["counts"]), indent=1))
    if a.publish:
        print("published design BOM and robot plan for", publish(bom))


if __name__ == "__main__":
    main()
