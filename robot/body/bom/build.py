"""The robot BOM: one place that says, per line, what to make or buy, how many, from which files and scripts, with which
print setup, and where it goes in the assembly order.

Inputs (source of truth): the selected body manifest (`body_manifest` in robot/design.json), the hand-maintained line
data robot/body/bom/lines.json (agents edit it: grouping, supplier, part numbers, prices, notes, manual lines,
per-part overrides), the hardware specs (robot/body/bom/hardware.py), the per-part print settings
(robot/body/checks/data/part_settings.json via checks/process.py), the assembly order (checks/data/assembly_order.json),
the builder chain (robot/body/bom/chain.json) and the producer table / file resolution of robot/body/bom/inventory.py.

Output (derived, recreated by this script; default .cache/work/bom/):
  bom.json      lines with parts, quantity (CAD count and any lines.json override), make / buy, files per part (STEP,
                mesh, how each resolves, producer script, regenerate command), print setup, hardware spec, assembly steps,
                flags; plus the build chain and the defect counts
  bom.csv       one row per line (buy / print list)
  BOM.md        the builder's page: print list, buy list, hardware, assembly order, how to regenerate, open defects
  dashboard-bom.json   rows for the CAD BOM panel (infra/dashboard/ui/cad-bom.js)

  .venv/bin/python -B robot/body/bom/build.py [--manifest PATH] [--out DIR] [--hardware-json PATH] [--publish]

--publish writes the design-dashboard BOM of the selected complete body (data/results.sqlite design_boms) and points the
robot planning BOM at it (robot_bom), replacing the stale plan (owner 2026-11-20).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "robot/body")); sys.path.insert(0, str(HERE))
import inventory as INV  # noqa: E402
import hardware as HW  # noqa: E402

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


def stem(name):
    s = re.sub(r"^(upper__|lower__|combined__)", "", name)
    s = re.sub(r"^[LR]_", "", s)
    s = re.sub(r"(_\d+)+$", "", s)
    return s


def sleeve_count(p):
    """Steel sleeves in a <part>_steel_sleeves STEP: (count, {size: count}) of its solids that are sleeves
    (checks/modules/bom.py rule), or (None, {}) when the STEP cannot be read."""
    try:
        from checks.modules.bom import _is_sleeve, sleeve_sizes
        ok = [s for s in sleeve_sizes(p) if _is_sleeve(s)]
        out = defaultdict(int)
        for a, b, c in ok:
            out[f"{a:g} x {b:g} x {c:g}"] += 1
        return len(ok), dict(out)
    except Exception:  # noqa: BLE001
        return None, {}


def chain_commands():
    f = HERE / "chain.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            return None
    return None


def regenerate_command(producer, chain):
    """The command that rewrites a producer's outputs: from chain.json when it names the script, else the script itself."""
    if not producer or producer.startswith("none") or producer == "unknown" or "@" in producer or producer.startswith("pinned"):
        return None
    script = producer.split(" ")[0]
    for st in (chain or {}).get("steps", []):
        if script in (st.get("command") or ""):
            return st["command"]
    return f".venv/bin/python -B {script}"


def hardware_specs(man, manifest_path, path=None):
    sha = hashlib.sha256((ROOT / manifest_path).read_bytes()).hexdigest()
    f = Path(path) if path else ROOT / ".cache/work/bom-hardware/hardware.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d["summary"].get("manifest_sha256") in (None, sha) and d["summary"].get("manifest") == manifest_path:
            names = {p["name"] for p in man["parts"]}
            if set(d["parts"]) <= names:
                return d["parts"], f"{f.relative_to(ROOT)} (manifest sha {'matches' if d['summary'].get('manifest_sha256') == sha else 'not recorded'})"
    sp = HW.specs(man)
    sp.pop("_errors", None)
    for s in sp.values():
        s.pop("_mesh", None)
    return sp, "computed now (no geometry)"


def build(manifest_path, hardware_json=None):
    man, items, rows, settings, order = INV.build(manifest_path)
    parts = {p["name"]: p for p in man["parts"]}
    lines_doc = json.loads((HERE / "lines.json").read_text())
    hw, hw_source = hardware_specs(man, manifest_path, hardware_json)
    chain = chain_commands()
    ctx = _Ctx(parts, settings)
    overrides = lines_doc.get("part_overrides", {})
    rowmap = {r["name"]: r for r in rows}

    def hand_line(n):
        p = parts[n]
        ov = overrides.get(n, {})
        if ov.get("line"):
            return next((l for l in lines_doc["lines"] if l["id"] == ov["line"]), None)
        for l in lines_doc.get("lines", []):
            if l.get("match") and any(fnmatch.fnmatchcase(n, g) for g in l["match"]):
                return l
            if l.get("roles") and p["role"] in l["roles"]:
                return l
            if l.get("motor") and p.get("motor") == l["motor"]:
                return l
        return None

    groups = defaultdict(list)       # line id -> part names
    sleeve_parts, sleeve_qty = {}, defaultdict(int)
    meta = {}
    excluded = {}
    for n, r in rowmap.items():
        p = parts[n]
        if overrides.get(n, {}).get("exclude"):
            excluded[n] = overrides[n]["exclude"]; continue
        hl = hand_line(n)
        if hl:
            lid = hl["id"]; meta.setdefault(lid, dict(hl, _from="lines.json"))
        elif r["kind"] == "printed":
            lid = f"print:{n}"
            meta.setdefault(lid, dict(id=lid, name=n, make_buy="make", _from="generated: printed part"))
        elif p["role"] == "compression_limit_sleeves":
            # bought by size: one line per sleeve size (outer x inner x length), the part listed on each of its sizes
            c, sizes = sleeve_count(parts[n])
            sleeve_parts[n] = (c, sizes)
            for sz, k in (sizes or {}).items():
                lid = "sleeve:" + sz.replace(" ", "")
                meta.setdefault(lid, dict(id=lid, name=f"304 steel compression sleeve {sz} mm (outer x inner x length)", make_buy="buy",
                                          _from="generated: sleeve size from the connector's sleeve STEP"))
                groups[lid].append(n); sleeve_qty[lid] += k
            if not sizes:
                lid = f"sleeve:uncounted:{n}"
                meta.setdefault(lid, dict(id=lid, name=f"steel sleeves of {n} (not counted)", make_buy="buy", _from="generated: sleeve STEP not read"))
                groups[lid].append(n)
            continue
        elif n in hw and hw[n].get("label"):
            lab = hw[n]["label"]
            lid = "hw:" + re.sub(r"[^A-Za-z0-9.]+", "-", lab).strip("-")
            meta.setdefault(lid, dict(id=lid, name=lab, make_buy="buy", spec={k: v for k, v in hw[n].items() if k not in ("measured", "source_of_spec", "step", "stl", "fit_to_old_mesh")},
                                      _from="generated: hardware spec (robot/body/bom/hardware.py)"))
        elif p.get("motor"):
            lid = "servo:" + re.sub(r"[^A-Za-z0-9]+", "-", p["motor"]).strip("-")
            meta.setdefault(lid, dict(id=lid, name=p["motor"], make_buy="buy", _from="generated: manifest motor field"))
        else:
            lid = f"{r['kind']}:{stem(n)}"
            meta.setdefault(lid, dict(id=lid, name=stem(n), make_buy="buy" if r["kind"] in ("purchased", "hardware") else "unknown",
                                      _from=f"generated: {r['kind']} part, name stem"))
        groups[lid].append(n)

    out_lines = []
    for lid, names in groups.items():
        m = meta[lid]
        first = rowmap[names[0]]
        kinds = sorted({rowmap[n]["kind"] for n in names})
        qty_cad = len(names)
        sizes = None
        if lid.startswith("sleeve:"):
            qty_cad = sleeve_qty.get(lid) if not lid.startswith("sleeve:uncounted:") else None
        files = []
        for n in names:
            r = rowmap[n]
            f = dict(part=n, owning_link=r["owning_link"], section=r["section"], role=r["role"],
                     step=r["step_resolved"] or r["native_step"], step_status=r["step_status"],
                     mesh=r["mesh_resolved"] or r["mesh"], mesh_status=r["mesh_status"],
                     producer=r["producer"], producer_class=r["producer_class"], regenerate=regenerate_command(r["producer"], chain),
                     first_geometry=r["first_geometry"], assembly_steps=r["assembly_steps"],
                     flags=r["flags"])
            if n in hw and hw[n].get("step"):
                f["generated_hardware"] = dict(step=hw[n]["step"], stl=hw[n].get("stl"), fit_to_old_mesh=hw[n].get("fit_to_old_mesh"),
                                               regenerate=".venv/bin/python -B robot/body/bom/hardware.py")
            if r["kind"] == "printed":
                f["print"] = print_setup(ctx, n)
                f["print_settings_entry"] = n in settings
            if overrides.get(n, {}).get("notes"):
                f["notes"] = overrides[n]["notes"]
            files.append(f)
        qty = m.get("quantity", qty_cad)
        masses = [parts[n].get("mass_kg") for n in names if parts[n].get("mass_kg") is not None]
        line = dict(id=lid, name=m.get("name", lid), kinds=kinds, make_buy=m.get("make_buy", "unknown"),
                    quantity=qty, quantity_cad=qty_cad, quantity_source="lines.json" if "quantity" in m else "CAD count",
                    unit=m.get("unit", "ea"), sections=sorted({rowmap[n]["section"] or "unassigned" for n in names}),
                    material=first["material"] if first["material"] else (files[0].get("print") or {}).get("material"),
                    unit_mass_g=round(1000 * masses[0], 3) if masses and len(set(masses)) == 1 else None,
                    total_mass_g=round(1000 * sum(masses), 2) if len(masses) == len(names) else None,
                    supplier=m.get("supplier"), part_number=m.get("part_number"), unit_cost=m.get("unit_cost"),
                    currency=m.get("currency"), url=m.get("url"), notes=m.get("notes"),
                    spec=m.get("spec"), origin=m["_from"], parts=files,
                    flags=sorted({f for x in files for f in x["flags"]}))
        out_lines.append(line)
    for ml in lines_doc.get("manual_lines", []):
        out_lines.append(dict(ml, kinds=["manual"], quantity_source="lines.json", origin="lines.json manual line", parts=[], flags=[]))
    order_key = {"printed": 0, "purchased": 1, "hardware": 2, "other": 3, "manual": 4}
    out_lines.sort(key=lambda l: (order_key.get(l["kinds"][0], 9), l["sections"][0] if l["sections"] else "", l["id"]))
    return dict(manifest=manifest_path, manifest_sha256=hashlib.sha256((ROOT / manifest_path).read_bytes()).hexdigest(),
                hardware_specs=hw_source, lines_file="robot/body/bom/lines.json", chain=chain or {"steps": [dict(command=a, output=b) for a, b in INV.MANIFEST_CHAIN],
                                                                                                    "note": "robot/body/bom/chain.json not written yet; chain from inventory.py"},
                assembly_order="robot/body/checks/data/assembly_order.json",
                defects=defects(rows), excluded=excluded, lines=out_lines, counts=counts(out_lines, rows))


def defects(rows):
    """robot/body/bom/defects.json groups with the manifest parts each still matches; printed parts that lack a scripted
    native STEP and no group covers are listed under `unlisted`."""
    doc = json.loads((HERE / "defects.json").read_text())
    names = [r["name"] for r in rows]
    out, covered = [], set()
    for g in doc["groups"]:
        hit = [n for n in names if any(fnmatch.fnmatchcase(n, pat) for pat in g["parts"])]
        covered |= set(hit)
        out.append(dict(g, matched=hit))
    open_printed = [r["name"] for r in rows if r["kind"] == "printed" and r["name"] not in covered
                    and (r["producer_class"] != "script" or r["step_status"] in ("none", "missing"))]
    return dict(groups=out, unlisted=open_printed)


def counts(lines, rows):
    by = defaultdict(int)
    for l in lines:
        by[l["make_buy"]] += 1
    pr = [r for r in rows if r["kind"] == "printed"]
    return dict(lines=len(lines), lines_by_make_buy=dict(by), parts=len(rows),
                printed_parts=len(pr), printed_with_native_step=sum(1 for r in pr if r["step_status"] not in ("none", "missing")),
                printed_with_print_settings_entry=sum(1 for r in pr if r["print_settings"]),
                parts_with_assembly_step=sum(1 for r in rows if r["assembly_steps"]),
                parts_from_script_in_tree=sum(1 for r in rows if r["producer_class"] == "script"),
                parts_mesh_only=sum(1 for r in rows if "mesh_only" in r["flags"]),
                parts_file_missing=sum(1 for r in rows if "file_missing" in r["flags"]),
                parts_without_geometry=sum(1 for r in rows if "no_geometry" in r["flags"]),
                hardware_kinds_to_buy=sum(1 for l in lines if l["id"].startswith("hw:")))


def write_csv(bom, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "make_buy", "quantity", "quantity_cad", "unit", "sections", "material", "printer", "nozzle_mm",
                    "layer_height_mm", "up", "supplier", "part_number", "unit_cost", "currency", "parts", "step_files", "producer", "assembly_steps", "flags"])
        for l in bom["lines"]:
            pr = next((p["print"] for p in l["parts"] if p.get("print")), {}) or {}
            w.writerow([l["id"], l["name"], l["make_buy"], l["quantity"], l["quantity_cad"], l.get("unit", "ea"), "; ".join(l.get("sections") or []),
                        l.get("material") or "", pr.get("printer") or "", pr.get("nozzle_mm") or "", pr.get("layer_height_mm") or "", pr.get("up") or "",
                        l.get("supplier") or "", l.get("part_number") or "", l.get("unit_cost") or "", l.get("currency") or "",
                        "; ".join(p["part"] for p in l["parts"]),
                        "; ".join((p.get("generated_hardware") or {}).get("step") or p.get("step") or "" for p in l["parts"]),
                        "; ".join(sorted({p.get("producer") or "" for p in l["parts"]})),
                        "; ".join(sorted({s for p in l["parts"] for s in p["assembly_steps"]})), "; ".join(l["flags"])])


def markdown(bom):
    c = bom["counts"]
    L = ["# Cove-50 robot BOM", "",
         f"Generated by `robot/body/bom/build.py` from `{bom['manifest']}` (sha256 {bom['manifest_sha256'][:12]}) and the hand-maintained "
         f"`robot/body/bom/lines.json`. Do not edit this page; edit lines.json or the part's script and rerun.", "",
         f"{c['lines']} lines over {c['parts']} CAD parts: " + ", ".join(f"{v} {k}" for k, v in c["lines_by_make_buy"].items()) + ". "
         f"Printed parts {c['printed_parts']}: with a native STEP {c['printed_with_native_step']}, with a print-settings entry "
         f"{c['printed_with_print_settings_entry']}. Parts with an assembly step: {c['parts_with_assembly_step']}. Parts written by a script "
         f"in the tree: {c['parts_from_script_in_tree']}; mesh only {c['parts_mesh_only']}; file missing {c['parts_file_missing']}; "
         f"no geometry {c['parts_without_geometry']}. Hardware kinds to buy: {c['hardware_kinds_to_buy']}.", "",
         "## How to rebuild the body files", ""]
    for i, s in enumerate(bom["chain"].get("steps", [])):
        L.append(f"{i + 1}. `{s.get('command')}` -> `{s.get('output') or ', '.join(s.get('outputs', []) or [])}`")
    L += ["", "Standard hardware STEP/STL: `.venv/bin/python -B robot/body/bom/hardware.py`. This page: `.venv/bin/python -B robot/body/bom/build.py`.", ""]
    d = bom["defects"]
    L += ["## Open defects: printed parts without a scripted native STEP (robot/body/bom/defects.json)", "",
          "| group | parts | state | fix | effort |", "|---|---|---|---|---|"]
    for g in d["groups"]:
        if g["matched"]:
            L.append(f"| {g['id']} | {', '.join(g['matched'])} | {g['state']} | {g['fix']} | {g['effort']} |")
    if d["unlisted"]:
        L.append(f"| (not yet listed) | {', '.join(d['unlisted'])} | no entry in defects.json | | |")
    L += [""]
    L += ["## Print list", "", "| line | qty | section | material | printer / nozzle / layer | up | STEP | producer | assembly | flags |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for l in bom["lines"]:
        if "printed" not in l["kinds"]:
            continue
        p = l["parts"][0]; pr = p.get("print") or {}
        L.append(f"| {l['name']} | {l['quantity']} | {', '.join(l['sections'])} | {pr.get('material') or '-'} | {pr.get('printer') or '-'} / "
                 f"{pr.get('nozzle_mm') or '-'} / {pr.get('layer_height_mm') or '-'} | {pr.get('up') or '-'} | {p['step_status']} | "
                 f"{(p['producer'] or '-')} | {', '.join(p['assembly_steps']) or '-'} | {', '.join(f for f in l['flags'] if f not in ('mesh_npz_not_stl', 'inspection_reference'))} |")
    L += ["", "## Buy list (servos, electronics, cells, other bought parts)", "", "| line | qty | make / buy | sections | supplier / part no. | notes |", "|---|---|---|---|---|---|"]
    for l in bom["lines"]:
        if l["kinds"][0] in ("purchased", "other", "manual") and "printed" not in l["kinds"]:
            L.append(f"| {l['name']} | {l['quantity']} | {l['make_buy']} | {', '.join(l.get('sections') or [])} | "
                     f"{' '.join(x for x in (l.get('supplier'), l.get('part_number')) if x) or '-'} | {l.get('notes') or ''} |")
    L += ["", "## Hardware", "", "| spec | qty | sections |", "|---|---|---|"]
    for l in bom["lines"]:
        if l["kinds"] == ["hardware"]:
            q = l["quantity"] if l["quantity"] is not None else "not counted"
            L.append(f"| {l['name']} | {q} | {', '.join(l['sections'])} |")
    L += ["", "## Assembly order", "", f"`{bom['assembly_order']}` (checked by `robot/body/checks/run.py --checks assembly`). "
          f"Parts with a step: {c['parts_with_assembly_step']} of {c['parts']}; every step is a seeded draft until confirmed."]
    return "\n".join(L) + "\n"


def dashboard_rows(bom):
    cat = {"make": "produced", "buy": "purchased", "included": "context", "context": "context"}
    rows = []
    for l in bom["lines"]:
        pr = next((p["print"] for p in l["parts"] if p.get("print")), None)
        rows.append(dict(id=l["id"], name=l["name"], category=cat.get(l["make_buy"], "unresolved"), quantity=l["quantity"],
                         part_names=[p["part"] for p in l["parts"]], material=l.get("material"), manufacturer=l.get("supplier"),
                         part_number=l.get("part_number"), unit_mass_g=l.get("unit_mass_g"),
                         process=pr and {k: pr[k] for k in ("printer", "nozzle_mm", "layer_height_mm", "up") if pr.get(k) is not None},
                         notes=[x for x in [l.get("notes"), ("flags: " + ", ".join(l["flags"])) if l["flags"] else None] if x]))
    return rows


def design_bom_payload(bom):
    cat = {"make": "Produced parts", "buy": "Purchased parts", "included": "Included with purchased parts", "context": "Context"}
    items = []
    for l in bom["lines"]:
        pr = next((p["print"] for p in l["parts"] if p.get("print")), None)
        items.append(dict(id=l["id"], name=l["name"], category=cat.get(l["make_buy"], "To specify"), quantity=l["quantity"],
                          unit=l.get("unit", "ea"), material=str(l.get("material") or ""), supplier=l.get("supplier") or "",
                          part_number=l.get("part_number") or "", unit_mass_g=l.get("unit_mass_g"),
                          unit_cost=l.get("unit_cost"), currency=l.get("currency") or "",
                          notes=" | ".join(x for x in [l.get("notes") or "", f"print: {pr.get('printer')} {pr.get('material')} {pr.get('nozzle_mm')} mm, up {pr.get('up')}" if pr else "",
                                                       f"CAD parts: {', '.join(p['part'] for p in l['parts'][:12])}{' ...' if len(l['parts']) > 12 else ''}"] if x)))
    return dict(scope=f"Generated from {bom['manifest']} by robot/body/bom/build.py with robot/body/bom/lines.json (sha256 {bom['manifest_sha256'][:12]}).",
                completeness="partial", notes="Prices and suppliers are filled in lines.json; lines without them are uncounted, not zero.",
                exclusions=[f"{k}: {v}" for k, v in bom["excluded"].items()], items=items)


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
                             notes=f"The selected complete body {rid}; its BOM is generated by robot/body/bom/build.py (owner 2026-11-20). "
                                   "Earlier plan (shoulder A20, knee v16, elbow r3, hip r3, electronics r1) replaced: none of those is a current selection.",
                             exclusions=[], assemblies=[dict(result_id=rid, count=1)]), plan.get("revision") if plan else None)
        db.commit()
    return rid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=json.loads((ROOT / "robot/design.json").read_text())["body_manifest"])
    ap.add_argument("--out", default=str(ROOT / ".cache/work/bom"))
    ap.add_argument("--hardware-json")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    bom = build(a.manifest, a.hardware_json)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "bom.json").write_text(json.dumps(bom, indent=1, default=str) + "\n")
    write_csv(bom, out / "bom.csv")
    (out / "BOM.md").write_text(markdown(bom))
    (out / "dashboard-bom.json").write_text(json.dumps(dashboard_rows(bom), indent=1, default=str) + "\n")
    print(json.dumps(bom["counts"], indent=1))
    if a.publish:
        print("published design BOM and robot plan for", publish(bom))


if __name__ == "__main__":
    main()
