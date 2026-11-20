"""BOM inventory of the selected body manifest: per part, where its geometry comes from and what a builder needs.

For every part of `body_manifest` in robot/design.json (or --manifest): name, role, kind (printed / hardware / purchased /
other), owning link and robot section, the native STEP and the mesh with how each resolves (on disk, through
infra/storage/path-moves.json, in the blob index, regenerable, missing), the script that writes the file (producer
table below, or the pinned input in robot/inputs.json with its producer commit), its assembly step (data/assembly_order.json), whether it has print settings
(data/part_settings.json), and flags. Then manifest-wide counts per flag, the manifest-level chain of builders, and the
BOM stores outside the manifest (results.sqlite design_boms / robot_bom).

Read-only: nothing is restored from the bucket and no geometry is loaded. Output: .cache/work/bom-inventory/inventory.json
and summary.md (recreated by running this script).

  .venv/bin/python -B robot/body/bom/inventory.py [--manifest PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from infra.storage.blob_store import lookup as blob_lookup, moved  # noqa: E402
from infra.storage.regenerable_artifacts import lookup as regen_lookup  # noqa: E402

DATA = ROOT / "robot/body/checks/data"

# Output folder -> the script in Git that writes it (longest prefix wins). A file under artifacts/robot/body without an
# entry here is reported as producer "unknown".
PRODUCERS = [
    ("artifacts/robot/body/ape-proportions/shell-pass", "robot/body/ape-proportions/shells.py"),
    ("artifacts/robot/body/ape-proportions", "robot/body/ape-proportions/build.py"),
    ("artifacts/robot/body/leg/roll-flip", "robot/body/leg/build_upper_hip_connectors.py + robot/body/leg/thigh_shell.py"),
    ("artifacts/robot/body/leg/combined", "robot/body/leg/build_combined.py"),
    ("artifacts/robot/body/leg/complete_leg_hip_combined", "robot/body/leg/package_complete_leg_hip.py"),
    ("artifacts/robot/body/leg", "robot/body/leg (unknown script)"),
    ("artifacts/robot/body/arm/shoulder-clevis", "robot/body/arm/shoulder_clevis.py"),
    ("artifacts/robot/body/arm/combined", "robot/body/arm/build_arm_connectors.py"),
    ("artifacts/robot/body/arm/shell_hardware", "robot/body/arm/package_and_render.py"),
    ("artifacts/robot/body/neck-head", "robot/body/neck-head/build_neck_head_connectors.py"),
    ("artifacts/robot/body/torso/neighbor_meshes_cache", "robot/body/torso/common.py (copy of a neighbour mesh)"),
    ("artifacts/robot/body/torso/shell_assets", "robot/body/torso/package_and_render.py (copy of a shell input)"),
    ("artifacts/robot/body/torso", "robot/body/torso/build_torso_carriers.py"),
    ("artifacts/robot/body/assembly", "robot/body/assembly/assemble_complete_fdm_body.py"),
    ("artifacts/robot/body/motor-refit", "robot/body/motor-refit/motor_refit.py"),
]

# The builders that turn the per-section outputs into the selected manifest, in order. Each starts from the previous
# manifest and writes the next (several edit it in place).
MANIFEST_CHAIN = [
    ("robot/body/assembly/assemble_complete_fdm_body.py", "artifacts/robot/body/assembly/assembly-parts.json"),
    ("robot/body/motor-refit/motor_refit.py", "artifacts/robot/body/motor-refit/assembly-parts.json"),
    ("robot/body/ape-proportions/build.py", "artifacts/robot/body/ape-proportions/assembly-parts.json"),
    ("robot/body/ape-proportions/shells.py", "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"),
    ("robot/body/leg/roll_flip_checks.py manifest", "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"),
    ("robot/body/leg/thigh_shell.py build", "(edits assembly-parts-roll-flip.json in place)"),
    ("robot/body/leg/thigh_shell.py shin-cover", "(edits assembly-parts-roll-flip.json in place)"),
    ("robot/body/arm/shoulder_clevis.py build + manifest", "artifacts/robot/body/arm/shoulder-clevis/assembly-parts-shoulder-u.json"),
]

PRINTED_ROLES = {"printed_structural_connector", "printed_structural_carrier", "body_shell", "produced_backing",
                 "produced_holder", "manufactured", "insulating_mount_spacer"}
HARDWARE_ROLES = {"fastener", "standard_fastener_reference", "nominal_fastener", "fastener_proxy", "standard_nominal",
                  "nominal_hardware", "compression_limit_sleeves", "spacer", "case_mount_spacer", "bearing"}
PURCHASED_ROLES = {"motor", "output", "output_horn", "horn", "actuator", "battery_cell", "purchased_module",
                   "candidate_sensor", "supplier_native", "source_power_board", "purchased_servo_output_family_reference",
                   "purchased_servo_case_family_reference"}
ODD_NAME = re.compile(r"(proposal|legacy|candidate|_old\b|_tmp|test|_v\d+\b|_r\d+\b|copy)", re.I)


def kind_of(p):
    r = p["role"]
    if r == "body_shell" and re.search(r"_nut(_\d+)?$", p["name"]):
        return "hardware"
    if r in PRINTED_ROLES:
        return "printed"
    if r in HARDWARE_ROLES:
        return "hardware"
    if r in PURCHASED_ROLES:
        return "purchased"
    return "other"   # cut stock, fins, insulators, adhesive, harness


def resolve(path):
    """(status, path to read) the way the checks find a file (bodylib.src -> blob_store.ensure), without restoring."""
    if not path:
        return "none", None
    if (ROOT / path).is_file():
        return "disk", path
    if regen_lookup(path):
        return "regenerable", path
    if blob_lookup(path):
        return "blob", path
    new = moved(path)
    if new is not None:
        new = str(new)
        if (ROOT / new).is_file():
            return "moved_disk", new
        if regen_lookup(new):
            return "moved_regenerable", new
        if blob_lookup(new):
            return "moved_blob", new
    return "missing", path


def producer_of(path, pinned):
    if not path:
        return None, None
    best = max((pre for pre, _ in PRODUCERS if path == pre or path.startswith(pre + "/")), key=len, default=None)
    if best:
        return dict(PRODUCERS)[best], "script"
    pin = max((i for i in pinned if path == i["path"] or path.startswith(i["path"].rstrip("/") + "/")),
              key=lambda i: len(i["path"]), default=None)
    if pin:
        prod = pin.get("producer")
        return (f"{prod['folder']} @ {prod['commit']} (not in the tree)" if prod else "pinned input, no producer recorded"), \
               ("legacy_producer" if prod else "pinned_no_producer")
    if path.startswith("artifacts/robot/body/"):
        return "unknown", "unknown"
    return "none (old study folder, not a pinned input)", "no_producer"


def study_folder(path):
    if not path:
        return None
    parts = path.split("/")
    return "/".join(parts[:3]) if parts[:3] != ["artifacts", "robot", "body"] else "/".join(parts[:4])


def upstream_paths(selected):
    """{part name: [(chain manifest, step path, mesh path)]} over the manifests the chain wrote before the selected one."""
    out = defaultdict(list)
    for _, path in MANIFEST_CHAIN:
        f = ROOT / path
        if not path.startswith("artifacts/") or not f.is_file() or path == selected:
            continue
        for q in json.loads(f.read_text())["parts"]:
            out[q["name"]].append((path, q.get("native_step"), q.get("mesh")))
    return out


def build(manifest_path):
    man = json.loads((ROOT / manifest_path).read_text())
    up = upstream_paths(manifest_path)
    parts = man["parts"]
    pinned = json.loads((ROOT / "robot/inputs.json").read_text())["inputs"]
    sections = {k: v for k, v in json.loads((DATA / "sections.json").read_text()).items() if not k.startswith("_")}
    settings = json.loads((DATA / "part_settings.json").read_text())
    order = json.loads((DATA / "assembly_order.json").read_text())
    names = [p["name"] for p in parts]
    base = lambda link: re.sub(r"^[LR]_", "", link or "")  # noqa: E731

    # assembly steps per part name
    step_of = defaultdict(list)
    for aname, a in order.get("assemblies", {}).items():
        for n in [n for n in names if any(fnmatch.fnmatchcase(n, g) for g in ([a.get("base")] if isinstance(a.get("base"), str) else a.get("base") or []))]:
            step_of[n].append(f"{aname}:base")
        for s in a.get("steps", []):
            spec = s.get("part") or s.get("screw")
            pats = [spec] if isinstance(spec, str) else list(spec or [])
            for n in names:
                if any(fnmatch.fnmatchcase(n, g) for g in pats):
                    step_of[n].append(f"{aname}:{s.get('id')}")

    # the manifest's own hand-written BOM (retired 2026-11-20; the BOM is robot/body/bom/build.py): listed if still present
    items = (man.get("bom") or {}).get("items", [])

    def section_of(p):
        for s, v in sections.items():
            if base(p["owning_link"]) in v["links"] or any(fnmatch.fnmatchcase(p["name"], g) for g in v["parts"]):
                return s
        return None

    mesh_users = Counter(p.get("mesh") for p in parts if p.get("mesh"))
    bounds_key = Counter(json.dumps(p.get("bounds_mm")) + "|" + str(p.get("mesh")) for p in parts)
    rows = []
    for p in parts:
        k = kind_of(p)
        st_status, st_path = resolve(p.get("native_step"))
        me_status, me_path = resolve(p.get("mesh"))
        geo = st_path if st_status not in ("none", "missing") else me_path if me_status not in ("none", "missing") else (p.get("native_step") or p.get("mesh"))
        prod, prod_class = producer_of(geo, pinned)
        flags = []
        if st_status == "missing" or me_status == "missing":
            flags.append("file_missing")
        if st_status in ("none", "missing") and me_status not in ("none", "missing"):
            flags.append("mesh_only")
        if st_status in ("none", "missing") and me_status in ("none", "missing"):
            flags.append("no_geometry")
        if prod_class in ("no_producer", "pinned_no_producer", "legacy_producer", "unknown"):
            flags.append(f"source_{prod_class}")
        for raw in (p.get("native_step"), p.get("mesh")):
            if raw and not raw.startswith("artifacts/robot/"):
                flags.append("path_old_study_folder"); break
        for raw in (p.get("native_step"), p.get("mesh")):
            if raw and moved(raw) is not None and not (ROOT / raw).is_file():
                flags.append("path_pre_restructure_name"); break
        if (p.get("mesh") or "").endswith(".npz"):
            flags.append("mesh_npz_not_stl")
        hist = up.get(p["name"], [])
        first = hist[0] if hist else None
        origin = (first[1] or first[2]) if first else None
        origin = (moved(origin) and str(moved(origin))) or origin if origin else None
        if not hist:
            flags.append("added_after_chain_start")
        upstream_src = p.get("integration_source") or p.get("upstream_native_assembly_path") or p.get("source")
        if not step_of.get(p["name"]):
            flags.append("no_assembly_step")
        if k == "printed" and p["name"] not in settings:
            flags.append("no_print_settings")
        if ODD_NAME.search(p["name"]):
            flags.append("odd_name")
        if p.get("mesh") and mesh_users[p["mesh"]] > 1:
            flags.append("mesh_shared")
        if bounds_key[json.dumps(p.get("bounds_mm")) + "|" + str(p.get("mesh"))] > 1:
            flags.append("duplicate_placement")
        if p.get("inspection_reference"):
            flags.append("inspection_reference")
        rows.append(dict(name=p["name"], role=p["role"], kind=k, owning_link=p["owning_link"], section=section_of(p),
                         native_step=p.get("native_step"), step_status=st_status, step_resolved=st_path,
                         mesh=p.get("mesh"), mesh_status=me_status, mesh_resolved=me_path,
                         source_folder=study_folder(geo), producer=prod, producer_class=prod_class,
                         first_manifest=first[0] if first else None, first_geometry=origin,
                         first_geometry_folder=study_folder(origin), recorded_upstream_source=upstream_src,
                         bom_group=p.get("bom_group"), assembly_steps=step_of.get(p["name"], []),
                         print_settings=p["name"] in settings, material=p.get("material"), mass_kg=p.get("mass_kg"),
                         flags=flags))
    return man, items, rows, settings, order


def dashboard_boms():
    db = ROOT / "data/results.sqlite"
    if not db.exists():
        return {}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        boms = [r[0] for r in c.execute("SELECT result_id FROM design_boms")]
        plan = c.execute("SELECT payload_json, revision FROM robot_bom WHERE id=1").fetchone()
    design = json.loads((ROOT / "robot/design.json").read_text())
    sel = {k: v.get("result_id") or v.get("study_id") for k, v in design.get("section_selections", {}).items()}
    plan_ids = [a["result_id"] for a in json.loads(plan[0])["assemblies"]] if plan else []
    return dict(design_bom_result_ids=boms, selected_results=sel,
                selected_without_design_bom=sorted({v for v in sel.values() if v not in boms}),
                robot_bom_plan_revision=plan[1] if plan else None, robot_bom_plan_assemblies=plan_ids,
                robot_bom_plan_not_selected=[i for i in plan_ids if i not in sel.values()])


def summarise(man, items, rows, settings, order, dash, manifest_path):
    flags = Counter(f for r in rows for f in r["flags"])
    by_kind = Counter(r["kind"] for r in rows)
    flag_kind = defaultdict(Counter)
    for r in rows:
        for f in r["flags"]:
            flag_kind[f][r["kind"]] += 1
    folders = Counter(r["source_folder"] for r in rows)
    first_folders = Counter(r["first_geometry_folder"] for r in rows)
    prod = Counter(r["producer_class"] for r in rows)
    names = {r["name"] for r in rows}
    settings_unused = sorted(k for k in settings if not k.startswith("_") and k not in names)
    order_unmatched = []
    for aname, a in order.get("assemblies", {}).items():
        for s in a.get("steps", []):
            spec = s.get("part") or s.get("screw")
            pats = [spec] if isinstance(spec, str) else list(spec or [])
            if not any(fnmatch.fnmatchcase(n, g) for n in names for g in pats):
                order_unmatched.append(f"{aname}:{s.get('id')}")
    printed_rows = [r for r in rows if r["kind"] == "printed"]
    out = dict(manifest=manifest_path, parts=len(rows), parts_by_kind=dict(by_kind), flag_counts=dict(flags.most_common()),
               flag_counts_by_kind={f: dict(c) for f, c in flag_kind.items()},
               producer_class_counts=dict(prod), source_folders=dict(folders.most_common()),
               first_geometry_folders=dict(first_folders.most_common()),
               legacy_manifest_bom_lines=[dict(id=i["id"], quantity=i.get("quantity")) for i in items],
               assemblies={a: len(v.get("steps", [])) for a, v in order.get("assemblies", {}).items()},
               assembly_steps_matching_no_part=order_unmatched,
               part_settings_for_parts_not_in_manifest=settings_unused,
               printed_parts=len(printed_rows),
               printed_with_step=sum(1 for r in printed_rows if r["step_status"] not in ("none", "missing")),
               printed_with_print_settings=sum(1 for r in printed_rows if r["print_settings"]),
               printed_in_assembly_order=sum(1 for r in printed_rows if r["assembly_steps"]),
               manifest_chain=MANIFEST_CHAIN, dashboard=dash)
    return out


def markdown(s, rows):
    L = [f"# BOM inventory of `{s['manifest']}`", "",
         f"{s['parts']} parts: " + ", ".join(f"{v} {k}" for k, v in s["parts_by_kind"].items()) + ".", "",
         "## Flags (parts, by kind)", "", "| flag | parts | printed | hardware | purchased | other |", "|---|---|---|---|---|---|"]
    for f, n in s["flag_counts"].items():
        c = s["flag_counts_by_kind"][f]
        L.append(f"| {f} | {n} | {c.get('printed', 0)} | {c.get('hardware', 0)} | {c.get('purchased', 0)} | {c.get('other', 0)} |")
    L += ["", "## Where the geometry comes from (producer class)", ""] + [f"- {k}: {v}" for k, v in s["producer_class_counts"].items()]
    L += ["", "## Source folders", ""] + [f"- `{k}`: {v}" for k, v in s["source_folders"].items()]
    L += ["", "## Earliest geometry in the chain (first manifest the part appears in)", ""] + [f"- `{k}`: {v}" for k, v in s["first_geometry_folders"].items()]
    if s["legacy_manifest_bom_lines"]:
        L += ["", f"The manifest still carries the retired hand-written `bom` ({len(s['legacy_manifest_bom_lines'])} lines); "
              "the BOM is robot/body/bom/build.py. It drops out at the next rebuild of the manifest chain."]
    L += ["", f"Printed parts: {s['printed_parts']}; with a native STEP {s['printed_with_step']}, with print settings "
          f"{s['printed_with_print_settings']}, in the assembly order {s['printed_in_assembly_order']}.",
          f"Assemblies in the order: {s['assemblies']}; steps matching no part: {len(s['assembly_steps_matching_no_part'])}.",
          f"Part settings for names not in the manifest: {len(s['part_settings_for_parts_not_in_manifest'])}.", "",
          "## Manifest chain (each step reads the previous manifest)", ""]
    L += [f"{i + 1}. `{a}` -> `{b}`" for i, (a, b) in enumerate(s["manifest_chain"])]
    d = s["dashboard"]
    if d:
        L += ["", "## Dashboard BOM stores (data/results.sqlite)", "",
              f"- design_boms rows: {len(d['design_bom_result_ids'])}; selected sections without one: {d['selected_without_design_bom']}",
              f"- robot_bom plan r{d['robot_bom_plan_revision']}: {d['robot_bom_plan_assemblies']}; not among the selected: {d['robot_bom_plan_not_selected']}"]
    worst = [r for r in rows if r["kind"] == "printed" and ({"mesh_only", "file_missing", "no_geometry"} & set(r["flags"])
             or r["producer_class"] != "script")]
    L += ["", f"## Printed parts without a scripted STEP ({len(worst)})", "", "| part | link | step | mesh | producer |", "|---|---|---|---|---|"]
    L += [f"| {r['name']} | {r['owning_link']} | {r['step_status']} | {r['mesh_status']} | {r['producer']} |" for r in worst]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=json.loads((ROOT / "robot/design.json").read_text())["body_manifest"])
    ap.add_argument("--out", default=str(ROOT / ".cache/work/bom-inventory"))
    a = ap.parse_args()
    man, items, rows, settings, order = build(a.manifest)
    s = summarise(man, items, rows, settings, order, dashboard_boms(), a.manifest)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(dict(summary=s, parts=rows), indent=1) + "\n")
    (out / "summary.md").write_text(markdown(s, rows))
    print(markdown(s, rows))


if __name__ == "__main__":
    main()
