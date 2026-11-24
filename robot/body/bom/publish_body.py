"""The design-dashboard record of a body study, generated from the same source as the BOM: the body definitions
(robot/**/<builder>.bom.yaml, with the study's `study` block), the study's manifest (default robot/design.json
body_manifest, the selected body) and the rebuild's outputs (the BOM and assembly order built by build.py, the latest
complete check reports on the same manifest sha256). Like build.py it knows no part and no study: what the dashboard says
about a part is what its definition declares plus what the files and checks measured, so a redesign or a renamed part
shows up without editing a publisher.

The record carries one section `body_view` that the study page, the BOM tab and the CAD viewer all draw from: the
subassemblies in assembly order with their steps, and per part its BOM line, STEP / mesh and how they resolve, generator
and regenerate command, assembly step(s) and direction, print setup, the latest check worst cases and the BOM findings,
and the definition file to fix. design.bom is the BOM of the same build (robot BOM & costs).

  .venv/bin/python -B robot/body/bom/publish_body.py                    # dry run: writes .cache/work/bom/<study_id>-record.json
  .venv/bin/python -B robot/body/bom/publish_body.py --out FILE         # the record to FILE (the rebuild copy uses this)
  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/bom/publish_body.py package
                                                                         # display GLB of the manifest (when it changed)
  .venv/bin/python -B robot/body/bom/publish_body.py --publish          # main checkout only: package if stale, then
                                                                         # results_hub.py publish (sources must be committed)
  --study ID          another declared study (default: the study selected for "Whole robot" in robot/design.json)
  --manifest PATH     another manifest (default: the study's `manifest`, else body_manifest)
  --definitions F...  explicit definition files (replay tests on temporary copies; robot/body/bom/definitions.py)
  --checks-dir DIR    check reports (default .cache/work/cad-speed/checks)

Derived output only: the record is recreated by this script whenever the body or a definition changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import build as BUILD  # noqa: E402
import definitions as DEF  # noqa: E402

CHECKS = ROOT / ".cache/work/cad-speed/checks"
OUT = ROOT / ".cache/work/bom"                     # default record file: OUT/<study_id>-record.json
GLB_DIR = "artifacts/robot/body/studies"   # display GLB per study (derived; recreated by `package`)


def glb_path(study_id):
    """The study's own display copy of its manifest. One path per study: studies that shared one file showed each
    other's model on the remote dashboard, which maps files by path (T-8prn3q)."""
    return f"{GLB_DIR}/{study_id}/assembly.glb"
WORST_PER_CHECK = 3


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def design():
    return json.loads((ROOT / "robot/design.json").read_text())


def pick_study(D, study_id=None):
    sid = study_id or design()["selected"].get("Whole robot")
    if sid not in D.studies:
        raise SystemExit(f"study {sid!r} is declared in no definition (`study` block); declared: {sorted(D.studies)}")
    return D.studies[sid]


# ------------------------------------------------------------------ check reports
def check_reports(manifest_sha, checks_dir=CHECKS):
    """Complete check reports on this manifest, newest first: [(mtime, label, report)]."""
    out = []
    for f in Path(checks_dir).glob("*/report.json"):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("complete") and r.get("manifest_sha256") == manifest_sha:
            out.append((f.stat().st_mtime, f.parent.name, r))
    return sorted(out, key=lambda x: -x[0])


def part_checks(reports, names):
    """Per part the newest measurement of each check: {part: [{check, label, at, worst: [...], measured}]} and the check
    list [{check, label, at, parts, bad_events}]. A part counts as measured by a check when the report lists it in the
    check's per_part or in its scope."""
    names = set(names)
    seen, per, summary = set(), defaultdict(list), []
    for mt, label, r in reports:
        at = time.strftime("%Y-%m-%d %H:%M", time.localtime(mt))
        scoped = set((r.get("scope") or {}).get("parts") or [])
        for check, v in (r.get("checks") or {}).items():
            if not isinstance(v, dict):
                continue
            pp = v.get("per_part") if isinstance(v.get("per_part"), dict) else {}
            covered = (set(pp) | scoped) & names
            worst = defaultdict(list)
            for w in v.get("worst_cases") or []:
                if isinstance(w, dict) and w.get("part") in names:
                    worst[w["part"]].append(w)
            covered |= set(worst)
            new = [n for n in covered if (n, check) not in seen]
            if not new:
                continue
            summary.append(dict(check=check, label=label, at=at, parts=len(new), bad_events=v.get("bad_events")))
            for n in new:
                seen.add((n, check))
                ws = [dict(value=w.get("value"), unit=w.get("unit"), what=w.get("what"), mating_part=w.get("mating_part"),
                           pose=w.get("pose"), location_mm=w.get("location_mm")) for w in worst[n][:WORST_PER_CHECK]]
                per[n].append(dict(check=check, label=label, at=at, worst=ws, worst_cases=len(worst[n])))
    return dict(per), summary


# ------------------------------------------------------------------ the body view
def regenerate(g):
    if not g:
        return None
    cmd = g["command"] if isinstance(g["command"], str) else " ".join(g["command"])
    if g.get("rebuild", True):
        return f"{cmd}   (isolated: .venv/bin/python -B robot/body/bom/rebuild_chain.py --step {g['id']})"
    return f"{cmd}   (section builder: the rebuild links its outputs read-only)"


def body_view(bom, D, study, reports, manifest_path):
    names = [p["part"] for l in bom["lines"] for p in l["parts"]]
    scope = study.get("scope") or {}
    own_files = {f["file"] for f in D.files if f["definition"] in (scope.get("definitions") or [])}
    in_scope = set(names)
    if scope.get("definitions"):
        in_scope = {p["part"] for l in bom["lines"] for p in l["parts"] if p["definition"] in own_files}
    if scope.get("parts"):
        in_scope &= set(DEF.match(names, scope["parts"]))
    seq = bom["assembly_sequence"]
    step_of = defaultdict(list)
    for a in seq["assemblies"]:
        for x in a["base"]:
            step_of[x["part"]].append(dict(subassembly=a["name"], step="base", number=0, **{"as": "base"}))
        for s in a["steps"]:
            for key, items in (("part" if s["kind"] == "part" else "screw", s["items"]), ("with", s["with_items"])):
                for x in items:
                    step_of[x["part"]].append(dict(subassembly=a["name"], step=s["id"], number=s["number"], **{"as": key}))
    checks, check_summary = part_checks(reports, in_scope)
    findings = defaultdict(list)
    general = []
    for f in bom["findings"]:
        row = dict(kind=f["kind"], what=f["what"], file=f.get("file"), entry=f.get("entry"))
        if f.get("part") in in_scope:
            findings[f["part"]].append(row)
        elif not f.get("part") and (not scope.get("definitions") or f.get("file") in own_files):
            general.append(row)
    parts, lines = {}, {}
    for l in bom["lines"]:
        mine = [p for p in l["parts"] if p["part"] in in_scope]
        contained = [n for n in l.get("in_parts", []) if n in in_scope]
        if not mine and not contained:
            continue
        lines[l["id"]] = dict(name=l["name"], kinds=l["kinds"], make_buy=l["make_buy"], quantity=l["quantity"],
                              quantity_declared=l["quantity_declared"], unit=l.get("unit"), spec=l.get("spec"), supplier=l.get("supplier"),
                              part_number=l.get("part_number"), material=l.get("material"), unit_mass_g=l.get("unit_mass_g"),
                              total_mass_g=l.get("total_mass_g"), sections=l.get("sections"), definitions=l["definitions"],
                              notes=l.get("notes"), tool=BUILD.tool_of(l.get("spec")), in_parts=contained or None,
                              contains_per_part=l.get("contains_per_part"))
        for p in mine:
            n = p["part"]
            e = getattr(D, "_claims", {}).get(n)
            parts[n] = dict(line=l["id"], kind=(e or {}).get("kind"), definition=p["definition"],
                            entry=e and f"parts[{e['index']}] line {e['line']}", owning_link=p["owning_link"], section=p["section"], role=p["role"],
                            step=_file(p["step"], p["step_status"], p["step_sha256"], p.get("step_cold_record")),
                            mesh=_file(p["mesh"], p["mesh_status"], p["mesh_sha256"], p.get("mesh_cold_record")),
                            volume_mm3=p["volume_mm3"], mass_g=None if p["mass_kg"] is None else round(1000 * p["mass_kg"], 3),
                            generator=p["generator"] and p["generator"]["id"], assembly=step_of.get(n, []), print=p.get("print"),
                            checks=checks.get(n, []), findings=findings.get(n, []), flags=p["flags"])
            parts[n] = {k: v for k, v in parts[n].items() if v not in (None, [], {})}
    subs = []
    for a in seq["assemblies"]:
        steps = []
        for s in a["steps"]:
            its = [x["part"] for x in s["items"] if x["part"] in in_scope]
            wi = [x["part"] for x in s["with_items"] if x["part"] in in_scope]
            if not its and not wi:
                continue
            steps.append(dict(number=s["number"], id=s["id"], kind=s["kind"], parts=its, with_parts=wi, direction=s["direction"],
                              status=s["status"], check=_check_word(s["check_supported"]), measured=s["measured"], definition=s["file"],
                              tools=sorted({x["tool"] for x in s["items"] + s["with_items"] if x.get("tool")})))
        base = [x["part"] for x in a["base"] if x["part"] in in_scope]
        if not steps and not base:
            continue
        subs.append(dict(number=a["number"], name=a["name"], definition=a["file"], doc=a.get("doc"), after=a["after"], base=base,
                         steps=steps, counts=a["counts"], tools=a["tools"],
                         consumes=[c for c in a["consumes"] if c["line"] in lines or c["line"] == "(no BOM line)"]))
    d_by_file = {f["file"]: f for f in D.files}
    return dict(kind="body_view", manifest=rel(manifest_path), manifest_sha256=bom["manifest_sha256"],
                generated_by="robot/body/bom/publish_body.py (build.py BOM and assembly order, check reports on this manifest sha256)",
                scope=dict(scope, parts_listed=len(parts), parts_in_manifest=len(names)),
                counts=dict(bom["counts"], assembly=seq["counts"]),
                assembly_report=seq.get("check_report"),
                definitions=[dict(file=f, definition=d_by_file[f]["definition"] if f in d_by_file else None,
                                  owner=d_by_file.get(f, {}).get("owner"),
                                  parts=sum(1 for p in parts.values() if p["definition"] == f)) for f in bom["definitions"]],
                generators=_generators(D, bom),
                subassemblies=subs, parts_in_no_step=[n for n in seq["parts_in_no_step"] if n in in_scope],
                lines=lines, parts=parts, findings_general=general, checks=check_summary,
                defects=[dict(id=d["id"], file=d["file"], parts=d["matched"], kind=d.get("kind"), why=d.get("why") or d.get("state"),
                              fix=d.get("fix"), task=d.get("task")) for d in bom["defects"]["listed"]
                         if any(n in in_scope for n in d["matched"])])


def _generators(D, bom):
    """Every declared generator: rebuild position (1-based, None for section builders the rebuild links read-only),
    definition file, command, manifest_out, the command that regenerates its outputs."""
    pos = {g["id"]: i + 1 for i, g in enumerate(bom["generators"])}
    out = [dict(id=g["id"], file=g["file"], rebuild_position=pos.get(g["id"]), command=" ".join(g["command"]) if isinstance(g["command"], list) else g["command"],
                manifest_out=g.get("manifest_out"), regenerate=regenerate(g)) for g in D.generators]
    return sorted(out, key=lambda g: (g["rebuild_position"] is None, g["rebuild_position"] or 0, g["id"]))


def _file(path, status, sha, cold):
    """How a manifest file resolves: path, status (disk / regenerable / blob / moved / bucket / missing / none), sha256 head."""
    return {k: v for k, v in dict(path=path, status=status, sha256=sha and sha[:12], cold_record=cold).items() if v is not None}


def _check_word(v):
    return {True: "supported", False: "not supported", None: "not measured"}[v]


# ------------------------------------------------------------------ the record
def sources(D, study):
    """The committed files the record is generated from: their last commit is the version."""
    src = [f["file"] for f in D.files]
    for g in D.generators:
        cmd = g["command"] if isinstance(g["command"], list) else g["command"].split()
        src += [c for c in cmd if c.endswith(".py") and (ROOT / c).exists()]
    src += ["robot/design.json", rel(HERE / "publish_body.py"), rel(HERE / "build.py"), rel(HERE / "definitions.py")]
    return sorted(dict.fromkeys(s for s in src if not Path(s).is_absolute()))


def link_asset(x, kind_default):
    p = x["path"]
    return dict(id="file-" + hashlib.sha256(p.encode()).hexdigest()[:12], title=x["label"], kind=x.get("kind") or kind_default, path=p)


def glb_state(manifest_sha, glb):
    side = ROOT / (glb + ".json")
    made = json.loads(side.read_text()).get("manifest_sha256") if side.exists() else None
    return made if (ROOT / glb).exists() else None, made == manifest_sha and (ROOT / glb).exists()


def record(study, D, manifest_path, checks_dir=CHECKS):
    bom = BUILD.build(manifest_path, definitions=D)
    man = json.loads(Path(ROOT / manifest_path).read_text())
    reports = check_reports(bom["manifest_sha256"], checks_dir)
    view = body_view(bom, D, study, reports, manifest_path)
    c, ac = bom["counts"], bom["assembly_sequence"]["counts"]
    sid = study["study_id"]
    GLB = glb_path(sid)
    glb_made, glb_current = glb_state(bom["manifest_sha256"], GLB)
    view["glb"] = dict(path=GLB, made_from_manifest_sha256=glb_made,
                       state="made from this manifest" if glb_current else "stale or missing: run publish_body.py package")
    overlay_ids = sorted({p.get("overlay_id") for p in man["parts"] if p.get("overlay_id")})
    declared = {o["id"]: o for o in (study.get("cad") or {}).get("overlays", [])}
    overlays = [dict({"id": i, "label": i.replace("_", " ").capitalize(), "visible": True}, **declared.get(i, {})) for i in overlay_ids]
    assets = [dict(id="assembly", title=(study.get("cad") or {}).get("label") or f"{study['title']} (display GLB of the manifest)", kind="model", path=GLB),
              dict(id="parts", title="Named parts and joints (the manifest)", kind="data", path=rel(manifest_path))]
    renders = [link_asset(x, "image") for x in study.get("renders", []) if x.get("path")]
    evidence_files = [link_asset(x, "data") for x in study.get("evidence", []) if x.get("path")]
    assets += renders + [a for a in evidence_files if a["path"] not in {x["path"] for x in assets}]
    n_checks = len({x["check"] for x in view["checks"]})
    worst_rows = [dict(name=f"{p}: {ch['check']} ({w['what'] or ''})"[:200], value=w["value"], unit=w["unit"] or "",
                       condition=f"{ch['label']} {ch['at']}" + (f", with {w['mating_part']}" if w.get("mating_part") else "") + (f", pose {w['pose']}" if w.get("pose") else ""))
                  for p, row in view["parts"].items() for ch in row.get("checks", []) for w in ch["worst"][:1] if isinstance(w.get("value"), (int, float))]
    observation = (f"{study.get('observation') or study['summary']} Generated from {len(bom['definitions'])} body definitions and {rel(manifest_path)} "
                   f"(sha256 {bom['manifest_sha256'][:12]}): {c['parts']} CAD parts on {c['lines']} BOM lines "
                   f"({', '.join(f'{v} {k}' for k, v in c['lines_by_make_buy'].items())}); {ac['assemblies']} subassemblies, {ac['steps']} assembly steps.")
    measurements = ((study["measurements"] + " ") if study.get("measurements") else "") + (f"Printed parts {c['printed_parts']}: native STEP {c['printed_with_native_step']}, print setup {c['printed_with_print_setup']}. "
                    f"Assembly steps measured by the assembly check: {ac['check_supported']} clear, {ac['not_supported']} with contact or obstruction, "
                    f"{ac['not_measured']} not measured. Check reports on this manifest: {len(reports)} ({n_checks} checks over "
                    f"{sum(1 for p in view['parts'].values() if p.get('checks'))} parts). Buy lines {c['buy_lines']}: supplier known {c['buy_lines_supplier_known']}, "
                    f"part number known {c['buy_lines_part_number_known']}.")
    if scope_note := (view["scope"].get("definitions") or view["scope"].get("parts")):
        measurements += f" Part view: {len(view['parts'])} of {c['parts']} manifest parts (scope {scope_note}); the rest is context in the CAD scene."
    fk = ", ".join(f"{k} {v}" for k, v in sorted(c["findings_by_kind"].items()))
    limitations = " ".join(x for x in [study.get("limitations"),
                                        f"{c['findings']} BOM findings name the definition file to fix ({fk})." if c["findings"] else None,
                                        f"Mesh only {c['parts_mesh_only']} parts; display GLB {view['glb']['state']}."] if x)
    prov = [f"{s.get('date', '')} {s.get('who', '')}: {s['text']}" + (f" ({s['source']})" if s.get("source") else "") for s in study.get("statements", [])]
    prov += [f"definition {study['file']} (study block)", "generated by robot/body/bom/publish_body.py"]
    cad = dict(model_units="m", views=[dict(label=(study.get("cad") or {}).get("label") or study["title"], model_path=GLB)],
               parts_path=rel(manifest_path), hide_shell_toggle=True, show_shells=(study.get("cad") or {}).get("show_shells", True),
               motion_note=(study.get("cad") or {}).get("motion_note") or "Joint sliders are inspection commands over the joint ranges declared in the body definitions (joints:), not qualified travel.",
               joint_ranges={n: j["range_deg"] for n, j in bom["joints"].items()},
               joint_range_sources={n: f"{j['file']} {j['entry']}" for n, j in bom["joints"].items()},
               scope="The display GLB holds one named node per manifest part (decimated display copy; native STEP and meshes stay exact). "
                     "Click a part for its BOM line, files, generator, assembly step and checks.",
               bom=BUILD.design_bom_payload(bom))
    if overlays:
        cad["overlays"] = overlays
    rec = dict(study_id=sid, id=sid, title=study["title"], source=sources(D, study), dashboard="design",
               workspace=study.get("workspace") or "Body", robot_sections=study["robot_sections"],
               area=study.get("version_note") or study["title"], summary=study["summary"],
               observation=observation, measurements=measurements, limitations=limitations, provenance=prov,
               assets=assets, clips=[],
               design=dict(id=sid, title=study["title"], summary=study["summary"], visual_kind="native_CAD_assembly",
                           maturity=study.get("maturity") or "Generated from the body definitions; physical behaviour unmeasured unless an evidence card says so.",
                           image_path=renders[0]["path"] if renders else None, cad=cad,
                           evidence=[dict(id="checks", title="Latest check worst cases per part (this manifest)", kind="analysis",
                                          observation=f"{len(reports)} complete check reports on manifest sha256 {bom['manifest_sha256'][:12]}; per part the newest report of each check. "
                                                      "Full per-part rows (all checks, findings, definition to fix) in the part view.",
                                          measurements=worst_rows[:60])]
                           + ([dict(id="evidence", title="Declared evidence", kind="analysis", observation="Files and links declared in the study block.",
                                    source_refs=[dict(label=x["label"], path=x["path"]) if x.get("path") else dict(label=x["label"], url=x["url"])
                                                 for x in study.get("evidence", [])])] if study.get("evidence") else [])),
               sections=[dict(title="Selected body: subassemblies, parts, files, assembly and checks", payload=view)])
    if rec["design"]["image_path"] is None:
        del rec["design"]["image_path"]
    return rec


# ------------------------------------------------------------------ package / publish
def package(manifest_path, study_id):
    sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
    from package import scene  # one named node per manifest part, metres, decimated display copy
    raw = (ROOT / manifest_path).read_bytes()          # one read: the sidecar sha is the sha of what was packaged
    man = json.loads(raw)
    GLB = glb_path(study_id)
    out = ROOT / GLB
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.glb")
    scene(man).export(str(tmp))
    os.replace(tmp, out)
    (ROOT / (GLB + ".json")).write_text(json.dumps(dict(manifest=rel(manifest_path), manifest_sha256=hashlib.sha256(raw).hexdigest(),
                                                        made=time.strftime("%Y-%m-%dT%H:%M:%S%z"), by="robot/body/bom/publish_body.py package"), indent=1) + "\n")
    print("packaged", GLB, round(out.stat().st_size / 1e6, 1), "MB")


def committed_definitions(D, study):
    """(Definitions as committed at HEAD, snapshot info): every tracked definition file's HEAD content, written under
    .cache/work/publish-snapshot/<commit>/ and named by its repository path; untracked definitions and uncommitted edits
    of sources are listed, not used."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    base = ROOT / ".cache/work/publish-snapshot" / head
    paths, untracked = [], []
    for f in DEF.discover():
        r = rel(f)
        got = subprocess.run(["git", "show", f"HEAD:{r}"], cwd=ROOT, capture_output=True)
        if got.returncode:
            untracked.append(r); continue
        q = base / r; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes(got.stdout)
        paths.append(q)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *sources(D, study)], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    snapD = DEF.Definitions(paths, strip=rel(base))
    return snapD, dict(commit=head, definitions_from="HEAD", untracked_definitions_not_included=untracked,
                       working_tree_edits_not_included=[l[3:] for l in dirty])


def main_checkout():
    return (ROOT / ".git").exists() and "/.cache/work/" not in str(ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", choices=["record", "package"], default="record")
    ap.add_argument("--study")
    ap.add_argument("--manifest")
    ap.add_argument("--definitions", nargs="+")
    ap.add_argument("--checks-dir", default=str(CHECKS))
    ap.add_argument("--out", help=f"record file (default {rel(OUT)}/<study_id>-record.json)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    D = DEF.load(a.definitions)
    study = pick_study(D, a.study)
    mp = a.manifest or study.get("manifest") or design()["body_manifest"]
    if a.action == "package":
        return package(mp, study["study_id"])
    if a.publish:
        if not main_checkout() or a.definitions:
            raise SystemExit("publish runs from the main checkout on its own definitions; this copy writes the record with --out only")
        # a consistent snapshot, never main's working tree: the definitions as committed at HEAD; work in progress in other
        # sessions stays out until they commit (the dashboard keeps its last committed state); never blocks the publish
        D, snap = committed_definitions(D, study)
        study = pick_study(D, a.study)
        if not glb_state(sha_file(ROOT / mp), glb_path(study["study_id"]))[1]:
            print("display GLB is stale: packaging it under memguard", flush=True)
            subprocess.run(["python3", "infra/tools/memguard.py", "--limit-gb", "6", "--timeout-s", "1800", "--",
                            sys.executable, "-B", rel(Path(__file__)), "package", "--study", study["study_id"], "--manifest", mp], cwd=ROOT, check=True)
    rec = record(study, D, mp, a.checks_dir)
    if a.publish:
        rec["snapshot"] = dict(snap, manifest=mp, manifest_sha256=sha_file(ROOT / mp))
        rec["sections"][0]["payload"]["snapshot"] = rec["snapshot"]
    out = Path(a.out) if a.out else OUT / f"{study['study_id']}-record.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    v = rec["sections"][0]["payload"]
    print(json.dumps(dict(record=rel(out), study=rec["study_id"], bytes=out.stat().st_size, parts_listed=len(v["parts"]), lines=len(v["lines"]),
                          subassemblies=len(v["subassemblies"]), steps=sum(len(s["steps"]) for s in v["subassemblies"]),
                          parts_with_checks=sum(1 for p in v["parts"].values() if p.get("checks")),
                          parts_with_findings=sum(1 for p in v["parts"].values() if p.get("findings")), check_rows=len(v["checks"]), glb=v["glb"]["state"]), indent=1))
    if a.publish:
        subprocess.run([sys.executable, "infra/dashboard/results_hub.py", "publish", rel(out)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
