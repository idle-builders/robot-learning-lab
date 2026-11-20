"""Focused CAD check for one design iteration (cad-speed task, 2026-11-17): changed parts in, native-unit scores out
in about a minute or two. Screening only; exact B-rep fit and FEM stay opt-in for the final candidate.

  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 600 -- .venv/bin/python -B robot/body/leg/fast_check.py \
      --parts R_hip_pitch_to_roll_roll_flip [--manifest artifacts/.../assembly-parts-roll-flip.json] \
      [--poses key|library|all] [--checks clearance,gaps,mass,press,render] [--exact-press] [--label NAME]

What it does (all continuous measurements; nothing accepts or rejects):
  clearance  every changed part against every part on another link (hip skeleton + thigh covers) and the other
             changed parts, over the pose set: new overlap vs standing (mm3) and minimum surface distance (mm,
             capped at the 5 mm search radius, --search-mm), worst pose per pair and family. Pose sets: key = standing, the 7 leg_motion key
             poses, the 8 fit_check extremes and the joint range ends (23 poses); library = the 151 realistic poses incl.
             paths; all = library + the 72 range combinations.
  gaps       standing: every part (any link) within 6 mm of a changed part: overlap (mm3) and distance (mm)
  mass       changed part volume from its native STEP x density vs the manifest mass; owning link mass / COM
  press      sleeves of the changed part: tool / anvil path obstruction (mm3) with mesh booleans (manifold3d, within
             1.5 % of the OCC booleans on the spine chassis, no zero / non-zero flips), counter-face coverage, flatness,
             tilt as press_lib; --exact-press uses press_lib.score_part (OCC, ~10x slower)
  render     the owner's six-side case drawing (owner_case_views.py) of this manifest, into the output folder

Solids: a changed part with a native STEP is tessellated from the STEP (0.03 mm deflection, closed, true shape;
no voxel fill, no convex hull); unchanged printed connectors and servo housings too, at 0.1 mm. Other parts use roll_flip_checks.closed() (smoothed voxel fill, never a hull for
open meshes). Every cache entry is keyed by the sha256 of the file it was made from, not the manifest's recorded hash,
so a regenerated STEP / mesh is never served stale. Output: .cache/work/cad-speed/fast-check/<label>/report.json.
Since the shared tool exists (robot/body/checks/run.py) the solids, caches, pose sets and the check functions live
there; this script keeps its old command line and report shape on top of them (retired later).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body"))
from checks.core import (CACHE, R, T, WORK, kit, pose_set, save_sha, screen_solid, sha, step_mesh, tick)  # noqa: E402,F401
from checks.modules.clearance import check_clearance as _clearance  # noqa: E402
from checks.modules.gaps import check_gaps  # noqa: E402
from checks.modules.mass import check_mass  # noqa: E402
from checks.modules.press import check_press  # noqa: E402
from checks.modules.render import check_render  # noqa: E402

SEARCH_MM = 5.0     # min_gap search radius: cost grows fast with it (10 mm: 6x slower), values below it are exact


def check_clearance(body, changed, sol, mf, poses):
    from joint_cuts import skeleton_parts
    others = {**skeleton_parts(body), **{n: body.parts[n] for n in body.parts if n[2:] in R.COVERS}}
    return _clearance(body, changed, others, sol, mf, poses, SEARCH_MM)


# ------------------------------------------------------------------------------------------------ main
def main():
    global SEARCH_MM
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", required=True, help="comma-separated changed part names (manifest names)")
    ap.add_argument("--manifest", default=str(R.VARIANT.relative_to(ROOT)))
    ap.add_argument("--poses", default="key", choices=("key", "library", "all"))
    ap.add_argument("--checks", default="clearance,gaps,mass,press,render")
    ap.add_argument("--exact-press", action="store_true")
    ap.add_argument("--search-mm", type=float, default=5.0, help="distances are measured up to this (mm)")
    ap.add_argument("--render-timeout", type=float, default=240.0)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    SEARCH_MM = a.search_mm
    t_all = time.time()
    checks = a.checks.split(",")
    man_path = ROOT / a.manifest
    label = a.label or "-".join(p.replace("R_", "").replace("L_", "")[:24] for p in a.parts.split(","))
    out_dir = WORK / "fast-check" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); body = kit.base(man_path); tick("load manifest", t0)
    changed = [n for n in a.parts.split(",") if n]
    miss = [n for n in changed if n not in body.parts]
    if miss:
        sys.exit(f"not in {a.manifest}: {miss}")
    sol, mf = {}, {}
    t0 = time.time()
    for n in changed:
        sol[n] = screen_solid(body, n, set(changed)); mf[n] = kit.mf(sol[n][0])
    tick("solids (changed)", t0)
    rep = dict(manifest=a.manifest, manifest_sha256=sha(man_path), changed=changed,
               changed_files={n: dict(native_step_sha256=sha(R.F.src(body.parts[n]["native_step"])) if body.parts[n].get("native_step") else None,
                                      mesh_sha256=sha(R.F.src(body.parts[n]["mesh"]))) for n in changed},
               solids={n: dict(how=h, volume_mm3=round(float(m.volume), 1), faces=len(m.faces)) for n, (m, h) in sol.items()})
    rep["other_solids"] = {}
    if "clearance" in checks:
        rep["clearance"] = check_clearance(body, set(changed), sol, mf, pose_set(body, a.poses))
        rep["clearance"]["pose_set"] = a.poses
    if "gaps" in checks:
        rep["gaps_standing"] = check_gaps(body, changed, sol, mf)
    if "mass" in checks:
        rep["mass"] = check_mass(body, changed, sol)
    if "press" in checks:
        rep["press"] = check_press(body, changed, a.exact_press)
    if "render" in checks:
        rep["render"] = check_render(man_path, out_dir, a.render_timeout)
    rep["other_solids"] = {n: h for n, (_, h) in sol.items() if n not in changed}
    save_sha()
    rep["seconds"] = dict(T, total=round(time.time() - t_all, 1))
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    summary(rep)
    print(f"\n{(out_dir / 'report.json').relative_to(ROOT)}")


def summary(rep):
    print(f"changed: {', '.join(rep['changed'])}")
    for n, s in rep["solids"].items():
        print(f"  solid {n}: {s['how']}, {s['volume_mm3']} mm3, {s['faces']} faces")
    odd = {n: h for n, h in rep.get("other_solids", {}).items() if "hull" in h}
    if odd:
        print(f"  other parts screened as convex hulls (over-state overlap): {', '.join(odd)}")
    if "clearance" in rep:
        c = rep["clearance"]
        print(f"clearance ({c['pose_set']}: {c['poses']} poses, {c['pairs']} pairs) - closest / most new overlap:")
        for k, w in list(c["by_pair"].items())[:12]:
            print(f"  {k:72s} new overlap {w['max_new_overlap_mm3']:7.1f} mm3 ({w['pose_of_max']}),"
                  f" min dist {w['min_distance_mm']:6.3f} mm ({w['pose_of_min']})")
    if "gaps_standing" in rep:
        print("gaps at standing (<= 6 mm):")
        for k, g in list(rep["gaps_standing"].items())[:12]:
            print(f"  {k:72s} {'same link' if g['same_link'] else 'moving  '} overlap {g['overlap_mm3']:8.3f} mm3, dist {g['distance_mm']:.3f} mm")
    if "mass" in rep:
        for n, r in rep["mass"]["parts"].items():
            print(f"mass {n}: {r['mass_from_volume_g']} g from volume (manifest {r['manifest_mass_g']} g)")
        for lk, r in rep["mass"]["links"].items():
            print(f"  link {lk}: {r['mass_g']} g, COM {r['com_neutral_mm']} mm")
    for n, r in rep.get("press", {}).items():
        print(f"press {n} [{r['method'].split(' (')[0]}]: " + ", ".join(f"{k} {float(v):g}" for k, v in r["summary"].items()))
    if "render" in rep:
        print(f"render: {rep['render']}")
    print("seconds:", rep["seconds"])


if __name__ == "__main__":
    main()
