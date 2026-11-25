"""Verification of the mount check itself: known defects injected into copies of real, good connections must come back
as the injected numbers (owner 2026-11-18: 'review the verification actually works correctly').

Base cases (current body): the right knee ST3025 PA2x8 screw 0 (thigh cheek clamped on the knee case, no sleeve) and
the right shoulder roll-case PA2x7 screw 1 (washer, sleeved roll-to-yaw connector, spacer, C047 case). Each variant moves
one thing by a known amount in a temporary manifest and reruns the check on that screw only.

  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 900 -- .venv/bin/python -B robot/body/fastening/verify_mount_check.py

Output: .cache/work/mount-check/verify/report.json (measured vs injected per variant).
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import mount_check as MC  # noqa: E402

WORK = ROOT / ".cache/work/mount-check/verify"
KNEE, KNEE_CASE, THIGH = "R_knee_ST3025_PA2x8_0", "R_knee_ST3025_case", "R_thigh_roll_flip"
ROLL, ROLL_SLEEVES = "upper__R_roll_case_PA2x7_1", "R_shoulder_roll_to_yaw_combined_steel_sleeves"


def save(name, mesh, tag):
    f = WORK / "parts" / f"{tag}__{name}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, vertices=np.asarray(mesh.vertices, float), faces=np.asarray(mesh.faces, np.int64))
    return str(f.relative_to(ROOT))


def variant(man, body, tag, edits):
    """edits: {part: 4x4 transform | 'remove' | trimesh (added part)}; returns a manifest path."""
    m = copy.deepcopy(man)
    parts = {p["name"]: p for p in m["parts"]}
    for n, e in edits.items():
        if isinstance(e, str) and e == "remove":
            m["parts"] = [p for p in m["parts"] if p["name"] != n]
            continue
        if isinstance(e, trimesh.Trimesh):
            m["parts"].append(dict(name=n, role="manufactured", owning_link="R_roll_link", mesh=save(n, e, tag),
                                   bounds_mm=np.round(e.bounds, 3).tolist()))
            continue
        mesh = body.mesh(n).copy()
        mesh.apply_transform(e)
        p = parts[n]
        p["mesh"] = save(n, mesh, tag)
        p.pop("placement_offset_mm", None)
        p["bounds_mm"] = np.round(mesh.bounds, 3).tolist()
    f = WORK / f"{tag}.json"
    f.write_text(json.dumps(m))
    return f


def T_translate(v):
    T = np.eye(4); T[:3, 3] = v; return T


def screw_frame(body, name):
    g = MC.screw_geometry(body.mesh(name))
    return g["_head"], g["_d"]


def radial(d):
    u, _ = MC.basis(d)
    return u


def main():
    man = json.loads(MC.default_manifest().read_text())
    body = MC.F.Body(man)
    kh, kd = screw_frame(body, KNEE)
    rh, rd = screw_frame(body, ROLL)
    tilt = trimesh.transformations.rotation_matrix(math.radians(3.0), radial(kd), kh)
    blocker = trimesh.creation.box((8, 8, 2))                     # 2 mm thick along its own z
    blocker.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], kd))
    blocker.apply_translation(kh - 11.0 * kd)                      # its near face 10 mm behind the head top
    cases = [
        ("baseline_knee", KNEE, {}, {}),
        ("knee_screw_radial_0p30", KNEE, {KNEE: T_translate(0.30 * radial(kd))},
         dict(terminal_off_axis_mm=0.30, clamped_hole_off_axis_mm=0.30)),
        ("knee_screw_tilt_3deg", KNEE, {KNEE: tilt}, dict(terminal_tilt_deg=3.0)),
        ("knee_screw_out_0p50", KNEE, {KNEE: T_translate(-0.50 * kd)}, dict(head_seat_gap_mm=0.50, engagement_drop_mm=0.50)),
        ("knee_screw_in_0p40", KNEE, {KNEE: T_translate(0.40 * kd)}, dict(head_seat_gap_mm=-0.40)),
        ("knee_thigh_lifted_0p30", KNEE, {THIGH: T_translate(-0.30 * kd)}, dict(layers_gap_mm=0.30, head_seat_gap_mm=-0.30)),
        ("knee_case_moved_0p25", KNEE, {KNEE_CASE: T_translate(0.25 * radial(kd))}, dict(terminal_off_axis_mm=0.25)),
        ("knee_driver_blocked_10mm", KNEE, {"verify_blocker": blocker}, dict(driver_nearest_mm=10.0)),
        ("baseline_roll", ROLL, {}, {}),
        ("roll_sleeves_removed", ROLL, {ROLL_SLEEVES: "remove"}, dict(clamped_printed_hole_without_sleeve=1)),
    ]
    out = []
    for tag, screw, edits, expect in cases:
        mp = variant(man, body, tag, edits) if edits else MC.default_manifest()
        rep = MC.run(mp, f"^{screw}$", WORK / f"{tag}-report.json")
        r = rep["rows"][0]
        t = r.get("terminal") or {}
        meas = dict(terminal_part=t.get("part"), terminal_off_axis_mm=t.get("axis_offset_mm"), terminal_tilt_deg=t.get("angle_deg"),
                    engagement_mm=t.get("engagement_mm"),
                    head_seat_gap_mm=(r.get("head_seat") or {}).get("min_gap_mm"),
                    clamped_hole_off_axis_mm=max([h["axis_offset_mm"] or 0 for h in r.get("clamped_holes", [])] or [None]),
                    layers=[(c["upper"], c["lower"], c["min_gap_mm"]) for c in r.get("contacts", []) if c["upper"] != screw],
                    sleeves=[(s["printed_part"], s["sleeve"]) for s in r.get("sleeves", [])],
                    driver_nearest_mm=r["driver_path_without_shells"]["nearest_mm"],
                    driver_nearest_part=r["driver_path_without_shells"]["nearest_part"],
                    findings=[(f["kind"], f["value"]) for f in r["findings"]])
        out.append(dict(variant=tag, screw=screw, injected=expect, measured=meas))
        print(tag, json.dumps(meas, default=str)[:600])
    (WORK / "report.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
