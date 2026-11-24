#!/usr/bin/env python3
"""Audit exact geometry, mating planes, fastener axes, and hard-shell mounts of the 4 APE0 arm connector stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from build123d import import_step, GeomType
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from infra.storage.blob_store import ensure

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
ART = ROOT / "artifacts/robot/body/arm"


def sha256_file(path: Path) -> str:
    ensure(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_step_bores_and_planes(step_path: Path) -> dict:
    ensure(step_path)
    solid = import_step(step_path)
    bb = solid.bounding_box()
    bores = []
    planes = []
    for f in solid.faces():
        adap = BRepAdaptor_Surface(f.wrapped)
        gtype = adap.GetType()
        if gtype == GeomAbs_Cylinder:
            cyl = adap.Cylinder()
            r = float(cyl.Radius())
            u_span = float(adap.LastUParameter() - adap.FirstUParameter())
            if r < 4.0 and u_span > 5.5:
                d = np.array(cyl.Axis().Direction().Coord(), dtype=float)
                loc = np.array(cyl.Location().Coord(), dtype=float)
                fc = np.array(list(f.center()), dtype=float)
                proj = loc + d * float(np.dot(fc - loc, d))
                bores.append({
                    "radius_mm": round(r, 4),
                    "diameter_mm": round(2.0 * r, 4),
                    "axis": [round(float(x), 6) for x in d],
                    "axis_center_mm": [round(float(x), 4) for x in proj],
                    "face_center_mm": [round(float(x), 4) for x in fc],
                    "area_mm2": round(float(f.area), 3),
                })
        elif gtype == GeomAbs_Plane and f.area > 25.0:
            n = np.array(list(f.normal_at()), dtype=float)
            fc = np.array(list(f.center()), dtype=float)
            planes.append({
                "area_mm2": round(float(f.area), 2),
                "normal": [round(float(x), 6) for x in n],
                "center_mm": [round(float(x), 4) for x in fc],
                "plane_d_mm": round(float(np.dot(n, fc)), 4),
            })
    bores.sort(key=lambda b: (b["radius_mm"], b["axis_center_mm"]))
    planes.sort(key=lambda p: -p["area_mm2"])
    return {
        "path": str(step_path.relative_to(ROOT)),
        "sha256": sha256_file(step_path),
        "volume_mm3": round(float(solid.volume), 3),
        "mass_al5052_g": round(float(solid.volume) * 0.00268, 3),
        "bounds_mm": [
            [round(float(bb.min.X), 4), round(float(bb.min.Y), 4), round(float(bb.min.Z), 4)],
            [round(float(bb.max.X), 4), round(float(bb.max.Y), 4), round(float(bb.max.Z), 4)],
        ],
        "bores": bores,
        "major_planes": planes[:10],
    }


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    src_manifest_path = ROOT / "artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json"
    ensure(src_manifest_path)
    src_manifest = json.loads(src_manifest_path.read_text())
    parts_by_name = {p["name"]: p for p in src_manifest["parts"]}

    shell_manifest_path = ROOT / "artifacts/cove50-arm-shell/r5-surface-20261029/manufactured-parts.json"
    ensure(shell_manifest_path)
    shell_manifest = json.loads(shell_manifest_path.read_text())
    shell_parts_by_name = {p["name"]: p for p in shell_manifest["parts"]}

    s_conn_step = ROOT / "artifacts/cove50-arm-shell/r4-roll70-onepiece-20261029/native/R_S_yaw_to_elbow.step"
    ensure(s_conn_step)

    stages = {
        "stage1_shoulder_pitch_to_roll_F1": inspect_step_bores_and_planes(
            ROOT / parts_by_name["upper__R_pitch_to_roll_one_piece"]["native_step"]
        ),
        "stage2_shoulder_roll_to_yaw_F2": inspect_step_bores_and_planes(
            ROOT / parts_by_name["upper__R_roll_to_yaw_one_piece"]["native_step"]
        ),
        "stage3_upper_arm_yaw_to_elbow_F3_v5": inspect_step_bores_and_planes(
            ROOT / parts_by_name["upper__R_yaw_to_elbow_one_piece"]["native_step"]
        ),
        "stage3_upper_arm_yaw_to_elbow_F3_S_web": inspect_step_bores_and_planes(
            s_conn_step
        ),
        "stage4_forearm_elbow_to_wrist_F4": inspect_step_bores_and_planes(
            ROOT / parts_by_name["upper__R_forearm_wrist_one_piece"]["native_step"]
        ),
    }

    actuators = {}
    for act_name in [
        "upper__R_pitch_C047_case",
        "upper__R_pitch_C047_main_horn",
        "upper__R_pitch_C047_rear_idler",
        "upper__R_roll_C047_case",
        "upper__R_roll_C047_main_horn",
        "upper__R_yaw_C047_case",
        "upper__R_yaw_C047_main_horn",
        "upper__R_elbow_C047_case",
        "upper__R_elbow_C047_main_horn",
    ]:
        p = parts_by_name[act_name]
        sp = ROOT / p["native_step"]
        ensure(sp)
        s = import_step(sp)
        bb = s.bounding_box()
        actuators[act_name] = {
            "owning_link": p["owning_link"],
            "role": p["role"],
            "path": str(sp.relative_to(ROOT)),
            "sha256": sha256_file(sp),
            "volume_mm3": round(float(s.volume), 3),
            "bounds_mm": [
                [round(float(bb.min.X), 4), round(float(bb.min.Y), 4), round(float(bb.min.Z), 4)],
                [round(float(bb.max.X), 4), round(float(bb.max.Y), 4), round(float(bb.max.Z), 4)],
            ],
        }

    arm_joints = [
        j for j in src_manifest["joints"]
        if any(k in j["name"] for k in ("shoulder", "elbow", "wrist", "hand", "finger"))
    ]

    report = {
        "schema": "cove50.arm-fdm-baseline-audit.v1",
        "source_electronics_manifest": {
            "path": str(src_manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(src_manifest_path),
        },
        "source_arm_shell_manifest": {
            "path": str(shell_manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(shell_manifest_path),
        },
        "arm_joints": arm_joints,
        "stages": stages,
        "actuators": actuators,
        "shell_mounts": shell_manifest.get("mounts", []),
        "direct_source_mounts": shell_manifest.get("direct_source_mounts", []),
        "source_seats": shell_manifest.get("source_seats", []),
    }

    out_path = ART / "arm-baseline-audit.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")
    for k, v in stages.items():
        print(f"\n--- {k}: mass_al5052={v['mass_al5052_g']} g, vol={v['volume_mm3']} mm^3 ---")
        for b in v["bores"]:
            print(f"  bore d={b['diameter_mm']} axis={b['axis']} axis_center={b['axis_center_mm']}")
        for pl in v["major_planes"][:4]:
            print(f"  plane area={pl['area_mm2']} normal={pl['normal']} center={pl['center_mm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
