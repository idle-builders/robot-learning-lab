"""Phase 1: Audit the 4-stage Cove-50 Neck & Head baseline connectors (`fixed_neck_ribbon_anchor_v21`,
`one_piece_bent_neck_L`, `compact_neck_U_strap_v23`, `single_shared_carrier_v23` + `microphone_compact_tab_v23`),
their exact screw-hole coordinates, and the active `neck_yaw` x `neck_pitch` flexion clearance space.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges from the body definitions)
from build123d import import_step
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

from common import (
    ART,
    AXIS_NECK_PITCH,
    AXIS_NECK_YAW,
    BASELINE_STEPS,
    EC_MANIFEST,
    MATERIALS,
    P_NECK_PITCH,
    P_NECK_YAW,
    REPLACED_NECK_HEAD_PARTS,
    ROOT,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    read_json,
    ref,
    rel,
    stage_interface_specs,
    write_json,
)


def inspect_step(step_path: Path, density_g_cm3: float) -> dict:
    shape = import_step(str(step_path))
    solids = shape.solids()
    bb = shape.bounding_box()
    vol = float(sum(s.volume for s in solids))
    mass_g = vol * (density_g_cm3 * 1e-3)

    cylinders = []
    planes = []
    for f in shape.faces():
        s = BRepAdaptor_Surface(f.wrapped)
        t = s.GetType()
        fc = f.center()
        area = float(f.area)
        if t == GeomAbs_Cylinder:
            cyl = s.Cylinder()
            r = float(cyl.Radius())
            ax = cyl.Axis().Direction()
            loc = cyl.Location()
            cylinders.append({
                "r_mm": round(r, 4),
                "d_mm": round(2.0 * r, 4),
                "axis": [round(ax.X(), 4), round(ax.Y(), 4), round(ax.Z(), 4)],
                "origin_mm": [round(loc.X(), 3), round(loc.Y(), 3), round(loc.Z(), 3)],
                "face_center_mm": [round(fc.X, 3), round(fc.Y, 3), round(fc.Z, 3)],
                "area_mm2": round(area, 3),
            })
        elif t == GeomAbs_Plane and area >= 15.0:
            pln = s.Plane()
            n = pln.Axis().Direction()
            planes.append({
                "normal": [round(n.X(), 4), round(n.Y(), 4), round(n.Z(), 4)],
                "face_center_mm": [round(fc.X, 3), round(fc.Y, 3), round(fc.Z, 3)],
                "area_mm2": round(area, 3),
            })

    return {
        "source": ref(step_path, step_path.stem),
        "solid_count": len(solids),
        "volume_mm3": round(vol, 3),
        "baseline_mass_g": round(mass_g, 3),
        "bbox_mm": {
            "min": [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)],
            "max": [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)],
            "size": [round(bb.size.X, 3), round(bb.size.Y, 3), round(bb.size.Z, 3)],
        },
        "cylindrical_faces": cylinders,
        "major_planar_faces": sorted(planes, key=lambda d: -d["area_mm2"])[:12],
    }


def main():
    specs = stage_interface_specs()
    al_rho = MATERIALS["Al5052_H32"]["density_g_cm3"]
    pa12_rho = 1.01  # PA12 baseline density for fixed_neck_ribbon_anchor_v21

    stages = {}
    total_vol = 0.0
    total_mass = 0.0
    for stg_key, path in BASELINE_STEPS.items():
        rho = pa12_rho if stg_key == "stage1_neck_root_cradle" else al_rho
        info = inspect_step(path, rho)
        # Stage 4 also replaces microphone_compact_tab_v23 (2.0 mm Al5052-H32)
        if stg_key == "stage4_head_sensor_compute_carrier":
            mic_tab_path = ROOT / "artifacts/cove50-sensor-integration/original-shell-layout-20261026/parts/microphone_compact_tab_v23.step"
            mic_info = inspect_step(mic_tab_path, al_rho)
            info["consolidated_secondary_part"] = mic_info
            info["combined_baseline_volume_mm3"] = round(info["volume_mm3"] + mic_info["volume_mm3"], 3)
            info["combined_baseline_mass_g"] = round(info["baseline_mass_g"] + mic_info["baseline_mass_g"], 3)
            total_vol += info["combined_baseline_volume_mm3"]
            total_mass += info["combined_baseline_mass_g"]
        else:
            total_vol += info["volume_mm3"]
            total_mass += info["baseline_mass_g"]

        info["part_basename"] = STAGE_PART_BASENAMES[stg_key]
        info["owning_link"] = STAGE_LINKS[stg_key]
        info["interface_spec"] = {
            k: {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else [x.tolist() for x in vv] if isinstance(vv, list) and vv and isinstance(vv[0], np.ndarray) else vv)
                for kk, vv in v.items()
            }
            if isinstance(v, dict)
            else [
                {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in item.items()}
                for item in v
            ]
            for k, v in specs[stg_key].items()
        }
        stages[stg_key] = info

    report = {
        "study": "cove50-neck-head-fdm-connectors-20261104",
        "baseline_manifest": ref(EC_MANIFEST, "cove50-electronics-integration-r1"),
        "replaced_legacy_parts": sorted(REPLACED_NECK_HEAD_PARTS),
        "kinematics_mm": {
            "neck_yaw": {
                "origin_mm": P_NECK_YAW.tolist(),
                "axis": AXIS_NECK_YAW.tolist(),
                "range_deg": BOM_CONFIG.joint_range("neck_yaw"),
                "audit_grid_deg": [-60.0, -30.0, 0.0, 30.0, 60.0],
            },
            "neck_pitch": {
                "origin_mm": P_NECK_PITCH.tolist(),
                "axis": AXIS_NECK_PITCH.tolist(),
                "range_deg": BOM_CONFIG.joint_range("neck_pitch"),
                "audit_grid_deg": [-30.0, -15.0, 0.0, 17.5, 35.0],
            },
        },
        "stages": stages,
        "total_baseline_volume_mm3": round(total_vol, 3),
        "total_baseline_mass_g": round(total_mass, 3),
        "baseline_structural_gaps_identified": {
            "stage1_neck_root_cradle": (
                "Legacy fixed_neck_ribbon_anchor_v21 (PA12, 2.35 g, 2330.8 mm^3) only held the rear ribbon cable cap at Z=375 mm "
                "via a thin 3 mm L-strip cantilevered off two M3 torso spine bolts, while leaving all 4 rear M2 tapped holes "
                "on upper__neck_yaw_case_XM335 unused after Common_power_carrier_mounting_r33 excised A_neck_single_twofold_bracket_r5's rear web."
            ),
            "stage2_neck_yaw_to_pitch_link": (
                "Legacy one_piece_bent_neck_L (5052-H32 Al, 15.16 g, 5656.4 mm^3) used only 2 of the 4 M2 tapped holes on "
                "upper__neck_yaw_front_horn and 2 of the 4 M2 tapped holes on upper__neck_pitch_front_horn, concentrating "
                "all head pitch and yaw moments onto two screws per horn across an unclosed 90-degree L-bend."
            ),
            "stage3_neck_pitch_to_head_cradle": (
                "Legacy compact_neck_U_strap_v23 (3 mm 5052-H32 Al, 12.83 g, 4787.3 mm^3) wrapped around the +X side of neck_pitch_case_XM335 "
                "as an unclosed U-strap with a narrow 8 mm tongue at X in [-4, 4] clamped by "
                "two vertical M3 bolts at X=0, Z in {413, 419}, offering limited lateral roll seating and open C-channel torsion."
            ),
            "stage4_head_sensor_compute_carrier": (
                "Legacy single_shared_carrier_v23 (1.5 mm 5052-H32 Al, 20.73 g, 7736.4 mm^3) + separate microphone_compact_tab_v23 "
                "(2.0 mm 5052-H32 Al, 1.60 g, 595.2 mm^3) + 2x M2x3 foot screws relied on a flat 1.5 mm sheet to cantilever ROCK4D, "
                "heatsink, fan, and two cameras from the narrow Z=413..419 neck root boss."
            ),
        },
    }

    out_path = ART / "neck-head-baseline-audit.json"
    write_json(out_path, report)
    print(f"Wrote {rel(out_path)} | total baseline mass = {total_mass:.2f} g ({total_vol:.1f} mm^3)")
    for k, v in stages.items():
        print(f"  {k}: vol={v['volume_mm3']:.1f} mm^3, mass={v['baseline_mass_g']:.2f} g, bbox={v['bbox_mm']['min']}..{v['bbox_mm']['max']}")


if __name__ == "__main__":
    main()
