"""Fit a source STEP clamping coupler and catalog-width roll pulleys to the ape.

This is a geometry/assembly screen, not a spline-fit or loaded duty proof.
The ZIP is fetched only when absent and rejected if its bytes change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import build123d as bd
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import (
    candidate, intersection, mesh,
)

OUT = ROOT / "artifacts/ape-frame50-roll-coupler"
SOURCE = OUT / "vendor-source"
ZIP = SOURCE / "4001-0025-0006.zip"
STEP = SOURCE / "4001-0025-0006 assembly.STEP"
URL = "https://www.gobilda.com/content/step_files/4001-0025-0006.zip"
ZIP_SHA = "44f87f1fb182129e07da5ee86608c2f454eee6bab78bb0629fd372e931a3cdac"
STEP_SHA = "bc27ad4a29019e0f4a5cb7731eae238d3b20f7146224de1510094871e9ce19cc"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_step() -> Path:
    SOURCE.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        with urllib.request.urlopen(urllib.request.Request(URL, headers={
                "User-Agent": "robot-learning-lab/1.0"})) as response:
            ZIP.write_bytes(response.read())
    if sha(ZIP) != ZIP_SHA:
        raise ValueError("goBILDA source ZIP changed; inspect new source before use")
    if not STEP.exists():
        with zipfile.ZipFile(ZIP) as archive:
            STEP.write_bytes(archive.read("4001-0025-0006 assembly.STEP"))
    if sha(STEP) != STEP_SHA:
        raise ValueError("goBILDA source STEP changed")
    return STEP


def installed_coupler() -> bd.Shape:
    # The pinned manufacturer STEP axis is +Y. Center its spline on the
    # Feetech output at X=59, Z=278.5 and put its input face at Y=-60.
    return bd.Pos(116.362905338, -66.773472927, 260.155190795) * bd.import_step(
        str(source_step()))


def exact_intersection(a: bd.Shape, b: bd.Shape) -> float:
    # build123d's Compound boolean returned zero for this two-solid vendor
    # STEP despite a large real overlap. Run BReps on each solid explicitly.
    return sum(intersection(x, y) for x in a.solids() for y in b.solids())


def nonmating_hits(stages: dict) -> list[dict]:
    yaw, roll, fixed = (stages[name] for name in ("yaw", "roll", "fixed"))
    pairs = (
        ("coupler", yaw["roll_clamping_coupler_4001"], "roll belt", yaw["roll_belt_123_3MGT_09"]),
        ("coupler", yaw["roll_clamping_coupler_4001"], "roll pulley", yaw["roll_drive_20T"]),
        ("coupler", yaw["roll_clamping_coupler_4001"], "motor brace", yaw["roll_motor_standoff"]),
        ("coupler", yaw["roll_clamping_coupler_4001"], "yaw fork", yaw["yaw_fork_front"]),
        ("roll pulley", yaw["roll_drive_20T"], "yaw fork", yaw["yaw_fork_front"]),
        ("driven pulley", roll["roll_driven_20T"], "front bearing seat", yaw["roll_seat_front"]),
        ("motor brace", yaw["roll_motor_standoff"], "driven pulley", roll["roll_driven_20T"]),
        ("motor case", yaw["roll_ST3250_case"], "yaw housing", fixed["yaw_housing_tube"]),
    )
    return [{"a": a_name, "b": b_name, "overlap_mm3": round(volume, 3)}
            for a_name, a, b_name, b in pairs
            if (volume := exact_intersection(a, b)) > .05]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual = installed_coupler()
    hw, _ = candidate()
    envelope = hw["yaw"]["roll_clamping_coupler_4001"]
    outside = sum((piece - envelope).volume for piece in actual.solids())
    # Full STEP for this specific manufactured coupler, instead of relying
    # only on its deliberately conservative source-size primitive envelope.
    hw["yaw"]["roll_clamping_coupler_4001"] = actual
    poses = {}
    for pose, angles in hip.POSES.items():
        poses[pose] = {}
        for side, sign in (("left", -1), ("right", 1)):
            poses[pose][side] = nonmating_hits(hip.frame(hw, sign, *angles))
    case = hw["yaw"]["roll_ST3250_case"]
    brace = hw["yaw"]["roll_motor_standoff"]
    supported_joins = {
        "brace_to_case_mm3": round(intersection(brace, case), 3),
        "brace_to_yaw_fork_mm3": round(intersection(brace, hw["yaw"]["yaw_fork_front"]), 3),
    }
    show = {
        "ST3250_case": (case, (63, 99, 129, 255)),
        "4001_coupler_exact_STEP": (actual, (221, 145, 78, 255)),
        "6mm_output_shaft": (hw["yaw"]["roll_6mm_output_shaft"], (154, 167, 173, 255)),
        "drive_20T_flanged_pulley": (hw["yaw"]["roll_drive_20T"], (76, 139, 162, 255)),
        "driven_20T_flanged_pulley": (hw["roll"]["roll_driven_20T"], (76, 139, 162, 255)),
        "9mm_3MGT_belt": (hw["yaw"]["roll_belt_123_3MGT_09"], (71, 154, 118, 255)),
        "motor_brace": (brace, (181, 181, 172, 255)),
        "front_yaw_fork": (hw["yaw"]["yaw_fork_front"], (181, 181, 172, 255)),
        "front_roll_bearing_seat": (hw["yaw"]["roll_seat_front"], (181, 181, 172, 255)),
        "roll_output_shaft": (hw["roll"]["roll_spindle_front"], (154, 167, 173, 255)),
    }
    scene = trimesh.Scene()
    for name, (solid, color) in show.items():
        scene.add_geometry(mesh(solid, color), geom_name=name, node_name=name)
    scene.export(OUT / "roll-coupler-package.glb")
    bd.export_step(bd.Compound(children=[item[0] for item in show.values()]),
                   str(OUT / "roll-coupler-package.step"))
    (OUT / "index.html").write_text(Path(__file__).with_name(
        "roll_coupler_viewer.html").read_text())
    result = {
        "status": "specific coupler STEP and catalog-size pulley assembly candidate; not load qualified",
        "source_step_url": URL, "source_zip_sha256": ZIP_SHA, "source_step_sha256": STEP_SHA,
        "source_coupler": "goBILDA 4001-0025-0006, H25T to 6mm clamping bore, 6g",
        "source_coupler_page": "https://www.gobilda.com/4001-series-clamping-servo-to-shaft-coupler-25-tooth-spline-to-6mm-round-bore/",
        "source_pulley": "MISUMI GPA20GT3090-A-P6, 20T/3GT, 9mm belt, 6mm bore, 14mm full width, M4 set screws",
        "source_pulley_catalog": "https://uk.misumi-ec.com/pdf/fa/2014/P1_1383-1384_F19_EN.pdf",
        "source_belt": "Gates 123-3MGT-09",
        "source_belt_catalog": "https://gates.com/content/dam/documents-library/catalogs/power-transmission-catalog.pdf",
        "actual_step_outside_source_bounds_envelope_mm3": round(outside, 3),
        "source_STEP_vs_nominal_6mm_shaft_hard_overlap_mm3": round(
            exact_intersection(actual, hw["yaw"]["roll_6mm_output_shaft"]), 3),
        "center_distance_mm": 31.5, "drive_ratio": "1:1",
        "case_y_mm": [-95, -60], "coupler_y_mm": [-60, -43],
        "drive_pulley_y_mm": [-39, -25], "belt_y_mm": [-36.5, -27.5],
        "coupler_to_pulley_axial_gap_mm": 4,
        "pulley_to_front_bearing_seat_axial_gap_mm": 1,
        "shaft_to_coupler_nominal_axial_overlap_mm": 11,
        "shaft_to_drive_pulley_nominal_axial_overlap_mm": 14,
        "supported_joins": supported_joins,
        "pose_nonmating_hits": poses,
        "open": ["H25T-to-ST3250 spline fit requires a real mating check",
                 "Clamp and M4 pulley set-screw torque capacities and preload are unverified",
                 "6mm shaft stock, tolerances, service access and belt radial-load deflection are unverified",
                 "The 1mm driven-pulley/seat gap is nominal; continuous/tolerance and loaded clearance not proved",
                 "Belt length, tooth/cord load and tension adjustment need qualification"],
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_envelope_excess_mm3": result["actual_step_outside_source_bounds_envelope_mm3"],
                      "joins": supported_joins,
                      "pose_hit_counts": {p: sum(len(h) for h in sides.values())
                                          for p, sides in poses.items()}}, indent=2))


if __name__ == "__main__":
    main()
