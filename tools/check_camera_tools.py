"""Supplementary positive tool reach and drawing-axis witnesses, before bezel fit."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build123d import Pos
from lab.camera_support import (
    carrier_parts,
    hardware_parts,
    module_parts,
    cylinder,
    specification,
)
from lab.camera_vendor import assembly_overlap as overlap


def check(spec=None):
    spec = spec or specification()
    # Independent drawing coordinates, rather than generated support centers.
    axes = [(-10.5, -12.4), (10.5, -12.4), (-10.5, 0.1), (10.5, 0.1)]
    carrier = carrier_parts(spec)["camera_rear_carrier"]
    pcb = module_parts(spec)[
        "camera_vendor_module" if spec["profile"] == "vendor_step" else "camera_pcb"
    ]
    hardware = hardware_parts(spec)
    rows = []
    for i, (y, z) in enumerate(axes):
        front = cylinder(9.5, y, z, 1.5, 32)
        rear = cylinder(-20, y, z, 3, 10)
        axis = cylinder(-11, y, z, 1, 10)
        screw = hardware[f"camera_board_screw_{i}"]
        nut = hardware[f"camera_board_nut_{i}"]
        row = dict(
            index=i,
            front_target_overlap_mm3=overlap(front, screw),
            rear_target_overlap_mm3=overlap(rear, nut),
            support_axis_overlap_mm3=overlap(axis, carrier),
            pcb_axis_overlap_mm3=overlap(axis, pcb),
            displaced_support_negative_control_mm3=overlap(
                axis, Pos(0, 0.5, 0) * carrier
            ),
            short_front_negative_control_mm3=overlap(Pos(40, 0, 0) * front, screw),
        )
        row["status"] = (
            "pass"
            if row["front_target_overlap_mm3"] > 1
            and row["rear_target_overlap_mm3"] > 1
            and row["support_axis_overlap_mm3"] < 0.02
            and row["pcb_axis_overlap_mm3"] < 0.02
            and row["displaced_support_negative_control_mm3"] > 0.02
            and row["short_front_negative_control_mm3"] < 0.02
            else "fail"
        )
        rows.append(row)
    return dict(
        status="pass" if all(r["status"] == "pass" for r in rows) else "fail",
        rows=rows,
        scope="nominal drawing-axis clearance and positive reach in bench assembly; tool profile, fastening torque and thread contact unqualified",
    )


if __name__ == "__main__":
    p = Path(sys.argv[1])
    p.mkdir(parents=True, exist_ok=False)
    r = check()
    r["source_hashes"] = {
        str(s): hashlib.sha256(s.read_bytes()).hexdigest()
        for s in [
            Path(__file__).relative_to(Path.cwd())
            if Path(__file__).is_absolute()
            else Path(__file__),
            Path("lab/camera_support.py"),
            Path("lab/camera_mount.py"),
            Path("electronics/camera-support.yaml"),
        ]
    }
    (p / "report.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r))
