"""Source-size knee motor relocation screen against the saved moving shin.

This only screens case occupancy and a minimum pulley separation. It does not
assert that a belt, pulley interface, shaft, or thigh skin has been designed.
"""

from __future__ import annotations

import json

import build123d as bd

from ankle_pitch_iteration import volume
from knee_pitch_iteration import CASE_SIZE, OUT, pitched


def main() -> None:
    shin = bd.import_step(OUT / "shin-and-knee-yoke.step")
    thigh = bd.import_step(OUT / "relieved-thigh-and-bearing-housings.step")
    hip_case = bd.Pos(0, 41, 182.2) * bd.Box(*CASE_SIZE)
    poses = range(0, 121, 10)
    rows = []
    for x in (0, -15, 15):
        for z in (132, 136, 140, 144):
            case = bd.Pos(x, 41, z) * bd.Box(*CASE_SIZE)
            overlaps = [round(volume(pitched(shin, deg).intersect(case)), 1)
                        for deg in poses]
            row = {
                "center_x_mm": x, "center_z_mm": z,
                "official_output_z_mm": z - 12.2,
                "output_to_knee_axis_mm": round(((z - 12.2 - 103)**2 + x*x)**.5, 2),
                "case_vs_hip_case_mm3": round(volume(case.intersect(hip_case)), 1),
                "case_vs_existing_thigh_mm3": round(volume(case.intersect(thigh)), 1),
                "max_case_vs_moving_shin_mm3": max(overlaps),
                "first_clashing_knee_deg": next((deg for deg, v in zip(poses, overlaps) if v > 1), None),
            }
            rows.append(row)
            print(row, flush=True)
    (OUT / "relocated-motor-screen.json").write_text(json.dumps({
        "stage": "case-only source-size knee relocation comparison; no belt, mounting or shell proof",
        "servo_source": "https://www.feetechrc.com/en/562636.html",
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
