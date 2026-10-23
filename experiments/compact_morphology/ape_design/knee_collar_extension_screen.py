"""Find moving-shin shell material that can extend into the direct knee gap.

An original-shell 75..95 mm band is clipped by the inverse sweep of the saved
fixed thigh and source-size motor. The result is only a zero-tolerance free
volume screen, not a finished collar or an assertion of continuous clearance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as bd

from knee_pitch_iteration import CASE_CENTER, CASE_SIZE, pitched


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
ANKLE = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"
OUT = ROOT / "artifacts/ape-knee-moving-collar-screen-20261004"
ANGLES = tuple(range(0, 121, 10))
REUSE = "--reuse-swept-step" in sys.argv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = bd.import_step(SOURCE / "relieved-thigh-and-bearing-housings.step")
    moving = bd.import_step(SOURCE / "shin-and-knee-yoke.step")
    original_shin = bd.import_step(ANKLE / "shin-and-ankle-sleeve.step")
    motor = bd.Pos(*CASE_CENTER) * bd.Box(*CASE_SIZE)
    # Begin just below the saved 75 mm cut so the surviving material can
    # actually connect to the moving shin if enough of the band remains.
    raw = original_shin & (bd.Pos(0, 41, 84.75) * bd.Box(200, 200, 20.5))
    lost = []
    saved = OUT / "collision-free-moving-shell-band.step"
    if REUSE:
        free = bd.import_step(saved)
    else:
        free = raw
        for angle in ANGLES:
            before = free.volume
            free = free - pitched(fixed, -angle) - pitched(motor, -angle)
            lost.append({"knee_flex_deg": angle,
                         "additional_shell_volume_removed_mm3": round(before - free.volume, 2),
                         "shell_volume_left_mm3": round(free.volume, 2)})
        bd.export_step(free, saved)
    combined = moving.fuse(free)
    bd.export_step(combined, OUT / "moving-shin-with-free-band.step")
    report = {
        "stage": "Inverse-swept moving shin band screen; no surface allowance, skin fastening or gaiter",
        "source_direct_knee": str(SOURCE.relative_to(ROOT)),
        "raw_band_z_mm": [74.5, 95.0],
        "raw_shell_band_volume_mm3": round(raw.volume, 2),
        "remaining_free_shell_band_volume_mm3": round(free.volume, 2),
        "remaining_percent": round(100 * free.volume / raw.volume, 1),
        "free_band_connected_solids": len(free.solids()),
        "combined_moving_connected_solids": len(combined.solids()),
        "removed_by_pose": lost,
        "screen_verdict": "REJECT: naive inverse-swept original shin band fragments instead of yielding a manufacturable moving collar" if len(free.solids()) > 1 else "requires further clearance and attachment engineering",
        "warning": "Boolean subtraction at 10 degree intervals has zero tolerance; a positive running gap and between-sample clearance have not been established. This fragmented output is rejected and was not sent through further costly posed-intersection checks. Shape a new collar with a real mating/retention interface instead.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("raw_shell_band_volume_mm3", "remaining_free_shell_band_volume_mm3", "remaining_percent", "free_band_connected_solids", "combined_moving_connected_solids")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
