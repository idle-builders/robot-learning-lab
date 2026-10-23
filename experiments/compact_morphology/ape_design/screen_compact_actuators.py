"""Cheap/small actuator geometry counterfactuals inside the approved shell.

XL330 uses the dimensions from the existing source-labelled layout; its 7 mm
shaft-to-case-center is an estimate, so passing geometry is not an approval.
This screen deliberately leaves the control/torque implications separate.
"""

from __future__ import annotations

import copy
import json

import build123d as bd
import numpy as np

import sketch_shell as shell
from search_shell_cases import OUT, corners, shell_volume
from sweep_shell_scale import choose_cases, replay_pairs

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ape_cad"))
import build as envelope  # noqa: E402


def battery_search(cases: list[dict]) -> dict:
    motors = [c for c in cases]
    candidates = []
    for x in np.arange(-.030, .031, .010):
        for z in np.arange(.160, .251, .010):
            battery = {"name": "battery_4s_850", "center": np.array([x, 0., z]),
                       "rot": np.eye(3), "half": np.array([.030, .015, .01525])}
            outside = sum(not shell.INTERIORS["torso_shell"].is_inside(
                bd.Vector(*(corner * 1000))) for corner in corners(battery))
            clashes = envelope.clashes(motors + [battery], 1)
            motor_clashes = [pair for pair in clashes if "battery_4s_850" in pair]
            candidates.append((len(motor_clashes) * 100 + outside,
                               outside, motor_clashes, battery))
    candidates.sort(key=lambda entry: entry[0])
    _, outside, clashes, battery = candidates[0]
    battery_solid = bd.Pos(*(battery["center"] * 1000)) * bd.Box(60, 30, 30.5)
    shell_hits = {part.name: round(shell_volume(battery_solid.intersect(part.shape)) / 1000, 3)
                  for part in shell.PARTS}
    shell_hits = {name: volume for name, volume in shell_hits.items() if volume > .001}
    return {"center_mm": (battery["center"] * 1000).round(1).tolist(),
            "outside_torso_interior_corners": outside,
            "case_clashes": clashes,
            "selected_shell_intersections_cm3": shell_hits,
            "grid_note": "Battery center x=-30..30mm and z=160..250mm in 10mm steps, y=0; 4S 850 case 60×30×30.5mm. Grid ranking uses interior corners and case clashes; exact shell BRep is checked on the selected point only. This is not retention or wiring."}


def main():
    shell.build()
    original = copy.deepcopy(envelope.CFG["parts"])
    rows = []
    for scenario in ("strong-legs-xl-arms", "all-xl330"):
        envelope.CFG["parts"] = copy.deepcopy(original)
        envelope.CFG["parts"]["arms"] = copy.deepcopy(original["small"])
        if scenario == "all-xl330":
            envelope.CFG["parts"]["legs"] = copy.deepcopy(original["small"])
        for axes in (22, 20):
            selected = choose_cases(axes, 1.0)
            poses = [c.pose for c in selected.values()]
            pairs = envelope.clashes(poses, 1)
            battery = battery_search(poses)
            reference = json.loads((OUT / f"case-screen-{axes}.json").read_text())
            samples = json.loads((OUT / f"motion-screen-{axes}.json").read_text())["samples"]
            chosen_battery = np.array(battery["center_mm"]) / 1000
            motion_counts = [len(replay_pairs(selected, sample, reference, 1.0, chosen_battery))
                             for sample in samples]
            row = {"scenario": scenario, "axes": axes,
                   "case_pairs_without_battery_neutral": len(pairs),
                   "pair_names": pairs,
                   "outside_interior_case_corners_neutral":
                       sum(c.outside_corners for c in selected.values()),
                   "best_static_battery_grid": battery,
                   "worst_saved_pose_case_pairs_with_battery": max(motion_counts),
                   "sampled_frames_with_pairs": sum(n > 0 for n in motion_counts),
                   "sampled_frames": len(motion_counts),
                   "layout_caveat": "XL330 case is 34×26×20mm, 18g, but shaft offset is an estimate in layout.yaml. Existing policy has not run with its lower torque or this mass distribution. No mount, moving-shell or battery restraint is designed."}
            rows.append(row)
            print(scenario, axes, "neutral pairs", len(pairs), "outside",
                  row["outside_interior_case_corners_neutral"], "battery",
                  battery["center_mm"], "moving worst", max(motion_counts), flush=True)
    (OUT / "compact-actuator-screen.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
