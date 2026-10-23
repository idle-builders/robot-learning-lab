"""Static shoulder package comparison: two direct cases vs a paired-axis case.

Case dimensions are sourced; paired module horn positions and mount structure
are deliberately unknown. This answers only whether an envelope can occupy
the approved sloped deltoid in neutral, before a moving joint is engineered.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
from hip_joint_coupon import add, case, cyl, overlaps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-shoulder-joint-coupon-20261002"


def run(height: int):
    s = height / 350
    scenarios = []
    for side, sign in (("left", 1), ("right", -1)):
        region = ("torso_shell", f"{side}_upper_arm_shell")
        direct = [
            case(f"{side}_direct_yaw_XL430", (-4*s, sign*60*s, 258*s),
                 (28.5, 34, 46.5), region, s),
            case(f"{side}_direct_roll_XL430", (-4*s, sign*69*s, 247*s),
                 (28.5, 34, 46.5), region, s),
        ]
        paired = case(f"{side}_paired_2XL430", (-4*s, sign*75*s, 244*s),
                      (36, 36, 46.5), region, s)
        scenarios.append({
            "side": side, "serial_direct": direct,
            "serial_direct_case_intersection": overlaps(*direct, gap=0),
            "paired": paired,
            "paired_center_to_model_yaw_axis_mm": round(((75-60)**2+(244-258)**2)**.5*s, 2),
            "paired_center_to_model_roll_axis_mm": round(((75-69)**2+(244-247)**2)**.5*s, 2),
        })
    return {"height_mm": height, "scenarios": scenarios,
            "caveat": "The paired motor's actual output axes are unknown here. A case that fits is not a kinematically attached shoulder. Static shell corner and direct-box overlap only; no cable, bearing, mount, torque, reach/crawl sweep, or revised policy."}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    results = [run(h) for h in (350, 500, 600)]
    (OUT / "screen.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        render(result)
        scenarios = result["scenarios"]
        print(result["height_mm"], "mm: paired outside corners",
              sum(len(x["paired"]["outside_corners"]) for x in scenarios),
              "; direct case overlaps", sum(x["serial_direct_case_intersection"] for x in scenarios))
    (OUT / "index.html").write_text(Path(__file__).with_name("shoulder_joint_viewer.html").read_text())


def render(result: dict):
    s = result["height_mm"] / 350
    for mode in ("direct", "paired"):
        scene = trimesh.Scene()
        for part in shell.PARTS:
            if part.name == "torso_shell" or part.name.endswith("_upper_arm_shell"):
                add(scene, f"approved_{part.name}", part.shape.scale(s), "shell")
        for item in result["scenarios"]:
            side = item["side"]
            chosen = item["serial_direct"] if mode == "direct" else [item["paired"]]
            for motor in chosen:
                add(scene, motor["name"], bd.Pos(*motor["center_mm"]) * bd.Box(*motor["size_mm"]),
                    "motor" if mode == "direct" else "paired")
            sign = 1 if side == "left" else -1
            add(scene, f"{side}_model_yaw_reference", cyl(1.7, 50, (-4*s, sign*60*s, 258*s)), "axis")
            add(scene, f"{side}_model_roll_reference", cyl(1.7, 50, (-4*s, sign*69*s, 247*s), "x"), "axis")
        scene.export(OUT / f"shoulder-{result['height_mm']}-{mode}.glb")


if __name__ == "__main__":
    main()
