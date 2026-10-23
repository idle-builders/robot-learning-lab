"""Source-sized battery in the approved V torso, with neighboring joint cases.

This is a packing and mass-location screen. The service reserve is explicitly
an assumption; no compute board, regulator or PCB has been selected.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
import hip_joint_coupon as hip
import shoulder_joint_coupon as shoulder
from hip_joint_coupon import add, case, overlaps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-torso-pack-coupon-20261002"
HEIGHTS = (350, 500, 600)
BATTERY_MM = (60.0, 30.0, 30.5)
BATTERY_KG = .107
# 45x35x20 mm is a design space reservation, not a purchased component.
SERVICE_MM = (45.0, 35.0, 20.0)


def volume_intersection(a: bd.Shape, b: bd.Shape) -> float:
    v = a.intersect(b)
    return round(sum(x.volume for x in v), 1) if v else 0.0


def screen(height: int) -> dict:
    s = height / 350
    near = hip.screen(height)["cases"]
    for side in shoulder.run(height)["scenarios"]:
        near.append(side["paired"])
    torso = next(p.shape for p in shell.PARTS if p.name == "torso_shell").scale(s)
    thighs = [p.shape.scale(s) for p in shell.PARTS if p.name.endswith("_thigh_shell")]
    candidates = []
    for z_nom in (155, 165, 175, 185, 195, 205, 215):
        for x_nom in (-10, 0, 10):
            center = (x_nom * s, 0, z_nom * s)
            battery = case("Tattu_4S850", center, BATTERY_MM, ("torso_shell",), s)
            solid = bd.Pos(*center) * bd.Box(*BATTERY_MM)
            shell_cross = volume_intersection(solid, torso)
            thigh_cross = sum(volume_intersection(solid, item) for item in thighs)
            clashes = [item["name"] for item in near if overlaps(battery, item, gap=1)]
            candidates.append({"center_mm": [round(v, 1) for v in center],
                               "outside_torso_corners": len(battery["outside_corners"]),
                               "torso_skin_intersection_mm3": shell_cross,
                               "thigh_skin_intersection_mm3": round(thigh_cross, 1),
                               "nearby_motor_envelope_clashes": clashes,
                               "static_screen_pass": not battery["outside_corners"] and
                               shell_cross < .1 and thigh_cross < .1 and not clashes})
    passing = [row for row in candidates if row["static_screen_pass"]]
    # Lower battery height is preferred; moving x from centre is penalized.
    chosen = min(passing, key=lambda row: (row["center_mm"][2], abs(row["center_mm"][0]))) if passing else min(
        candidates, key=lambda row: (row["outside_torso_corners"],
                                     len(row["nearby_motor_envelope_clashes"]),
                                     row["torso_skin_intersection_mm3"] + row["thigh_skin_intersection_mm3"]))
    battery_as_prior = (0, 0, 191 * s)
    dz = chosen["center_mm"][2] - battery_as_prior[2]
    service = case("assumed_compute_service_reserve", (0, 0, 230*s),
                   SERVICE_MM, ("torso_shell",), s)
    service_clashes = [item["name"] for item in near if overlaps(service, item, gap=1)]
    service_battery_clash = overlaps(service, {
        "center_mm": chosen["center_mm"], "size_mm": BATTERY_MM}, gap=1)
    # The retained 35cm MJCF mass is source-labelled elsewhere; scaled models
    # have no established body mass. Use a 35cm-only sensitivity number.
    return {
        "height_mm": height,
        "battery_source": "Tattu 850mAh 4S, 60x30x30.5 mm, 107 g; candidate only; needs 12V/5V conversion",
        "assumed_service_reserve_mm": SERVICE_MM,
        "assumed_service_reserve_outside_torso_corners": len(service["outside_corners"]),
        "assumed_service_reserve_motor_clashes": service_clashes,
        "assumed_service_reserve_battery_clash": service_battery_clash,
        "checked_battery_centres": len(candidates),
        "static_passing_centres": len(passing),
        "lowest_selected_battery_candidate": chosen,
        "prior_model_battery_height_mm": round(battery_as_prior[2], 1),
        "battery_height_change_mm": round(dz, 1),
        "whole_robot_COM_shift_if_other_mass_fixed_mm": round(BATTERY_KG * dz / 2.26553, 2) if height == 350 else None,
        "candidate_source_motor_cases": [item["name"] for item in near],
        "all_battery_candidates": candidates,
        "warning": "Battery corner/solid and AABB motor screens only; 2XC/2XL case positions are unresolved architectural hypotheses. No battery mount, wiring service path, regulator, actual compute module, heat, or complete mass/inertia model. 35cm COM sensitivity uses unchanged 2.26553kg MJCF mass and no new motor layout.",
    }


def render(report: dict) -> None:
    s = report["height_mm"] / 350
    scene = trimesh.Scene()
    add(scene, "approved_V_torso_skin",
        next(p.shape for p in shell.PARTS if p.name == "torso_shell").scale(s), "shell")
    for p in shell.PARTS:
        if p.name.endswith("_thigh_shell") or p.name.endswith("_upper_arm_shell"):
            add(scene, "approved_" + p.name, p.shape.scale(s), "shell")
    for item in hip.screen(report["height_mm"])["cases"]:
        add(scene, item["name"], bd.Pos(*item["center_mm"]) * bd.Box(*item["size_mm"]),
            "motor" if "yaw" in item["name"] else "paired")
    for side in shoulder.run(report["height_mm"])["scenarios"]:
        item = side["paired"]
        add(scene, item["name"], bd.Pos(*item["center_mm"]) * bd.Box(*item["size_mm"]), "paired")
    c = report["lowest_selected_battery_candidate"]["center_mm"]
    add(scene, "source_Tattu_4S850", bd.Pos(*c) * bd.Box(*BATTERY_MM), "gear")
    # Deliberately visible design-space reservation, not a selected board.
    add(scene, "hypothetical_compute_service_reserve",
        bd.Pos(0, 0, 230*s) * bd.Box(*SERVICE_MM), "bearing")
    scene.export(OUT / f"torso-{report['height_mm']}.glb")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    reports = [screen(h) for h in HEIGHTS]
    for report in reports:
        render(report)
        print(report["height_mm"], "mm:", report["static_passing_centres"],
              "of", report["checked_battery_centres"], "grid centres pass; chosen",
              report["lowest_selected_battery_candidate"]["center_mm"], flush=True)
    (OUT / "screen.json").write_text(json.dumps(reports, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("torso_pack_viewer.html").read_text())


if __name__ == "__main__":
    main()
