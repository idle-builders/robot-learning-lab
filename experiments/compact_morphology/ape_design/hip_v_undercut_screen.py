"""One-factor symmetric V-pelvis underside relief against saved knee CAD.

The source-size hip case, output and bearing seats stay at their original axes.
This is a cosmetic/clearance screen, not a structural approval of the shell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as bd
import trimesh

from ankle_pitch_iteration import cylinder_y, volume
from coupled_leg_clearance import HIP, KNEE, turn
from hip_pitch_iteration import add


ROOT = Path(__file__).resolve().parents[3]
MOTOR_SAFE = "--motor-safe" in sys.argv
OUT = ROOT / ("artifacts/ape-hip-v-undercut-motor-safe-20261003" if MOTOR_SAFE else
              "artifacts/ape-hip-v-undercut-20261003")
HIP_OUT = ROOT / "artifacts/ape-hip-pitch-iteration-20261003"
KNEE_OUT = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
ANKLE_OUT = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"
POSES = ((-54.43, 108.86), (-60, 120), (-40, 0), (0, 0))


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pelvis = bd.import_step(HIP_OUT / "V-pelvis-with-pitch-socket.step")
    thigh = bd.import_step(HIP_OUT / "thigh-with-hip-pitch-yoke.step")
    shin = bd.import_step(KNEE_OUT / "shin-and-knee-yoke.step")
    boot = bd.import_step(ANKLE_OUT / "boot-and-yoke.step").fuse(
        bd.import_step(ANKLE_OUT / "oval-outsole.step"))
    # Same smooth undercut on both sides. The centre and 3 radii are a screen
    # selected from the exact lower-pelvis clash bounds, not a claimed print.
    radii = (50, 39, 39) if MOTOR_SAFE else (50, 39, 42)
    cutters = [bd.Pos(20, side * 35, 120) *
               bd.scale(bd.Sphere(1), by=radii) for side in (1, -1)]
    undercut = pelvis.cut(*cutters)
    # The original hip socket was relieved at 20-degree samples, so the real
    # -54.43-degree training reset exposes a separate upper-thigh intersection.
    # Relieve that exact candidate pose as an additional, visibly reported
    # local socket change. Neighbouring angles still need a denser sweep.
    cut = undercut - turn(thigh, HIP, -54.43)
    solids = sorted(cut.solids(), key=lambda part: part.volume, reverse=True)
    revised = solids[0]
    removed = round(pelvis.volume - revised.volume, 1)

    hip_motor = bd.Pos(0, 41, 182.2) * bd.Box(24.72, 35, 45.22)
    bearing_seats = [cylinder_y(11.5, 6, 0, y, 170) -
                     cylinder_y(8.05, 7, 0, y, 170) for y in (9, 73)]
    rows = []
    for h, k in POSES:
        leg = turn(thigh, HIP, h)
        calf = turn(turn(shin, KNEE, k), HIP, h)
        foot = turn(turn(boot, KNEE, k), HIP, h)
        clash = calf.intersect(revised)
        row = {
            "hip_pitch_deg": h, "knee_flex_deg": k,
            "source_floor_reach_squat_reset": (h, k) == (-54.43, 108.86),
            "shin_vs_revised_pelvis_mm3": round(volume(clash), 2),
            "boot_vs_revised_pelvis_mm3": round(volume(foot.intersect(revised)), 2),
            "thigh_vs_revised_pelvis_mm3": round(volume(leg.intersect(revised)), 2),
        }
        rows.append(row)
        scene = trimesh.Scene()
        add(scene, "revised_V_pelvis", revised, (232, 222, 205, 255))
        add(scene, "moving_thigh", leg, (242, 233, 215, 255))
        add(scene, "moving_shin", calf, (247, 236, 218, 255))
        add(scene, "moving_boot", foot, (227, 217, 195, 255))
        add(scene, "ST3250_hip_case", hip_motor, (77, 112, 148, 180))
        for i, bearing in enumerate(bearing_seats):
            add(scene, f"bearing_seat_{i+1}", bearing, (92, 98, 102, 255))
        scene.export(OUT / f"hip{h}-knee{k}.glb")
        print(row, flush=True)
    report = {
        "stage": "smooth symmetric V-pelvis underside relief; geometry candidate only",
        "source_floor_reach_pose": "docs/research/ape-left-reach-screen-20260930.md",
        "unchanged_hip_axis_mm": HIP,
        "cutters": f"two mirrored ellipsoids, centres (20, ±35, 120) mm, radii {radii} mm",
        "motor_safe_undercut_variant": MOTOR_SAFE,
        "original_fixed_pelvis_volume_mm3": round(pelvis.volume, 1),
        "removed_fixed_volume_mm3": removed,
        "additional_task_pose_thigh_socket_relief_mm3": round(undercut.volume - cut.volume, 1),
        "removed_pct_of_fixed_shell": round(100 * removed / pelvis.volume, 1),
        "revised_connected_solids": len(revised.solids()),
        "discarded_chip_volume_mm3": round(sum(s.volume for s in solids[1:]), 1),
        "cut_vs_left_hip_motor_mm3": round(volume(cutters[0].intersect(hip_motor)), 2),
        "cut_vs_left_bearing_seats_mm3": round(sum(volume(cutters[0].intersect(s)) for s in bearing_seats), 2),
        "poses": rows,
        "limits": ["only selected poses; continuous path and tolerances remain open",
                   "a clean Boolean shell is not proof of joint load path or printability",
                   "flexible hip skin, cable and roll/yaw carrier are not present",
                   "independent appearance review required"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    bd.export_step(revised, OUT / "V-pelvis-symmetric-relief.step")
    (OUT / "index.html").write_text(Path(__file__).with_name("hip_v_undercut_viewer.html").read_text())
    return report


if __name__ == "__main__":
    main()
