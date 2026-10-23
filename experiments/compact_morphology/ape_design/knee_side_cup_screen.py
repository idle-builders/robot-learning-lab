"""Two removable outboard knee-cup envelopes around the direct-drive bearing housings.

The cups screen shape and swept rigid clearance only. They do not close the
front/rear gap or include their clips, seals or compliant gusset.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

from ankle_pitch_iteration import volume
from knee_pitch_iteration import pitched


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
FLAT = "--flat" in sys.argv
OUT = ROOT / ("artifacts/ape-knee-flat-cheek-screen-20261004" if FLAT else
              "artifacts/ape-knee-side-cup-screen-20261004")
POSES = (0, 30, 60, 90, 120)


def ellipse(y: float, rx: float, rz: float) -> bd.Wire:
    return bd.Wire.make_polygon([
        (rx * math.cos(2 * math.pi * index / 48), y,
         103 + rz * math.sin(2 * math.pi * index / 48))
        for index in range(48)
    ], close=True)


def loft(sections: tuple[tuple[float, float, float], ...]) -> bd.Solid:
    return bd.Solid.make_loft([ellipse(*section) for section in sections])


def cup(mirror: bool) -> bd.Shape:
    # 82-y mirrors the first cup around the leg centre y=41. The crown is
    # outboard; the hollow opens toward the 688 housing and output shaft.
    flip = (lambda section: (82 - section[0], *section[1:])) if mirror else (lambda section: section)
    if FLAT:
        # Review-driven second form: crown 3.5 mm nearer the thigh, narrow
        # oval margin beyond the 11.5 mm bearing housing, feathered perimeter.
        outer_sections = ((4.5, 10.5, 13.0), (5.5, 12.3, 15.4),
                          (7.5, 13.5, 17.0), (8.5, 13.5, 17.0))
        hollow_sections = ((5.8, 9.0, 11.0), (7.0, 11.0, 13.5),
                           (9.0, 12.0, 15.0))
    else:
        outer_sections = ((1.0, 9.5, 11.5), (2.5, 15.0, 17.0),
                          (5.5, 16.5, 18.5), (8.0, 16.5, 18.5))
        hollow_sections = ((2.4, 10.0, 12.0), (4.0, 13.0, 15.0),
                           (8.6, 14.0, 16.0))
    outer = loft(tuple(map(flip, outer_sections)))
    hollow = loft(tuple(map(flip, hollow_sections)))
    return outer - hollow


def colored(scene: trimesh.Scene, shape: bd.Shape, name: str) -> None:
    verts, faces = shape.tessellate(.18, .18)
    mesh = trimesh.Trimesh(vertices=np.array([[p.X, p.Y, p.Z] for p in verts]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = (234, 217, 190, 255)
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = bd.import_step(SOURCE / "relieved-thigh-and-bearing-housings.step")
    moving = bd.import_step(SOURCE / "shin-and-knee-yoke.step")
    rows = []
    cups = []
    for label, mirror in (("inside", False), ("outside", True)):
        raw = cup(mirror)
        fixed_overlap = volume(raw.intersect(fixed))
        # A relief makes this a separate, removable skin instead of a second
        # printed solid occupying the same volume as the structural thigh.
        clean = raw - fixed
        collisions = {pose: round(volume(clean.intersect(pitched(moving, pose))), 3)
                      for pose in POSES}
        bd.export_step(clean, OUT / f"{label}-knee-cup-envelope.step")
        cups.append((label, clean))
        rows.append({
            "side": label,
            "raw_skin_vs_fixed_thigh_mm3_removed": round(fixed_overlap, 2),
            "remaining_skin_volume_mm3": round(clean.volume, 2),
            "connected_solids_after_relief": len(clean.solids()),
            "skin_vs_moving_shin_mm3_by_pose": collisions,
            "attachment": "UNBUILT: relief avoids hard overlap but clip/fastener and insertion path are not designed",
        })
    for pose in POSES:
        scene = trimesh.load(SOURCE / f"knee-{pose:03d}.glb", force="scene")
        for label, shape in cups:
            colored(scene, shape, f"{label}_removable_side_cup")
        scene.export(OUT / f"covered-knee-{pose:03d}.glb")
    report = {
        "stage": "Outboard bearing-cover envelope coupon, not a finished knee skin",
        "appearance_variant": "shallow feathered oval cheek" if FLAT else "round protruding cup",
        "source_mechanism": str(SOURCE.relative_to(ROOT)),
        "source_cad": "The saved direct-drive supported ST3250 knee; cups are separate 1-3 mm-class printed shell envelopes",
        "screened_flex_deg": list(POSES),
        "rows": rows,
        "unresolved": ["Only side bearings covered; front/rear opening remains", "No actual cup mounting, cable exit, or print orientation", "No TPU gusset, shin collar, continuous clearance or cyclic strain model", "Appearance remains subject to independent review"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("knee_side_cup_viewer.html").read_text())
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
