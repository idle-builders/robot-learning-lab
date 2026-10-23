"""Place a scanned owner in one Replica view and check joint RGB-D/range change."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from real_capture_dynamic_scene import ScannedSittingPerson
from replica_capture_scene import ReplicaCaptureScene


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--source-scene", type=Path, required=True)
    p.add_argument("--converted-mesh", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--scanned-person", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = json.loads((args.reference / "trace.json").read_text())
    frame = trace["frames"][0]
    x, y, z = frame["position_m"]
    yaw = float(frame["yaw_rad"])
    eye = np.array([x, y + .42, z])
    forward = np.array([-math.sin(yaw), 0., -math.cos(yaw)])
    heading = math.degrees(math.atan2(forward[0], forward[2]))
    person = ScannedSittingPerson(args.scanned_person)
    with ReplicaCaptureScene(converted_mesh=args.converted_mesh,
            source_scene=args.source_scene, renderer=args.renderer,
            render_output=args.output / "rgb_requests", floor_y=y,
            scanned_person=person, person_facing_deg=heading+180.) as scene:
        candidates = []
        for distance in (1.4, 1.6, 1.8, 2.0, 2.2):
            point = np.array([x + distance*forward[0],
                              z + distance*forward[2]])
            room_scan = scene.range_fan(np.array([point[0], y+.42, point[1]]),
                heading, owner_xz=None, angles_deg=np.arange(-90.,91.,5.))
            clear = bool(np.isfinite(room_scan).all() and
                         float(room_scan.min()) >= .35)
            candidates.append({"distance_m": distance, "xz": point.tolist(),
                "floor_supported": scene.floor_supported(*point),
                "static_clear_m": float(room_scan.min()),
                "candidate": clear and scene.floor_supported(*point)})
        chosen = next((c for c in candidates if c["candidate"]), None)
        if chosen is None:
            raise ValueError("No geometry-supported owner insertion in this view")
        owner = tuple(chosen["xz"])
        background_rgb, background_depth = scene.observe(
            eye, forward, owner_xz=None)
        owner_rgb, owner_depth = scene.observe(eye, forward, owner_xz=owner)
        empty_range = scene.range_fan(eye, heading, owner_xz=None,
            angles_deg=np.arange(-90.,91.,5.))
        owner_range = scene.range_fan(eye, heading, owner_xz=owner,
            angles_deg=np.arange(-90.,91.,5.))
    Image.fromarray(background_rgb).save(args.output / "background.png")
    Image.fromarray(owner_rgb).save(args.output / "scanned_owner.png")
    for name, depth in (("background_depth", background_depth),
                        ("owner_depth", owner_depth)):
        encoded = np.where(np.isfinite(depth),
            np.clip(np.rint(depth*1000),0,65535),0).astype(np.uint16)
        Image.fromarray(encoded).save(args.output / f"{name}.png")
    closer = np.isfinite(owner_depth) & (
        ~np.isfinite(background_depth) | (owner_depth < background_depth-.01))
    result = {"schema": "ape.replica-scanned-owner-probe.v1",
        "source_scene": args.source_scene.name,
        "chosen_owner_xz_m": chosen["xz"], "candidate_geometry": candidates,
        "changed_rgb_pixel_fraction": float(np.any(owner_rgb!=background_rgb,axis=-1).mean()),
        "closer_owner_depth_pixel_fraction": float(closer.mean()),
        "changed_range_ray_count": int(np.sum(owner_range < empty_range-.01)),
        "limits": "Evaluator chooses a floor-supported placement. The inserted person is one rigid photogrammetry scan with no articulated movement, speech or natural interaction; this is a sensor-composition probe, not an owner task."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
