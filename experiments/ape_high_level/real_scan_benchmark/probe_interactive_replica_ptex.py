"""Check actor-requested PTex views against the same saved official route poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from interactive_replica_ptex import ReplicaPTexStream


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    trace = json.loads((args.reference / "trace.json").read_text())
    frames = trace["frames"]
    chosen = sorted(set((0, len(frames)//2, len(frames)-1)))
    args.output.mkdir(parents=True, exist_ok=True)
    glass = args.scene / "glass.sur"
    records = []
    with ReplicaPTexStream(args.renderer, args.scene / "mesh.ply",
                           args.scene / "textures", args.output,
                           glass if glass.is_file() else None) as renderer:
        for index in chosen:
            frame = frames[index]
            actual, path = renderer.render(frame["position_m"],
                frame["yaw_rad"], trace["camera_height_above_nav_m"])
            expected = np.asarray(Image.open(args.reference / frame["image"])
                                  .convert("RGB"))
            if actual.shape != expected.shape:
                raise ValueError("Interactive RGB shape differs from saved route")
            difference = actual.astype(float) - expected.astype(float)
            mse = float(np.mean(difference**2))
            psnr = None if mse == 0 else 10. * math.log10(255.**2/mse)
            records.append({"source_frame": index, "rendered_frame": path.name,
                "reference_frame": frame["image"], "mse": mse,
                "pixel_identical": mse == 0,
                "psnr_db": psnr})
        protocol = renderer.log
    report = {"schema": "ape.interactive-replica-ptex-probe.v1",
        "scene": args.scene.name,
        "camera_height_above_nav_m": trace["camera_height_above_nav_m"],
        "actor_requested_nonadjacent_poses": len(records),
        "reference_trace": str(args.reference / "trace.json"),
        "frames": records,
        "protocol_tail": protocol[-8:],
        "limits": "Same saved poses, static scene and source HDR textures; this checks an interactive RGB renderer and coordinate convention, not an owner task, metric depth fusion or body-qualified navigation."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
