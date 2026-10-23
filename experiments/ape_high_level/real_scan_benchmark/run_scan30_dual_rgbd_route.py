"""Full Scan30 policy with dual RGB-D route evidence and live range safety.

The owner approach and item walk receive two projected head-depth views.
The independent 10 Hz two-height range worker remains the near-body/freshness
gate, so this compares navigation evidence in a deployable hybrid bundle.
Unknown projected forward sectors hold the body. No room-specific changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import run_scan30_unified_policy as unified
import run_real_capture_teach_find as baseline
import run_hidden_item_rearrangement as motion
from rgbd_range_projection import dual_view_fan


ROOT = unified.ROOT
ART = unified.ART
OUT = ROOT / "assets/real_indoor_scenes/scan30_dual_rgbd_route_20261015"
PRECOMMIT = ART / "scan30_dual_rgbd_route_precommit.json"
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)
FRONT = np.abs(ANGLES) <= 15.
ORIGINAL_RUN_OWNER = baseline.run_owner
ORIGINAL_SAVE_FRAME = motion.base.save_frame
ORIGINAL_CHOOSE = motion.approach_v2.choose_with_resolution
_last_down_depth: np.ndarray | None = None


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def freeze() -> dict:
    source = ART / "scan30_unified_policy_precommit.json"
    return {"schema": "ape.scan30-dual-rgbd-route-precommit.v1",
        "frozen_scan30_precommit": relative(source),
        "frozen_scan30_precommit_sha256": sha(source),
        "source_report": relative(ART / "scan30_same_runtime_healthy_report.json"),
        "source_report_sha256": sha(ART / "scan30_same_runtime_healthy_report.json"),
        "policy_sources": [{"path": relative(path), "sha256": sha(path)}
            for path in (Path(__file__),
                ROOT / "experiments/ape_high_level/real_scan_benchmark/rgbd_range_projection.py")],
        "cases": [{"scene_key": c["scene_key"],
                   "capture_group": c["capture_group"]}
                  for c in unified.cohort()],
        "sensor_bundle": "The frozen Scan30 owner task uses upright and simultaneous 45-degree-down ideal RGB-D at one 0.42 m optical center for owner-route and item-route two-height projections. A separate 10 Hz two-height range worker still gates motion health/near-body clearance. RGB-D still localizes the taught item; historical native-shaped pose increments remain. Only observed projected low/mid range is route evidence; unknown forward sectors hold.",
        "score": ["whole-task completion and honest unresolved", "owner reshow burden",
                  "unknown-coverage stops", "false item offers",
                  "fresh range/pose gates", "saved upright/downward RGB-D and route fan"],
        "limits": "Previously exposed 30 captured static rooms, scripted rigid person and staged objects; ideal aligned RGB-D and separate virtual range safety gate; unselected 17 cm proxy, not a calibrated 50 cm body. This tests a dual-camera navigation bundle, not removal of all range sensors."}


def camera(scene, xy: np.ndarray, yaw_deg: float,
           owner_xz: tuple[float, float] | None,
           width: int | None = None, height: int | None = None):
    """Return projected route fan from only the two rendered depth views."""
    global _last_down_depth
    width = width or getattr(scene, "WIDTH", 640)
    height = height or getattr(scene, "HEIGHT", 360)
    eye = (scene.eye_at(float(xy[0]), float(xy[1]))
           if hasattr(scene, "eye_at") else
           np.array([xy[0], scene.floor_y + .42, xy[1]]))
    if eye is None:
        raise ValueError(f"No local floor under camera at {xy}")
    yaw = math.radians(yaw_deg)
    forward = np.array([math.sin(yaw), 0., math.cos(yaw)])
    rgb, depth, ids = scene.observe(eye, forward, owner_xz=owner_xz,
                                    width=width, height=height,
                                    return_prop_ids=True)
    tilt = math.radians(45.)
    down_forward = forward * math.cos(tilt) + \
        np.array([0., -math.sin(tilt), 0.])
    _, down_depth = scene.observe(eye, down_forward, owner_xz=owner_xz,
                                  width=width, height=height)
    _last_down_depth = down_depth
    return rgb, depth, ids, dual_view_fan(depth, down_depth, ANGLES)


def save_with_down(out: Path, writer, frame: int, rgb: np.ndarray,
                   depth: np.ndarray) -> None:
    ORIGINAL_SAVE_FRAME(out, writer, frame, rgb, depth)
    if _last_down_depth is None or _last_down_depth.shape != depth.shape:
        raise RuntimeError("Missing aligned downward depth for route frame")
    Image.fromarray(np.where(np.isfinite(_last_down_depth),
        np.clip(_last_down_depth * 1000., 0., 65535.), 0).astype(np.uint16)
        ).save(out / f"down_depth_{frame:03d}.png")


def choose_known_forward(actor: np.ndarray, yaw: float, target: np.ndarray,
                         ranges: np.ndarray) -> tuple[str, float, str]:
    # Preserve the frozen target/standoff/turn rule. Demand observed clearance
    # on both planes before any projected-depth forward move.
    delta = target - actor
    if float(np.linalg.norm(delta)) <= motion.base.STANDOFF_M + \
            motion.approach_v2.TOLERANCE_M:
        return "hold", 0., "observed_target_standoff"
    bearing = math.degrees(math.atan2(delta[0], delta[1]))
    turn = motion.base.signed_angle(bearing - yaw)
    if abs(turn) > 5.:
        return "turn", float(np.clip(turn, -motion.base.TURN_DEG,
                                     motion.base.TURN_DEG)), "face_observed_target"
    if np.isnan(ranges[:, FRONT]).any():
        return "hold", 0., "forward_clearance_unknown"
    return ORIGINAL_CHOOSE(actor, yaw, target, ranges)


def run_owner_dual(*args, **kwargs):
    if "range_mode" in kwargs:
        raise ValueError("Owner sensor arm already selected")
    return ORIGINAL_RUN_OWNER(*args, range_mode="dual_rgbd_projection",
                              **kwargs)


def main() -> None:
    definition = freeze()
    if PRECOMMIT.is_file():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen dual RGB-D arm changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    unified.OUT = OUT
    unified.BASE_OUT = OUT / "base"
    unified.SHOW_OUT = OUT / "show"
    unified.REPORT = ART / "scan30_dual_rgbd_route_report.json"
    baseline.run_owner = run_owner_dual
    baseline.camera = camera
    motion.camera = camera
    motion.base.save_frame = save_with_down
    motion.approach_v2.choose_with_resolution = choose_known_forward
    unified.main()


if __name__ == "__main__":
    main()
