"""Reconstruct the dual-depth route packets saved by a full Scan30 run."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_range_projection import dual_view_fan


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
OUT = ROOT / "assets/real_indoor_scenes/scan30_dual_rgbd_route_20261015"
PRECOMMIT = ART / "scan30_dual_rgbd_route_precommit.json"
REPORT = ART / "scan30_dual_rgbd_route_report.json"
TASK_AUDIT = ART / "scan30_dual_rgbd_route_audit.json"
AUDIT = ART / "scan30_dual_rgbd_provenance_audit.json"
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)
FRONT = np.abs(ANGLES) <= 15.


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def depth(path: Path) -> np.ndarray:
    millimetres = np.asarray(Image.open(path), dtype=np.float32)
    return np.where(millimetres > 0., millimetres / 1000., np.nan)


def packet(folder: Path, frame: int) -> np.ndarray:
    return dual_view_fan(depth(folder / f"depth_{frame:03d}.png"),
                         depth(folder / f"down_depth_{frame:03d}.png"), ANGLES)


def decoded(saved: list[list]) -> np.ndarray:
    return np.asarray([[np.nan if x is None else np.inf if x == "clear"
                        else float(x) for x in plane] for plane in saved])


def equal_packet(expected: np.ndarray, saved: np.ndarray) -> bool:
    if expected.shape != saved.shape or not np.array_equal(
            np.isnan(expected), np.isnan(saved)) or not np.array_equal(
            np.isinf(expected), np.isinf(saved)):
        return False
    finite = np.isfinite(expected)
    return not finite.any() or bool(np.max(np.abs(expected[finite] -
                                                    saved[finite])) <= .03)


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    report = json.loads(REPORT.read_text())
    task_audit = json.loads(TASK_AUDIT.read_text())
    issues = list(task_audit["issues"])
    for source in frozen["policy_sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or sha(path) != source["sha256"]:
            issues.append(f"Frozen dual-depth source changed: {source['path']}")
    if sha(ROOT / frozen["frozen_scan30_precommit"]) != frozen[
            "frozen_scan30_precommit_sha256"] or sha(ROOT / frozen[
            "source_report"]) != frozen["source_report_sha256"]:
        issues.append("Frozen Scan30 source changed")
    if len(report["rows"]) != 30 or len({r["capture_group"] for r in report[
            "rows"]}) != 30:
        issues.append("Not thirty distinct whole tasks")
    owners = walks = owner_frames = walk_frames = unknown_holds = 0
    png_quantization_ambiguities = []
    side_beam_quantization_mismatches = 0
    outcomes = Counter(row["result"] for row in report["rows"])
    for row in report["rows"]:
        base = ROOT / row["base_summary"]
        owner_path = base.parent / "initial/owner_approach/trace.json"
        if not owner_path.is_file():
            issues.append(f"Missing owner approach: {row['scene_key']}")
            continue
        owner = json.loads(owner_path.read_text())
        owners += 1
        if owner.get("range_mode") != "dual_rgbd_projection":
            issues.append(f"Owner used different range source: {row['scene_key']}")
        for frame, observation in enumerate(owner["trace"]):
            try:
                replay = packet(owner_path.parent, frame)
                finite_saved = np.asarray([[np.nan if value is None else
                    float(value) for value in observation[name]] for name in
                    ("range_low_m", "range_mid_m")])
                finite = np.isfinite(finite_saved)
                both = finite & np.isfinite(replay)
                different = (finite != np.isfinite(replay)) | (both &
                    (np.abs(finite_saved - replay) > .03))
                count_differs = int((~np.isnan(replay)).sum()) != observation[
                    "known_range_beam_count"]
                if different.any() or count_differs:
                    detail = {"stage": "owner", "scene_key": row["scene_key"],
                        "frame": frame, "forward_beams": int(different[:, FRONT].sum()),
                        "side_beams": int(different[:, ~FRONT].sum()),
                        "known_count_changed": bool(count_differs),
                        "commanded_step_m": observation.get("commanded_step_m", 0.)}
                    # The actor used original floating-point depth, while
                    # retained PNGs round to millimetres. At a grazing pixel,
                    # that can change the first-hit sample by many centimetres.
                    png_quantization_ambiguities.append(detail)
                    if detail["commanded_step_m"] > 0 and detail[
                            "forward_beams"]:
                        issues.append(f"Moving-owner forward depth mismatch: {row['scene_key']}:{frame}")
                owner_frames += 1
            except Exception as error:
                issues.append(f"Unreadable owner depth: {row['scene_key']}:{frame}: {error}")
    for path in OUT.rglob("trace.json"):
        try:
            trace = json.loads(path.read_text())
        except Exception as error:
            issues.append(f"Unreadable trace {path.relative_to(ROOT)}: {error}")
            continue
        actions = trace.get("actions") or []
        if not actions or not any("range_planes_m" in a for a in actions):
            continue
        walks += 1
        for action in actions:
            frame = action["frame"]
            try:
                replay = packet(path.parent, frame)
                saved = decoded(action["range_planes_m"])
                # The PNGs keep millimetres, while the actor projects the
                # original float depth. A 1 mm rounding can flip a grazing
                # side beam between a distant hit and observed clear. Require
                # exact masks on the decision-critical forward sectors and
                # count the remaining wide-fan ambiguity explicitly.
                if not equal_packet(replay[:, FRONT], saved[:, FRONT]):
                    issues.append(f"Forward depth mismatch: {path.relative_to(ROOT)}:{frame}")
                side = ~FRONT
                side_beam_quantization_mismatches += int(np.count_nonzero(
                    (np.isnan(replay[:, side]) != np.isnan(saved[:, side])) |
                    (np.isinf(replay[:, side]) != np.isinf(saved[:, side]))))
                finite_side = np.isfinite(replay[:, side]) & \
                    np.isfinite(saved[:, side])
                if finite_side.any() and np.max(np.abs(replay[:, side][
                        finite_side] - saved[:, side][finite_side])) > .03:
                    detail = {"stage": "item_walk", "path": str(path.relative_to(ROOT)),
                        "frame": frame, "decision": action["decision"],
                        "side_distance_difference_m": float(np.max(np.abs(
                            replay[:, side][finite_side] - saved[:, side][finite_side])))}
                    png_quantization_ambiguities.append(detail)
                    if action["decision"] == "forward":
                        issues.append(f"Moving-walk side depth mismatch: {path.relative_to(ROOT)}:{frame}")
                if action["decision"] == "forward" and action.get(
                        "reason") != "route_to_observed_free_waypoint" and \
                        np.isnan(replay[:, FRONT]).any():
                    issues.append(f"Unobserved forward move: {path.relative_to(ROOT)}:{frame}")
                if action["decision"] == "hold" and action.get(
                        "reason") == "forward_clearance_unknown":
                    unknown_holds += 1
                walk_frames += 1
            except Exception as error:
                issues.append(f"Unreadable walk depth: {path.relative_to(ROOT)}:{frame}: {error}")
    result = {"schema": "ape.scan30-dual-rgbd-provenance-audit.v1",
        "attempts": len(report["rows"]), "outcomes": dict(outcomes),
        "owner_approaches": owners, "owner_depth_packets_replayed": owner_frames,
        "item_walks": walks, "item_route_packets_replayed": walk_frames,
        "unknown_clearance_holds": unknown_holds,
        "side_beam_quantization_mismatches": side_beam_quantization_mismatches,
        "png_quantization_ambiguities": png_quantization_ambiguities,
        "task_audit_issues": len(task_audit["issues"]), "issues": issues,
        "limits": frozen["limits"]}
    AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("attempts", "outcomes",
        "owner_depth_packets_replayed", "item_route_packets_replayed",
        "unknown_clearance_holds", "issues")}, indent=2))


if __name__ == "__main__":
    main()
