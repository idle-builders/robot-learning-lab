"""Exercise the existing furnished owner actor with native-shaped odometry error.

The owner actor and image models are imported unchanged. Its bounded body
actions remain kinematic surrogates. We synthesize an odometry increment from
a saved native 50 cm walking trace for each executed translation; this is an
error-shape transfer, not native policy execution or a physical sensor packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
from experiments.ape_high_level.closed_loop import run as loop


SOURCE_QPOS = ROOT / "artifacts/ape-cove50-transfer-20261008/varied150-forward-seed144.npz"
SOURCE_ODOM = (ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1"
               / "varied150-forward-seed144-trace.json")
DEFAULT_OUT = ROOT / "artifacts/ape-native-owner-bridge/v1"
SCENARIO = "furnished_moving"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeShapedRoom(loop.Room):
    def __init__(self) -> None:
        with np.load(SOURCE_QPOS, allow_pickle=False) as saved:
            self.native_true = saved["qpos"][:, :2].astype(float)
            self.native_dt = float(saved["dt"])
        rows = json.loads(SOURCE_ODOM.read_text())
        self.native_estimate = np.asarray([
            row["estimated_xy_m"]["combined_illustrative_impairments"]
            for row in rows])
        self.native_arc = np.r_[0, np.cumsum(
            np.linalg.norm(np.diff(self.native_true, axis=0), axis=1))]
        self.native_cursor = 0
        self.native_packets: list[dict] = []
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self.native_cursor = 0
        self.native_packets = []

    def move(self, vx: float, vy: float, *, slip: float = .88,
             safety_probe=None) -> None:
        old_actual = self.actual.copy()
        old_estimate = self.estimated.copy()
        old_cursor = self.native_cursor
        super().move(vx, vy, slip=slip, safety_probe=safety_probe)
        actual_delta = self.actual[:2] - old_actual[:2]
        traveled = float(np.linalg.norm(actual_delta))
        if traveled <= 1e-9:
            return
        destination_arc = self.native_arc[old_cursor] + traveled
        end = int(np.searchsorted(self.native_arc, destination_arc, side="left"))
        if end >= len(self.native_true):
            raise RuntimeError("Native source trace exhausted during owner route")
        source_true_delta = self.native_true[end] - self.native_true[old_cursor]
        source_sensor_delta = self.native_estimate[end] - self.native_estimate[old_cursor]
        source_error = source_sensor_delta - source_true_delta
        source_direction = math.atan2(source_true_delta[1], source_true_delta[0])
        actor_direction = math.atan2(actual_delta[1], actual_delta[0])
        angle = actor_direction - source_direction
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        scale = traveled / max(self.native_arc[end] - self.native_arc[old_cursor], 1e-9)
        injected_error = rotation @ (source_error * scale)
        measured_delta = actual_delta + injected_error
        self.estimated[:2] = old_estimate[:2] + measured_delta
        self.native_cursor = end
        self.native_packets.append({
            "time_s": round(self.time_s, 4),
            "source_start_frame": old_cursor, "source_end_frame": end,
            "source_elapsed_s": round((end - old_cursor) * self.native_dt, 4),
            "source_true_delta_evaluator_only_m": source_true_delta.round(5).tolist(),
            "source_foot_estimate_delta_m": source_sensor_delta.round(5).tolist(),
            "actor_true_delta_evaluator_only_m": actual_delta.round(5).tolist(),
            "actor_sensor_delta_m": measured_delta.round(5).tolist(),
            "actor_estimate_xy_m": self.estimated[:2].round(5).tolist(),
            "actor_true_xy_evaluator_only_m": self.actual[:2].round(5).tolist(),
        })


class OptimisticNoCorrectionLocalizer:
    """Deliberately unsafe control: keep range planning, bypass pose correction."""

    def initialize(self, scan, pose):
        return {"status": "mapped", "ablation": "no_pose_correction"}

    def correct(self, scan, predicted):
        return predicted, {"status": "registered", "ablation": "no_pose_correction"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[85])
    parser.add_argument("--configs", nargs="+", choices=loop.CONFIGS,
                        default=["chest_rgbd_range_guard_scan_memory"])
    parser.add_argument("--modes", nargs="+",
                        choices=("command", "native_shaped",
                                 "native_shaped_no_scan_correction"),
                        default=["command", "native_shaped"])
    parser.add_argument("--present-only", action="store_true",
                        help="Skip absent control when extending a paired present cohort")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if abs(float(np.load(SOURCE_QPOS)["dt"]) - .02) > 1e-9:
        raise ValueError("Expected 50 Hz native source")
    vision = loop.Vision("generic")
    teaching = loop.Room()
    try:
        teaching.teach()
        teach_rgb, _ = teaching.capture("upper_rgb", depth=False)
        truth_box = teaching.truth_box("TomatoSauce", "upper_rgb")
        if truth_box is None:
            raise ValueError("Scripted teaching object not visible")
        exemplar = vision.feature([teach_rgb.crop(truth_box)])[0]
        teach_rgb.save(output / "teach.png")
    finally:
        teaching.close()
    results = []
    for mode in args.modes:
        loop.OUT = output / mode
        loop.OUT.mkdir(parents=True, exist_ok=True)
        room = loop.Room() if mode == "command" else NativeShapedRoom()
        original_localizer = loop.ScanLocalizer
        if mode == "native_shaped_no_scan_correction":
            loop.ScanLocalizer = OptimisticNoCorrectionLocalizer
        try:
            for seed in args.seeds:
                for absent in ((False,) if args.present_only else (False, True)):
                    for config in args.configs:
                        scored = loop.run_episode(
                            room, vision, exemplar, seed=seed, absent=absent,
                            config=config, scenario=SCENARIO,
                            observer_video=True)
                        episode = loop.OUT / (f"{SCENARIO}-seed{seed}-{int(absent)}-"
                                              f"{config}-none")
                        if mode != "command":
                            packets = episode / "native-shaped-odometry-packets.json"
                            packets.write_text(json.dumps(room.native_packets, indent=2) + "\n")
                            scored["native_packet_trace"] = str(packets.relative_to(ROOT))
                            scored["native_packet_trace_sha256"] = sha(packets)
                            scored["native_source_end_frame"] = room.native_cursor
                        scored["mode"] = mode
                        scored["episode_video"] = str(
                            (episode / "episode.mp4").relative_to(ROOT))
                        scored["episode_video_sha256"] = sha(episode / "episode.mp4")
                        scored["episode_trace"] = str(
                            (episode / "trace.json").relative_to(ROOT))
                        scored["episode_trace_sha256"] = sha(episode / "trace.json")
                        results.append(scored)
                        print(mode, seed, int(absent), config, scored["outcome"],
                              scored.get("pose_error_m"),
                              scored.get("min_clearance_m"),
                              scored.get("native_source_end_frame"))
        finally:
            loop.ScanLocalizer = original_localizer
            room.close()
    report = {
        "schema": "robot-learning-lab.native-shaped-owner-bridge.v1",
        "task": "T-5ybbx8", "scenario": SCENARIO,
        "owner_actor_source_sha256": sha(Path(loop.__file__)),
        "adapter_source_sha256": sha(Path(__file__)),
        "native_qpos_source": str(SOURCE_QPOS.relative_to(ROOT)),
        "native_qpos_source_sha256": sha(SOURCE_QPOS),
        "native_odometry_source": str(SOURCE_ODOM.relative_to(ROOT)),
        "native_odometry_source_sha256": sha(SOURCE_ODOM),
        "mapping": "For each executed kinematic actor translation, consume the next source native-walk arc-length interval. Rotate and scale that interval's (contact-gated foot estimate minus native true displacement) into the actor's actual step direction, then add it to simulator step displacement to synthesize the actor-visible odometry increment. True actor displacement is used only by this simulator packet generator and evaluator. No native joint/IMU/contact packet reaches the actor. The no-scan-correction control deliberately reports registered without changing pose while range obstacle planning remains active; it is not a safe controller.",
        "scope": "Furnished moving owner task with unchanged image actor and bounded kinematic base action. Native-shaped error transfer, not native policy execution; no selected physical body or sensor qualification.",
        "rows": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
