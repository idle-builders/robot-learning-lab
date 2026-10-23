"""V300: insert time-aligned RGB-D pose fusion into a live owner search.

This deliberately preserves V284's point-only shin route gate so a pose
correction's effect on action and owner outcome can be seen. It does not make
that route gate safe, nor replace the kinematic base with the native policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_localization_gate_v298 import motion_checked
from audit_direct_localization_v297 import depth_variant
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from preflight_lateral_clutter_v282 import LowClutterScreenRoom
from rgbd_odometry import RGBDOdometry
from run import ROOT, Vision, sha
from run_lateral_clutter_shin_search_v284 import run_case


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v300"
SEED = 138
HEIGHTS_M = (.04, .08)
MODES = ("present", "absent", "identical_lookalike")
POSE_SOURCES = ("command", "motion_checked_rgbd")


class LocalizedLowClutterRoom(LowClutterScreenRoom):
    def __init__(self, *, mode: str, crate_height_m: float,
                 pose_source: str, branch: Path) -> None:
        self.crate_height_m = crate_height_m
        self.pose_source = pose_source
        self.branch = branch
        self.odom = RGBDOdometry()
        self.localization_events: list[dict] = []
        super().__init__(kind="screen", mode=mode, side=-1)

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        geom = self.model.geom("loop_crate_geom")
        geom.size[2] = self.crate_height_m / 2
        body = self.model.body("loop_crate")
        self.model.body_pos[body.id, 2] = self.crate_height_m / 2
        self._forward()

    def capture_localization(self, index: int):
        rgb, ideal = self.capture("upper_rgb", depth=True)
        if ideal is None:
            raise AssertionError("Rendered depth missing")
        # Same deterministic defect model in both paired branches.
        depth = depth_variant(ideal, "noisy_long", SEED, index)
        rgb_path = self.branch / f"localization-{index:02d}-rgb.png"
        depth_path = self.branch / f"localization-{index:02d}-depth.npz"
        rgb.save(rgb_path)
        np.savez_compressed(depth_path, depth_m=depth)
        return (rgb, depth, self.camera_pose_estimate("upper_rgb")[1],
                {"rgb_path": str(rgb_path.relative_to(ROOT)),
                 "depth_path": str(depth_path.relative_to(ROOT)),
                 "rgb_sha256": sha(rgb_path),
                 "depth_sha256": sha(depth_path)})

    def move(self, vx: float, vy: float, *, slip: float = .88,
             safety_probe=None) -> None:
        if abs(self.head) > .01:
            self.look(0., "lower_rgb")
        if self.odom.previous is None:
            rgb, depth, rotation, files = self.capture_localization(0)
            status = self.odom.initialize(rgb, depth, self.estimated.copy(), rotation)
            self.localization_events.append({
                "step": 0, "capture_time_s": round(self.time_s, 3),
                "actor_estimated_xyyaw": self.estimated.round(5).tolist(),
                "evaluator_actual_xyyaw": self.actual.round(5).tolist(),
                "status": status, **files})
        index = len(self.localization_events)
        previous = self.estimated.copy()
        super().move(vx, vy, slip=slip, safety_probe=safety_probe)
        predicted = self.estimated.copy()
        rgb, depth, rotation, files = self.capture_localization(index)
        corrected, status = self.odom.correct(rgb, depth, predicted, rotation)
        measured = (corrected[:2] - previous[:2]
                    if status["status"] == "registered" else None)
        applied, gate = motion_checked(np.array((vx, vy)), measured)
        if self.pose_source == "motion_checked_rgbd":
            self.estimated[:2] = previous[:2] + applied
        # The old image and old pose must name the same capture instant, even
        # when a visual update is rejected or the ablation ignores it.
        keys, descriptors, old_depth, _, old_rotation = self.odom.previous
        self.odom.previous = (keys, descriptors, old_depth,
                              self.estimated.copy(), old_rotation)
        self.localization_events.append({
            "step": index, "capture_time_s": round(self.time_s, 3),
            "motion_command_xy_m": [vx, vy], "gate": gate,
            "measured_rgbd_delta_xy_m": None if measured is None else
                measured.round(5).tolist(),
            "actor_estimated_xyyaw": self.estimated.round(5).tolist(),
            "evaluator_actual_xyyaw": self.actual.round(5).tolist(),
            "evaluator_pose_error_m": round(float(np.linalg.norm(
                self.estimated[:2] - self.actual[:2])), 5),
            "status": status, **files})


def run_pair(vision: Vision, exemplar: np.ndarray,
             owner_hist: np.ndarray) -> list[dict]:
    rows = []
    for height in HEIGHTS_M:
        for mode in MODES:
            for source in POSE_SOURCES:
                height_cm = round(height * 100)
                branch = OUT / f"height{height_cm}cm-{source}" / f"seed{SEED}-{mode}"
                branch.mkdir(parents=True, exist_ok=True)
                holder = []

                def factory(seed: int, requested_mode: str):
                    if seed != SEED or requested_mode != mode:
                        raise ValueError("Unexpected room request")
                    room = LocalizedLowClutterRoom(
                        mode=mode, crate_height_m=height,
                        pose_source=source, branch=branch)
                    room.layout(seed, absent=mode != "present", furnished=True)
                    holder.append(room)
                    return room

                root = OUT / f"height{height_cm}cm-{source}"
                result = run_case(SEED, mode, vision, exemplar, owner_hist,
                                  room_factory=factory, output_root=root,
                                  sensor_case_prefix=f"v300-{height_cm}")
                localization = holder[0].localization_events
                trace = branch / "localization-trace.json"
                trace.write_text(json.dumps({
                    "height_m": height, "mode": mode, "pose_source": source,
                    "events": localization}, indent=2) + "\n")
                row = {"height_m": height, "mode": mode, "pose_source": source,
                       "status": result["status"],
                       "owner_questions": result["owner_questions"],
                       "motion_steps": len(localization) - 1 if localization else 0,
                       "evaluator_pose_error_m": result["evaluator_pose_error_m"],
                       "evaluator_min_model_clearance_m":
                           result["evaluator_min_model_clearance_m"],
                       "evaluator_wrong_candidate_views":
                           result["evaluator_wrong_candidate_views"],
                       "route_events": [event for event in result["events"]
                                        if event["action"] in (
                                            "lateral_sector", "return_sector")],
                       "localization_statuses": [event["status"]["status"]
                                                for event in localization],
                       "localization_gates": [event.get("gate")
                                              for event in localization[1:]],
                       "localization_trace_path": str(trace.relative_to(ROOT)),
                       "localization_trace_sha256": sha(trace),
                       "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                       "video_sha256": result["video_sha256"]}
                rows.append(row)
                (OUT / "localized-low-clutter-audit.json").write_text(
                    json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
                print(height_cm, mode, source, row["status"],
                      row["motion_steps"], row["evaluator_pose_error_m"],
                      flush=True)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = run_pair(vision, exemplar, owner_hist)
    report = {"scope": "V300 exact selected Direct Frame live visual teach/find search in the seed138 low-clutter screen room, with full floor-box heights 4/8cm, present/absent/identical-lookalike modes. Paired command-only and frozen V298 motion-checked half-weight upper RGB-D pose fusion each capture the same ideal RGB/depth with deterministic 1cm depth noise and 15% holes after 8cm kinematic lateral moves. V284 four-shin point-only actor route gate and owner exemplar/inference are unchanged; RGB-D pose updates affect future sensor placement and target world coordinates. Evaluator true pose/clearance/instance score only. Head is returned to yaw0 for odometry captures in both branches. No native dynamics, calibrated physical depth, moving obstacle, sensor age, free-volume proof or actual owner contact.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "localized-low-clutter-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
