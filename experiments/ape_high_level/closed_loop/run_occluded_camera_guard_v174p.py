"""Join owner RGB-D low-region change to a live occluded-range owner turn.

This is a diagnostic counterfactual from a far held pose that V167p's
pre-show range-only admission would defer. The owner-facing camera can issue
an early request if its pose-compensated depth shows a closing low surface;
the scripted prop may halt or ignore it. The ape has no physical escape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
import run_continuous_owner_guard_v95 as v95
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import run_standoff_yield_recovery_v153p as v153p
from run_standoff_front_response_v150p import FrontMonitor
from compare_standoff_range_layouts_v165p import front_left_roi
import screen_standoff_camera_depth_v171p as v171p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-occluded-camera-guard"
SEEDS = (170, 172)
STATION_X_M = -.30
RESPONSES_S = (.2, None)


class CameraFrontMonitor(FrontMonitor):
    def __init__(self, room, branch: Path, seed: int) -> None:
        self.camera_history: list[float | None] = []
        super().__init__(room, branch, seed)

    def capture(self) -> None:
        super().capture()
        room = self.room
        index = len(self.frames)-1
        rgb, depth = room.capture("lower_rgb", depth=True)
        if depth is None:
            raise AssertionError("Owner camera depth unavailable")
        camera_pose = room.camera_pose_estimate("lower_rgb")
        fovy = float(room.model.cam_fovy[
            room.model.camera("lower_rgb").id])
        support, p10, valid = v171p.project_low_region(
            depth, camera_pose, room.estimated.copy(), fovy)
        self.camera_history.append(p10)
        closing = None
        cue = False
        if len(self.camera_history) >= 3:
            older = self.camera_history[-3]
            if older is not None and p10 is not None:
                closing = round(older-p10, 5)
                cue = closing >= v171p.MIN_CLOSING_M_PER_0P2S
        first_cue = cue and getattr(room, "front_alarm_s", None) is None
        if first_cue:
            room.front_alarm_s = round(float(room.time_s), 4)
            self.alarm = {"capture_time_s": room.front_alarm_s,
                          "capture_index": index,
                          "side": "owner_rgbd_low_region",
                          "closing_0p2s_m": closing}
            room.motion_interrupted = True
        path = self.branch / f"owner-depth-watch-{index:02d}.npz"
        np.savez_compressed(
            path,
            sampled_depth_m=depth[
                v171p.DEPTH_OFFSET::v171p.DEPTH_STRIDE,
                v171p.DEPTH_OFFSET::v171p.DEPTH_STRIDE].astype(np.float32),
            full_shape_hw=np.asarray(depth.shape),
            estimated_pose_xyyaw=room.estimated.copy(),
            camera_origin_xyz_m=camera_pose[0],
            camera_rotation=camera_pose[1],
            fovy_deg=fovy,
            capture_time_s=room.time_s)
        self.frames[-1].update({
            "owner_depth_packet_path": str(path.relative_to(ROOT)),
            "owner_depth_packet_sha256": sha(path),
            "owner_depth_valid_samples": valid,
            "owner_depth_low_support": support,
            "owner_depth_low_p10_x_m": p10,
            "owner_depth_closing_0p2s_m": closing,
            "owner_depth_cue_now": cue,
            "owner_depth_first_cue": first_cue,
        })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    old_scan = v95.front_scan
    old_room = v153p.ResponsiveRoom
    old_monitor = v153p.FrontMonitor
    old_out = v153p.OUT
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v153p.ResponsiveRoom = v160p.VariantRoom
    v153p.FrontMonitor = CameraFrontMonitor
    rows = []
    try:
        for variant_name, variant in v160p.VARIANTS.items():
            v160p.VariantRoom.variant = variant
            v153p.OUT = OUT / variant_name
            for seed in SEEDS:
                prior = next(row for row in v98["rows"]
                             if row["condition"] == "moving" and
                             row["case"][0] == seed)
                source = (v100.V98 / "moving" / f"seed{seed}-present" /
                          f"seed{seed}-present/side-probe-00.npz")
                with np.load(source, allow_pickle=False) as packet:
                    qpos = packet["actor_qpos"].copy()
                    estimate = packet["estimated_pose_xyyaw"].copy()
                probe = v160p.VariantRoom(
                    case=tuple(prior["case"]),
                    pose_source="command", branch=OUT)
                try:
                    shoulder = probe.model.joint(
                        "left_shoulder_pitch").qposadr[0]
                finally:
                    probe.close()
                qpos[0] += STATION_X_M
                estimate[0] += STATION_X_M
                qpos[shoulder] = .6
                for delay in RESPONSES_S:
                    removed = []

                    def occluded_scan(room, robot, case):
                        clouds, counts = old_scan(room, robot, case)
                        world = clouds["four"]
                        first_count = len(clouds["two"])
                        estimated = v95.estimated_shin_points(
                            world, room.actual.copy(),
                            room.estimated.copy())
                        blocked = (front_left_roi(
                            estimated, room.estimated.copy()) &
                            (np.arange(len(world)) < first_count))
                        missing = int(np.count_nonzero(blocked))
                        clouds["two"] = clouds["two"][~blocked[:first_count]]
                        clouds["four"] = world[~blocked]
                        counts["two_environment_points"] = len(clouds["two"])
                        counts["injected_forward_pair_roi_removed"] = missing
                        removed.append({"time_s": round(float(room.time_s), 4),
                                        "removed_forward_roi_returns": missing})
                        return clouds, counts

                    v95.front_scan = occluded_scan
                    row = v153p.run_one(
                        seed, STATION_X_M, delay,
                        tuple(prior["case"]), qpos, estimate,
                        vision, exemplar, owner_hist)
                    row.update({
                        "variant": variant_name,
                        "staging_packet": str(source.relative_to(ROOT)),
                        "staging_packet_sha256": sha(source),
                        "removed_by_live_packet": removed,
                    })
                    rows.append(row)
                    print(variant_name, seed, delay,
                          row["front_alarm"], row["task_result"],
                          row["evaluator_minimum_scene_gap_m"], flush=True)
    finally:
        v95.front_scan = old_scan
        v153p.ResponsiveRoom = old_room
        v153p.FrontMonitor = old_monitor
        v153p.OUT = old_out
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-occluded-camera-guard.v1",
        "source_sha256": sha(SOURCE),
        "v171_source_sha256": sha(Path(v171p.__file__)),
        "v153_source_sha256": sha(Path(v153p.__file__)),
        "v98_report_sha256": sha(v98_file),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "scope": "Diagnostic complete owner turn from a far held pose whose "
                 "forward-pair-occluded range preflight was not admitted. "
                 "Live range packets remain occluded, while a 10Hz "
                 "owner-facing RGB-D world-low-region cue can request a "
                 "scripted prop stop 0.2s later or be ignored. The head "
                 "camera is pose-compensated with ideal joint state and "
                 "saved base estimate. No real owner request, selected-body "
                 "action or safe task recovery is claimed. Evaluator gap "
                 "scores only.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
