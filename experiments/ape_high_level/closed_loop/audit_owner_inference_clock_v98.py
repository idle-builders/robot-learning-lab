"""Audit V98's retained alarm, image-validity, video and held-path evidence."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_continuous_owner_guard_v95 import CLOSING_THRESHOLD_M_PER_0P2S
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_shin_intrusion_posture_v324 import actor_alarm
from run_owner_inference_clock_v98 import OUT, SOURCE
import run_blocked_owner_help_v317 as owner_module


AUDIT_SOURCE = Path(__file__)


def main() -> None:
    report = json.loads((OUT / "live-moving-owner-offer-audit.json").read_text())
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    check(report["source_sha256"] == sha(SOURCE), "source hash")
    check(report["search_source_sha256"] == sha(Path(
        owner_module.search.__file__)), "search source hash")
    check(len(report["results"]) == 4, "four full owner episodes")
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    results = []
    try:
        for row in report["results"]:
            seed, condition = row["seed"], row["condition"]
            check(seed in (170, 172) and condition in ("moving", "parked"),
                  "condition")
            branch = (OUT / condition / f"seed{seed}-present" /
                      f"seed{seed}-present")
            trace = json.loads((branch / "episode-trace.json").read_text())
            monitor = json.loads((branch / "continuous-owner-guard.json").read_text())
            frames = monitor["frames"]
            check(len(frames) == row["monitor_packet_count"],
                  "monitor frame count")
            packets = []
            for index, frame in enumerate(frames):
                path = ROOT / frame["packet_path"]
                check(sha(path) == frame["packet_sha256"], "range packet hash")
                with np.load(path) as packet:
                    check(abs(float(packet["capture_time_s"]) -
                              frame["capture_time_s"]) < 1e-6,
                          "packet capture time")
                    packets.append({key: packet[key].copy()
                                    for key in packet.files})
                if index:
                    check(abs(frame["capture_time_s"] -
                              frames[index-1]["capture_time_s"] - .1) < 1e-5,
                          "10Hz monitoring")
            alarms = []
            for index, frame in enumerate(frames):
                if index < 2:
                    check(frame["left_closing_0p2s_m"] is None and not
                          frame["actor_alarm"], "first two packets")
                    continue
                replay = actor_alarm(
                    packets[index-2]["front_four_points_xyz_m"],
                    packets[index]["front_four_points_xyz_m"],
                    packets[index]["estimated_pose_xyyaw"])
                closing = replay["side_returns"]["left"]["closing_0p2s_m"]
                check(closing == frame["left_closing_0p2s_m"],
                      "recomputed closure")
                first_alarm = (closing is not None and
                               closing >= CLOSING_THRESHOLD_M_PER_0P2S and
                               not alarms)
                check(frame["actor_alarm"] == first_alarm,
                      "alarm from sensor packets")
                if first_alarm:
                    alarms.append(frame["capture_time_s"])
            check((alarms[0] if alarms else None) ==
                  (monitor["alarm"]["capture_time_s"] if monitor["alarm"]
                   else None), "first alarm time")
            events = trace["owner_intervention_events"]
            check(any(e["event"] == "scripted_owner_presentation" for e in events),
                  "owner presented item")
            views = [event for event in trace["controller_events"]
                     if event.get("action") == "view" and event.get(
                         "label", "").startswith(("owner-presented", "post-owner"))]
            check(views == row["inference_views"], "reported views")
            frame_hashes = {sha(path): path for path in branch.glob("frame-*.npz")}
            for view in views:
                path = frame_hashes.get(view["actor_input_sha256"])
                check(path is not None, "source image packet")
                with np.load(path) as frame:
                    check(abs(float(frame["capture_time_s"]) -
                              view["capture_time_s"]) < 1e-5,
                          "image captured before inference")
                check(abs(view["time_s"] - view[
                    "inference_completion_time_s"]) < 1e-5, "decision clock")
            video_path = ROOT / row["episode_video_path"]
            check(sha(video_path) == row["episode_video_sha256"],
                  "episode video hash")
            video = imageio.get_reader(video_path)
            try:
                first = video.get_data(0)
                last = video.get_data(video.count_frames() - 1)
                check(first.shape == last.shape and first.ndim == 3,
                      "decoded video")
            finally:
                video.close()
            if condition == "moving":
                check(alarms == [17.4], "inference-time alarm")
                check(len(views) == 1, "one invalidated view")
                view = views[0]
                check(view["capture_time_s"] == 17.0 and
                      view["inference_completion_time_s"] == 17.4,
                      "alarm inside inference interval")
                check(view["raw_candidate_box_before_validity"] is not None
                      and view["accepted_box"] is None and
                      not view["inference_validity"]["valid"],
                      "candidate invalidated")
                check(row["status"] == "owner_turn_near_body_unresolved" and
                      not any(e["event"] == "scripted_owner_identity_answer"
                              for e in events), "no stale owner answer")
                check(monitor["response"]["gate"]["status"] ==
                      "no_observed_free_swept_arm_corridor" and
                      not monitor["response"]["gate"]["allowed"],
                      "unknown side corridor blocks action")
                # Evaluator-only counterfactual: keep the same recorded body
                # still while the delayed low prop finishes its approach.
                qpos = packets[-1]["actor_qpos"]
                room = OfferDriftRoom(case=tuple(next(x for x in report["rows"]
                                    if x["condition"] == condition and
                                    x["case"][0] == seed)["case"]),
                                      pose_source="command", branch=OUT / "audit")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    room.data.qpos[:] = qpos
                    room.actual[:] = (qpos[0], qpos[1],
                                      2*math.atan2(qpos[6], qpos[3]))
                    room.estimated[:] = packets[-1]["estimated_pose_xyyaw"]
                    room.head = float(qpos[room.head_addr])
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    samples = []
                    video_path = (OUT / f"seed{seed}-held-counterfactual.mp4")
                    writer_context = (imageio.get_writer(
                        video_path, fps=5, codec="libx264", quality=7)
                        if seed == 170 else nullcontext(None))
                    with writer_context as writer:
                        for tick in range(31):
                            time_s = 17.4 + .02*tick
                            fraction = min(1., max(0.,
                                (time_s - report["delayed_prop"]["start_s"]) /
                                report["delayed_prop"]["duration_s"]))
                            y = (report["delayed_prop"]["start_y_m"] +
                                 fraction * (report["delayed_prop"]["end_y_m"] -
                                             report["delayed_prop"]["start_y_m"]))
                            room.obstacle_xy[3] = (0., y)
                            room.model.body_pos[room.model.body(
                                "loop_stool").id] = (0., y, STOOL_SIZE[2])
                            room.time_s = time_s
                            room._forward()
                            gap = float(room._current_obstacle_clearance()[0])
                            samples.append({"time_s": round(time_s, 4),
                                            "scene_gap_m": round(gap, 5)})
                            check(math.isfinite(gap), "held gap")
                            if tick % 5 == 0 and seed == 170:
                                rgb, _ = room.capture("lower_rgb", depth=False)
                                writer.append_data(owner_module.panel(
                                    room, rgb, f"V98 evaluator-only hold | "
                                    f"t{time_s:.1f} gap {gap*100:.1f}cm"))
                    check(samples[0]["scene_gap_m"] >= .05 and
                          min(x["scene_gap_m"] for x in samples) < .05,
                          "holding later breaches provisional margin")
                    results.append({"seed": seed, "condition": condition,
                                    "alarm_s": 17.4,
                                    "held_first_below_5cm_s": next(
                                        x["time_s"] for x in samples
                                        if x["scene_gap_m"] < .05),
                                    "held_minimum_gap_m": min(
                                        x["scene_gap_m"] for x in samples),
                                    "held_video_path": (str(video_path.relative_to(ROOT))
                                        if seed == 170 else None),
                                    "held_video_sha256": (sha(video_path)
                                        if seed == 170 else None)})
                finally:
                    room.close()
            else:
                check(alarms == [] and len(views) == 2,
                      "clear owner controls")
                check(all(v["accepted_box"] is not None and
                          v["inference_validity"]["valid"] and
                          abs(v["inference_validity"]["elapsed_s"] - .8) < 1e-5
                          for v in views), "fresh delayed owner results")
                check(row["status"] == "owner_assisted_visual_reacquisition"
                      and any(e["event"] == "scripted_owner_identity_answer"
                              for e in events), "owner control completes")
                results.append({"seed": seed, "condition": condition,
                                "alarm_s": None,
                                "fresh_owner_views": len(views)})
    finally:
        structural.structural_model = old_model
    output = {"schema": "robot-learning-lab.ape-owner-inference-clock-audit.v1",
              "passed": True, "checks": checks,
              "source_sha256": sha(AUDIT_SOURCE),
              "results": results,
              "scope": "Replays all 10Hz owner monitor alarms from retained actor points, verifies image capture time and invalidation before owner answer, decodes four full videos, and scores an evaluator-only 20ms held counterfactual after the alert. The logical inference scheduler and 5cm historical-body margin are not physical qualification."}
    path = OUT / "independent-audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path, checks, "checks passed")


if __name__ == "__main__":
    main()
