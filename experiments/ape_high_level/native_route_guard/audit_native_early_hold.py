"""Audit actual paired native stop traces and V29 full-body gap evidence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import held_body_intrusion_trial as late
import kitchen_trial as trial
import native_early_hold_screen as screen


def main() -> None:
    path = screen.OUT / "report.json"
    report = json.loads(path.read_text())
    early_source, _ = screen.MODES["early_2p5s"]
    late_source, _ = screen.MODES["late_3p0s"]
    prefix_comparison = {}
    with np.load(early_source, allow_pickle=False) as early, \
            np.load(late_source, allow_pickle=False) as later:
        prefix = round(2.5/.02)-1
        for world in range(4):
            xy = np.linalg.norm(early["root_pose"][:prefix+1, world, :2] -
                                later["root_pose"][:prefix+1, world, :2],
                                axis=1)
            joints = np.max(np.abs(
                early["joint_pos"][:prefix+1, world] -
                later["joint_pos"][:prefix+1, world]), axis=1)
            prefix_comparison[str(world)] = {
                "max_root_xy_difference_m": round(float(xy.max()), 6),
                "max_joint_difference_rad": round(float(joints.max()), 6),
                "at_stop_root_xy_difference_m": round(float(xy[-1]), 6),
                "at_stop_joint_difference_rad": round(float(joints[-1]), 6),
                "near_identical_prefix": bool(xy.max() < .0001 and
                                              joints.max() < .001),
            }
    info = report["native_sources"]
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(screen.__file__)),
        "scene": report["scene_sha256"] ==
                 trial.sha(late.previous.OUT / "scene.xml"),
        "body": report["body_sha256"] == trial.sha(screen.kitchen.BODY),
        "v27": report["v27_report_sha256"] == trial.sha(late.OUT / "report.json"),
        "same_initial_state": (info["early_2p5s"]["initial_state_sha256"] ==
                               info["late_3p0s"]["initial_state_sha256"]),
        "same_body": all(item["body_sha256"] == report["body_sha256"]
                         for item in info.values()),
        "same_policy_checkpoint": (
            json.loads(early_source.with_suffix(".json").read_text())["checkpoint"] ==
            json.loads(late_source.with_suffix(".json").read_text())["checkpoint"]),
        "matrix": len(report["results"]) == 16 and len({
            (r["mode"], r["world"], r["case"]) for r in report["results"]}) == 16,
    }
    for mode, (source, stop_s) in screen.MODES.items():
        item = info[mode]
        checks[f"native_source:{mode}"] = (
            item["trace_sha256"] == trial.sha(source) and
            item["report_sha256"] == trial.sha(source.with_suffix(".json")) and
            item["stop_at_s"] == stop_s)
    summary = {}
    old_rows = {(r["world"], r["case"]): r for r in
                json.loads((late.OUT / "report.json").read_text())["results"]
                if r["configuration"] == "bounded_guard"}
    videos = 0
    for mode in screen.MODES:
        rows = [r for r in report["results"] if r["mode"] == mode]
        summary[mode] = {
            "below_5cm": sum(r["below_5cm"] for r in rows),
            "overlaps": sum(r["overlap"] for r in rows),
            "min_gap_m": min(r["min_body_person_gap_m"] for r in rows),
            "mean_final_progress_m": round(sum(r["root_final_progress_m"]
                                             for r in rows)/len(rows), 5),
            "min_fixture_gap_m": min(r["min_sampled_body_fixture_gap_m"]
                                     for r in rows),
        }
        for row in rows:
            key = f'{mode}:{row["world"]}:{row["case"]}'
            trace = Path(row["trace"]["path"])
            checks[f"trace:{key}"] = trial.sha(trace) == row["trace"]["sha256"]
            with np.load(trace, allow_pickle=False) as saved:
                gaps = np.r_[saved["gap_m"], saved["midstep_gap_m"]]
                times = np.r_[saved["time_s"], saved["midstep_time_s"]]
                root = saved["root_xy_m"]
            bad_times = times[gaps < trial.MIN_MARGIN_M]
            checks[f"gap:{key}"] = (
                round(float(gaps.min()), 5) == row["min_body_person_gap_m"]
                and bool((gaps < trial.MIN_MARGIN_M).any()) == row["below_5cm"]
                and bool((gaps <= 0).any()) == row["overlap"]
                and (round(float(bad_times.min()), 5) if len(bad_times) else None)
                    == row["first_below_5cm_s"])
            index = round(row["native_stop_command_s"]/.02)-1
            checks[f"progress:{key}"] = (
                round(float(np.linalg.norm(root[index]-root[0])), 5) ==
                row["root_command_progress_m"] and
                round(float(np.linalg.norm(root[-1]-root[0])), 5) ==
                row["root_final_progress_m"])
            if mode == "late_3p0s":
                previous = old_rows[row["world"], row["case"]]
                checks[f"v27_match:{key}"] = (
                    row["min_body_person_gap_m"] ==
                    previous["min_modeled_body_person_gap_m"] and
                    row["below_5cm"] == previous["below_5cm_person_margin"]
                    and row["overlap"] == previous["modeled_person_overlap"])
            if row["video"] is not None:
                media = row["video"]
                cap = cv2.VideoCapture(media["path"])
                decoded = 0
                while True:
                    okay, frame = cap.read()
                    if not okay:
                        break
                    decoded += int(frame.size > 0)
                cap.release()
                checks[f"video:{key}"] = (
                    trial.sha(Path(media["path"])) == media["sha256"] and
                    decoded == media["frames"])
                videos += 1
    checks["four_videos"] = videos == 4
    audit = {"schema": "robot-learning-lab.ape-native-early-hold-audit.v1",
             "report_sha256": trial.sha(path), "summary": summary,
             "pre_command_trace_comparison": prefix_comparison,
             "videos_checked": videos, "checks": checks,
             "pass": all(checks.values())}
    (screen.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "summary": summary}, indent=2))
    if not audit["pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"V29 audit failed: {failed}")


if __name__ == "__main__":
    main()
