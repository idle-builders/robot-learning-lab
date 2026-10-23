"""Audit V90's task-only range gate against identical V87/V88 owner paths."""

from __future__ import annotations

import json
from pathlib import Path

import audit_owner_joint_hazard_response as media
import owner_social_view_near_surface_claim_gate as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def normalized_pre(rows):
    stages = {"post_alarm_range_watch", "pre_show_restage_step",
              "pre_show_restage_wait", "fresh_new_show_capture"}
    selected = [row for row in rows if row["stage"] in stages]
    return [dict(row, raw_rgbd=row["raw_rgbd"]["sha256"])
            if row["stage"] == "fresh_new_show_capture" else row
            for row in selected]


def main():
    sha = trial.v86.v82.v77.v72.v52.sha
    current = json.loads((trial.OUT / "report.json").read_text())
    refs = {
        "outward_normal": json.loads((trial.v87.OUT / "report.json").read_text()),
        **{name: json.loads((trial.v88.OUT / "report.json").read_text())
           for name, _, _ in trial.v88.PATHS}}
    checks = []
    check(current["source_sha256"] == sha(trial.__file__) and
          current["v86_source_sha256"] == sha(trial.v86.__file__) and
          current["v87_report_sha256"] == sha(trial.v87.OUT / "report.json")
          and current["v88_report_sha256"] == sha(trial.v88.OUT / "report.json")
          and current["runner_source_sha256"] == sha(
              trial.v86.runner.__file__) and
          current["current_design_pointer_sha256"] == sha(
              trial.v86.ROOT / "robot/ape-current-design.json") and
          current["task_claim_nearest_surface_gate_m"] == .27,
          "frozen task-gate provenance", checks)
    findings = []
    for item in current["results"]:
        name = item["path"]
        run = item["run"]
        reference = next(x for x in refs[name]["results"] if
                         x["path"] == name)
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        old = json.loads(Path(reference["run"]["trace"]["path"]).read_text())
        check(normalized_pre(rows) == normalized_pre(old),
              f"{name} identical pre-show and RGB-D capture", checks)
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"] and
              media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"],
              f"{name} decoded full video", checks)
        flank = next(row for row in rows if row["stage"] ==
                     "flank_sensor_video")
        check(sha(flank["path"]) == flank["sha256"] and
              media.video_frames(flank["path"], (480, 640, 3)) ==
              flank["frames"], f"{name} decoded flank video", checks)
        watch = [row for row in rows if row["stage"] ==
                 "second_inference_range_watch"]
        old_watch = [row for row in old if row["stage"] ==
                     "second_inference_range_watch"]
        check(len(watch) == len(old_watch) ==
              trial.v86.runner.FIXED_MONITOR_TICKS,
              f"{name} full watch clock", checks)
        streak = 0
        for index, (new, prior) in enumerate(zip(watch, old_watch)):
            valid = [x for x in new["actor_packet"] if x is not None]
            nearest = min(valid, default=None)
            approaching = bool(new["near_body_observation"] and
                               new["near_body_observation"]["alarm"])
            risk = bool((nearest is not None and nearest < .27) or approaching)
            streak = streak+1 if risk else 0
            check(new["time_s"] == prior["time_s"] and
                  new["actor_packet"] == prior["actor_packet"] and
                  new["ape_xy_m"] == prior["ape_xy_m"] and
                  new["task_risk_decision"]["risk"] == risk and
                  new["task_risk_decision"][
                      "minimum_observed_surface_m"] == nearest and
                  new["task_risk_decision"][
                      "approaching_track_alarm"] == approaching and
                  new["risk_streak"] == streak,
                  f"{name} identical raw watch and task rule {index}", checks)
        # A changed task outcome can omit the old final 240 ms item-check
        # tail; compare the common commanded-scene clock, not list length.
        prior_gap = {round(row["time_s"], 5): row for row in
                     reference["run"]["evaluator_gap_samples"]}
        overlap = 0
        for sample in run["evaluator_gap_samples"]:
            older = prior_gap.get(round(sample["time_s"], 5))
            if older is None:
                continue
            overlap += 1
            check(abs(sample["prop_gap_m"]-older["prop_gap_m"]) < 1e-8 and
                  sample["ape_xy_m"] == older["ape_xy_m"],
                  f"{name} matched 20ms geometry {sample['time_s']:.2f}",
                  checks)
        check(overlap > 300 and
              run["minimum_prop_gap_m"] == reference[
                  "run"]["minimum_prop_gap_m"],
              f"{name} unchanged minimum and common geometry", checks)
        first_risk = next((row["time_s"] for row in watch if row[
            "task_risk_decision"]["risk"]), None)
        item_candidate = next((row for row in rows if row["stage"] ==
                               "fresh_item_candidate"), None)
        if name != "outward_normal":
            check(run["minimum_prop_gap_m"] < 0.,
                  f"{name} changed-turn overlap remains", checks)
        findings.append({"path": name, "original_outcome":
                         reference["run"]["outcome"],
                         "task_gate_outcome": run["outcome"],
                         "minimum_prop_gap_m": run["minimum_prop_gap_m"],
                         "first_task_risk_s": first_risk,
                         "item_depth_pixels": item_candidate[
                             "depth_valid_pixels"] if item_candidate else None,
                         "worker_wall_s": next(row["worker_wall_s"]
                             for row in rows if row["stage"] ==
                             "second_inference_result"),
                         "video": run["video"]["path"]})
    claims_after_overlap = [row["path"] for row in findings if
        row["path"] != "outward_normal" and
        row["task_gate_outcome"] == "new_show_metric_item_seen_unattributed"]
    check(claims_after_overlap == ["grid_x1_y0", "grid_x2_y1", "grid_x2_y2"],
          "three item claims persist after pre-watch overlap", checks)
    control = next(row for row in findings if row["path"] == "outward_normal")
    check(control["first_task_risk_s"] is None and
          control["task_gate_outcome"] == "new_show_unconfirmed_unknown",
          "control changed only through image-age gate, not range rule", checks)
    audit = {"schema": "robot-learning-lab.ape-near-surface-claim-gate-audit.v1",
             "passed": True, "checks": len(checks), "results": findings,
             "remaining_claims_after_overlap": claims_after_overlap,
             "scope": "Identical pre-show RGB-D and 20ms geometry, identical full raw range watches, independently recomputed task risk/streak, decoded media. Image worker wall-clock age changed in the control; no causal false-proximity conclusion. Threshold is task honesty only, not a physical safety action."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
