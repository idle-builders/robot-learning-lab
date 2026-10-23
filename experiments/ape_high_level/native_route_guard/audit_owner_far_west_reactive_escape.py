"""Independent full-angle response and gap audit for V62 negative result."""

from __future__ import annotations

import json
from pathlib import Path

import audit_owner_fixed_clock_pair as matched
import audit_owner_approach_aware_meeting as choice_audit
import owner_far_west_reactive_escape as trial


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-far-west-reactive-escape.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v59_source_sha256", trial.v59.__file__),
                      ("v61_report_sha256", trial.v61.OUT / "report.json"),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(path)
    rows = json.loads(Path(report["run"]["trace"]["path"]).read_text())
    choice = choice_audit.choice_from_rows(rows)
    assert choice["choice"] == "west_60cm"
    alarm = next(row for row in rows if row["stage"] ==
                 "near_body_escape_plan")
    assert alarm["time_s"] == report["near_body_alarm_s"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    original_paths = dict(trial.v59.v50.PATHS)
    try:
        trial.v59.v50.PATHS["far_west_turn"] = trial.v61.END_XY_M
        checked = matched.score_item(
            {"run": report["run"], "case": trial.v54.CASE,
             "path": "far_west_turn", "policy": "surface_track",
             "near_body_alarm_s": report["near_body_alarm_s"],
             "near_body_commanded_m": report["near_body_commanded_m"]},
            {"checkpoint_s": report["checkpoint_s"],
             "monitor_ticks": trial.runner.FIXED_MONITOR_TICKS,
             "monitor_period_s": trial.runner.SHOW_MONITOR_PERIOD_S},
            source, scenario)
    finally:
        trial.v59.v50.PATHS.clear()
        trial.v59.v50.PATHS.update(original_paths)
    assert checked["commanded_m"] == 0.
    assert checked["minimum_prop_gap_m"] < 0.
    alarm_sample = next(row for row in rows if row["stage"] ==
                        "second_inference_range_watch" and
                        row["time_s"] == report["near_body_alarm_s"])
    assert alarm_sample["evaluator_prop_gap_m"] < .05
    result = {"passed": True, "choice": choice, "row": checked,
              "alarm_prop_gap_m": alarm_sample["evaluator_prop_gap_m"],
              "scope": "Frozen V59 actor meeting choice and V52 all-angle surface-track escape on new fixture-clear farther-west turn. Independently replayed fixed-clock packets, cluster prediction, blocked prior-packet commands, 20 ms MuJoCo gaps and decoded full video. Alarm occurs after modeled prop gap is already below 5 cm; no escape step is issued."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
