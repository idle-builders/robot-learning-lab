"""Compare paired contact-feedback native replays with exact open-loop baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from summarize_phase_feedback import REPO, first, sha, summarize


ROOT = REPO / "artifacts/ape-cove50-contact-transition-20261010"


def case(path: Path, label: str) -> dict:
    raw = json.loads(path.read_text())
    row = summarize(path, label)
    samples = raw["timeseries"]
    row["swing_load_feedback_gain"] = raw.get("swing_load_feedback_gain_rad_per_fraction", 0)
    row["peak_swing_roll_feedback_rad"] = raw.get("peak_swing_load_feedback_roll_rad", 0)
    row["stance_load_at_4p2"] = (first(samples, lambda x: x["time_s"] >= 4.2)
                                     ["stance_foot_load_fraction"])
    row["stance_load_at_5p2"] = (first(samples, lambda x: x["time_s"] >= 5.2)
                                     ["stance_foot_load_fraction"]
                                  if raw["duration_s"] >= 5.2 else None)
    row["slip_at_5p2_mm"] = (round(first(samples, lambda x: x["time_s"] >= 5.2)
                                   ["stance_slip_m"] * 1000, 2)
                              if raw["duration_s"] >= 5.2 else None)
    return row


def main() -> None:
    plan_path = ROOT / "plan.json"
    plan = json.loads(plan_path.read_text())
    for key in ("selected_design_pointer", "selected_body", "actuator_catalog",
                "probe_source", "static_screen"):
        item = plan[key]
        assert sha(REPO / item["path"]) == item["sha256"], key
    cases = []
    exact_replays = {}
    for side in ("right", "left"):
        item = plan["baseline_reports"][side]
        path = REPO / item["path"]
        assert sha(path) == item["sha256"]
        replay = ROOT / f"trials/sameworker-openloop-{side}-nominal.json"
        feedback = ROOT / f"trials/loadroll-{side}-nominal.json"
        assert replay.exists() and feedback.exists()
        old = json.loads(path.read_text())
        fresh = json.loads(replay.read_text())
        for field in ("initial_state_sha256", "duration_s", "longest_single_support_s",
                      "rated_leg_fraction", "max_stance_slip_m", "lift_skill_screen_pass"):
            assert old[field] == fresh[field], (side, field)
        with np.load(REPO / old["state_trace"]) as a, np.load(REPO / fresh["state_trace"]) as b:
            for field in ("qpos", "qvel", "joint_target"):
                assert np.array_equal(a[field], b[field]), (side, field)
        exact_replays[side] = {"prior_report": str(path.relative_to(REPO)),
                               "same_worker_report": str(replay.relative_to(REPO)),
                               "exact_qpos_qvel_targets": True}
        cases.extend((case(path, "prior-open-loop"),
                      case(replay, "same-worker-open-loop"),
                      case(feedback, "load-roll-feedback")))
    assert len(list((ROOT / "trials").glob("*.json"))) == 4
    report = {
        "schema": "robot-learning-lab.cove50-contact-transition-comparison.v1",
        "method": "Paired selected-body native MJLab motor replays from upright. The authored controller reads simulator foot force; no learned or deployable policy claim.",
        "selected_design_pointer_sha256": plan["selected_design_pointer"]["sha256"],
        "selected_body_sha256": plan["selected_body"]["sha256"],
        "actuator_catalog_sha256": plan["actuator_catalog"]["sha256"],
        "plan": {"path": str(plan_path.relative_to(REPO)), "sha256": sha(plan_path)},
        "same_worker_baseline_replay": exact_replays,
        "cases": cases,
    }
    out = ROOT / "comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(out), "cases": len(cases),
                      "passes": [x["case"] for x in cases if x["full_gate_pass"]]}, indent=2))


if __name__ == "__main__":
    main()
