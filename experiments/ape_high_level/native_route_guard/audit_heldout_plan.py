"""Audit V26 changed furniture/timing against saved actor and evaluator traces."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import heldout_plan_scene as heldout
import kitchen_trial as trial
import plan_bounded_trial as plan


def person_fixture_gaps(scene: Path) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    person = model.geom("person_capsule").id
    fixtures = [i for i in range(model.ngeom)
                if model.geom_group[i] != 4 and i != person and
                model.geom(i).name != "pet_box" and
                model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE and
                "floor" not in model.geom(i).name.lower()]
    holder = np.zeros(6)
    gaps = {}
    for case in heldout.CASES:
        smallest = float("inf")
        for t in np.arange(.02, 8., .1):
            model.body_pos[model.body("approaching_person").id, :2] = \
                heldout.hazard_xy(case, float(t))
            model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
            mujoco.mj_forward(model, data)
            smallest = min(smallest, *(float(mujoco.mj_geomDistance(
                model, data, person, i, 2., holder)) for i in fixtures))
        gaps[case] = round(smallest, 5)
    return gaps


def main() -> None:
    path = heldout.OUT / "report.json"
    report = json.loads(path.read_text())
    rows = report["results"]
    checks = {
        "scene": report["scene_sha256"] == trial.sha(heldout.OUT / "scene.xml"),
        "holdout_source": report["source_sha256"] == trial.sha(Path(heldout.__file__)),
        "actor_source": report["actor_source_sha256"] == trial.sha(Path(plan.__file__)),
        "native_trace": report["native_trace_sha256"] == trial.sha(heldout.kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(heldout.kitchen.BODY),
        "v24_frozen": report["v24_report_sha256"] == trial.sha(plan.OUT / "report.json"),
        "matrix": len(rows) == 48 and len({
            (r["configuration"], r["world"], r["case"]) for r in rows}) == 48,
    }
    summary = {"fast_false_stops": 0, "threats_below_5cm": 0,
               "threat_overlaps": 0, "min_guarded_threat_gap_m": float("inf"),
               "control_fast_clear": 0, "control_threats_below_5cm": 0,
               "control_threat_overlaps": 0, "guard_packets": 0,
               "videos_checked": 0}
    for row in rows:
        key = f'{row["configuration"]}:{row["world"]}:{row["case"]}'
        fast = row["case"].startswith("fast_")
        guarded = row["configuration"] == "online_plan_guard"
        trace = Path(row["trace"]["path"])
        checks[f"trace:{key}"] = trial.sha(trace) == row["trace"]["sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"],
                         saved["midstep_gap_evaluator_m"]]
            packets = [json.loads(raw) for raw in saved["packet_json"]]
        checks[f"gap:{key}"] = (
            round(float(gaps.min()), 5) == row["min_modeled_body_person_gap_m"]
            and bool((gaps <= 0).any()) == row["modeled_person_overlap"]
            and bool((gaps < trial.MIN_MARGIN_M).any()) ==
            row["below_5cm_person_margin"])
        if guarded:
            summary["guard_packets"] += len(packets)
            checks[f"actor_packets:{key}"] = bool(packets) and all(
                len(p["ranges_m"]) == 73 and
                p["route_guard"]["map_source"] == "none_online_clusters" and
                abs(p["route_guard"]["forecast_horizon_s"] - min(
                    heldout.guard.HORIZON_S,
                    max(0., plan.SKILL_END_S-p["captured_at_s"]) +
                    plan.STOP_EQUIVALENT_S)) < 1e-4
                for p in packets)
            risk_times = [p["captured_at_s"] for a, p in enumerate(packets)
                          if a and p["route_guard"]["risk"] and
                          packets[a-1]["route_guard"]["risk"]]
            first_stop = risk_times[0] if risk_times else 3.
            checks[f"stop_rule:{key}"] = abs(row["stop_command_s"]-first_stop) < 1e-4
            summary["fast_false_stops"] += int(fast and row["stop_command_s"] < 3.)
            summary["threats_below_5cm"] += int(not fast and
                                                row["below_5cm_person_margin"])
            summary["threat_overlaps"] += int(not fast and
                                              row["modeled_person_overlap"])
            if not fast:
                summary["min_guarded_threat_gap_m"] = min(
                    summary["min_guarded_threat_gap_m"],
                    row["min_modeled_body_person_gap_m"])
        else:
            checks[f"no_sensor_packets:{key}"] = len(packets) == 0
            summary["control_fast_clear"] += int(
                fast and not row["below_5cm_person_margin"])
            summary["control_threats_below_5cm"] += int(
                not fast and row["below_5cm_person_margin"])
            summary["control_threat_overlaps"] += int(
                not fast and row["modeled_person_overlap"])
        if row["video"] is not None:
            video = row["video"]
            capture = cv2.VideoCapture(video["path"])
            decoded = 0
            while True:
                okay, frame = capture.read()
                if not okay:
                    break
                decoded += int(frame.size > 0)
            capture.release()
            checks[f"video:{key}"] = (
                trial.sha(Path(video["path"])) == video["sha256"]
                and decoded == video["frames"])
            summary["videos_checked"] += 1
    fixture_gaps = person_fixture_gaps(heldout.OUT / "scene.xml")
    checks["person_fixture_clear"] = all(value >= .05 for value in
                                         fixture_gaps.values())
    checks["control_challenge"] = (
        summary["control_fast_clear"] == 8 and
        summary["control_threats_below_5cm"] == 16)
    audit = {"schema": "robot-learning-lab.ape-robocasa-plan-holdout-audit.v1",
             "report_sha256": trial.sha(path), "summary": summary,
             "person_fixture_gaps_m": fixture_gaps,
             "checks": checks, "pass": all(checks.values())}
    (heldout.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "summary": summary,
                      "person_fixture_gaps_m": fixture_gaps}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V26 audit failed")


if __name__ == "__main__":
    main()
