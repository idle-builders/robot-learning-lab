"""Audit V27's post-deadline body/person gaps and missing guard packets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import held_body_intrusion_trial as late
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
    for case in late.CASES:
        smallest = float("inf")
        for t in np.arange(.02, 8., .1):
            model.body_pos[model.body("approaching_person").id, :2] = \
                late.hazard_xy(case, float(t))
            model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
            mujoco.mj_forward(model, data)
            smallest = min(smallest, *(float(mujoco.mj_geomDistance(
                model, data, person, i, 2., holder)) for i in fixtures))
        gaps[case] = round(smallest, 5)
    return gaps


def main() -> None:
    report_path = late.OUT / "report.json"
    report = json.loads(report_path.read_text())
    rows = report["results"]
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(late.__file__)),
        "actor": report["actor_source_sha256"] == trial.sha(Path(plan.__file__)),
        "runner": report["runner_source_sha256"] == trial.sha(Path(trial.__file__)),
        "scene": report["scene_sha256"] == trial.sha(late.previous.OUT / "scene.xml"),
        "native": report["native_trace_sha256"] == trial.sha(late.kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(late.kitchen.BODY),
        "v26": report["v26_report_sha256"] ==
               trial.sha(late.previous.OUT / "report.json"),
        "matrix": len(rows) == 16 and len({
            (r["configuration"], r["world"], r["case"]) for r in rows}) == 16,
    }
    summary = {"control_below_5cm": 0, "guarded_below_5cm": 0,
               "control_overlaps": 0, "guarded_overlaps": 0,
               "all_guard_last_packet_s": [],
               "first_guarded_below_5cm_s": [],
               "videos_checked": 0}
    for row in rows:
        key = f'{row["configuration"]}:{row["world"]}:{row["case"]}'
        guarded = row["configuration"] == "bounded_guard"
        trace = Path(row["trace"]["path"])
        checks[f"trace:{key}"] = trial.sha(trace) == row["trace"]["sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"],
                         saved["midstep_gap_evaluator_m"]]
            times = np.r_[saved["time_s"], saved["midstep_time_s"]]
            packets = [json.loads(raw) for raw in saved["packet_json"]]
        checks[f"gap:{key}"] = (
            round(float(gaps.min()), 5) == row["min_modeled_body_person_gap_m"]
            and bool((gaps <= 0).any()) == row["modeled_person_overlap"]
            and bool((gaps < trial.MIN_MARGIN_M).any()) ==
            row["below_5cm_person_margin"])
        checks[f"deadline:{key}"] = row["stop_command_s"] == plan.SKILL_END_S
        if guarded:
            last_packet = max(p["captured_at_s"] for p in packets)
            summary["all_guard_last_packet_s"].append(last_packet)
            checks[f"packet_gap:{key}"] = (
                bool(packets) and last_packet <= plan.SKILL_END_S and
                all(p["route_guard"]["map_source"] == "none_online_clusters"
                    for p in packets))
            summary["guarded_below_5cm"] += int(row["below_5cm_person_margin"])
            summary["guarded_overlaps"] += int(row["modeled_person_overlap"])
            if row["below_5cm_person_margin"]:
                first = float(times[np.where(gaps < trial.MIN_MARGIN_M)[0]].min())
                summary["first_guarded_below_5cm_s"].append(round(first, 5))
        else:
            checks[f"no_packets:{key}"] = len(packets) == 0
            summary["control_below_5cm"] += int(row["below_5cm_person_margin"])
            summary["control_overlaps"] += int(row["modeled_person_overlap"])
        if row["video"] is not None:
            video = row["video"]
            cap = cv2.VideoCapture(video["path"])
            decoded = 0
            while True:
                okay, frame = cap.read()
                if not okay:
                    break
                decoded += int(frame.size > 0)
            cap.release()
            checks[f"video:{key}"] = (
                trial.sha(Path(video["path"])) == video["sha256"] and
                decoded == video["frames"])
            summary["videos_checked"] += 1
    fixture_gaps = person_fixture_gaps(late.previous.OUT / "scene.xml")
    checks["fixture_clear"] = all(gap >= .05 for gap in fixture_gaps.values())
    checks["post_deadline_intrusion"] = (
        summary["guarded_below_5cm"] > 0 and
        all(t > plan.SKILL_END_S for t in
            summary["first_guarded_below_5cm_s"]))
    audit = {"schema": "robot-learning-lab.ape-held-body-intrusion-audit.v1",
             "report_sha256": trial.sha(report_path), "summary": summary,
             "person_fixture_gaps_m": fixture_gaps,
             "checks": checks, "pass": all(checks.values())}
    (late.OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pass": audit["pass"], "summary": summary,
                      "person_fixture_gaps_m": fixture_gaps}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V27 audit failed")


if __name__ == "__main__":
    main()
