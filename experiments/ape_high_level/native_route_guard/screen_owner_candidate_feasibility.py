"""Evaluator-only bounded-direction screen for V41's closer prop repeats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

import owner_candidate_occupancy_response as v42


OUT = v42.OUT / "closer-finish-feasibility.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    main_path = v42.v41.OUT / "report.json"
    repeat_path = v42.v41.OUT / "timing-repeats/report.json"
    main_report = json.loads(main_path.read_text())
    repeat_report = json.loads(repeat_path.read_text())
    cases = [next(item for item in main_report["results"]
                  if item["prop_case"] == "holdout_closer_finish")]
    cases.extend(item for item in repeat_report["results"]
                 if item["prop_case"] == "holdout_closer_finish")
    source = v42.v41.v40.v38.original()
    summary = source["summary"]
    start = np.asarray(summary["ape_after_episode_xy_m"])
    scene = v42.v41.v40.v38.v35.moving.SCENE_XML
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    rows = []
    for index, item in enumerate(cases):
        run = item["run"]
        trace = json.loads(Path(run["trace"]["path"]).read_text())
        inference = next(row for row in trace if row["stage"] ==
                         "second_inference_result")
        first = inference["first_response_s"]
        assert first is not None
        candidates = {}
        for name, direction in v42.CANDIDATES.items():
            person_gap = prop_gap = float("inf")
            for t in np.arange(run["second_inference_start_s"], 24.81, .02):
                distance = min(.25, max(0., .25*(t-first)))
                ape = start+distance*direction
                person = v42.v41.v40.v38.human_xy(
                    "blind_return_flank_fast_response", summary, float(t),
                    run["request_time_s"], run["second_inference_start_s"])
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v42.v41.v40.prop_at(
                        float(t), main_report["checkpoint_s"],
                        v42.v41.PROP_CASES["holdout_closer_finish"])
                person_gap = min(person_gap, v42.v41.v40.v38.geometry.gap(
                    model, data, ape, person))
                prop_gap = min(prop_gap, v42.v41.v40.v38.prop_gap(
                    model, data, ape))
            candidates[name] = {"minimum_person_gap_m": person_gap,
                                "minimum_prop_gap_m": prop_gap}
        rows.append({"source_run": "main" if index == 0 else
                     f"timing_repeat_{index}",
                     "source_trace_sha256": run["trace"]["sha256"],
                     "first_response_s": first,
                     "candidates": candidates})
    result = {"schema": "robot-learning-lab.ape-owner-response-feasibility.v1",
              "scope": "Evaluator-only 20ms MuJoCo geometry for fixed 25cm directions at four observed V41 first-command times. This is a feasibility bound, not an actor choice or selected-body safety result.",
              "source_sha256": sha(Path(__file__)),
              "v42_source_sha256": sha(v42.__file__),
              "main_report_sha256": sha(main_path),
              "repeat_report_sha256": sha(repeat_path),
              "scene_sha256": sha(scene),
              "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    for row in rows:
        print(row["source_run"], row["first_response_s"],
              row["candidates"]["southwest"], flush=True)


if __name__ == "__main__":
    main()
