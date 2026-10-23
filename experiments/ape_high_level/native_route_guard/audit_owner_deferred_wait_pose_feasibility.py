"""Independent saved-gap and sampled MuJoCo audit for the V75 pose screen."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import owner_deferred_wait_pose_feasibility as v75
import render_owner_deferred_wait_pose_feasibility as render


def check(ok, message, checks):
    if not ok:
        raise AssertionError(message)
    checks.append(message)


def first_below(times, gaps):
    which = np.flatnonzero(gaps < .05)
    return float(times[which[0]]) if len(which) else None


def main():
    report = json.loads((v75.OUT / "report.json").read_text())
    checks, sha = [], v75.v72.v52.sha
    for key, source in (
        ("source_sha256", v75.__file__),
        ("v60_report_sha256", v75.V60_REPORT),
        ("v59_source_sha256", v75.v72.v64.v60.v59.__file__),
        ("v65_source_sha256", v75.v74.v65.__file__),
        ("v74_report_sha256", v75.v74.OUT / "report.json"),
        ("scene_sha256", v75.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         v75.ROOT / "robot/ape-current-design.json")):
        check(report[key] == sha(source), key, checks)
    check(report["period_s"] == .02 and
          report["provisional_margin_m"] == .05,
          "20ms/5cm screen", checks)
    base = json.loads(v75.V60_REPORT.read_text())
    trace = json.loads(Path(base["results"][0]["run"]["trace"][
        "path"]).read_text())
    first = next(row for row in trace if row["stage"] ==
                 "pre_show_restage_step")
    eligible = [row for row in first["actor_decision"][
        "candidate_scores"] if row["eligible"]]
    check(len(eligible) == len(report["candidates"]) == 16,
          "all 16 actor-eligible candidates", checks)
    check(report["variant_names"] == [x[0] for x in v75.VARIANTS],
          "six declared variants", checks)
    raw = Path(report["raw"]["path"])
    check(sha(raw) == report["raw"]["sha256"],
          "raw gap hash", checks)
    with np.load(raw) as saved:
        times = saved["time_s"]
        props = saved["prop_gap_m"]
        people = saved["person_gap_m"]
    check(props.shape == people.shape ==
          (16, 6, len(times)) and len(times) > 300,
          "raw matrix dimensions", checks)
    check(np.allclose(np.diff(times), .02, atol=1e-8) and
          abs(times[0]-report["move_start_s"]) < 1e-8,
          "fixed evaluator clock", checks)
    source = v75.runner.original()
    summary = source["summary"]
    policy = json.loads(v75.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        v75.runner.v35.moving.SCENE_XML))
    sim = mujoco.MjData(model)
    start, _, t0, _, _ = v75.v72.v64.v51.v46.low_prop_geometry(
        v75.runner.geometry_at)(
            summary, source["result"]["result"]["episode"],
            scenario, model, sim)
    check(np.allclose(start, report["start_xy_m"]) and
          abs(t0-report["checkpoint_s"]) < 1e-8,
          "checkpoint pose/time", checks)
    for ci, (expected, row) in enumerate(zip(eligible,
                                              report["candidates"])):
        check(expected["name"] == row["name"] and
              abs(expected["distance_m"]-row["distance_m"]) < 1e-8 and
              np.allclose(expected["target_xy_m"], row["target_xy_m"]),
              f"candidate {ci} matches actor menu", checks)
        target = np.asarray(row["target_xy_m"])
        distance = row["distance_m"]
        for vi, (variant, path, finish_s) in enumerate(v75.VARIANTS):
            cell = row["paths"][vi]
            prop = props[ci, vi]
            person = people[ci, vi]
            check(cell["variant"] == variant and cell["path"] == path and
                  abs(cell["turn_finish_after_checkpoint_s"]-finish_s) < 1e-8,
                  f"{ci}/{vi} variant", checks)
            check(abs(float(prop.min())-cell[
                "minimum_prop_gap_m"]) < 1e-8 and
                  abs(float(person.min())-cell[
                      "minimum_person_gap_m"]) < 1e-8,
                  f"{ci}/{vi} raw minima", checks)
            check(first_below(times, prop) == cell[
                "first_prop_below_5cm_s"] and
                  first_below(times, person) == cell[
                      "first_person_below_5cm_s"],
                  f"{ci}/{vi} first breach", checks)
            endpoint = v75.v72.v64.PATHS[path]
            for metric, samples in (("prop", prop),
                                    ("person", person)):
                ti = int(samples.argmin())
                t = float(times[ti])
                portion = np.clip((t-report["move_start_s"])*
                                  v75.SPEED_MPS/distance, 0., 1.)
                pose = start+portion*(target-start)
                other = v75.runner.human_xy(
                    v75.v72.v64.v54.CASE, summary, t, t0,
                    row["scripted_person_return_s"])
                v75.runner.v35.two.set_distractor(model, other)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v75.v74.v65.prop_at(t, t0, endpoint, finish_s)
                mujoco.mj_forward(model, sim)
                measured = (v75.runner.prop_gap(model, sim, pose)
                            if metric == "prop" else
                            v75.runner.geometry.gap(model, sim, pose, other))
                check(abs(measured-float(samples[ti])) < 1e-5,
                      f"{ci}/{vi} independent {metric} minimum", checks)
        check(abs(min(x["minimum_prop_gap_m"] for x in row["paths"])-
                  row["worst_prop_gap_m"]) < 1e-8 and
              abs(min(x["minimum_person_gap_m"] for x in row["paths"])-
                  row["worst_person_gap_m"]) < 1e-8,
              f"candidate {ci} worst gaps", checks)
    supported = sum(row["worst_prop_gap_m"] >= .05 and
                    row["worst_person_gap_m"] >= .05
                    for row in report["candidates"])
    check(supported == report["supported_candidate_count"] == 0,
          "no eligible candidate clears all six paths", checks)
    plot = Path(report["plot"]["path"])
    check(sha(plot) == report["plot"]["sha256"],
          "plot hash", checks)
    check(report["plot_renderer_sha256"] == sha(render.__file__),
          "plot renderer source", checks)
    with Image.open(plot) as image:
        check(image.size == (1450, 930), "plot dimensions", checks)
    output = {"schema": "robot-learning-lab.ape-wait-pose-audit.v1",
              "source_sha256": sha(__file__),
              "status": "pass", "checks": len(checks),
              "candidate_count": len(report["candidates"]),
              "variant_count": len(v75.VARIANTS),
              "supported_candidate_count": supported,
              "independent_minimum_replays": 2*len(report["candidates"])*
                  len(v75.VARIANTS)}
    target = v75.OUT / "audit.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(target, len(checks), "checks passed")


if __name__ == "__main__":
    main()
