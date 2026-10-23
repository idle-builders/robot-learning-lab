"""V75: oracle evaluator screen of V59's entire eligible wait-pose menu.

This is not a policy: future prop/person paths remain evaluator-only. It asks
whether any *existing* actor-eligible destination could have provided a 5cm
modeled response-free reserve across the authored path/speed set. The body
follows the same 25cm/s candidate move, then waits through V60's horizon.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import owner_deferred_lower_rgbd_turn_speed as v74


v72, runner = v74.v72, v74.v72.runner
ROOT = v74.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v75-wait-pose-feasibility"
V60_REPORT = ROOT / "artifacts/ape-native-route-guard/v60-deferred-owner-show/report.json"
SPEED_MPS = .25
PERIOD_S = .02
PROVISIONAL_MARGIN_M = .05
VARIANTS = (
    ("outward_normal", "outbound_control", 5.20),
    ("inward_normal", "inbound_holdout", 5.20),
    ("west_south_normal", "new_west_south_turn", 5.20),
    ("far_west_normal", "far_west_turn", 5.20),
    ("far_west_fast", "far_west_turn", 4.85),
    ("far_west_slow", "far_west_turn", 5.55),
)


def first_below(times, values):
    positions = np.flatnonzero(values < PROVISIONAL_MARGIN_M)
    return (float(times[positions[0]]) if len(positions) else None)


def main():
    source = runner.original()
    summary = source["summary"]
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    base = json.loads(V60_REPORT.read_text())
    run = base["results"][0]["run"]
    trace = json.loads(open(run["trace"]["path"]).read())
    first_step = next(row for row in trace if row["stage"] ==
                      "pre_show_restage_step")
    scores = first_step["actor_decision"]["candidate_scores"]
    candidates = [row for row in scores if row["eligible"]]
    checkpoint_s = base["checkpoint_s"]
    move_start_s = run["resume_time_s"]
    watch_end_s = checkpoint_s+v72.v64.v60.WATCH_END_AFTER_CHECKPOINT_S
    times = np.arange(move_start_s, watch_end_s+.00001, PERIOD_S)
    model = mujoco.MjModel.from_xml_path(str(
        runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    start, _, geometry_t0, _, _ = v72.v64.v51.v46.low_prop_geometry(
        runner.geometry_at)(summary,
            source["result"]["result"]["episode"], scenario,
            model, data)
    assert abs(geometry_t0-checkpoint_s) < 1e-8
    raw_prop = np.empty((len(candidates), len(VARIANTS), len(times)),
                        dtype=np.float64)
    raw_person = np.empty_like(raw_prop)
    results = []
    for ci, candidate in enumerate(candidates):
        target = np.asarray(candidate["target_xy_m"])
        distance = float(candidate["distance_m"])
        move_end_s = move_start_s+distance/SPEED_MPS
        return_s = move_end_s+v72.v64.v60.SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S
        candidate_paths = []
        for vi, (variant, path, finish_s) in enumerate(VARIANTS):
            endpoint = v72.v64.PATHS[path]
            for ti, time_s in enumerate(times):
                portion = np.clip((time_s-move_start_s)*
                                  SPEED_MPS/distance, 0., 1.)
                pose = start+portion*(target-start)
                person = runner.human_xy(
                    v72.v64.v54.CASE, summary, float(time_s),
                    checkpoint_s, return_s)
                runner.v35.two.set_distractor(model, person)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v74.v65.prop_at(float(time_s), checkpoint_s,
                                    endpoint, finish_s)
                mujoco.mj_forward(model, data)
                raw_prop[ci, vi, ti] = runner.prop_gap(model, data, pose)
                raw_person[ci, vi, ti] = runner.geometry.gap(
                    model, data, pose, person)
            prop = raw_prop[ci, vi]
            human = raw_person[ci, vi]
            first_prop = first_below(times, prop)
            first_person = first_below(times, human)
            candidate_paths.append({
                "variant": variant, "path": path,
                "turn_finish_after_checkpoint_s": finish_s,
                "minimum_prop_gap_m": float(np.min(prop)),
                "minimum_person_gap_m": float(np.min(human)),
                "first_prop_below_5cm_s": first_prop,
                "first_person_below_5cm_s": first_person,
                "first_prop_below_5cm_stage": (
                    "move" if first_prop is not None and
                    first_prop < move_end_s else "wait" if
                    first_prop is not None else None)})
        worst_prop = min(x["minimum_prop_gap_m"] for x in candidate_paths)
        worst_person = min(x["minimum_person_gap_m"] for x in candidate_paths)
        results.append({"name": candidate["name"],
                        "distance_m": distance,
                        "target_xy_m": target.tolist(),
                        "actor_static_margin_m": candidate[
                            "actor_static_margin_m"],
                        "actor_approach_margin_m": candidate[
                            "actor_approach_margin_m"],
                        "owner_station_range_m": candidate[
                            "owner_station_range_m"],
                        "move_start_s": move_start_s,
                        "move_end_s": move_end_s,
                        "scripted_person_return_s": return_s,
                        "worst_prop_gap_m": worst_prop,
                        "worst_person_gap_m": worst_person,
                        "all_six_above_5cm": bool(
                            worst_prop >= PROVISIONAL_MARGIN_M and
                            worst_person >= PROVISIONAL_MARGIN_M),
                        "paths": candidate_paths})
        print(candidate["name"], distance,
              round(worst_prop, 3), round(worst_person, 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "all-candidate-20ms-gaps.npz"
    np.savez_compressed(raw, time_s=times, prop_gap_m=raw_prop,
                        person_gap_m=raw_person)
    canvas = Image.new("RGB", (1200, 900), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    x_min, x_max, y_min, y_max = .4, 2.45, -5.7, -4.05
    left, top, width, height = 105, 90, 1000, 690

    def screen(xy):
        return (round(left+(xy[0]-x_min)*width/(x_max-x_min)),
                round(top+(y_max-xy[1])*height/(y_max-y_min)))

    draw.text((100, 25),
              "V75: every tested waiting pose loses the 5 cm reserve",
              fill=(20, 28, 35), font=font)
    draw.rectangle((left, top, left+width, top+height),
                   outline=(55, 65, 75), width=2)
    for x_tick in np.arange(.5, 2.46, .25):
        x_pos, _ = screen((x_tick, y_min))
        draw.line((x_pos, top, x_pos, top+height),
                  fill=(220, 224, 227), width=1)
        draw.text((x_pos-12, top+height+8), f"{x_tick:.2f}",
                  fill=(40, 47, 54), font=small)
    for y_tick in np.arange(-5.5, -4.0, .25):
        _, y_pos = screen((x_min, y_tick))
        draw.line((left, y_pos, left+width, y_pos),
                  fill=(220, 224, 227), width=1)
        draw.text((left-62, y_pos-8), f"{y_tick:.2f}",
                  fill=(40, 47, 54), font=small)
    colors = {"outbound_control": (38, 130, 199),
              "inbound_holdout": (216, 144, 31),
              "new_west_south_turn": (145, 95, 194),
              "far_west_turn": (208, 58, 61)}
    for path, color in colors.items():
        curve = [screen(v72.v64.v60.v50.prop_at(
            t, checkpoint_s, v72.v64.PATHS[path]))
            for t in np.linspace(checkpoint_s+4.2,
                                 checkpoint_s+5.6, 90)]
        draw.line(curve, fill=color, width=4)
    for row in results:
        x_pos, y_pos = screen(row["target_xy_m"])
        gap_cm = 100*row["worst_prop_gap_m"]
        dot = (170, 37, 42) if gap_cm < -5 else (228, 86, 48)
        draw.ellipse((x_pos-9, y_pos-9, x_pos+9, y_pos+9),
                     fill=dot, outline=(30, 30, 30), width=2)
        draw.text((x_pos+12, y_pos-10),
                  f"{row['name']} {int(row['distance_m']*100)}: {gap_cm:+.1f}cm",
                  fill=(18, 25, 31), font=small)
    x_start, y_start = screen(start)
    draw.ellipse((x_start-13, y_start-13, x_start+13, y_start+13),
                 fill=(10, 15, 20), outline=(255, 255, 255), width=2)
    draw.text((x_start+16, y_start+8), "ape before move",
              fill=(10, 15, 20), font=small)
    draw.text((left+width//2-60, 835), "World X (m)",
              fill=(25, 30, 35), font=font)
    draw.text((left, 810), "Color lines: outward blue; inward orange; west-south purple; far-west red",
              fill=(45, 52, 58), font=small)
    plot = OUT / "candidate-wait-pose-screen.png"
    canvas.save(plot)
    report = {"schema": "robot-learning-lab.ape-wait-pose-feasibility.v1",
              "source_sha256": v72.v52.sha(__file__),
              "v60_report_sha256": v72.v52.sha(V60_REPORT),
              "v59_source_sha256": v72.v52.sha(
                  v72.v64.v60.v59.__file__),
              "v65_source_sha256": v72.v52.sha(v74.v65.__file__),
              "v74_report_sha256": v72.v52.sha(v74.OUT / "report.json"),
              "scene_sha256": v72.v52.sha(
                  runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "move_start_s": move_start_s,
              "watch_end_s": watch_end_s,
              "period_s": PERIOD_S,
              "provisional_margin_m": PROVISIONAL_MARGIN_M,
              "start_xy_m": start.tolist(),
              "candidates": results,
              "variant_names": [name for name, _, _ in VARIANTS],
              "supported_candidate_count": sum(
                  x["all_six_above_5cm"] for x in results),
              "raw": {"path": str(raw), "sha256": v72.v52.sha(raw)},
              "plot": {"path": str(plot), "sha256": v72.v52.sha(plot)},
              "scope": "Evaluator-only six-path future screen of all 16 V59 actor-eligible waiting poses, with 20ms kinematic full-body/person/prop gaps. Not an actor oracle, a new pose planner, fixture-sweep qualification or selected-body safety claim."}
    target = OUT / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target, report["supported_candidate_count"],
          "supported candidates")


if __name__ == "__main__":
    main()
