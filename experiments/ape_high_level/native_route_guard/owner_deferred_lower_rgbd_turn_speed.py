"""V74: frozen held-camera cue under faster and slower far-west turns.

The body remains at V61's west waiting pose. Prop timing varies as in V65,
but no range response or camera-triggered action is executed. This isolates
whether the V73 camera observation arrives before a modeled body breach.
"""

from __future__ import annotations

import json

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

import owner_deferred_lower_rgbd_changed_routes as v73
import owner_deferred_turn_speed_stress as v65


v72 = v73.v72
ROOT = v72.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v74-lower-rgbd-turn-speed"
V61_REPORT = ROOT / "artifacts/ape-native-route-guard/v61-far-west-turn-holdout/report.json"
VARIANTS = (("fast", 4.85), ("normal", 5.20), ("slow", 5.55))


def sampled_gap(source, summary, scenario, checkpoint_s, held_xy, finish_s):
    model = mujoco.MjModel.from_xml_path(str(
        v72.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    v72.v64.v51.v46.low_prop_geometry(v72.runner.geometry_at)(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    endpoint = v72.v64.PATHS["far_west_turn"]
    start_s, end_s = 24.31078, 25.81078
    rows = []
    for time_s in np.arange(start_s, end_s+.00001, .02):
        model.body_pos[model.body("audit_ape").id, :2] = held_xy
        model.body_pos[model.body("route_obstacle").id, :2] = \
            v65.prop_at(time_s, checkpoint_s, endpoint, finish_s)
        mujoco.mj_forward(model, data)
        rows.append({"time_s": round(float(time_s), 5),
                     "prop_gap_m": float(v72.runner.prop_gap(
                         model, data, held_xy))})
    breach = next((row["time_s"] for row in rows
                   if row["prop_gap_m"] < .05), None)
    return {"minimum_prop_gap_m": min(row["prop_gap_m"] for row in rows),
            "first_below_5cm_s": breach,
            "samples_20ms": rows}


def main():
    source = v72.runner.original()
    summary = source["summary"]
    policy = json.loads(v72.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    held_report = json.loads(V61_REPORT.read_text())
    run = held_report["run"]
    checkpoint_s = held_report["checkpoint_s"]
    OUT.mkdir(parents=True, exist_ok=True)
    results, frames = {}, {}
    for name, finish_s in VARIANTS:
        obstacle = lambda t, cp, endpoint, finish_s=finish_s: \
            v65.prop_at(t, cp, endpoint, finish_s)
        rows, rgb, depth, prop_mask = v72.render_path(
            source, summary, scenario, checkpoint_s, "far_west_turn",
            run, prop_at_fn=obstacle)
        gap = sampled_gap(source, summary, scenario, checkpoint_s,
                          np.asarray(rows[0]["ape_xy_m"]), finish_s)
        raw = OUT / f"{name}-rgbd.npz"
        np.savez_compressed(raw, time_s=np.asarray([
            row["time_s"] for row in rows]), rgb=rgb, depth=depth,
            evaluator_prop_mask=prop_mask)
        alarms = v73.cue(rows)
        results[name] = {"finish_after_checkpoint_s": finish_s,
                         "raw": {"path": str(raw),
                                 "sha256": v72.v52.sha(raw)},
                         "rows": rows,
                         "actor_camera_cue": alarms,
                         "evaluator_held_gap": gap}
        frames[name] = rgb
        print(name, "cue", alarms, "breach",
              gap["first_below_5cm_s"], "min",
              gap["minimum_prop_gap_m"], flush=True)
    video = OUT / "held-far-west-turn-speed-rgbd.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for index in v72.WATCH_INDICES:
            panels = []
            for name, _ in VARIANTS:
                canvas = frames[name][index].copy()
                row = results[name]["rows"][index]
                cv2.rectangle(canvas, (0, 0), (640, 55),
                              (15, 22, 27), -1)
                cv2.putText(canvas,
                    f"{name} t={row['time_s']:.2f} close={row['actor_lower_closer_depth_pixels']}",
                    (7, 33), cv2.FONT_HERSHEY_SIMPLEX, .58,
                    (240, 240, 240), 2)
                panels.append(canvas)
            writer.append_data(np.concatenate(panels, axis=1))
    report = {"schema": "robot-learning-lab.ape-lower-rgbd-turn-speed.v1",
              "source_sha256": v72.v52.sha(__file__),
              "v61_report_sha256": v72.v52.sha(V61_REPORT),
              "v65_source_sha256": v72.v52.sha(v65.__file__),
              "v73_report_sha256": v72.v52.sha(v73.OUT / "report.json"),
              "scene_sha256": v72.v52.sha(
                  v72.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "camera": {"height_m": v72.HEIGHT_M,
                         "fovy_deg": v72.FOVY_DEG,
                         "yaw_deg": v72.YAW_DEG,
                         "pitch_deg": v72.PITCH_DEG},
              "checkpoint_s": checkpoint_s,
              "cue_source": "V73 fixed stationary lower-depth reversal cue",
              "results": results,
              "video": {"path": str(video),
                        "sha256": v72.v52.sha(video),
                        "frames": len(v72.WATCH_INDICES),
                        "shape": [480, 1920, 3]},
              "scope": "Held historical proxy and exact V61 west pose; V65 scripted prop turn durations; no camera action, selected-body or physical safety claim."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
