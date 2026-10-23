"""V73: freeze a held-pose camera cue on the other authored prop routes.

The cue was chosen after V72, so this is a diagnostic extension, not a blind
holdout. It never receives evaluator labels and never dispatches motion.
"""

from __future__ import annotations

import json

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_deferred_lower_rgbd_pair as v72


ROOT, OUT = v72.ROOT, v72.ROOT / (
    "artifacts/ape-native-route-guard/v73-lower-rgbd-changed-routes")
V60_REPORT = ROOT / "artifacts/ape-native-route-guard/v60-deferred-owner-show/report.json"
PATHS = ("outbound_control", "inbound_holdout",
         "new_west_south_turn", "far_west_turn")
NEW_PATHS = ("inbound_holdout", "new_west_south_turn")


def cue(rows):
    """Actor-side held-camera reversal cue, fixed from the V72 pair."""
    values = [row["actor_lower_closer_depth_pixels"] for row in rows]
    alarms = []
    for index in range(4, len(rows)):
        preceding = values[index-3:index]
        if (preceding[0] > preceding[1] > preceding[2] and
                values[index] > 1000 and
                values[index] > 4*max(preceding[2], 1)):
            alarms.append({"watch_index": index,
                           "time_s": rows[index]["time_s"],
                           "lower_closer_depth_pixels": values[index],
                           "previous_three": preceding})
    return alarms


def main():
    source = v72.runner.original()
    summary = source["summary"]
    policy = json.loads(v72.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    v60 = json.loads(V60_REPORT.read_text())
    base = {row["path"]: row["run"] for row in v60["results"]}
    v72_report = json.loads((v72.OUT / "report.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in NEW_PATHS:
        rows, rgb, depth, mask = v72.render_path(
            source, summary, scenario, v60["checkpoint_s"],
            path, base[path])
        raw = OUT / f"{path}-rgbd.npz"
        np.savez_compressed(raw, time_s=np.array([
            x["time_s"] for x in rows]), rgb=rgb, depth=depth,
            evaluator_prop_mask=mask)
        results[path] = {"rows": rows,
                         "raw": {"path": str(raw),
                                 "sha256": v72.v52.sha(raw)},
                         "actor_camera_cue": cue(rows)}
        print(path, [(x["watch_index"],
                     x["actor_lower_closer_depth_pixels"])
                    for x in rows[6:13]], "cue", cue(rows), flush=True)
    for path in ("outbound_control", "far_west_turn"):
        original = v72_report["results"][path]
        results[path] = {"rows": original["rows"],
                         "raw": original["raw"],
                         "actor_camera_cue": cue(original["rows"])}

    video = OUT / "four-held-routes-lower-rgbd.mp4"
    frames = {path: np.load(results[path]["raw"]["path"])["rgb"]
              for path in PATHS}
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for index in v72.WATCH_INDICES:
            panels = []
            for path in PATHS:
                canvas = frames[path][index].copy()
                row = results[path]["rows"][index]
                cv2.rectangle(canvas, (0, 0), (640, 55), (15, 22, 27), -1)
                cv2.putText(canvas,
                    f"{path} t={row['time_s']:.2f} close={row['actor_lower_closer_depth_pixels']}",
                    (7, 33), cv2.FONT_HERSHEY_SIMPLEX, .54,
                    (240, 240, 240), 2)
                panels.append(canvas)
            writer.append_data(np.concatenate((
                np.concatenate(panels[:2], axis=1),
                np.concatenate(panels[2:], axis=1)), axis=0))
    report = {"schema": "robot-learning-lab.ape-lower-rgbd-changed-routes.v1",
              "source_sha256": v72.v52.sha(__file__),
              "v60_report_sha256": v72.v52.sha(V60_REPORT),
              "v72_report_sha256": v72.v52.sha(v72.OUT / "report.json"),
              "scene_sha256": v72.v52.sha(
                  v72.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "camera": {"height_m": v72.HEIGHT_M,
                         "fovy_deg": v72.FOVY_DEG,
                         "yaw_deg": v72.YAW_DEG,
                         "pitch_deg": v72.PITCH_DEG},
              "cue": "After three strictly falling frame-to-frame lower-image closer-depth counts, trigger when next count exceeds 1000 pixels and four times the last. Fixed after V72; stationary camera only; no action.",
              "results": results,
              "video": {"path": str(video),
                        "sha256": v72.v52.sha(video),
                        "frames": len(v72.WATCH_INDICES),
                        "shape": [960, 1280, 3]},
              "scope": "Two additional already-authored held prop paths plus the original V72 pair. No motion response or physical camera selection."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
