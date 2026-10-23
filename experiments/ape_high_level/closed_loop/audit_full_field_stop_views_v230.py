"""V230: frozen RGB identity on full-field rerenders at live stop views."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_entry_color_v150 import descriptor
from run import H, ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import DINO_MIN, HUE_MIN
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v230"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V226 = ROOT / "artifacts/ape-closed-loop-cove50-v226"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
WIDTH = 896


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    rows = []
    for seed in (107, 111, 113):
        scene = next(row for row in scenes if row["seed"] == seed)
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                              "Parmesan": tuple(positions["Parmesan"])})
        try:
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
            room.set_table_height(.37)
            room._forward()
            room.renderer.close()
            room.renderer = mujoco.Renderer(room.model, width=WIDTH, height=H)
            branch = V226 / f"scene{seed}/seed{seed}"
            trace = json.loads((branch / "trace.json").read_text())
            events = [event for event in trace["events"]
                      if event["action"] == "reacquire_head_view" and
                      event["time_s"] >= (17. if seed == 107 else 9.)]
            for event in events:
                time_s = float(event["time_s"])
                frame_path = next(path for path in branch.glob("frame-*.npz")
                                  if abs(float(np.load(path)["capture_time_s"]) - time_s) < .001)
                with np.load(frame_path) as saved:
                    room.data.qpos[:] = saved["actor_qpos"]
                    room.estimated[:] = saved["estimated_pose_xyyaw"]
                    cropped_rgb = saved["rgb"]
                room.actual[:] = event["evaluator_actual_pose_xyyaw"]
                room.head = float(room.data.qpos[room.head_addr])
                room.time_s = time_s
                room._forward()
                rgb, _ = room.capture("lower_rgb", depth=False)
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                proposals = vision.detect(rgb)
                features = vision.feature([rgb.crop(p["box"]) for p in proposals]) if proposals else []
                candidates = []
                for proposal, feature in zip(proposals, features):
                    score = float(feature @ exemplar)
                    hue = float(np.minimum(owner, np.asarray(
                        descriptor(rgb, proposal["box"])["hue_histogram"])).sum())
                    if score >= DINO_MIN and hue >= HUE_MIN:
                        candidates.append((score, hue, proposal["box"]))
                selected = max(candidates, key=lambda x: x[0]) if candidates else None
                image = OUT / f"seed{seed}-time{time_s:.1f}-full-field.png"
                rgb.save(image)
                row = {"seed": seed, "time_s": round(time_s, 3),
                       "head_yaw_rad": round(room.head, 3),
                       "cropped_actor_frame_sha256": sha(frame_path),
                       "cropped_image_shape": list(cropped_rgb.shape),
                       "cropped_actor_accepted": event.get("accepted_box") is not None,
                       "full_field_image_sha256": sha(image),
                       "full_field_proposals": len(proposals),
                       "full_field_true_can_box_evaluator_only": truth,
                       "full_field_accepted_box": None if selected is None else selected[2],
                       "full_field_similarity": None if selected is None else round(selected[0], 4),
                       "full_field_hue": None if selected is None else round(selected[1], 4),
                       "full_field_correct_evaluator_only": bool(
                           selected is not None and truth is not None and
                           iou(selected[2], truth) >= .25)}
                rows.append(row)
                print(seed, time_s, row["cropped_actor_accepted"],
                      row["full_field_correct_evaluator_only"], flush=True)
        finally:
            room.close()
    result = {"scope": "V230 offline diagnostic on exact V226 end-stage selected-body actual poses and head yaws. Re-render static V203 room at 896x480/102.03x67deg pinhole Pi Wide field; run frozen RGB proposal/DINO/hue identity gate. Compare saved 640x480 actor acceptance at same action time. Evaluator actual pose and segmentation are used only for replay/scoring, not fed to live actor. No full-field live loop, camera calibration/distortion, or physical sensor behavior.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "owner_exemplar_sha256": sha(V149), "owner_hue_audit_sha256": sha(V150),
              "rows": rows,
              "summary": {"views": len(rows),
                          "cropped_actor_accepted_views": sum(r["cropped_actor_accepted"] for r in rows),
                          "full_field_correct_views": sum(r["full_field_correct_evaluator_only"] for r in rows)}}
    (OUT / "full-field-stop-view-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
