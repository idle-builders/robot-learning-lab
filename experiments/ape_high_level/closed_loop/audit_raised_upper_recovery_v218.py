"""V218: upper-RGB owner identity screen from V217 raised-target stop poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_entry_color_v150 import descriptor
from run import ROOT, Vision, iou, locate_world, sha
from run_raised_target_v217 import RaisedCanRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v218"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V217 = ROOT / "artifacts/ape-closed-loop-cove50-v217/raised-target-live-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
ANGLES = (-1.2, -.6, 0., .6, 1.2)


def screen(seed: int, vision: Vision, exemplar: np.ndarray,
           owner_hist: np.ndarray) -> dict:
    scene = next(s for s in json.loads(V203.read_text())["rows"] if s["seed"] == seed)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    room = RaisedCanRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                          "Parmesan": tuple(positions["Parmesan"])})
    trace = json.loads((ROOT / f"artifacts/ape-closed-loop-cove50-v217/scene{seed}/seed{seed}/trace.json").read_text())
    last = trace["events"][-1]
    frames = sorted((ROOT / f"artifacts/ape-closed-loop-cove50-v217/scene{seed}/seed{seed}").glob("frame-*.npz"),
                    key=lambda path: int(path.stem.split("-")[1]))
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    with np.load(frames[-1]) as z:
        qpos = z["actor_qpos"].copy()
    rows = []
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.data.qpos[:] = qpos
        room.actual[:] = last["evaluator_actual_pose_xyyaw"]
        room.estimated[:] = last["estimated_pose_xyyaw"]
        room.head = float(qpos[room.head_addr])
        room.time_s = float(last["time_s"])
        room._forward()
        with imageio.get_writer(branch / "upper-search.mp4", fps=3,
                                codec="libx264", quality=7) as video:
            for angle in ANGLES:
                room.look(angle, "lower_rgb")
                rgb, depth = room.capture("upper_rgb", depth=True)
                pose = room.camera_pose_estimate("upper_rgb")
                truth = room.truth_box("TomatoSauce", "upper_rgb")
                proposals = vision.detect(rgb)
                features = (vision.feature([rgb.crop(p["box"]) for p in proposals])
                            if proposals else np.zeros((0, len(exemplar))))
                candidates = []
                for proposal, feature in zip(proposals, features):
                    score = float(feature @ exemplar)
                    color = descriptor(rgb, proposal["box"])
                    overlap = float(np.minimum(owner_hist,
                        np.asarray(color["hue_histogram"])).sum())
                    accepted = score >= .35 and overlap >= .70
                    world = locate_world(proposal["box"], pose, depth) if accepted else None
                    candidates.append({"box": proposal["box"],
                                       "owner_similarity": round(score, 4),
                                       "owner_hue_intersection": round(overlap, 4),
                                       "identity_only_gate": accepted,
                                       "dense_xyz_diagnostic_m": world,
                                       "evaluator_target_iou": round(iou(proposal["box"], truth), 4)})
                selected = max((r for r in candidates if r["identity_only_gate"]),
                               key=lambda r: r["owner_similarity"], default=None)
                annotated = rgb.copy()
                draw = ImageDraw.Draw(annotated)
                for row in candidates:
                    draw.rectangle(row["box"],
                                   outline=(0, 255, 0) if row is selected else (255, 200, 0),
                                   width=3)
                annotated.save(branch / f"upper-yaw-{angle:+.1f}.png")
                panel = Image.fromarray(np.concatenate((np.asarray(annotated),
                           np.asarray(room.capture_external())), axis=1))
                video.append_data(np.asarray(panel))
                rows.append({"yaw_rad": angle, "time_s": round(room.time_s, 3),
                             "evaluator_true_box": truth,
                             "candidates": candidates,
                             "identity_only_selected_box": None if selected is None else selected["box"],
                             "identity_only_selected_correct_evaluator_only":
                                 selected is not None and selected["evaluator_target_iou"] >= .25,
                             "image_sha256": sha(branch / f"upper-yaw-{angle:+.1f}.png")})
    finally:
        room.close()
    return {"seed": seed, "v217_status": trace["status"],
            "replay_last_frame_sha256": sha(frames[-1]),
            "rows": rows, "visible_views": sum(r["evaluator_true_box"] is not None for r in rows),
            "identity_only_correct_views": sum(r["identity_only_selected_correct_evaluator_only"] for r in rows),
            "video_sha256": sha(branch / "upper-search.mp4")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[107])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    rows = [screen(seed, vision, exemplar, owner_hist) for seed in args.seeds]
    result = {"scope": "V218 revealed static upper-RGB head-yaw screen replayed from V217 raised-can stop pose. Existing upper camera, frozen owner exemplar/DINO>=.35 and hue>=.70 identity-only rule; no old-XY gate so moved identity visibility can be diagnosed. Dense depth available only as a diagnostic location; evaluator segmentation/actual pose used for replay and post-score, never candidate selection. Head look is kinematic, no guard/native timing. No resumed route or success claim.",
              "source_sha256": sha(SOURCE), "v217_sha256": sha(V217),
              "rows": rows}
    (OUT / "raised-upper-search-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "visible", row["visible_views"],
              "identity_correct", row["identity_only_correct_views"], flush=True)


if __name__ == "__main__":
    main()
