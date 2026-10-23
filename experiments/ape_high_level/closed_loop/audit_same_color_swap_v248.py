"""V248: sudden target-absent same-color prop swap after a confirmed track.

This is an adversarial optical diagnostic, not a physical swap simulation.
The actor sees only the rendered RGB, its saved recent appearance bank and a
remembered floor location. Object names/identity are evaluator-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from run import ROOT, Vision, iou, sha
from run_moved_props_v134 import MovedPropRoom
from run_temporal_identity_valid_stool_v247 import BoundedTrackBank


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v248"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V247 = ROOT / "artifacts/ape-closed-loop-cove50-v247/temporal-identity-valid-stool-live-audit.json"
V247_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v247"
YAW_RAD = (.2, .6, 1.0)
LIGHT_MULTIPLIERS = (.7, 1.0, 1.3)


def recent_features(seed: int, vision: Vision) -> tuple[list[np.ndarray], list[dict]]:
    branch = V247_ROOT / f"scene{seed}" / f"seed{seed}"
    trace = json.loads((branch / "trace.json").read_text())
    accepted = [event for event in trace["events"]
                if event.get("accepted_box") is not None and any(
                    proposal["box"] == event["accepted_box"] and
                    proposal.get("identity_basis") == "owner_confirmed" and
                    proposal.get("owner_similarity", 0.) >= .35
                    for proposal in event.get("proposals", []))]
    if len(accepted) < 2:
        raise AssertionError("Missing two original-owner-confirmed prototypes")
    selected = accepted[-2:]
    paths = list(branch.glob("frame-*.npz"))
    found = {}
    for path in paths:
        digest = sha(path)
        if digest in {row["actor_input_sha256"] for row in selected}:
            found[digest] = path
    if len(found) != 2:
        raise AssertionError("Cannot reconstruct final confirmed crops")
    features = []
    sources = []
    for event in selected:
        path = found[event["actor_input_sha256"]]
        with np.load(path) as frame:
            image = Image.fromarray(frame["rgb"])
        crop = image.crop(event["accepted_box"])
        features.append(np.asarray(vision.feature([crop])[0], dtype=float))
        sources.append({"frame_sha256": sha(path),
                        "box": event["accepted_box"],
                        "time_s": event["time_s"]})
    return features, sources


def room_for_swap(seed: int, location: np.ndarray, mode: str,
                  light: float, final_qpos: np.ndarray,
                  actual: np.ndarray, estimated: np.ndarray) -> MovedPropRoom:
    scene = next(row for row in json.loads(V203.read_text())["rows"]
                 if row["seed"] == seed)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                          "Parmesan": tuple(location)})
    room.reset()
    room.layout(seed, absent=True, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
    room.set_table_height(.37)
    room.model.body_pos[room.model.body("loop_stool").id] = (.84, .35, .20)
    room.model.light_diffuse[0] *= light
    other = room.model.geom("loop_Parmesan_geom").id
    target = room.model.geom("loop_TomatoSauce_geom").id
    if mode in ("same_red_material", "identical_replica"):
        room.model.geom_matid[other] = room.model.geom_matid[target]
    if mode == "identical_replica":
        room.model.geom_dataid[other] = room.model.geom_dataid[target]
    room.data.qpos[:] = final_qpos
    room.actual[:] = actual
    room.estimated[:] = estimated
    room.head = float(final_qpos[room.head_addr])
    room.time_s = 28.
    room._forward()
    return room


def main() -> None:
    if not V247.exists():
        raise FileNotFoundError("Run V247 first")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    owner = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    bank_sources = {}
    video_path = OUT / "same-color-swaps.mp4"
    with imageio.get_writer(video_path, fps=3, codec="libx264", quality=7) as video:
        for seed in (111, 113):
            branch = V247_ROOT / f"scene{seed}" / f"seed{seed}"
            trace = json.loads((branch / "trace.json").read_text())
            location = np.asarray(trace["final_actor_target_xy_m"], dtype=float)
            actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
            estimated = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
            final_frame = max(branch.glob("frame-*.npz"),
                              key=lambda path: int(path.stem.split("-")[1]))
            with np.load(final_frame) as saved:
                qpos = saved["actor_qpos"].copy()
            features, sources = recent_features(seed, vision)
            bank_sources[str(seed)] = sources
            for mode in ("ordinary_absent", "same_red_material",
                         "identical_replica"):
                for light in LIGHT_MULTIPLIERS:
                    room = room_for_swap(seed, location, mode, light,
                                         qpos, actual, estimated)
                    try:
                        lookalike = room.model.geom("loop_Parmesan_geom").id
                        gaps = []
                        for gid in range(room.model.ngeom):
                            name = room.model.geom(gid).name
                            if gid == lookalike or not name.startswith("loop_") or name == "loop_rug_visual":
                                continue
                            gaps.append((float(mujoco.mj_geomDistance(
                                room.model, room.data, lookalike, gid,
                                10., None)), name))
                        min_scene_gap, nearest = min(gaps)
                        if min_scene_gap < .03:
                            rows.append({"seed": seed, "mode": mode,
                                         "light_multiplier": light,
                                         "valid_scene": False,
                                         "minimum_scene_gap_m": round(min_scene_gap, 4),
                                         "nearest_geom": nearest})
                            continue
                        for yaw in YAW_RAD:
                            room.head = yaw
                            room._forward()
                            rgb, depth = room.capture("lower_rgb", depth=True)
                            camera = room.camera_pose_estimate("lower_rgb")
                            bank = BoundedTrackBank(owner, owner_hist)
                            bank.recent = [feature.copy() for feature in features]
                            bank.last_owner_confirmed_call = 0
                            bank.calls = 0
                            stem = f"seed{seed}-{mode}-light{light:.1f}-yaw{yaw:.1f}"
                            rgb_path = OUT / f"{stem}-rgb.png"
                            rgb.save(rgb_path)
                            result = bank.infer(vision, owner, rgb, depth, camera,
                                                location, None,
                                                OUT / f"{stem}-annotated.png")
                            truth_box = room.truth_box("Parmesan", "lower_rgb")
                            selected = next((proposal for proposal in result["proposals"]
                                             if proposal["box"] == result["accepted_box"]), None)
                            selected_iou = (0. if selected is None else
                                            iou(selected["box"], truth_box))
                            video.append_data(np.asarray(rgb))
                            rows.append({"seed": seed, "mode": mode,
                                         "light_multiplier": light,
                                         "yaw_rad": yaw, "valid_scene": True,
                                         "minimum_scene_gap_m": round(min_scene_gap, 4),
                                         "nearest_geom": nearest,
                                         "candidate_accepted": selected is not None,
                                         "lookalike_iou_evaluator_only":
                                             round(selected_iou, 4),
                                         "selected_identity_basis": None if selected is None else
                                             selected["identity_basis"],
                                         "selected_hue": None if selected is None else
                                             selected["owner_hue_intersection"],
                                         "selected_bank_similarity": None if selected is None else
                                             selected["similarity"],
                                         "rgb_sha256": sha(rgb_path),
                                         "proposals": result["proposals"]})
                    finally:
                        room.close()
    false_accepts = sum(row.get("candidate_accepted", False) and
                        row.get("lookalike_iou_evaluator_only", 0) >= .25
                        for row in rows)
    report = {"scope": "V248 revealed end-pose abrupt target-absent optical swap. The Parmesan mesh is placed at remembered target XY; ordinary texture, target-red material on distinct Parmesan geometry, and identical target mesh+material replica are compared at three yaws and three light levels. Final V247 last two owner-confirmed appearance prototypes are reconstructed from hashed saved frames. Each view starts immediately after a confirmed track (one-call continuation opportunity). Existing V247 detector/thresholds use only RGB and remembered XY; evaluator identity and collision geometry score afterward. Exact replica is intentionally unidentifiable without continuity through the swap or an external marker/owner input. Stationary optical diagnostic only.",
              "source_sha256": sha(SOURCE), "v247_sha256": sha(V247),
              "bank_sources": bank_sources, "rows": rows,
              "false_lookalike_acceptances": false_accepts,
              "video_sha256": sha(video_path)}
    (OUT / "same-color-swap-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({"views": sum("yaw_rad" in row for row in rows),
                      "invalid_cases": sum(row.get("valid_scene") is False for row in rows),
                      "false_lookalike_acceptances": false_accepts}, indent=2))


if __name__ == "__main__":
    main()
