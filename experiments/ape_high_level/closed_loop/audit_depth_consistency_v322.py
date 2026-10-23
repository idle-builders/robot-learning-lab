"""Evaluator-only provenance for V321/V322's isolated rug depth spike.

Segmentation identifies the *cause* after the live actor decision. It never
feeds the actor's depth filter, map, route gate or owner identity decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v322"
SEEDS = (170, 172)


def trace_for(version: int, seed: int) -> tuple[dict, Path]:
    path = (ROOT / f"artifacts/ape-closed-loop-cove50-v{version}" /
            f"seed{seed}-present/seed{seed}-present/episode-trace.json")
    return json.loads(path.read_text()), path


def main() -> None:
    rows = []
    for seed in SEEDS:
        raw, raw_path = trace_for(321, seed)
        clean, clean_path = trace_for(322, seed)
        raw_frames = raw["recovery_inputs"][0]["head_inputs"]
        clean_frames = clean["recovery_inputs"][0]["head_inputs"]
        actual = next(e["evaluator_actual_pose_xyyaw"]
                      for e in reversed(raw["controller_events"])
                      if "evaluator_actual_pose_xyyaw" in e)
        room = HeldoutMovingRoom(case=tuple(raw["case"]),
                                 pose_source="command",
                                 branch=OUT / f"seed{seed}-present" /
                                 f"seed{seed}-present")
        try:
            room.layout(seed, absent=False, furnished=True)
            stool_id = room.model.geom("loop_stool_geom").id
            frame_rows = []
            for first, second in zip(raw_frames, clean_frames, strict=True):
                with np.load(ROOT / first["actor_frame_path"]) as original:
                    before = {key: original[key].copy()
                              for key in original.files}
                with np.load(ROOT / second["actor_frame_path"]) as filtered:
                    after = {key: filtered[key].copy()
                             for key in filtered.files}
                if (not np.array_equal(before["rgb"], after["rgb"]) or
                    not np.allclose(before["filtered_depth_m"],
                                    after["filtered_depth_m"],
                                    equal_nan=True)):
                    raise AssertionError("Paired actor input changed")
                room.time_s = float(before["capture_time_s"])
                room._update_moving_obstacle()
                room.actual[:] = actual
                room.head = first["yaw_rad"]
                room.data.qpos[:] = before["actor_qpos"]
                room._forward()
                room.renderer.enable_segmentation_rendering()
                room.renderer.update_scene(room.data, camera="lower_rgb")
                segmentation = room.renderer.render().copy()
                room.renderer.disable_segmentation_rendering()
                original_depth = before["filtered_depth_m"]
                mapped_depth = after["map_depth_m"]
                grid_y, grid_x = np.mgrid[2:original_depth.shape[0]:4,
                                          2:original_depth.shape[1]:4]
                prop = ((segmentation[grid_y, grid_x, 0] == stool_id) &
                        (segmentation[grid_y, grid_x, 1] ==
                         int(mujoco.mjtObj.mjOBJ_GEOM)))
                valid = np.isfinite(original_depth[grid_y, grid_x])
                lost = valid & ~np.isfinite(mapped_depth[grid_y, grid_x])
                frame_rows.append({
                    "yaw_rad": first["yaw_rad"],
                    "capture_time_s": first["capture_time_s"],
                    "raw_actor_frame_sha256": sha(
                        ROOT / first["actor_frame_path"]),
                    "filtered_actor_frame_sha256": sha(
                        ROOT / second["actor_frame_path"]),
                    "filter_rejected_sampled_pixels": second[
                        "depth_quality"]["rejected_isolated_pixels"],
                    "true_stool_valid_sampled_pixels_evaluator_only":
                        int((prop & valid).sum()),
                    "true_stool_rejected_sampled_pixels_evaluator_only":
                        int((prop & lost).sum())})
                if seed == 172 and first["yaw_rad"] == 1.2:
                    row, col = 426, 282
                    gid = int(segmentation[row, col, 0])
                    geom = room.model.geom(gid).name
                    if geom != "loop_rug_visual":
                        raise AssertionError("Limiting pixel source changed")
                    patch = original_depth[row - 2:row + 3,
                                           col - 2:col + 3]
                    critical = {"row": row, "col": col,
                                "evaluator_geom": geom,
                                "measured_depth_m": round(float(
                                    original_depth[row, col]), 5),
                                "neighbor_median_depth_m": round(float(
                                    np.nanmedian(patch)), 5),
                                "rejected_by_actor_filter": bool(
                                    np.isnan(mapped_depth[row, col]))}
            result = {"seed": seed, "frame_rows": frame_rows,
                      "v321_trace_sha256": sha(raw_path),
                      "v322_trace_sha256": sha(clean_path)}
            if seed == 172:
                result["limiting_pixel_evaluator_only"] = critical
            rows.append(result)
        finally:
            room.renderer.close()
    report = {"scope": "Post-hoc evaluator-only segmentation audit of the paired V321/V322 blocked-viewpoint head frames. Raw RGB and self-masked depth arrays are verified identical; only V322's map-depth rays differ. The segmentation labels explain whether an isolated rejected pixel belonged to the visual rug or the real moving stool. No segmentation, room pose or truth enters actor route decisions.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "depth-consistency-provenance.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], [(x["yaw_rad"],
                              x["true_stool_valid_sampled_pixels_evaluator_only"],
                              x["true_stool_rejected_sampled_pixels_evaluator_only"])
                             for x in row["frame_rows"]], flush=True)


if __name__ == "__main__":
    main()
