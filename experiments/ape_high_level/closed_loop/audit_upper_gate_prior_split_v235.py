"""V235: frozen V234 route gate on earlier independent third-split rooms."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_origin_route_v152 as route
from audit_chest_short_depth_v223 import ideal_mask
from audit_upper_free_evidence_gate_v234 import UpperFreeEvidenceClearance
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom
import run_rgb_floor_target_v213 as rgb_floor


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v235"
FRAMES = OUT / "origin"
V192 = ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
V193 = ROOT / "artifacts/ape-closed-loop-cove50-v193/third-split-origin-audit.json"
V234 = ROOT / "artifacts/ape-closed-loop-cove50-v234/upper-free-evidence-route-audit.json"
SHORT = (.07, .50, 84., 58.)


def crate_mask(room: MovedPropRoom) -> np.ndarray:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera="lower_rgb")
    labels = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    gid = room.model.geom("loop_crate_geom").id
    return ((labels[:, :, 0] == gid) &
            (labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V192.read_text())["rows"]
    entries = json.loads(V193.read_text())["rows"]
    rows = []
    old_clearance = route.TwoHeightClearance
    old_floor_frames = rgb_floor.V204_FRAMES
    rgb_floor.V204_FRAMES = ROOT / "artifacts/ape-closed-loop-cove50-v193"
    try:
        for scene in scenes:
            if scene["label"] == "invalid_object_separation":
                continue
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            entry = next(r for r in entries if r["seed"] == seed and
                         not r["absent"] and r["light_multiplier"] == 1.)
            room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                                  "Parmesan": tuple(positions["Parmesan"])})
            try:
                room.reset()
                room.layout(seed, absent=False, furnished=True)
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40, (1 if seed % 2 else -1) * .46, 0.)
                room.set_table_height(.37)
                room._forward()
                counts = []
                for view in entry["rows"]:
                    yaw = float(view["yaw_command_rad"])
                    name = f"lower-angle-{yaw:+.1f}.npz"
                    original = (ROOT / "artifacts/ape-closed-loop-cove50-v193" /
                                f"seed{seed}-light1.00-present" / name)
                    with np.load(original) as z:
                        values = {key: z[key].copy() for key in z.files}
                    room.data.qpos[:] = values["actor_qpos"]
                    room.actual[:] = 0.
                    room.head = float(room.data.qpos[room.head_addr])
                    room._forward()
                    mask = crate_mask(room)
                    short = ideal_mask(values["depth_m"], SHORT)
                    hole = short.copy()
                    hole[mask] = np.nan
                    for label, depth in (("ideal_short", short),
                                         ("crate_hole_short", hole)):
                        branch = FRAMES / label / f"seed{seed}-light1.00-present"
                        branch.mkdir(parents=True, exist_ok=True)
                        output = branch / name
                        np.savez_compressed(output, **{**values, "depth_m": depth})
                    counts.append({"yaw_rad": yaw,
                                   "short_crate_pixels_removed": int(
                                       (mask & np.isfinite(short)).sum())})
            finally:
                room.close()
            variant = {"variant": f"seed{seed}-third-split-upper-evidence",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            choices = {}
            for label in ("ideal_short", "crate_hole_short"):
                for gate in (False, True):
                    route.TwoHeightClearance = (
                        UpperFreeEvidenceClearance if gate else old_clearance)
                    chosen = route.screen(seed, variant, rgb_floor.floor_entry(entry),
                                          frame_root=FRAMES / label,
                                          light_multiplier=1.)
                    choices[f"{label}_{'gated' if gate else 'baseline'}"] = chosen
                    print(seed, label, gate, chosen["status"],
                          chosen.get("selected_kind"),
                          (chosen.get("evaluator_selected_path") or {}).get(
                              "min_full_scene_m"), flush=True)
            rows.append({"seed": seed, "geometry_label": scene["label"],
                         "depth_counts": counts, "choices": choices})
    finally:
        route.TwoHeightClearance = old_clearance
        rgb_floor.V204_FRAMES = old_floor_frames
    result = {"scope": "V235 retrospective independent split check of frozen V234 necessary upper-free-evidence route gate. All seven V192 geometry-valid generous/tight seeds and V193 frozen owner RGB origin frames, ideal short 7-50cm lower-head depth versus evaluator-generated all-crate depth holes, each with ungated/gated same planner and virtual 4/8/11cm scans. Evaluator segmentation only generates observation holes and scores selected route, never actor input. Earlier split is independent of V234 tuning but not freshly held out for the overall project. No continuous action or physical sensor.",
              "source_sha256": sha(SOURCE), "v192_sha256": sha(V192),
              "v193_sha256": sha(V193), "v234_sha256": sha(V234),
              "rows": rows}
    (OUT / "upper-free-evidence-third-split-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
