"""V270: selected Direct Frame in compact, structurally different room probes.

This is a scene and sensor preflight. It does not pass simulator identity,
segmentation or true pose to an actor, and it does not claim dynamic walking.
"""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from run import H, ROOT, W, Room, model_with_room, sha
from run_route_valid_moving_stool_v244 import RouteValidDynamicRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v270"
DESIGN = ROOT / "robot/ape-current-design.json"
SCENES = ((127, "open"), (128, "doorway"), (129, "screen"), (130, "raised"))
MODES = ("present", "absent", "identical_lookalike")


def selected_visual() -> Path:
    design = json.loads(DESIGN.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("The current mechanical design changed")
    visual = ROOT / design["evidence"]["visual_body"]
    if sha(visual) != design["evidence"]["visual_body_sha256"]:
        raise ValueError("Selected Direct Frame visual body hash changed")
    return visual


def structural_model(kind: str, side: int) -> mujoco.MjModel:
    visual = selected_visual()
    base = model_with_room(visual=visual)
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=visual.parent,
                                     delete=False) as stream:
        saved = Path(stream.name)
    try:
        mujoco.mj_saveLastXML(str(saved), base)
        root = ET.parse(saved).getroot()
        asset, world = root.find("asset"), root.find("worldbody")
        floor = next(x for x in asset.findall("texture")
                     if x.get("name") == "room_floor_texture")
        palette = {
            "open": ("0.79 0.74 0.64", "0.64 0.57 0.48"),
            "doorway": ("0.66 0.72 0.69", "0.43 0.54 0.52"),
            "screen": ("0.74 0.68 0.60", "0.58 0.51 0.45"),
            "raised": ("0.76 0.75 0.70", "0.58 0.60 0.61"),
        }
        floor.set("rgb1", palette[kind][0])
        floor.set("rgb2", palette[kind][1])
        # Geometric partition and freestanding screen give different optical
        # and route topology; neither is an actor map prior.
        if kind == "doorway":
            for name, y, halfy in (("north", .775, .455),
                                   ("south", -.775, .455)):
                ET.SubElement(world, "geom", name=f"loop_partition_{name}",
                              type="box", pos=f"0.78 {y} 0.52",
                              size=f"0.025 {halfy} 0.52",
                              rgba="0.52 0.61 0.56 1", group="0",
                              contype="0", conaffinity="0")
        if kind == "screen":
            ET.SubElement(world, "geom", name="loop_freestanding_screen",
                          type="box", pos=f"0.47 {side * .14} 0.38",
                          size="0.025 0.09 0.38", rgba="0.46 0.34 0.30 1",
                          group="0", contype="0", conaffinity="0")
        # Small wall art creates repeatable visual landmarks without changing
        # the collision field. It is intentionally not a semantic cue to item ID.
        for name, y, rgba in (("a", -.42, "0.76 0.34 0.28 1"),
                              ("b", .62, "0.28 0.47 0.68 1")):
            ET.SubElement(world, "geom", name=f"room_wall_art_{name}",
                          type="box", pos=f"1.301 {y} 0.64",
                          size="0.004 0.11 0.13", rgba=rgba, group="0",
                          contype="0", conaffinity="0")
        imposter = ET.SubElement(world, "body", name="loop_Imposter",
                                pos="10 0 0.052")
        ET.SubElement(imposter, "geom", name="loop_Imposter_geom",
                      type="mesh", mesh="loop_TomatoSauce",
                      material="loop_TomatoSauce_mat",
                      quat="0.70710678 0.70710678 0 0", group="0",
                      rgba="1 1 1 1", contype="0", conaffinity="0")
        saved.write_text(ET.tostring(root, encoding="unicode"))
        return mujoco.MjModel.from_xml_path(str(saved))
    finally:
        saved.unlink(missing_ok=True)


def target_xy(kind: str, side: int) -> tuple[float, float]:
    return {"open": (.70, side * .16),
            "doorway": (1.12, 0.),
            "screen": (.74, side * .14),
            "raised": (.40, side * .46)}[kind]


class DirectStructuralRoom(RouteValidDynamicRoom):
    def __init__(self, positions: dict | None = None, *, kind: str,
                 mode: str = "present", side: int = 1) -> None:
        self.moved_positions = {} if positions is None else positions
        self.kind, self.mode, self.side = kind, mode, side
        self.depth_mode = "ideal"
        Room.__init__(self, model=structural_model(kind, side))

    def capture(self, camera: str, *, depth: bool):
        rgb, ideal = Room.capture(self, camera, depth=depth)
        if camera != "lower_rgb" or ideal is None or self.depth_mode == "ideal":
            return rgb, ideal
        if self.depth_mode == "short":
            from audit_chest_short_depth_v223 import ideal_mask
            from audit_head_short_depth_v225 import SHORT
            return rgb, ideal_mask(ideal, SHORT)
        raise ValueError(f"Unknown depth mode: {self.depth_mode}")

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        side = 1 if seed % 2 else -1
        if side != self.side:
            raise ValueError("Scene/model side mismatch")
        target = target_xy(self.kind, side)
        target_z = .45 if self.kind == "raised" else .052
        self.positions["TomatoSauce"] = (10., 0.) if absent else target
        self.model.body_pos[self.model.body("loop_TomatoSauce").id] = (
            *(self.positions["TomatoSauce"]), target_z)
        self.model.body_pos[self.model.body("loop_Imposter").id] = (
            *(target if absent and self.mode == "identical_lookalike" else
              (10., 0.)), target_z)
        # Reconfigure the old movable furniture into a compact apartment-like
        # corner; the partition/screen, not centimeter jitter, drives novelty.
        fixture = {
            "crate": (.30, -side * .60, .09),
            "box": (1.08, side * .80, .14),
            "shelf": (1.16, -side * .60, .32),
            "stool": (1.17, side * .48, .20),
            "sofa": (10., 0., 0.), "lamp": (10., 0., 0.),
            "rug": (.56, 0., 0.),
            "table": (.40, side * .46, 0.),
        }
        for name, xyz in fixture.items():
            self.model.body_pos[self.model.body(f"loop_{name}").id] = xyz
        self.obstacle_xy = np.array([fixture[name][:2]
                                     for name in ("crate", "box", "shelf", "stool")])
        # V270 deliberately tests structural novelty with stationary fixtures;
        # the inherited moving-stool motion belongs to a later dynamic trial.
        self.motion_side = None
        self.set_table_height(.37)
        self._forward()


def make_room(seed: int, kind: str, mode: str) -> DirectStructuralRoom:
    room = DirectStructuralRoom(kind=kind, mode=mode,
                                side=1 if seed % 2 else -1)
    room.layout(seed, absent=mode != "present", furnished=True)
    return room


def clearance_path(kind: str, side: int) -> list[tuple[float, float]]:
    if kind == "doorway":
        return [(0., 0.), (.18, 0.), (.35, 0.), (.52, 0.), (.70, 0.), (.78, 0.)]
    if kind == "screen":
        return [(0., 0.), (.18, -side * .19), (.35, -side * .25),
                (.52, -side * .25)]
    if kind == "open":
        return [(0., 0.), (.15, 0.), (.30, 0.), (.36, 0.)]
    return [(0., 0.)]


def preflight(room: DirectStructuralRoom) -> dict:
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    start = full_body_clearance(room, obstacles, bodies)
    path = clearance_path(room.kind, room.side)
    witness = start
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0., 1., 21):
            room.actual[:2] = (1 - t) * np.asarray(a) + t * np.asarray(b)
            room._forward()
            row = full_body_clearance(room, obstacles, bodies)
            if row[0] < witness[0]:
                witness = row
    room.actual[:] = 0.
    room._forward()
    check_name = ("loop_TomatoSauce_geom" if room.mode == "present" else
                  "loop_Imposter_geom" if room.mode == "identical_lookalike" else
                  None)
    table_geoms = {room.model.geom(f"loop_table_{name}").id
                   for name in ("top", "leg_-1_-1", "leg_-1_1",
                                "leg_1_-1", "leg_1_1")}
    check = None if check_name is None else room.model.geom(check_name).id
    nearest = (None if check is None else min((float(mujoco.mj_geomDistance(
        room.model, room.data, check, other, 10., None)),
        room.model.geom(other).name) for other in obstacles
        if other != check and
        room.model.geom(other).name not in ("loop_TomatoSauce_geom",
                                            "loop_Imposter_geom") and
        (room.kind != "raised" or other not in table_geoms)))
    return {"start_body_clearance_m": round(start[0], 5),
            "path_min_body_clearance_m": round(witness[0], 5),
            "path_limiting_robot_geom": witness[1],
            "path_limiting_scene_geom": witness[2],
            "evaluator_fixed_yaw_path_xy_m": path,
            "visible_item_other_scene_gap_m": (None if nearest is None else
                                                round(nearest[0], 5)),
            "visible_item_nearest_other_scene_geom": (None if nearest is None
                                                       else nearest[1]),
            "route_witness_at_9cm": len(path) > 1 and witness[0] >= .09,
            "visible_item_other_scene_valid": (None if nearest is None else
                                                nearest[0] >= .025)}


def preview(room: DirectStructuralRoom, output: Path) -> dict:
    side = room.side
    with imageio.get_writer(output, fps=3, codec="libx264", quality=7) as video:
        for yaw in (-1.2, -.6, 0., .6, 1.2):
            room.head = yaw
            room._forward()
            rgb, _ = room.capture("lower_rgb", depth=False)
            external = room.capture_external()
            panel = Image.fromarray(np.concatenate(
                (np.asarray(rgb), np.asarray(external)), axis=1))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 29), fill=(0, 0, 0))
            draw.text((8, 6), f"Direct Frame | {room.kind} | {room.mode} | head {yaw:+.1f}",
                      fill=(255, 255, 255))
            video.append_data(np.asarray(panel))
    return {"video_sha256": sha(output), "frames": 5,
            "side": side}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, kind in SCENES:
        for mode in MODES:
            room = make_room(seed, kind, mode)
            try:
                audit = preflight(room)
                name = f"seed{seed}-{kind}-{mode}"
                video = OUT / f"{name}.mp4"
                video_info = preview(room, video)
                row = {"seed": seed, "kind": kind, "mode": mode,
                       "selected_body_source_sha256": sha(selected_visual()),
                       "body_nq": room.model.nq,
                       "body_joints": [room.model.joint(i).name
                                       for i in range(room.model.njnt)],
                       "model_ngeom": room.model.ngeom,
                       "target_xy_m_evaluator_only": target_xy(kind, room.side),
                       "target_support": "table" if kind == "raised" else "floor",
                       "preflight": audit, "preview": video_info}
                rows.append(row)
                print(name, "clearance", audit["path_min_body_clearance_m"],
                      "item-gap", audit["visible_item_other_scene_gap_m"], flush=True)
            finally:
                room.close()
    report = {"scope": "V270 predeclared seeds127-130; selected exact Direct Frame visual MJCF plus compact furnished open, transverse doorway, freestanding screen, and table-supported object scenes. Paired target present, ordinary absent, visually identical impostor. Evaluator-only 21-sample-per-edge fixed-yaw full-body route/scene validity, no actor exposure or controller claim. Previews are RGB + independent external camera, not an action rollout. Geometries are still stylized primitives and base dynamics are not transferred to the selected frame.",
              "source_sha256": sha(SOURCE), "design_sha256": sha(DESIGN),
              "selected_visual_sha256": sha(selected_visual()),
              "scenes": rows}
    (OUT / "structural-room-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
