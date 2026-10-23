"""Probe owner-show to floor-find images on the selected Cove-50 viewpoints.

This mounts a virtual low-chest view on the pinned Cove-50 visual body and
renders textured HOPE objects. The simulated owner click is chosen from an
evaluator mask, but SAM 2 receives RGB plus that pixel only. Search receives
fresh RGB only: generic Grounding DINO proposals and DINOv2 crop similarity.
Evaluator masks score decisions afterward and provide an oracle-crop upper
bound. No physical optics, owner, qualified body turn or identity claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from cove_view_alternatives import DESIGN, ROOT, SHELL_REPORT, SHELL_VISUAL, sha256, xyaxes


MESHES = ROOT / ".cache/ape-high-level/hope_meshes_eval"
OBJECTS = ("TomatoSauce", "Parmesan", "CreamCheese")
CAMERAS = ("lower_rgb", "chest_low50_x45")
WIDTH, HEIGHT = 640, 480
SHOW_X, SHOW_Z = 0.35, 0.55
LAYOUTS = {
    "center_and_sides": {
        "TomatoSauce": (0.25, 0.0),
        "Parmesan": (0.40, -0.14),
        "CreamCheese": (0.40, 0.14),
    },
    "swapped": {
        "TomatoSauce": (0.38, 0.12),
        "Parmesan": (0.25, 0.0),
        "CreamCheese": (0.45, -0.16),
    },
    "tomato_absent": {
        "TomatoSauce": (10.0, 0.0),
        "Parmesan": (0.30, -0.08),
        "CreamCheese": (0.42, 0.12),
    },
    "parmesan_absent": {
        "TomatoSauce": (0.30, -0.08),
        "Parmesan": (10.0, 0.0),
        "CreamCheese": (0.42, 0.12),
    },
    "cream_absent": {
        "TomatoSauce": (0.30, -0.08),
        "Parmesan": (0.42, 0.12),
        "CreamCheese": (10.0, 0.0),
    },
}


def source_checks() -> dict:
    design = json.loads(DESIGN.read_text())
    shell = json.loads(SHELL_REPORT.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("Current mechanical reference changed")
    if sha256(ROOT / design["generator"]["path"]) != design["generator"]["sha256"]:
        raise ValueError("Selected CAD generator changed")
    if shell["source_visual_body_sha256"] != design["evidence"]["visual_body_sha256"]:
        raise ValueError("Saved shell is not based on the selected visual body")
    if sha256(SHELL_VISUAL) != shell["shell_visual_body_sha256"]:
        raise ValueError("Saved shell visual changed")
    for name in OBJECTS:
        for extension in ("obj", "png"):
            if not (MESHES / f"{name}.{extension}").exists():
                raise FileNotFoundError(MESHES / f"{name}.{extension}")
    return {"design_id": design["id"],
            "selected_source_sha256": design["generator"]["sha256"],
            "shell_visual_sha256": shell["shell_visual_body_sha256"],
            "hope_mesh_sha256": {name: sha256(MESHES / f"{name}.obj") for name in OBJECTS},
            "model_weights_sha256": {name: sha256(
                ROOT / ".cache/ape-high-level/models" / name / "model.safetensors")
                for name in ("sam2-hiera-tiny", "grounding-dino-tiny", "dinov2-small")}}


def build_model() -> mujoco.MjModel:
    xml = ET.parse(SHELL_VISUAL)
    root = xml.getroot()
    torso = next(b for b in root.iter("body") if b.get("name") == "torso")
    ET.SubElement(torso, "camera", name="chest_low50_x45",
                  pos="0.045 0 0.034", xyaxes=xyaxes(-50), fovy="67")
    asset = root.find("asset")
    world = root.find("worldbody")
    # The viewer hides its floor and defaults new visual geoms to alpha zero.
    # Override both in this experiment-local XML so RGB is actually inspectable.
    floor = next(g for g in world.findall("geom") if g.get("name") == "floor")
    floor.set("group", "0")
    for name in OBJECTS:
        ET.SubElement(asset, "mesh", name=f"eval_{name}",
                      file=str(MESHES / f"{name}.obj"), scale="0.01 0.01 0.01")
        ET.SubElement(asset, "texture", name=f"eval_{name}_tex", type="2d",
                      file=str(MESHES / f"{name}.png"))
        ET.SubElement(asset, "material", name=f"eval_{name}_mat",
                      texture=f"eval_{name}_tex", specular="0.1")
        body = ET.SubElement(world, "body", name=f"eval_{name}", pos="10 0 0.052")
        ET.SubElement(body, "geom", name=f"eval_{name}_geom", type="mesh",
                      mesh=f"eval_{name}", material=f"eval_{name}_mat",
                      quat="0.70710678 0.70710678 0 0", group="0",
                      rgba="1 1 1 1",
                      contype="0", conaffinity="0")
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=SHELL_VISUAL.parent,
                                     delete=False) as temporary:
        temporary.write(ET.tostring(root, encoding="unicode"))
        path = Path(temporary.name)
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink()


class CoveScene:
    def __init__(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=WIDTH, height=HEIGHT)

    def place(self, positions: dict[str, tuple[float, float, float]],
              headings: dict[str, float] | None = None) -> None:
        for name in OBJECTS:
            xyz = positions.get(name, (10.0, 0.0, 0.052))
            body = self.model.body(f"eval_{name}")
            body.pos[:] = xyz
            angle = (headings or {}).get(name, 0.0)
            body.quat[:] = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
        mujoco.mj_forward(self.model, self.data)

    def image(self, camera: str) -> Image.Image:
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        return Image.fromarray(self.renderer.render().copy())

    def view_pose(self, camera: str) -> dict:
        camera_id = self.model.camera(camera).id
        return {"camera_world_xyz_m": self.data.cam_xpos[camera_id].tolist(),
                "camera_world_rotation": self.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
                "body_world_xyz_m": self.data.qpos[:3].tolist(),
                "body_world_quat_wxyz": self.data.qpos[3:7].tolist(),
                "pose_source": "exact_visual_model_fk_not_physical_estimator",
                "calibration_id": "virtual_640x480_fovy67_v1"}

    def truth_box(self, name: str, camera: str) -> list[int] | None:
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        segment = self.renderer.render().copy()
        gid = self.model.geom(f"eval_{name}_geom").id
        ys, xs = np.where((segment[:, :, 0] == gid) &
                          (segment[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def close(self) -> None:
        self.renderer.close()


def intersection_over_union(a: list[float], b: list[float] | None) -> float:
    if b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def capture(scene: CoveScene, output: Path) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    teach, search = {}, {}
    for name in OBJECTS:
        scene.place({name: (SHOW_X, 0.0, SHOW_Z)})
        image = scene.image("upper_rgb")
        path = output / f"teach-{name}.png"
        image.save(path)
        truth = scene.truth_box(name, "upper_rgb")
        teach[name] = {"path": str(path), "rgb_sha256": sha256(path),
                       "view_pose": scene.view_pose("upper_rgb"),
                       "truth_box_evaluator_only": truth,
                       "truth_bbox_area_px_evaluator_only": 0 if truth is None else
                           (truth[2] - truth[0]) * (truth[3] - truth[1])}
    for layout, xy in LAYOUTS.items():
        scene.place({name: (x, y, 0.052) for name, (x, y) in xy.items()},
                    {name: 0.15 * i for i, name in enumerate(OBJECTS)})
        for camera in CAMERAS:
            image = scene.image(camera)
            path = output / f"find-{layout}-{camera}.png"
            image.save(path)
            search[(layout, camera)] = {
                "path": str(path), "rgb_sha256": sha256(path),
                "view_pose": scene.view_pose(camera),
                "truth_boxes_evaluator_only": {
                    name: scene.truth_box(name, camera) for name in OBJECTS}}
    return teach, search


def show_height_screen(scene: CoveScene) -> list[dict]:
    rows = []
    for height in (0.45, 0.50, 0.55, 0.63):
        scene.place({"TomatoSauce": (SHOW_X, 0.0, height)})
        box = scene.truth_box("TomatoSauce", "upper_rgb")
        rows.append({"object_center_m": [SHOW_X, 0.0, height],
                     "visible_box": box,
                     "touches_image_bottom": bool(box and box[3] == HEIGHT)})
    return rows


def evaluate_rgb(teach: dict, search: dict) -> dict:
    # Import the existing released-model path without modifying its experiment.
    import sys
    module_dir = ROOT / "experiments/ape_high_level/teach_find"
    sys.path.insert(0, str(module_dir))
    try:
        from active_scan import Vision
    finally:
        sys.path.pop(0)
    planner_dir = ROOT / "experiments/ape_high_level/planner"
    sys.path.insert(0, str(planner_dir))
    try:
        from turn_loop import Turn
    finally:
        sys.path.pop(0)
    vision = Vision("sam2-click")
    exemplars = {}
    for name, row in teach.items():
        truth = row["truth_box_evaluator_only"]
        if truth is None:
            row["teach_status"] = "out_of_view"
            continue
        click = [(truth[0] + truth[2]) // 2, (truth[1] + truth[3]) // 2]
        image = Image.open(row["path"]).convert("RGB")
        selected, selection = vision.select_click(image, click, [])
        row["owner_click_emulator_px"] = click
        row["selected_rgb_box"] = selected
        row["selection_info"] = selection
        row["selected_iou_evaluator_only"] = intersection_over_union(selected, truth) if selected else 0
        if selected is not None:
            exemplars[name] = vision.feature([image.crop(selected)])[0]
    rows = []
    for (layout, camera), frame in search.items():
        image = Image.open(frame["path"]).convert("RGB")
        proposals = vision.detect(image)
        crops = [image.crop(tuple(round(v) for v in p["box"])) for p in proposals]
        features = vision.feature(crops) if crops else np.empty((0, 384))
        oracle = [(name, box) for name, box in frame["truth_boxes_evaluator_only"].items()
                  if box is not None]
        oracle_features = vision.feature([image.crop(box) for _, box in oracle]) if oracle else None
        for name in OBJECTS:
            truth = frame["truth_boxes_evaluator_only"][name]
            query = exemplars.get(name)
            similarities = features @ query if query is not None and len(features) else np.array([])
            chosen = int(np.argmax(similarities)) if len(similarities) else None
            oracle_scores = oracle_features @ query if query is not None and oracle_features is not None else None
            oracle_choice = oracle[int(np.argmax(oracle_scores))][0] if oracle_scores is not None else None
            chosen_box = proposals[chosen]["box"] if chosen is not None else None
            turn = Turn(name, floor_camera=camera)
            teach_action = turn.teach(name_source="text_control", name_healthy=True,
                                      frame_id=f"teach:{name}", crop_confirmed=query is not None)
            if query is None:
                first_action = teach_action
            else:
                turn.find()
                first_action = turn.observe(
                    frame_id=f"{layout}:{camera}", camera=camera,
                    captured_at_ms=1000, age_ms=0, image_healthy=True,
                    view_pose_healthy=True, candidate_box=chosen_box)
            rows.append({
                "layout": layout, "camera": camera, "name": name,
                "target_visible_evaluator_only": truth is not None,
                "target_box_evaluator_only": truth,
                "proposal_count": len(proposals),
                "best_target_overlap_evaluator_only": max(
                    (intersection_over_union(p["box"], truth) for p in proposals), default=0.0),
                "chosen_box": chosen_box,
                "chosen_similarity": float(similarities[chosen]) if chosen is not None else None,
                "chosen_overlap_evaluator_only": intersection_over_union(
                    proposals[chosen]["box"], truth) if chosen is not None else 0.0,
                "controller_first_action": first_action["action"],
                "oracle_crop_top_class_evaluator_only": oracle_choice,
                "oracle_crop_top_similarity_evaluator_only":
                    float(np.max(oracle_scores)) if oracle_scores is not None else None,
            })
        print(f"{layout}/{camera}: {len(proposals)} generic proposals", flush=True)
    return {"teach": teach, "rows": rows,
            "summary": {camera: {
                "visible_present": sum(r["target_visible_evaluator_only"] for r in rows
                                       if r["camera"] == camera and
                                       LAYOUTS[r["layout"]][r["name"]][0] < 5),
                "present_trials": sum(LAYOUTS[r["layout"]][r["name"]][0] < 5 for r in rows
                                      if r["camera"] == camera),
                "chosen_right_when_present": sum(
                    r["chosen_overlap_evaluator_only"] >= 0.25 for r in rows
                    if r["camera"] == camera and LAYOUTS[r["layout"]][r["name"]][0] < 5),
                "would_suggest_absent_distractor": sum(
                    r["chosen_box"] is not None for r in rows
                    if r["camera"] == camera and LAYOUTS[r["layout"]][r["name"]][0] >= 5),
                "controller_actions": {
                    action: sum(r["controller_first_action"] == action for r in rows
                                if r["camera"] == camera)
                    for action in sorted({r["controller_first_action"] for r in rows
                                          if r["camera"] == camera})},
            } for camera in CAMERAS}}


def run(output: Path, *, geometry_only: bool) -> dict:
    evidence = source_checks()
    scene = CoveScene()
    try:
        heights = show_height_screen(scene)
        teach, search = capture(scene, output)
    finally:
        scene.close()
    result = {**evidence, "views": {"show_height_screen": heights, "teach": teach,
                                   "search": {f"{layout}/{camera}": value
                                              for (layout, camera), value in search.items()}},
              "scope": "Pinned selected-body/saved-shell visual MJCF, virtual chest view, "
                       "textured released HOPE object assets. Owner click is evaluator-chosen, "
                       "then SAM 2 uses RGB. Search uses generic RGB proposals and crop "
                       "features; evaluator truth never enters the RGB model or turn. The shown "
                       "object floats without an owner hand; fixed base and simple synthetic "
                       "background. No physical sensor or qualified turn."}
    if not geometry_only:
        result["rgb_model"] = evaluate_rgb(teach, search)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    result = run(args.output, geometry_only=args.geometry_only)
    print(json.dumps(result.get("rgb_model", {}).get("summary", {
        "teach_boxes": {name: row["truth_box_evaluator_only"]
                        for name, row in result["views"]["teach"].items()},
        "search_boxes": {key: row["truth_boxes_evaluator_only"]
                         for key, row in result["views"]["search"].items()},
    }), indent=2))


if __name__ == "__main__":
    main()
