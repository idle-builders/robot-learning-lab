"""Pilot a 50 cm ape's camera sightlines in an upstream RoboCasa kitchen.

This is a scene and sensor-geometry probe, not a closed-loop ape episode. The
camera follows a scripted aisle path; no walking policy, object detector, or
planner receives simulator labels. Two identical textured groceries let the
evaluator measure visibility exactly.

Run with the isolated RoboCasa environment described in the accompanying
research note. Assets and upstream checkouts stay in the ignored .cache tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-kitchen-pilot"
HOPE = ROOT / ".cache/ape-high-level/hope_meshes_eval"
WIDTH, HEIGHT = 640, 480


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_can(world: ET.Element, name: str, xyz: tuple[float, float, float]) -> None:
    body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, xyz)))
    ET.SubElement(body, "geom", name=name + "_geom", type="mesh",
                  mesh="probe_tomato_sauce", material="probe_tomato_sauce_mat",
                  quat="0.70710678 0.70710678 0 0", group="1",
                  contype="0", conaffinity="0")


def scene(*, style_id: int = 1) -> mujoco.MjModel:
    arena = KitchenArena(layout_id=1, style_id=style_id,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.asset, "mesh", name="probe_tomato_sauce",
                  file=str(HOPE / "TomatoSauce.obj"), scale="0.01 0.01 0.01")
    ET.SubElement(task.asset, "texture", name="probe_tomato_sauce_tex", type="2d",
                  file=str(HOPE / "TomatoSauce.png"))
    ET.SubElement(task.asset, "material", name="probe_tomato_sauce_mat",
                  texture="probe_tomato_sauce_tex", specular="0.1")
    # Identical textured groceries: one behind the adult-height counter, the
    # other lowered to ape eye height as a human-show visibility proxy.
    add_can(task.worldbody, "counter_can", (3.4, -0.25, 0.975))
    add_can(task.worldbody, "shown_can", (1.65, -1.15, 0.43))
    for height_name, z in (("chest", 0.30), ("head", 0.46)):
        ET.SubElement(task.worldbody, "camera", name=height_name,
                      pos=f"2.75 -2.3 {z}", xyaxes="1 0 0 0 0 1", fovy="70")
    return task.get_model()


def capture(renderer: mujoco.Renderer, data: mujoco.MjData,
            camera: str, can_ids: dict[str, int]
            ) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, list[int] | None]]:
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    visible = {name: int(np.count_nonzero(seg[:, :, 0] == geom_id))
               for name, geom_id in can_ids.items()}
    boxes = {}
    for name, geom_id in can_ids.items():
        ys, xs = np.nonzero(seg[:, :, 0] == geom_id)
        boxes[name] = ([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
                       if len(xs) else None)
    return rgb, depth, visible, boxes


def panel(rgb: np.ndarray, depth: np.ndarray, label: str,
          visible: dict[str, int]) -> Image.Image:
    # The visualization clips depth at 3 m; metrics use unmodified simulator
    # depth and segmentation. A dark patch is near, a light patch is far.
    depth_vis = (np.clip(depth / 3.0, 0, 1) * 255).astype(np.uint8)
    depth_rgb = np.repeat(depth_vis[:, :, None], 3, axis=2)
    image = Image.new("RGB", (2 * WIDTH, HEIGHT + 40), "#14191d")
    image.paste(Image.fromarray(rgb), (0, 40))
    image.paste(Image.fromarray(depth_rgb), (WIDTH, 40))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), label, fill="white")
    draw.text((WIDTH + 12, 10), "metric depth, white >= 3 m", fill="white")
    draw.text((WIDTH - 258, 10), f"pixels: shown {visible['shown_can']}, counter {visible['counter_can']}",
              fill="white")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = scene()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    can_ids = {name: model.geom(name + "_geom").id for name in ("shown_can", "counter_can")}
    cameras = {name: model.camera(name).id for name in ("head", "chest")}
    records: list[dict] = []
    frames: list[np.ndarray] = []
    # One slow aisle translation; the scripted path is an eye-position survey,
    # not an ape locomotion claim. Both cameras see the exact same world state.
    for step in range(48):
        x = 1.05 + 3.30 * step / 47
        rows: list[Image.Image] = []
        sample: dict = {"step": step, "camera_x_m": round(x, 5),
                        "visible_pixels": {}, "evaluator_boxes": {}}
        for name in ("head", "chest"):
            model.cam_pos[cameras[name], 0] = x
            mujoco.mj_forward(model, data)
            rgb, depth, visible, boxes = capture(renderer, data, name, can_ids)
            sample["visible_pixels"][name] = visible
            sample["evaluator_boxes"][name] = boxes
            rows.append(panel(rgb, depth, f"50 cm ape {name} camera: {x:.2f} m along aisle", visible))
            if step % 4 == 0 or step in (35, 47):
                Image.fromarray(rgb).save(out / f"step{step:02d}-{name}.png")
        composite = Image.new("RGB", (2 * WIDTH, 2 * (HEIGHT + 40)))
        composite.paste(rows[0], (0, 0))
        composite.paste(rows[1], (0, HEIGHT + 40))
        frames.append(np.asarray(composite))
        records.append(sample)
    renderer.close()
    video = out / "ape-eye-height-kitchen-survey.mp4"
    imageio.mimsave(video, frames, fps=10, codec="libx264", quality=7)
    report = {
        "scope": "scripted camera and sensor-geometry survey; no ape control or closed-loop plan",
        "upstream": {"robocasa_commit": "456174f62b89b8fca99eaaf33949c29fec9cfc2a",
                     "robosuite_commit": "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"},
        "source_sha256": sha256(Path(__file__)),
        "room": {"layout_id": 1, "style_id": 1, "fixtures": 44, "bodies": model.nbody,
                 "geoms": model.ngeom},
        "cameras": {"head": {"z_m": 0.46, "fovy_deg": 70},
                    "chest": {"z_m": 0.30, "fovy_deg": 70}},
        "targets": {"shown_can_xyz_m": [1.65, -1.15, 0.43],
                    "counter_can_xyz_m": [3.4, -0.25, 0.975],
                    "mesh": "HOPE TomatoSauce", "mesh_sha256": sha256(HOPE / "TomatoSauce.obj"),
                    "texture_sha256": sha256(HOPE / "TomatoSauce.png")},
        "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
        "summary": {name: {target: {"any_pixel_frames": sum(r["visible_pixels"][name][target] > 0 for r in records),
                                     "at_least_100_pixels_frames": sum(r["visible_pixels"][name][target] >= 100 for r in records),
                                     "max_pixels": max(r["visible_pixels"][name][target] for r in records)}
                           for target in can_ids} for name in cameras},
        "samples": records,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"video": str(video), "frames": len(frames),
                      "shown_visible_frames": {name: sum(r["visible_pixels"][name]["shown_can"] > 0 for r in records)
                                               for name in cameras},
                      "counter_visible_frames": {name: sum(r["visible_pixels"][name]["counter_can"] > 0 for r in records)
                                                 for name in cameras}}, indent=2))


if __name__ == "__main__":
    main()
