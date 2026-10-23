"""Render retained supported episodes without running or modifying their physics."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.visuals import visual_geometry

import mujoco
import numpy as np
from PIL import Image


def review(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sections, evidence = [], []
    for source in map(Path, runs):
        report = json.loads((source / "report.json").read_text())
        if report.get("fixture") != "supported":
            raise ValueError("this viewer requires a supported episode trace")
        rows = [
            json.loads(s)
            for s in (source / "trajectory.jsonl").read_text().splitlines()
        ]
        name = report["robot"]
        dest = output / name
        dest.mkdir()
        model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
        fidelity = visual_geometry(model)
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        base = json.loads((source / "interfaces.json").read_text())["base_height_m"]
        camera.lookat[:] = [0.04, 0.03, base + 0.02]
        camera.distance, camera.azimuth, camera.elevation = 0.62, 135, -22
        snapshots = []
        camera_coverage = []
        with mujoco.Renderer(model, width=800, height=600) as renderer:
            for label, t in [
                ("grasp", 4.5),
                ("lift", 6.5),
                ("transfer", 9.0),
                ("settled", rows[-1]["joint_state"]["time_s"]),
            ]:
                row = min(rows, key=lambda r: abs(r["joint_state"]["time_s"] - t))
                mujoco.mj_resetData(model, data)
                for joint, q in zip(
                    row["joint_state"]["names"], row["joint_state"]["position_rad"]
                ):
                    data.qpos[model.joint(joint).qposadr[0]] = q
                adr = model.joint("task_block_free").qposadr[0]
                state = row["task_state"]
                data.qpos[adr : adr + 3] = state["block_position_m"]
                data.qpos[adr + 3 : adr + 7] = state["block_quaternion_wxyz"]
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(
                    data.body("task_block").xpos, state["block_position_m"], atol=1e-10
                )
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(dest / (label + ".png"))
                renderer.update_scene(data, camera="head_camera")
                Image.fromarray(renderer.render()).save(dest / (label + "-head.png"))
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera="head_camera")
                segmentation = renderer.render()
                mask = (segmentation[:, :, 0] == model.geom("task_block_geom").id) & (
                    segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
                )
                camera_coverage.append(
                    {
                        "phase": label,
                        "time_s": row["joint_state"]["time_s"],
                        "block_visible_pixels": int(np.count_nonzero(mask)),
                        "image_size": [800, 600],
                        "scope": "renderer object IDs, not vision inference",
                    }
                )
                renderer.disable_segmentation_rendering()
                snapshots.append(
                    f'<figure><img src="{name}/{label}.png"><figcaption>{label} · {row["joint_state"]["time_s"]:.2f} s</figcaption></figure>'
                )
        for file in ["report.json", "episode.json", "episode.gif", "head-camera.png"]:
            if (source / file).exists():
                shutil.copyfile(source / file, dest / file)
        sha = hashlib.sha256((source / "trajectory.jsonl").read_bytes()).hexdigest()
        evidence.append(
            {
                "source": str(source),
                "trajectory_sha256": sha,
                "status": report["status"],
                "final_block_position_m": report["final"]["block_position_m"],
                "camera_coverage": camera_coverage,
                "visual_geometry": fidelity,
            }
        )
        playback = (
            f'<details><summary>Play full episode</summary><img src="{name}/episode.gif"></details>'
            if (dest / "episode.gif").exists()
            else ""
        )
        head_image = (
            "head-camera.png"
            if (dest / "head-camera.png").exists()
            else "settled-head.png"
        )
        sections.append(
            f"<section><h2>{html.escape(name)}</h2><p><strong>{report['status'].upper()}</strong> · {report['duration_s']:.2f} s · maximum block-bottom clearance {report['max_lift_m'] * 1000:.1f} mm</p>"
            + f"<p>Camera surfaces: {fidelity['mode']} ({fidelity['robot_mesh_count']}/{fidelity['expected_robot_mesh_count']} robot meshes). Physics contacts remain simplified.</p>"
            + playback
            + '<div class="grid">'
            + "".join(snapshots)
            + f'</div><p><a href="{name}/report.json">Measurements</a> · <a href="{name}/episode.json">Episode provenance</a> · <a href="{name}/{head_image}">Head-camera image</a></p></section>'
        )
    (output / "report.json").write_text(
        json.dumps(
            {
                "scope": "reconstruction of retained supported episodes, no simulation rerun",
                "sources": evidence,
            },
            indent=2,
        )
    )
    (output / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Learning robot · contact tasks</title><style>body{font:17px/1.5 system-ui;color:#213934;background:#f4f2eb;max-width:1150px;margin:40px auto;padding:0 22px}h1{font-size:38px}h2{font-size:28px}section{background:white;padding:24px;border-radius:20px;margin:28px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}img{width:100%}figure{margin:0}figcaption{padding:8px}a{color:#196258}details{margin:18px 0}strong{color:#196258}.note{border-left:4px solid #d8aa55;padding-left:16px}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style><h1>A task the design can learn from</h1><p>Two articulated-leg companions grasp a free 30 mm, 50 g block, lift it, move it over a tray and release it. Success requires the whole block to settle inside the tray for half a second.</p><p class="note"><strong>Supported torso fixture.</strong> Object motion comes from contact physics. The reference controller uses simulator object position; this is not visual perception, a learned policy, free-standing manipulation or a physical prototype.</p>"""
        + "".join(sections)
        + """<p>Camera fidelity is reported per episode. Detailed mechanism fit is checked separately; task success does not prove mechanical clearance or strength. Full manufacturing costs and the $1,000 selling-price target remain unverified.</p>"""
    )
    shutil.copyfile(__file__, output / Path(__file__).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review(args.runs, args.output)
