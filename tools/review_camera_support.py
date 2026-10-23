"""Review two bounded body studies and the retained vendor-model disagreement."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from lab.config import load_config
from lab.robot import make_robot
from lab.camera_support import visual_recipes, specification
from tools.review_camera_mount import verify_study


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(folder, output, robot=None, azimuth=145):
    m = mujoco.MjModel.from_xml_path(str((folder / "mechanism.xml").absolute()))
    # Uniform skybox for a neutral CAD review background.
    m.tex_data[:] = np.resize(
        np.array([224, 230, 224], dtype=np.uint8), m.tex_data.shape
    )
    if robot:
        colors = {
            name: color
            for name, _, _, color in visual_recipes(
                specification(robot.config.camera.support_profile)
            )
        }
        colors["head_shell"] = (
            next(p for p in robot.links if p.name == "head").geometry[0].color
        )
        for name, color in colors.items():
            for i in range(m.ngeom):
                if (
                    m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                    and m.mesh(m.geom_dataid[i]).name == name
                ):
                    m.geom_rgba[i] = color
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = mujoco.MjvCamera()
    c.azimuth = azimuth
    c.elevation = -12
    if robot:
        c.lookat[:] = robot.forward_kinematics()["head"][:3, 3]
        c.distance = 0.30
    else:
        c.lookat[:] = [-0.007, 0, -0.003]
        c.distance = 0.12
    with mujoco.Renderer(m, 800, 960) as r:
        r.update_scene(d, camera=c)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(r.render()).save(output)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    sections = []
    bindings = {}
    for name in ("pip", "scout"):
        folder = Path(f"artifacts/{name}-camera-support-v3")
        r = verify_study(folder)
        robot = make_robot(load_config(Path(f"variants/{name}-camera-support.yaml")))
        render(folder / "assembled", output / f"{name}-front.png", robot, 180)
        render(folder / "assembled", output / f"{name}-head.png", robot)
        row = dict(
            robot=name,
            status=r["status"],
            motion_conflicts=sum(len(p["conflicts"]) for p in r["motion"]),
            report=str(folder / "report.json"),
        )
        rows.append(row)
        bindings[row["report"]] = sha(row["report"])
        rel = os.path.relpath(folder, output)
        sections.append(
            f'<section><h2>{name.title()}</h2><p>Drawing-based assembly study: <b>{r["status"]}</b>. {row["motion_conflicts"]} reported face/body intersections across sampled poses.</p><div class="images"><img src="{name}-front.png"><img src="{name}-head.png"></div><p><a href="{rel}/camera-support.step">Split assembly STEP</a> · <a href="{rel}/exploded/mechanism.png">Exploded carrier study</a> · <a href="{rel}/report.json">Full findings</a></p></section>'
        )
    vendor = Path("artifacts/camera-vendor-fit-v6")
    vr = verify_study(vendor)
    original = Path("artifacts/camera-vendor-fit-v4")
    original_report = verify_study(original)
    for folder in (vendor, original):
        bindings[str(folder / "report.json")] = sha(folder / "report.json")
    render(vendor, output / "vendor-front.png")
    render(vendor, output / "vendor-rear.png", azimuth=30)
    witnesses = Path("artifacts/camera-tool-witnesses-v1/report.json")
    wr = json.loads(witnesses.read_text())
    for path, h in wr["source_hashes"].items():
        if sha(path) != h:
            raise ValueError("tool witness source changed")
    bindings[str(witnesses)] = sha(witnesses)
    report = dict(
        status="fail"
        if any(row["status"] == "fail" for row in rows)
        or original_report["status"] == "fail"
        else "partial",
        scope="candidate review: retained Scout brow collisions and drawing/vendor discrepancy remain open",
        integrity="pass",
        robots=rows,
        vendor_drawing_fit=original_report["status"],
        vendor_adjusted_stack_fit=vr["status"],
        vendor_registration=vr["registration"],
        tool_witnesses=wr["status"],
        source_bindings=bindings,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    source = output / "source"
    source.mkdir()
    shutil.copyfile(__file__, source / "review_camera_support.py")
    (output / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Camera assembly review</title><style>body{font:17px/1.5 system-ui;background:#e9eeea;color:#1e3631;max-width:1100px;margin:32px auto;padding:0 24px}h1,h2{line-height:1.15}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;border-radius:12px}section{margin:32px 0;padding:24px;background:#fff;border-radius:16px}a{color:#186452}.open{padding:18px;background:#fff0da;border-radius:12px}</style><h1>A camera we can assemble</h1><p>Split bezel and rear carrier, four supported PCB holes, screws and nuts, and conditional service paths. Engineering candidates; no physical-fit or finished-style claim.</p><p class="open"><b>Open findings:</b> the drawing-based carrier collides with the official STEP PCB. An adjusted standalone carrier clears it, but has not replaced the full-body model. Scout also has brow/neck-hardware intersections that need a separate geometry correction.</p>"""
        + "".join(sections)
        + """<section><h2>Official STEP changes the mounting decision</h2><p>The old support intersects the vendor substrate by about 8.32 mm³. Changing the support length from 4.6 to 4.095 mm clears all tested vendor solids. Four BRep mounting-hole centers agree with the optical-frame registration to within 0.000002 mm; a deliberately displaced registration fails.</p><div class="images"><img src="vendor-front.png"><img src="vendor-rear.png"></div><p>This is a standalone carrier/fastener experiment against the official model, without a whole-body, physical or tolerance qualification.</p><p><a href="../camera-vendor-fit-v6/candidate-assembly.step">Vendor-model assembly STEP</a> · <a href="../camera-vendor-fit-v6/report.json">Adjusted stack and registration</a> · <a href="../camera-vendor-fit-v4/report.json">Retained failed drawing stack</a></p></section><p>Independent drawing-axis and positive tool-reach witnesses pass, including displaced-support and unreachable-tool negative controls. Physical clamping surfaces, loose-nut retention, flex routing, Pi mounting and revised mass/dynamics remain open.</p><p><a href="report.json">Review provenance</a> · <a href="../camera-tool-witnesses-v1/report.json">Tool witnesses</a> · <a href="../../docs/design/camera-support.md">Design and reproducible commands</a></p>"""
    )
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
