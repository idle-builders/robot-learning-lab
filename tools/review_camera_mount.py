"""Audit and present two retained camera-mount studies without changing their results."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_study(folder):
    report = json.loads((folder / "report.json").read_text())
    if sha(folder / "report.json") != (folder / "report.sha256").read_text().strip():
        raise ValueError("study report changed")
    for relative, expected in report["source_hashes"].items():
        if sha(folder / "source" / relative) != expected:
            raise ValueError(f"study source changed: {relative}")
    for relative, expected in report["output_hashes"].items():
        if sha(folder / relative) != expected:
            raise ValueError(f"study output changed: {relative}")
    return report


def render_head(folder, output, *, azimuth=155):
    """Derived presentation view; leave the sealed study assets unchanged."""
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(
        str((folder / "appearance/mechanism.xml").absolute())
    )
    interface = json.loads((folder / "interfaces.json").read_text())
    head = next(link for link in interface["links"] if link["name"] == "head")
    shell_geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_dataid[i] == model.mesh("head_shell").id
    ]
    if len(shell_geoms) != 1:
        raise ValueError("expected exactly one head-shell display mesh")
    model.geom_rgba[shell_geoms[0]] = head["geometry"][0]["color"]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    links = {link["name"]: link for link in interface["links"]}
    center = [0.0, 0.0, interface["base_height_m"] - 0.015]
    link = head
    while link:
        center = [a + b for a, b in zip(center, link["pos"], strict=True)]
        link = links.get(link["parent"])
    camera.lookat[:] = center
    camera.distance = 0.32
    camera.azimuth = azimuth
    camera.elevation = -12
    with mujoco.Renderer(model, height=800, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(renderer.render().copy()).save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pip", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    sections = []
    for name, folder in [("Pip", args.pip), ("Scout", args.scout)]:
        report = verify_study(folder)
        drift = [
            name
            for name, expected in report["source_hashes"].items()
            if Path(name).is_file() and sha(name) != expected
        ]
        row = dict(
            robot=name,
            study=str(folder),
            report_sha256=sha(folder / "report.json"),
            integrity="pass",
            current_source_drift=drift,
            allocation_status=report["allocation_status"],
            power_status=report["power_allocation"]["status"],
            optics_status=report["optics_status"],
            selected_computer=report["selected_computer_reservation"],
            original_shell_negative_control_detected=bool(
                report["camera_reservation"]["original_shell_negative_control"]
            ),
            misplaced_computer_negative_control_detected=bool(
                report["negative_control"]
            ),
        )
        rows.append(row)
        face = report.get("face_layout")
        face_text = ""
        if face:
            row["face_layout"] = face
            row["face_motion_status"] = report.get("face_motion_status", "not_run")
            face_text = (
                f"<p>Intact eyes and nominal service gap: {html.escape(face['status'])}. "
                f"Eye/body motion: {html.escape(row['face_motion_status'])}. "
                "Eye lighting and physical attachment remain allocations.</p>"
            )
        preview = args.output / f"{name.lower()}-head.png"
        render_head(folder, preview)
        row["derived_preview_sha256"] = sha(preview)
        front = args.output / f"{name.lower()}-front.png"
        render_head(folder, front, azimuth=180)
        row["derived_front_sha256"] = sha(front)
        relative = os.path.relpath(folder, args.output)
        prefix = html.escape(relative, quote=True)
        camera = report["camera_mount"]
        sections.append(
            f"<section><h2>{name}</h2><p>Camera/cassette/Pi allocation: <strong>{row['allocation_status']}</strong>. "
            f"Camera endpoint visibility: {row['optics_status']}. Power coverage: {row['power_status']}.</p>"
            f"<p>Pi datum in head mm: {html.escape(str(row['selected_computer']['position_head_mm']))}. "
            "15 sampled neck poses; four sampled cassette removal positions. No continuous sweep or physical-fit qualification.</p>"
            + face_text
            + f'<div class="images"><figure><img src="{preview.name}"><figcaption>Candidate head and cassette; derived presentation view</figcaption></figure>'
            f'<figure><img src="{front.name}"><figcaption>Front view of the retained CAD meshes</figcaption></figure>'
            f'<figure><img src="{prefix}/mechanism.png"><figcaption>Mechanism and component reservations</figcaption></figure>'
            f'<figure><img src="{prefix}/selected_wide-transfer.png"><figcaption>Prior task transfer endpoint re-rendered; no new dynamics</figcaption></figure></div>'
            f'<p><a href="{prefix}/camera-mount.step">Head STEP</a> · <a href="{prefix}/report.json">Study report</a></p>'
            f"<p>Fastening depth: {html.escape(camera['camera_mount_depth_status'])}. Camera spacers, fasteners and cable support remain incomplete.</p></section>"
        )
    result = dict(
        status="pass",
        scope="integrity review only; retain per-domain engineering results",
        entries=rows,
        reviewer_source_sha256=sha(__file__),
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Camera cassette review</title>'
        "<style>body{font:17px/1.5 system-ui;background:#eef1eb;color:#19332e;max-width:1280px;margin:40px auto;padding:0 24px}section{background:white;border-radius:18px;padding:24px;margin:24px 0}.images{display:flex;gap:14px}figure{margin:0;flex:1}img{width:100%;border-radius:12px}figcaption{font-size:14px}a{color:#176751}@media(max-width:750px){.images{display:block}}</style>"
        "<h1>Pip & Scout · removable camera cassette</h1><p>Mechanical integration experiment. "
        "Power, actual camera fastening, Pi mounting, harness, thermal, revised mass/dynamics and production cost remain open.</p>"
        + "".join(sections)
        + '<p><a href="report.json">Artifact integrity audit</a></p>'
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
