"""Proxy geom-distance screen for the frozen virtual head camera module."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from screen_head_mount import BODY, OUT, ROOT, sha


def main() -> None:
    frozen = json.loads((OUT / "heldout-freeze.json").read_text())
    plan = json.loads((OUT / "plan.json").read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    root = ET.parse(BODY).getroot()
    head = root.find(".//body[@name='head']")
    assert head is not None
    ET.SubElement(head, "geom", name="virtual_camera_module_clearance_only",
                  type="box", size="0.0125 0.012 0.0062", mass="0",
                  pos=" ".join(map(str, frozen["head_local_camera_xyz_m"])),
                  rgba="1 0 0 1", contype="0", conaffinity="0")
    cube = ET.SubElement(root.find("worldbody"), "body", name="offline_visual_cube")
    ET.SubElement(cube, "freejoint")
    ET.SubElement(cube, "geom", type="box", size=".01 .01 .01",
                  mass=".00625", rgba=".95 .36 .05 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nq == 35
    data = mujoco.MjData(model)
    camera_module = model.geom("virtual_camera_module_clearance_only").id
    head_yaw = model.joint("head_yaw").qposadr[0]
    own_head = {"head_frame_sensor_budget", "upper_camera_module",
                "lower_camera_module"}
    rows = []
    for split, seed, times in ([
            ("development", seed, (6.0, 7.5, 8.5, 10.0))
            for seed in (9921, 9922, 9941, 9942)] + [
            ("reserved", seed, (7.5,)) for seed in (9951, 9952, 9957, 9958)]):
        path = ROOT / f"artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/{split}/seed-{seed}.npz"
        with np.load(path) as trace:
            states = trace["qpos"].astype(float)
        for time_s in times:
            data.qpos[:] = states[round(time_s / .02)]
            data.qpos[head_yaw] = frozen["fixed_head_yaw_rad"]
            mujoco.mj_forward(model, data)
            distances = []
            for geom in range(model.ngeom):
                if (geom == camera_module or model.geom_bodyid[geom] == 0 or
                        model.body(model.geom_bodyid[geom]).name == "offline_visual_cube"):
                    continue
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom-{geom}"
                distance = float(mujoco.mj_geomDistance(
                    model, data, camera_module, geom, 1.0, None))
                distances.append({"geom": name, "signed_distance_mm": round(distance*1000, 3)})
            named = {r["geom"]: r["signed_distance_mm"] for r in distances}
            other = min((r for r in distances if r["geom"] not in own_head),
                        key=lambda r: r["signed_distance_mm"])
            rows.append({"case": f"{split}-seed-{seed}-{time_s:g}s",
                         "head_sphere_gap_mm": named["head_frame_sensor_budget"],
                         "upper_module_gap_mm": named["upper_camera_module"],
                         "lower_module_gap_mm": named["lower_camera_module"],
                         "nearest_other_body": other})
    report = {
        "schema": "robot-learning-lab.cove50-virtual-camera-clearance-screen.v1",
        "classification": "In-memory module box against selected training-XML geoms; not mount, bracket, fastener or physical clearance qualification",
        "selected_body_sha256": plan["selected_body_sha256"],
        "frozen_view": "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json",
        "method": "Insert a massless collision-disabled 25x24x12.4 mm camera module at the frozen head-local position; compute signed MuJoCo geom distances on retained states with offline head yaw -1.5. Existing exposed-frame physics proxies and camera modules are included. A positive head-sphere gap requires an unmodeled mount, not proof of attachability.",
        "minimum_other_body_gap_mm": min(r["nearest_other_body"]["signed_distance_mm"]
                                      for r in rows),
        "minimum_upper_module_gap_mm": min(r["upper_module_gap_mm"] for r in rows),
        "head_sphere_gap_mm": min(r["head_sphere_gap_mm"] for r in rows),
        "rows": rows,
    }
    (OUT / "mount-clearance.json").write_text(json.dumps(report, indent=2,
                                                      allow_nan=False) + "\n")
    print("other body gap",report["minimum_other_body_gap_mm"],
          "upper module gap",report["minimum_upper_module_gap_mm"],
          "head sphere gap",report["head_sphere_gap_mm"])


if __name__ == "__main__":
    main()
