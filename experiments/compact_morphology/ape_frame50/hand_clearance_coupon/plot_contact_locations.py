"""Show where the nominal fixed-cube contact screen touches the cube."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import with_cube


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = ROOT / "artifacts/ape-cove50-hand-clearance-coupon"
    body = base / "physics-v2/candidate-training.xml"
    trace = base / "native-aperture-entry-v3/development/seed-2906.npz"
    waypoint_path = base / "contact-manifold-screen-v1/arm-waypoints.json"
    out = base / "native-aperture-comparison-v1/contact-location-geometry.png"
    model = with_cube(body)
    data = mujoco.MjData(model)
    source = np.load(trace)["qpos"][620]
    waypoints = json.loads(waypoint_path.read_text())["waypoints"]
    arm = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
    adr = [model.joint(name).qposadr[0] for name in arm]
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube = model.geom("audit_cube_geom").id
    pads = {"Palm": model.geom("right_hand").id,
            "Moving finger": model.geom("right_finger_tip").id}
    rows = {name: [] for name in pads}
    angles = np.array([row["jaw_rad"] for row in waypoints])
    positions = np.array([row["arm_rad"] for row in waypoints])
    for jaw in np.linspace(0, .5, 101):
        data.qpos[:] = source
        data.qpos[adr] = [np.interp(jaw, angles, positions[:, i]) for i in range(4)]
        data.qpos[jaw_adr] = jaw
        mujoco.mj_forward(model, data)
        center = data.geom_xpos[cube]
        to_local = data.geom_xmat[cube].reshape(3, 3).T
        for name, pad in pads.items():
            nearest = np.zeros(6)
            distance = mujoco.mj_geomDistance(model, data, cube, pad, .05, nearest)
            point = (to_local @ (nearest[:3] - center)) * 1000
            rows[name].append((jaw, *point, distance * 1000))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    colors = {"Palm": "#d34f44", "Moving finger": "#1b80aa"}
    axes[0].add_patch(Rectangle((-10, -10), 20, 20, facecolor="#eef2f1",
                                edgecolor="#56616b", linewidth=1.8))
    for name, data_rows in rows.items():
        arr = np.asarray(data_rows)
        axes[0].plot(arr[:, 1], arr[:, 2], color=colors[name], label=name, linewidth=2)
        axes[0].scatter(arr[::10, 1], arr[::10, 2], s=15, color=colors[name])
    axes[0].set(xlabel="Cube-local x (mm)", ylabel="Cube-local y (mm)",
                title="Contact points on cube, top view", xlim=(-13, 13), ylim=(-13, 13))
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=.2)
    axes[1].axhspan(5, 10, color="#f7e7bc", alpha=.7,
                    label="Top-rim zone above side-face target")
    axes[1].axhline(10, color="#aa6b20", linestyle="--", linewidth=1)
    for name, data_rows in rows.items():
        arr = np.asarray(data_rows)
        axes[1].plot(arr[:, 0], arr[:, 3], color=colors[name], label=name, linewidth=2)
    axes[1].set(xlabel="Jaw angle (rad)", ylabel="Cube-local contact height z (mm)",
                title="Static contact height through closure", ylim=(-11, 11), xlim=(0, .5))
    axes[1].grid(alpha=.2)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("101/101 geometric overlaps on a fixed cube still miss a robust side-face pinch",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    report = {"schema": "robot-learning-lab.cove50-contact-location-figure.v1",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "body_sha256": sha(body), "trace_sha256": sha(trace),
              "waypoints_sha256": sha(waypoint_path), "script_sha256": sha(Path(__file__)),
              "figure": str(out.relative_to(ROOT)), "figure_sha256": sha(out),
              "samples": 101,
              "palm_top_rim_fraction": sum(np.asarray(rows["Palm"])[:, 3] > 5) / 101,
              "finger_top_rim_fraction": sum(np.asarray(rows["Moving finger"])[:, 3] > 5) / 101,
              "limit": "Nearest points of static collision geometry; not contact force or dynamic grasp stability."}
    (out.parent / "contact-location-geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
