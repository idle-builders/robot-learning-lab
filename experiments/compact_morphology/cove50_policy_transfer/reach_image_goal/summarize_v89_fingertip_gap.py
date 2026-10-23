"""Explain the opened 10433 contact regression from saved native qpos traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUT = BASE / "v89"
VISITS = (
    ("V41 prior pass", "v41/opened-native/seed-10433", "#12816a"),
    ("V87 locked miss", "v87/native-prior75-retention/seed-10433-prior-10433", "#c08013"),
    ("V89 specialist miss", "v89/native-rear-boundary-probe/seed-10433-prior-10433", "#bc4358"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    state = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_tip")
    fig, (path_ax, time_ax) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    summary = {"schema": "robot-learning-lab.cove50-v89-10433-fingertip-gap.v1",
               "role": "posthoc_geometry_diagnostic_not_policy_evaluation",
               "body_role": "rejected_historical_direct_frame_sandbox",
               "body_sha256": sha(BODY), "source_sha256": sha(Path(__file__)), "visits": []}
    for label, relative, color in VISITS:
        report_path, trace_path = BASE / f"{relative}.json", BASE / f"{relative}.npz"
        report = json.loads(report_path.read_text())
        with np.load(trace_path) as trace:
            qpos = trace["qpos"]
        if len(qpos) != len(report["timeseries"]):
            raise ValueError(f"Trace/report length mismatch: {relative}")
        times, deltas, contacts = [], [], []
        for q, row in zip(qpos, report["timeseries"], strict=True):
            if row["phase"] != "reach":
                continue
            state.qpos[:] = q[:model.nq]
            mujoco.mj_forward(model, state)
            cube_xyz = q[model.nq:model.nq + 3]
            times.append(row["time_s"])
            deltas.append((state.geom_xpos[tip_id] - cube_xyz) * 1000)
            contacts.append(bool(row["finger_contact"]))
        times = np.asarray(times)
        deltas = np.asarray(deltas)
        contacts = np.asarray(contacts)
        distance = np.linalg.norm(deltas, axis=1)
        nearest = int(np.argmin(distance))
        near_window = times >= 12.0
        path_ax.plot(deltas[near_window, 0], deltas[near_window, 1], color=color,
                     alpha=.75, linewidth=2, label=label)
        path_ax.scatter(deltas[nearest, 0], deltas[nearest, 1], color=color, s=55,
                        edgecolor="white", linewidth=1.2, zorder=3)
        time_ax.plot(times, distance, color=color, linewidth=2, label=label)
        if contacts.any():
            time_ax.scatter(times[contacts], distance[contacts], color=color, s=13,
                            marker="s", zorder=3)
        summary["visits"].append({"label": label,
                                  "report": str(report_path.relative_to(ROOT)),
                                  "report_sha256": sha(report_path),
                                  "trace": str(trace_path.relative_to(ROOT)),
                                  "trace_sha256": sha(trace_path),
                                  "route": report["selection_mode"],
                                  "nearest_time_s": float(times[nearest]),
                                  "nearest_tip_center_to_cube_center_mm": round(float(distance[nearest]), 3),
                                  "nearest_tip_minus_cube_xyz_mm": np.round(deltas[nearest], 3).tolist(),
                                  "finger_contact_frames": int(contacts.sum())})
    path_ax.set(title="Fingertip center, relative to cube", xlabel="world X offset (mm)",
                ylabel="world Y offset (mm)")
    path_ax.grid(alpha=.2)
    path_ax.legend(loc="best", fontsize=8)
    time_ax.set(title="Closest approach is a narrow contact edge", xlabel="episode time (s)",
                ylabel="finger center to cube center (mm)")
    time_ax.grid(alpha=.2)
    time_ax.legend(loc="best", fontsize=8)
    fig.suptitle("10433: same native start, fingertip contact only on older V41 replay")
    OUT.mkdir(parents=True, exist_ok=True)
    figure = OUT / "10433-fingertip-gap-comparison.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    summary["figure"] = str(figure.relative_to(ROOT))
    summary["figure_sha256"] = sha(figure)
    summary["limitation"] = ("Tip-center distance is a kinematic explanation from saved native qpos, "
                             "not signed cube/geom separation. Native finger-contact sensor is the outcome.")
    (OUT / "10433-fingertip-gap-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"figure": summary["figure"], "visits": summary["visits"]}, indent=2))


if __name__ == "__main__":
    main()
