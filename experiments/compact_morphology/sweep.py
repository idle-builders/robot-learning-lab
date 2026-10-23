"""Quick kinematic screen using the pinned ToddlerBot MuJoCo model.

This measures geometry and Jacobians, not a trained policy or dynamic task success.
Run from the repository root with .venv-learning/bin/python.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".cache/upstream/toddlerbot/toddlerbot/descriptions/toddlerbot_2xc_gripper/scene.xml"
OUT = ROOT / "artifacts/compact-morphology-20260927"


def set_joint(model, data, name, value):
    joint = model.joint(name)
    data.qpos[model.jnt_qposadr[joint.id]] = value


def foot_pitch(data):
    rotation = data.site("left_foot_center").xmat.reshape(3, 3)
    return float(np.arctan2(-rotation[2, 0], rotation[0, 0]))


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    home = data.qpos.copy()
    home_foot = data.site("left_foot_center").xpos.copy()
    shoulder = data.body("left_shoulder_pitch_link").xpos.copy()
    reference_hip = data.body("left_hip_pitch_link").xpos.copy()
    reference_knee = data.body("left_knee_link").xpos.copy()
    reference_ankle = data.body("left_ankle_pitch_link").xpos.copy()
    reference_leg_span = float(reference_hip[2] - reference_ankle[2])
    thigh_fraction = float((reference_hip[2] - reference_knee[2]) / reference_leg_span)

    # Lock the elbow at a long-reach angle to proxy a rigid arm. The published
    # model has more DOF than our proposed body; this does not claim a new robot.
    arm_lengths = []
    for elbow in np.linspace(0, 0.7, 71):
        data.qpos[:] = home
        set_joint(model, data, "left_elbow_roll", elbow)
        mujoco.mj_forward(model, data)
        arm_lengths.append((float(np.linalg.norm(data.site("left_hand_center").xpos - shoulder)), elbow))
    arm_length, locked_elbow = max(arm_lengths)

    def squat(knee_flexion, hip, ankle):
        data.qpos[:] = home
        for name, value in (
            ("left_hip_pitch", hip), ("right_hip_pitch", -hip),
            ("left_knee", -knee_flexion), ("right_knee", knee_flexion),
            ("left_ankle_pitch", ankle), ("right_ankle_pitch", -ankle),
        ):
            set_joint(model, data, name, value)
        mujoco.mj_forward(model, data)
        return data.site("left_foot_center").xpos.copy(), foot_pitch(data)

    squat_rows = []
    for knee_flexion in (0.381, 0.7, 1.0, 1.3, 1.6, 1.9, 2.05):
        def residual(values):
            foot, pitch = squat(knee_flexion, *values)
            return ((foot[0] - home_foot[0]) * 10, pitch)

        fit = least_squares(
            residual, [-knee_flexion / 2, -knee_flexion / 2],
            bounds=([-2.0, -1.8], [1.8, 1.0]), max_nfev=100,
        )
        foot, pitch = squat(knee_flexion, *fit.x)
        squat_rows.append({
            "knee_flexion_deg": round(float(np.degrees(knee_flexion)), 1),
            "pelvis_lowering_m": round(float(foot[2] - home_foot[2]), 4),
            "foot_x_error_m": round(float(foot[0] - home_foot[0]), 6),
            "foot_pitch_error_deg": round(float(np.degrees(pitch)), 3),
        })
    deep_squat = squat_rows[-1]["pelvis_lowering_m"]

    data.qpos[:] = home
    for name, value in (
        ("left_elbow_roll", locked_elbow),
        ("left_shoulder_pitch", 0.4),
        ("left_shoulder_roll", 0.3),
        ("left_shoulder_yaw_driven", -1.2),
    ):
        set_joint(model, data, name, value)
    mujoco.mj_forward(model, data)
    linear = np.zeros((3, model.nv))
    angular = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, linear, angular, model.site("left_hand_center").id)
    shoulder_axes = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw_driven"]
    jacobians = {}
    for count in (1, 2, 3):
        cols = [model.jnt_dofadr[model.joint(name).id] for name in shoulder_axes[:count]]
        position = np.linalg.svd(linear[:, cols], compute_uv=False)
        pose = np.linalg.svd(np.vstack((linear[:, cols], 0.1 * angular[:, cols])), compute_uv=False)
        jacobians[f"shoulder_{count}_axes"] = {
            "hand_position_singular_m_per_rad": np.round(position, 5).tolist(),
            "hand_pose_singular_weighted": np.round(pose, 5).tolist(),
        }

    # Reach a fixed side button while the base and elbow stay still. This is
    # a concrete workspace comparison, not a button-press controller.
    side_button = np.array([0.14, 0.20, 0.30])
    side_reach = {}
    for count, initial, lower, upper in (
        (1, [-0.8], [-1.3], [1.3]),
        (2, [-0.8, 0.5], [-1.3, -0.436], [1.3, 1.57]),
    ):
        def position_error(values):
            data.qpos[:] = home
            set_joint(model, data, "left_elbow_roll", locked_elbow)
            for name, value in zip(shoulder_axes[:count], values):
                set_joint(model, data, name, value)
            mujoco.mj_forward(model, data)
            return data.site("left_hand_center").xpos - side_button

        fit = least_squares(position_error, initial, bounds=(lower, upper), max_nfev=100)
        side_reach[f"shoulder_{count}_axes_error_m"] = round(float(np.linalg.norm(fit.fun)), 4)

    def hand_frame(yaw):
        data.qpos[:] = home
        set_joint(model, data, "left_elbow_roll", locked_elbow)
        set_joint(model, data, "left_shoulder_pitch", -0.892)
        set_joint(model, data, "left_shoulder_roll", 0.535)
        set_joint(model, data, "left_shoulder_yaw_driven", yaw)
        mujoco.mj_forward(model, data)
        site = data.site("left_hand_center")
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    first_pos, first_frame = hand_frame(-1.57)
    second_pos, second_frame = hand_frame(-0.57)
    orientation_angle = np.arccos(np.clip((np.trace(first_frame.T @ second_frame) - 1) / 2, -1, 1))
    side_reach["third_axis_1rad_hand_position_change_m"] = round(float(np.linalg.norm(second_pos - first_pos)), 4)
    side_reach["third_axis_1rad_hand_orientation_change_deg"] = round(float(np.degrees(orientation_angle)), 1)

    foot_axes = {}
    for name in ("left_hip_roll", "left_hip_yaw_driven", "left_ankle_roll"):
        samples = []
        for angle in (-0.35, 0.0, 0.35):
            data.qpos[:] = home
            joint = model.joint(name)
            set_joint(model, data, name, home[model.jnt_qposadr[joint.id]] + angle)
            mujoco.mj_forward(model, data)
            foot = data.site("left_foot_center")
            rotation = foot.xmat.reshape(3, 3)
            yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
            roll = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
            samples.append({
                "command_deg": round(float(np.degrees(angle)), 1),
                "lateral_foot_shift_m": round(float(foot.xpos[1] - home_foot[1]), 4),
                "foot_yaw_deg": round(float(yaw), 1),
                "foot_roll_deg": round(float(roll), 1),
            })
        foot_axes[name] = samples

    # A swing-foot *geometry* check with the other leg held at home. It is not
    # a step simulation; the ranges intentionally stop short of joint limits.
    best_step = None
    for hip in np.linspace(-0.9, 0.5, 55):
        for knee in np.linspace(-1.4, -0.38, 45):
            ankle = knee - hip  # near-level foot in this ToddlerBot joint ordering
            if not -1.832 <= ankle <= 0.96:
                continue
            data.qpos[:] = home
            for name, value in (("left_hip_pitch", hip), ("left_knee", knee), ("left_ankle_pitch", ankle)):
                set_joint(model, data, name, value)
            mujoco.mj_forward(model, data)
            foot = data.site("left_foot_center").xpos
            forward = float(foot[0] - home_foot[0])
            clearance = float(foot[2] - home_foot[2])
            if 0.04 <= forward <= 0.12 and (best_step is None or clearance > best_step["clearance_m"]):
                best_step = {"clearance_m": round(clearance, 4), "forward_m": round(forward, 4)}

    # Assumed silhouettes, screened against one stationary left-foot floor
    # grasp target. Arm/shoulder dimensions vary; no new CAD or dynamics implied.
    shapes = [
        ("ToddlerBot reference", 0.56, float(shoulder[2]), arm_length, deep_squat, 0.17, 0.074),
        ("uniform compact", 0.45, 0.331, 0.173, 0.080, 0.137, 0.060),
        ("leggy compact", 0.45, 0.360, 0.160, 0.090, 0.130, 0.074),
        ("balanced compact", 0.45, 0.320, 0.200, 0.080, 0.120, 0.074),
        ("low shoulder, long arm", 0.45, 0.300, 0.220, 0.080, 0.120, 0.074),
        ("wide stance, balanced", 0.45, 0.320, 0.200, 0.080, 0.120, 0.110),
    ]
    target = np.array([0.13, 0.10, 0.05])  # floor toy grasp center, meters
    shape_rows = []
    for name, height, shoulder_z, arm, squat_depth, torso, stance in shapes:
        hip_height = shoulder_z - torso
        ankle_height = float(reference_ankle[2]) * height / 0.56
        leg_span = hip_height - ankle_height
        for lean_deg in (0, 35):
            lean = np.radians(lean_deg)
            shoulder_xyz = np.array([
                float(shoulder[0]) + torso * np.sin(lean),
                float(shoulder[1]) * stance / 0.074,
                shoulder_z - squat_depth - torso * (1 - np.cos(lean)),
            ])
            distance = float(np.linalg.norm(target - shoulder_xyz))
            shape_rows.append({
                "shape": name,
                "height_m": height,
                "shoulder_height_m": shoulder_z,
                "arm_length_m": round(arm, 3),
                "torso_length_m": torso,
                "hip_height_m": round(hip_height, 3),
                "assumed_thigh_length_m": round(leg_span * thigh_fraction, 3),
                "assumed_shin_length_m": round(leg_span * (1 - thigh_fraction), 3),
                "max_squat_m": squat_depth,
                "stance_width_m": stance,
                "torso_lean_deg": lean_deg,
                "floor_target_distance_m": round(distance, 3),
                "outside_arm_reach_m": round(max(0, distance - arm), 3),
                "horizontal_200g_payload_torque_Nm": round(0.2 * 9.81 * arm, 3),
                "stance_line_tip_angle_deg": round(float(np.degrees(np.arctan(stance / height))), 1),
                "scaled_swing_clearance_estimate_m": round(best_step["clearance_m"] * leg_span / reference_leg_span, 3),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(MODEL.relative_to(ROOT)),
        "scene_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "robot_xml_sha256": hashlib.sha256((MODEL.parent / "toddlerbot_2xc_gripper.xml").read_bytes()).hexdigest(),
        "upstream_commit": subprocess.run(
            ["git", "-C", str(ROOT / ".cache/upstream/toddlerbot"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "method": "MuJoCo forward kinematics plus explicitly assumed proportions; no policy, torque, contact, collision, or dynamic stability evaluation",
        "reference": {
            "shoulder_height_m": round(float(shoulder[2]), 4),
            "long_reach_elbow_lock_rad": round(float(locked_elbow), 3),
            "shoulder_to_hand_m": round(arm_length, 4),
            "home_foot_spacing_m": round(float(2 * abs(home_foot[1])), 4),
            "deep_squat_lowering_m": deep_squat,
            "hip_to_knee_vertical_m": round(float(reference_hip[2] - reference_knee[2]), 4),
            "knee_to_ankle_vertical_m": round(float(reference_knee[2] - reference_ankle[2]), 4),
            "hip_to_shoulder_vertical_m": round(float(shoulder[2] - reference_hip[2]), 4),
        },
        "squat": squat_rows,
        "shoulder_jacobians": jacobians,
        "side_button_reach": side_reach,
        "foot_axes": foot_axes,
        "swing_foot_screen": best_step,
        "shape_floor_reach": shape_rows,
    }
    path = OUT / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")

    # A visual of the *assumed* side-view envelopes, not a CAD fit drawing.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, shape in zip(axes.flat, shapes):
        name, height, shoulder_z, arm, squat_depth, torso, stance = shape
        hip_z = shoulder_z - torso - squat_depth
        shoulder_crouch = shoulder_z - squat_depth
        ax.plot([0, 0.015, 0], [hip_z, hip_z / 2, 0.015], color="#263447", lw=3)
        ax.plot([-0.045, 0.045], [0.01, 0.01], color="#263447", lw=5)
        ax.plot([0, 0], [hip_z, shoulder_crouch], color="#263447", lw=5)
        head_z = height - 0.05 - squat_depth
        ax.plot([0, 0], [shoulder_crouch, head_z - 0.04], color="#263447", lw=3)
        ax.add_patch(plt.Circle((0, head_z), 0.04, color="#e6f4f4", ec="#263447", lw=2))
        ax.scatter([0, 0], [hip_z, shoulder_crouch], color="#19a49a", s=45, zorder=4)
        ax.add_patch(plt.Circle((0, shoulder_crouch), arm, fill=False, ls="--", lw=1.6, color="#4489ba"))
        ax.scatter([target[0]], [target[2]], marker="*", s=120, color="#df5e45", zorder=5)
        gap = next(row["outside_arm_reach_m"] for row in shape_rows if row["shape"] == name and row["torso_lean_deg"] == 0)
        ax.set_title(
            f"{name}\narm {arm:.2f} m · shoulder {shoulder_z:.2f} m\n"
            f"stance {stance*100:.1f} cm · gap {gap*100:.0f} cm", fontsize=10
        )
        ax.set_xlim(-0.12, 0.32)
        ax.set_ylim(-0.02, 0.48)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel("forward distance (m)")
        ax.set_ylabel("height (m)")
    fig.suptitle("Crouched floor-toy reach bound · dashed circle is maximum arm reach", fontsize=15)
    fig.savefig(OUT / "shape-reach.png", dpi=160)
    plt.close(fig)
    print(path)
    print(json.dumps({"reference": summary["reference"], "swing_foot_screen": best_step, "shape_floor_reach": shape_rows}, indent=2))


if __name__ == "__main__":
    main()
