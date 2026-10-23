"""Native MuJoCo contact screen for compact body proportions.

This uses one diagnostic pose solver and identical position-servo limits. It is
NOT policy training, an electrical motor model, or a claim of learned skills.
Run: .venv-learning/bin/python experiments/compact_morphology/dynamic_screen.py
"""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/compact-morphology-dynamics-20260927-v7"
SHAPES = {
    "balanced": dict(hip=0.201, shoulder=0.320, arm=0.200, thigh=0.083, shin=0.086, mass=2.161, waist=False),
    "leggy": dict(hip=0.230, shoulder=0.360, arm=0.160, thigh=0.097, shin=0.101, mass=2.177, waist=False),
    "low_shoulder": dict(hip=0.181, shoulder=0.300, arm=0.220, thigh=0.073, shin=0.076, mass=2.161, waist=False),
    "waist_pitch": dict(hip=0.201, shoulder=0.320, arm=0.200, thigh=0.083, shin=0.086, mass=2.216, waist=True),
    "ape_short_torso": dict(hip=0.181, shoulder=0.221, arm=0.190, thigh=0.073, shin=0.076, mass=2.220, waist=False, height=0.350, head_radius=0.065, head_mass=0.160),
    "high_hip_short_torso": dict(hip=0.260, shoulder=0.300, arm=0.220, thigh=0.112, shin=0.116, mass=2.161, waist=False),
}


def joint(name, axis, limit, torque, kp, kv):
    return (
        f'<joint name="{name}" axis="{axis}" range="{limit}" damping="0.012" armature="0.001"/>'
        f'<position name="{name}_servo" joint="{name}" kp="{kp}" kv="{kv}" '
        f'forcerange="-{torque} {torque}"/>'
    )


def make_xml(shape, target=(0.13, 0.10, 0.045)):
    hip, shoulder, arm = shape["hip"], shape["shoulder"], shape["arm"]
    thigh, shin = shape["thigh"], shape["shin"]
    waist = shape["waist"]
    torso_len = shoulder - hip
    head_radius = shape.get("head_radius", 0.05)
    head_mass = shape.get("head_mass", 0.09)
    head_center = shape.get("height", 0.45) - shoulder - head_radius
    mass_scale = shape["mass"] / (2.16 + (head_mass - 0.09) + (0.055 if waist else 0))
    actuators = []

    def make_joint(name, axis, limit, torque, kp, kv):
        spec = joint(name, axis, limit, torque, kp, kv)
        actuators.append(spec[spec.index("<position"):])
        return spec[:spec.index("<position")]

    legs = []
    for side, sign in (("left", 1), ("right", -1)):
        y = sign * 0.037
        legs.append(f'''
      <body name="{side}_hip_yaw" pos="0 {y} 0">
        {make_joint(side + '_hip_yaw', '0 0 1', '-1.2 1.2', 0.92, 18, 0.45)}
        <geom type="sphere" size="0.018" mass="{0.04 * mass_scale}" rgba="0.35 0.40 0.52 1"/>
        <body name="{side}_hip_roll">
          {make_joint(side + '_hip_roll', '1 0 0', '-0.8 0.8', 1.8, 28, 0.7)}
          <geom type="sphere" size="0.018" mass="{0.04 * mass_scale}" rgba="0.35 0.40 0.52 1"/>
          <body name="{side}_thigh">
            {make_joint(side + '_hip_pitch', '0 1 0', '-1.5 1.5', 1.8, 30, 0.8)}
            <geom type="capsule" fromto="0 0 0 0 0 -{thigh}" size="0.018" mass="{0.10 * mass_scale}" rgba="0.50 0.66 0.80 1"/>
            <body name="{side}_shin" pos="0 0 -{thigh}">
              {make_joint(side + '_knee', '0 1 0', '-0.05 2.3', 3.0, 34, 0.9)}
              <geom type="capsule" fromto="0 0 0 0 0 -{shin}" size="0.016" mass="{0.11 * mass_scale}" rgba="0.50 0.66 0.80 1"/>
              <body name="{side}_foot" pos="0 0 -{shin}">
                {make_joint(side + '_ankle_pitch', '0 1 0', '-1.5 1.5', 3.0, 32, 0.8)}
                <geom name="{side}_sole" type="box" pos="0.014 0 -0.020" size="0.045 0.030 0.012" mass="{0.085 * mass_scale}" rgba="0.24 0.30 0.37 1" friction="1.25 0.01 0.001"/>
                <site name="{side}_sole_center" pos="0.014 0 -0.032" size="0.004"/>
              </body>
            </body>
          </body>
        </body>
      </body>''')

    arms = []
    for side, sign in (("left", 1), ("right", -1)):
        y = sign * 0.080
        arms.append(f'''
      <body name="{side}_shoulder_yaw_body" pos="0 {y} {torso_len}">
        {make_joint(side + '_shoulder_yaw', '0 0 1', '-2.0 2.0', 1.4, 15, 0.35)}
        <geom type="sphere" size="0.019" mass="{0.04 * mass_scale}" rgba="0.35 0.40 0.52 1"/>
        <body name="{side}_shoulder_roll_body">
          {make_joint(side + '_shoulder_roll', '1 0 0', '-1.6 1.6', 1.4, 18, 0.4)}
          <geom type="sphere" size="0.018" mass="{0.04 * mass_scale}" rgba="0.35 0.40 0.52 1"/>
          <body name="{side}_arm">
            {make_joint(side + '_shoulder_pitch', '0 1 0', '-2.2 2.2', 1.9, 20, 0.4)}
            <geom type="capsule" fromto="0 0 0 0 0 -{arm}" size="0.013" mass="{0.080 * mass_scale}" rgba="0.85 0.67 0.45 1"/>
            <geom name="{side}_hand" type="sphere" pos="0 0 -{arm}" size="0.017" mass="{0.005 * mass_scale}" rgba="0.95 0.75 0.50 1"/>
            <site name="{side}_hand_center" pos="0 0 -{arm}" size="0.004"/>
            <body name="{side}_finger" pos="0 0 -{arm}">
              {make_joint(side + '_gripper', '0 1 0', '-0.5 0.5', 0.5, 8, 0.12)}
              <geom name="{side}_finger_tip" type="box" pos="0.016 0 -0.005" size="0.014 0.008 0.008" mass="{0.005 * mass_scale}" rgba="0.95 0.75 0.50 1"/>
            </body>
          </body>
        </body>
      </body>''')

    waist_joint = make_joint("waist_pitch", "0 1 0", "-0.6 0.6", 1.4, 20, 0.6) if waist else ""
    waist_geom = f'<geom type="sphere" size="0.025" mass="{0.055 * mass_scale}" rgba="0.35 0.40 0.52 1"/>' if waist else ""
    xml = f'''<mujoco model="compact_{'waist' if waist else 'fixed'}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default><geom condim="3" friction="1 0.01 0.001"/><joint limited="true"/></default>
  <visual><global offwidth="960" offheight="720"/></visual>
  <worldbody>
    <light pos="0 -0.3 1.5" dir="0 0 -1" directional="true"/>
    <camera name="front" pos="0.65 -0.65 0.55" xyaxes="1 1 0 -1 1 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.18 0.22 0.28 1" friction="1.25 0.01 0.001"/>
    <geom name="toy" type="sphere" pos="{target[0]} {target[1]} 0.020" size="0.020" rgba="0.95 0.35 0.25 1"/>
    <body name="pelvis" pos="0 0 {hip}">
      <freejoint name="root"/>
      <geom type="box" pos="0 0 0.015" size="0.048 0.065 0.026" mass="{0.35 * mass_scale}" rgba="0.55 0.65 0.75 1"/>
      {''.join(legs)}
      <body name="torso">
        {waist_joint}
        {waist_geom}
        <geom type="box" pos="0 0 {torso_len / 2}" size="0.048 0.060 {torso_len / 2}" mass="{0.60 * mass_scale}" rgba="0.78 0.85 0.90 1"/>
        <geom type="capsule" fromto="0 0 {torso_len} 0 0 {torso_len + head_center}" size="0.018" density="0" contype="0" conaffinity="0" rgba="0.72 0.77 0.80 1"/>
        <body name="head_yaw_body" pos="0 0 {torso_len + head_center}">
          {make_joint('head_yaw', '0 0 1', '-1.5 1.5', 0.5, 8, 0.12)}
          <geom type="sphere" size="0.018" mass="{0.03 * mass_scale}" rgba="0.72 0.77 0.80 1"/>
          <body name="head">
            {make_joint('head_pitch', '0 1 0', '-0.8 0.8', 0.5, 8, 0.12)}
            <geom type="sphere" size="{head_radius}" mass="{head_mass * mass_scale}" rgba="0.90 0.91 0.87 1"/>
          </body>
        </body>
        {''.join(arms)}
      </body>
    </body>
  </worldbody>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>'''
    return xml


def pose(model, data, values):
    data.qpos[0] = values[0]  # root x
    data.qpos[2] = values[1]  # root z
    data.qpos[3:7] = [1, 0, 0, 0]
    for side in ("left", "right"):
        for name, value in (("hip_pitch", values[2]), ("knee", values[3]), ("ankle_pitch", values[4])):
            data.qpos[model.jnt_qposadr[model.joint(f"{side}_{name}").id]] = value
    data.qpos[model.jnt_qposadr[model.joint("left_shoulder_pitch").id]] = values[5]
    data.qpos[model.jnt_qposadr[model.joint("left_shoulder_roll").id]] = values[6]
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch") >= 0:
        data.qpos[model.jnt_qposadr[model.joint("waist_pitch").id]] = values[7]
    mujoco.mj_forward(model, data)


def ik(model, shape, target):
    data = mujoco.MjData(model)
    target = np.asarray(target)
    home_z = shape["hip"]
    foot_targets = {side: np.array([0.014, sign * 0.037, 0]) for side, sign in (("left", 1), ("right", -1))}

    def residual(values):
        pose(model, data, values)
        out = []
        for side in ("left", "right"):
            foot = data.site(f"{side}_sole_center")
            out.extend(80 * (foot.xpos - foot_targets[side]))
            out.extend(8 * (foot.xmat.reshape(3, 3)[:, 2] - np.array([0, 0, 1])))
        out.extend(50 * (data.site("left_hand_center").xpos - target))
        out.extend([0.8 * values[2], 0.35 * values[3], 0.7 * values[4], 0.2 * values[5], 0.2 * values[6]])
        if shape["waist"]:
            out.append(0.5 * values[7])
        # The root is free in dynamics, but the target posture should not demand
        # a lateral COM shift beyond the soles' support region.
        com_x = data.subtree_com[model.body("pelvis").id, 0]
        out.append(100 * max(0, com_x - 0.045, -com_x - 0.025))
        return np.array(out)

    initial = [0, home_z, -0.7, 1.5, -0.8, -0.7, -0.25]
    lower = [-0.055, home_z - 0.11, -1.3, 0.0, -1.5, -2.2, -1.4]
    upper = [0.055, home_z + 0.01, 1.3, 2.3, 1.5, 2.2, 1.4]
    if shape["waist"]:
        initial += [0.25]
        lower += [-0.6]
        upper += [0.6]
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=180)
    pose(model, data, result.x)
    hand = data.site("left_hand_center").xpos.copy()
    return result.x, dict(hand_error_m=float(np.linalg.norm(hand - target)), hand_xyz=hand.tolist(), target_xyz=target.tolist(), root_z_m=float(result.x[1]), joint_targets_rad={
        name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]]) for name in ("left_hip_pitch", "left_knee", "left_ankle_pitch", "left_shoulder_pitch", "left_shoulder_roll") + (("waist_pitch",) if shape["waist"] else ())
    })


def simulate(name, shape, target=(0.13, 0.10, 0.045), render=True):
    target_xyz = np.asarray(target)
    xml = make_xml(shape, target)
    (OUT / f"{name}.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[2] = shape["hip"]
    data.qpos[3] = 1
    mujoco.mj_forward(model, data)
    goal, ik_info = ik(model, shape, target_xyz)
    home = np.zeros(model.nu)
    joint_goal = home.copy()
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jname = actuator_name.removesuffix("_servo")
        jid = model.joint(jname).id
        if jname.startswith(("left_hip_pitch", "right_hip_pitch")):
            joint_goal[i] = goal[2]
        elif jname.startswith(("left_knee", "right_knee")):
            joint_goal[i] = goal[3]
        elif jname.startswith(("left_ankle_pitch", "right_ankle_pitch")):
            joint_goal[i] = goal[4]
        elif jname == "left_shoulder_pitch":
            joint_goal[i] = goal[5]
        elif jname == "left_shoulder_roll":
            joint_goal[i] = goal[6]
        elif jname == "waist_pitch":
            joint_goal[i] = goal[7]
        else:
            joint_goal[i] = 0
        joint_goal[i] = np.clip(joint_goal[i], *model.jnt_range[jid])

    # Same setpoint schedule for every morphology; this is a diagnostic probe.
    root_home = shape["hip"]
    data.qpos[:] = 0
    data.qpos[2] = root_home
    data.qpos[3] = 1
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    samples = []
    work_j = 0.0
    peak_torque = 0.0
    min_hand = float("inf")
    max_torso_pitch = 0.0
    peak_foot_force = 0.0
    peak_moving_sole_normal = 0.0
    peak_toy_normal = 0.0
    touchdowns = 0
    toy_contacts = 0
    fallen = False
    toy_id = model.geom("toy").id
    floor_id = model.geom("floor").id
    sole_ids = {model.geom(f"{side}_sole").id for side in ("left", "right")}
    previous_foot_contact = {gid: False for gid in sole_ids}
    end_effector_ids = {model.geom("left_hand").id, model.geom("left_finger_tip").id}
    pitch_rates = []
    saturation_steps = np.zeros(model.nu, dtype=int)
    renderer = mujoco.Renderer(model, height=448, width=640) if render else None
    camera = mujoco.MjvCamera() if render else None
    if render:
        camera.lookat[:] = [0.055, 0, 0.20]
        camera.distance = 0.66
        camera.azimuth = 135
        camera.elevation = -16
    writer = imageio.get_writer(OUT / f"{name}.mp4", fps=20, codec="libx264", quality=7) if render else None
    total_steps = int(5.6 / dt)
    for step in range(total_steps):
        t = step * dt
        progress = np.clip(((t - 0.7) if t <= 3.3 else (5.1 - t)) / 1.8, 0, 1)
        blend = progress * progress * (3 - 2 * progress)
        data.ctrl[:] = home + blend * (joint_goal - home)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            fallen = True
            break
        torque = np.abs(data.actuator_force)
        saturation_steps += torque >= 0.9 * model.actuator_forcerange[:, 1]
        velocity = np.array([data.qvel[model.jnt_dofadr[model.joint(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).removesuffix("_servo")).id]] for i in range(model.nu)])
        work_j += float(np.maximum(data.actuator_force * velocity, 0).sum() * dt)
        peak_torque = max(peak_torque, float(torque.max()))
        hand_distance = float(np.linalg.norm(data.site("left_hand_center").xpos - target_xyz))
        min_hand = min(min_hand, hand_distance)
        rot = data.body("torso").xmat.reshape(3, 3)
        pitch = float(np.arctan2(-rot[2, 0], rot[0, 0]))
        max_torso_pitch = max(max_torso_pitch, abs(pitch))
        pitch_rates.append(float(data.cvel[model.body("torso").id, 1]))
        if data.body("head").xpos[2] < 0.18:
            fallen = True
        sole_normal = 0.0
        foot_contact = {gid: False for gid in sole_ids}
        for ci in range(data.ncon):
            contact = data.contact[ci]
            geoms = {contact.geom1, contact.geom2}
            if toy_id in geoms and geoms.intersection(end_effector_ids):
                toy_contacts += 1
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                peak_toy_normal = max(peak_toy_normal, abs(float(wrench[0])))
            if floor_id in geoms and geoms.intersection(sole_ids):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                peak_foot_force = max(peak_foot_force, float(np.linalg.norm(wrench[:3])))
                sole_normal += abs(float(wrench[0]))
                foot_contact[next(iter(geoms.intersection(sole_ids)))] = True
        if t > 0.7:
            peak_moving_sole_normal = max(peak_moving_sole_normal, sole_normal)
            touchdowns += sum(foot_contact[gid] and not previous_foot_contact[gid] for gid in sole_ids)
        previous_foot_contact = foot_contact
        if step % 10 == 0:
            samples.append(dict(t=round(t, 3), root_z=round(float(data.qpos[2]), 4), torso_pitch_deg=round(float(np.degrees(pitch)), 2), hand_error_m=round(hand_distance, 4), peak_actuator_nm=round(float(torque.max()), 3), foot_contacts=int(sum(1 for ci in range(data.ncon) if floor_id in {data.contact[ci].geom1, data.contact[ci].geom2} and bool({data.contact[ci].geom1, data.contact[ci].geom2}.intersection(sole_ids))))))
        if render and step % 25 == 0:
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    if render:
        writer.close()
        renderer.close()
    rates = np.asarray(pitch_rates)
    smooth = np.convolve(rates, np.ones(151) / 151, mode="same")
    saturation = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).removesuffix("_servo"): round(float(count / total_steps), 4) for i, count in enumerate(saturation_steps) if count > 0}
    rot = data.body("torso").xmat.reshape(3, 3)
    end_pitch = float(np.degrees(np.arctan2(-rot[2, 0], rot[0, 0])))
    return dict(name=name, model_mass_kg=float(model.body_mass.sum()), xml_sha256=hashlib.sha256(xml.encode()).hexdigest(), ik=ik_info, dynamic=dict(fell=fallen, returned_upright=not fallen and abs(float(data.qpos[2]) - root_home) < 0.015 and abs(end_pitch) < 10, min_hand_error_m=min_hand, toy_touch_s=toy_contacts * dt, peak_toy_normal_n=peak_toy_normal, positive_mechanical_work_j=work_j, mean_positive_mechanical_power_w=work_j / (total_steps * dt), peak_actuator_torque_nm=peak_torque, actuator_saturation_fraction=saturation, max_torso_pitch_deg=float(np.degrees(max_torso_pitch)), rms_torso_pitch_rate_deg_s=float(np.degrees(np.sqrt(np.mean(np.square(rates))))), high_frequency_pitch_rate_deg_s=float(np.degrees(np.sqrt(np.mean(np.square(rates - smooth))))), peak_single_sole_contact_n=peak_foot_force, peak_total_sole_normal_n=peak_moving_sole_normal, foot_touchdowns=touchdowns, end_root_z_m=float(data.qpos[2]), end_torso_pitch_deg=end_pitch), trace=samples)


def write_viewer(rows, sweep):
    labels = {"balanced": "Balanced", "leggy": "Long legs, short arms", "low_shoulder": "Low shoulder, long arms", "waist_pitch": "Balanced + waist pitch", "ape_short_torso": "Ape-like: 4 cm torso, 35 cm tall", "high_hip_short_torso": "Short torso via high hip, 45 cm tall"}
    cards = []
    for row in rows:
        name = row["name"]
        cards.append(f'''<section><h2>{labels[name]}</h2><video controls muted playsinline preload="metadata" src="{name}.mp4"></video>
          <p>End-effector floor-target touch: <strong>{row['toy_touch_s']:.2f} s</strong> · Return upright: <strong>{'yes' if row['returned_upright'] else 'no'}</strong></p>
          <p>Positive joint work: {row['positive_mechanical_work_j']:.2f} J · Peak sole normal force: {row['peak_total_sole_normal_n']:.1f} N · High-frequency torso pitch rate: {row['high_frequency_pitch_rate_deg_s']:.1f}°/s</p>
          <a href="{name}.json">Trace + metrics</a> · <a href="{name}.xml">MuJoCo model</a></section>''')
    sweep_rows = []
    for name in SHAPES:
        cells = []
        for distance in ("near", "standard", "far"):
            metrics = sweep[distance][name]
            touched = metrics["toy_touch_s"] > 0
            succeeded = touched and metrics["returned_upright"]
            status = "touch + recover" if succeeded else "touch + fall" if touched else "miss"
            video = f' · <a href="{name}-far.mp4">video</a>' if distance == "far" else ""
            cells.append(f'<td class="{"pass" if succeeded else "miss"}">{status}<br><small>{metrics["min_hand_error_m"]:.3f} m closest{video}</small></td>')
        sweep_rows.append(f'<tr><th>{labels[name]}</th>{"".join(cells)}</tr>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compact body dynamics comparison</title>
<style>:root{color-scheme:dark;font-family:system-ui,sans-serif}body{max-width:1400px;margin:auto;padding:20px;background:#10141c;color:#e7edf4}h1{margin-bottom:5px}p{line-height:1.4}a{color:#91caff}.grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:18px}section{background:#1d2632;padding:15px;border-radius:12px}h2{margin:0 0 10px}video{width:100%;display:block;border-radius:8px}button{font:inherit;margin:8px 8px 18px 0;padding:8px 13px}small{color:#aebbc9}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>Compact body: floor-touch probe</h1><p>Five 45 cm bodies and one 35 cm ape-like candidate share contact physics, joint limits and the same diagnostic crouch/reach/return controller. Play together to inspect the movement.</p>
<button onclick="document.querySelectorAll('video').forEach(v=>v.play())">Play all</button><button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button><button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.pause()})">Reset all</button>
<div class="grid">''' + "".join(cards) + '''</div><h2>Floor target placement sweep</h2><p>Same floor height and lateral offset, three forward positions: 8, 13, and 18 cm from the starting pelvis origin. Green means actual hand-or-finger contact followed by an upright recovery. The far-case video links show the extra reach and a fall.</p><table><thead><tr><th>Body</th><th>Near · 8 cm</th><th>Standard · 13 cm</th><th>Far · 18 cm</th></tr></thead><tbody>''' + "".join(sweep_rows) + '''</tbody></table><p><a href="reach_sweep.json">Reach sweep data</a></p><p><small>This is a mechanical screen, not learned locomotion or grasping. The red target is fixed to the floor. Joint work is a lower-bound mechanical measure, not battery electricity. A deliberately bending torso raises pitch-rate metrics; that is not automatically jitter. Standing feet have no repeated landing impacts in this task.</small></p></html>'''
    html = html.replace("</style>", "table{border-collapse:collapse;width:100%;margin:10px 0 22px}td,th{text-align:left;border-bottom:1px solid #465261;padding:12px}.pass{color:#9de0b3}.miss{color:#ffb5ad}</style>")
    (OUT / "index.html").write_text(html)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, shape in SHAPES.items():
        result = simulate(name, shape)
        (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
        rows.append({"name": name, "model_mass_kg": result["model_mass_kg"], **result["ik"], **result["dynamic"]})
        print(f"{name}: IK {result['ik']['hand_error_m']:.3f} m, dynamic {result['dynamic']['min_hand_error_m']:.3f} m, touch {result['dynamic']['toy_touch_s']:.2f} s, fell {result['dynamic']['fell']}, work {result['dynamic']['positive_mechanical_work_j']:.2f} J")
    sweep = {"standard": {row["name"]: {key: row[key] for key in ("toy_touch_s", "min_hand_error_m", "fell", "returned_upright")} for row in rows}}
    for label, target_xyz in (("near", (0.08, 0.10, 0.045)), ("far", (0.18, 0.10, 0.045))):
        sweep[label] = {}
        for name, shape in SHAPES.items():
            run_name = f"{name}-{label}"
            result = simulate(run_name, shape, target_xyz, render=label == "far")
            (OUT / f"{run_name}.json").write_text(json.dumps(result, indent=2))
            sweep[label][name] = {key: result["dynamic"][key] for key in ("toy_touch_s", "min_hand_error_m", "fell", "returned_upright")}
            print(f"{run_name}: dynamic {result['dynamic']['min_hand_error_m']:.3f} m, touch {result['dynamic']['toy_touch_s']:.2f} s, fell {result['dynamic']['fell']}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    (OUT / "reach_sweep.json").write_text(json.dumps(sweep, indent=2))
    (OUT / "manifest.json").write_text(json.dumps({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "mujoco_version": mujoco.__version__, "task": "fixed-floor-target touch, 0.7-2.5 s approach, 3.3-5.1 s return; diagnostic setpoint control, not learned policy", "model_xml_sha256": {row["name"]: hashlib.sha256((OUT / f'{row["name"]}.xml').read_bytes()).hexdigest() for row in rows}}, indent=2))
    write_viewer(rows, sweep)


if __name__ == "__main__":
    main()
