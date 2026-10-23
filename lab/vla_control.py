"""Robot-specific VLA boundary. No task phases, object poses or teacher at inference."""
import numpy as np
import mujoco
from lab.floor_pickup import FloorProbe

FPS = 10
INSTRUCTION = 'Pick up the orange block from the floor and put it on the green area.'
HOLD = 'Stay still and leave the orange block on the floor.'


class VisualRobot:
    def __init__(self, scene, motor_checkpoint=None, motor_contract=None, motor_plant_reference=None):
        self.motor = FloorProbe(scene, 1.)
        self.positions = np.flatnonzero(~self.motor.wheels)
        self.grip = self.motor.order.index('left_gripper_joint')
        self.low = np.array([self.motor.joints[i]['limits'][0] for i in self.positions] + [-.06, -.15, -.12])
        self.high = np.array([self.motor.joints[i]['limits'][1] for i in self.positions] + [.06, .15, .12])
        self.learned_motor = None
        self.reset()
        if motor_checkpoint is not None:
            from lab.learned_motor import LearnedMotor
            self.learned_motor = LearnedMotor(self.motor,motor_checkpoint,motor_contract,motor_plant_reference)
        self.camera = mujoco.Renderer(self.motor.model, height=180, width=320)

    def reset(self, object_x=.015, object_y=.140):
        m = self.motor
        m.reset(seed=91)
        m.data.qpos[0] = -.25
        addr = m.model.jnt_qposadr[m.model.joint('task_block_free').id]
        m.data.qpos[addr:addr+3] = [object_x, object_y, .015]
        m.pack.effort_limit[self.grip] = .4
        mujoco.mj_forward(m.model, m.data)
        if self.learned_motor is not None:
            self.learned_motor.reset()

    def state(self):
        m = self.motor
        _, g, gyro, x, v, _ = m.motion()
        q = np.where(m.wheels, m.data.qvel[m.vidx], m.data.qpos[m.qidx])
        return np.r_[q, g, gyro, x, v, m.electrical['applied_effort_nm'][self.grip]].astype(np.float32)

    def image(self):
        self.camera.update_scene(self.motor.data, camera='head_camera')
        return self.camera.render().copy()

    def encode_action(self, target, speed, yaw_rate):
        m = self.motor
        x = m.motion()[3]
        return np.r_[target[self.positions], speed, yaw_rate, m.reference_x-x].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (17,) or not np.isfinite(action).all():
            raise ValueError('Expected 17 finite VLA commands')
        accepted = np.clip(action, self.low, self.high)
        m = self.motor
        if self.learned_motor is not None:
            self.learned_motor.set_goal(accepted)
            rows = []
            for _ in range(5):
                row = self.learned_motor.advance()
                rows.append(row)
                if abs(row['pitch_rad']) > .28:
                    break
            return rows, accepted
        target = m.targets.copy()
        target[self.positions] = accepted[:14]
        m.reference_x = m.motion()[3] + accepted[16]
        m.heading_error = -accepted[15] / 3
        rows = []
        for _ in range(5):
            row = m.advance(target, accepted[14])
            addr = m.model.jnt_dofadr[m.model.joint('task_block_free').id]
            row['block_velocity_m_s'] = m.data.qvel[addr:addr+3].tolist()
            rows.append(row)
            if abs(row['pitch_rad']) > .28:
                break
        return rows, accepted

    def contract(self):
        m = self.motor
        joint_names = [n + ('.velocity_rad_s' if w else '.position_rad') for n, w in zip(m.order, m.wheels)]
        return {'version': 1, 'fps': FPS, 'physics_hz': 500, 'servo_hz': 50,
                'state_names': joint_names + ['gravity_body.'+x for x in 'xyz'] + ['gyro_body_rad_s.'+x for x in 'xyz'] + ['encoder_odometry_m', 'encoder_velocity_m_s', 'gripper_effort_proxy_nm'],
                'action_names': [m.order[i]+'.position_rad' for i in self.positions] + ['forward_velocity_m_s', 'yaw_rate_rad_s', 'odometry_reference_error_m'],
                'action_min': self.low.tolist(), 'action_max': self.high.tolist(),
                'low_level': ('Learned PPO reference tracker, defined servo/electrical proxy; no analytic balance or gravity feedforward.'
                              if self.learned_motor is not None else
                              'Defined encoder/IMU balance, quasi-static gravity feedforward and delayed electrical servo proxy; not learned whole-body control.'),
                'timing': 'Fresh camera and state before the first labeled command; inference simulation pauses, so wall latency is reported separately.',
                'policy_excludes': ['task phase', 'object truth', 'teacher estimates', 'future image', 'episode clock']}

    def close(self):
        self.camera.close()
        self.motor.close()


def evaluate(rows, initial=(.015,.14,.015), goal=(.365,.14), task='cleanup'):
    """Physical outcome only; task phase labels cannot change acceptance."""
    if not rows:
        return {'success': False, 'reason': 'empty_episode'}
    initial = np.asarray(initial)
    hold = max_hold = max_lift = transport = settle = 0.
    lifted = supported = released = dropped = fallen = False
    previous_t = 0.
    for row in rows:
        dt = row['time_s'] - previous_t
        previous_t = row['time_s']
        pos = np.asarray(row['block_m'])
        contacts = row['block_contacts_n']
        grasp = all(contacts.get(n, 0) > .05 for n in ('left_forearm_1', 'left_gripper_0'))
        hold = hold + dt if grasp else 0.
        max_hold = max(max_hold, hold)
        max_lift = max(max_lift, float(pos[2]-initial[2]))
        lifted |= pos[2] > initial[2]+.02 and max_hold >= .15
        inside = bool(np.all(np.abs(pos[:2]-goal) < .06))
        floor = contacts.get('floor', 0) > .05
        if lifted:
            transport = max(transport, float(np.linalg.norm(pos[:2]-initial[:2])))
            supported |= floor and inside and grasp
            if not grasp and not (floor and inside):
                dropped = True
            released |= supported and floor and inside and not grasp
        speed = np.linalg.norm(row['block_velocity_m_s'])
        settled = released and inside and floor and not grasp and speed < .02
        settle = settle + dt if settled else 0.
        fallen |= abs(row['pitch_rad']) > .28
    checks = {'bilateral_grasp': max_hold >= .15, 'lift': bool(lifted),
              'transport_at_least_30cm': transport >= .30, 'no_drop': not dropped,
              'supported_set_down': bool(supported), 'released_and_settled': settle >= .5,
              'no_fall': not fallen}
    if task == 'hold':
        root0 = np.asarray(rows[0]['root_position_world_m'])
        q0 = np.asarray(rows[0]['q'])
        nonwheel = [i for i in range(16) if i not in (8, 15)]
        checks = {'base_stays_within_2cm': all(np.linalg.norm(np.asarray(r['root_position_world_m'])-root0) < .02 for r in rows),
                  'posture_stays_within_0_1rad': all(np.max(np.abs(np.asarray(r['q'])[nonwheel]-q0[nonwheel])) < .1 for r in rows),
                  'object_undisturbed': all(np.linalg.norm(np.asarray(r['block_m'])-initial) < .01 for r in rows),
                  'no_fall': not fallen, 'full_hold_duration': rows[-1]['time_s'] >= 7.9}
    return {'success': all(checks.values()), 'checks': checks, 'max_lift_m': max_lift,
            'max_transport_m': transport, 'max_grasp_hold_s': max_hold, 'settle_s': settle}
