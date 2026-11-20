"""Torch translation of R3 Runtime, called once BEFORE each native physics step.

Target radians and sensors use exact ordered source names. Actor raw history is
owned by the caller. Random draws follow Torch streams (same source distribution,
not NumPy byte identity); bias/noise injection enables matched algebra audits.
"""
from pathlib import Path
import hashlib
import json
import math
import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "artifacts/robot/model/timestep"


def source_contract():
    pointer = json.loads((ROOT / "robot/model/current-model.json").read_text())
    for field in ("rigid_MJCF", "actuators", "sensors", "timing"):
        record = pointer[field]
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Selected {field} hash changed")
        if path.parent != MODEL:
            raise ValueError("This boundary implements only the selected R3 contract")
    return pointer


def interp(x, xp, fp, *, left, right):
    """np.interp-equivalent piecewise linear interpolation for batched tensors."""
    index = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, len(xp)-1)
    value = fp[index-1] + (x-xp[index-1])*(fp[index]-fp[index-1])/(xp[index]-xp[index-1])
    return torch.where(x < xp[0], left, torch.where(x > xp[-1], right, value))


class BatchedRuntime:
    physics_dt_s = .00025
    policy_dt_s = .02
    decimation = 80

    def __init__(self, num_envs, *, device="cpu", dtype=torch.float32, rail_V=12.,
                 command_delay_s=.02, feedback_delay_s=.01, peak=False,
                 backlash_rad=0., seed=0, c047=None, duty=None, ranges_deg=None, joint_models=None, physics_dt_s=None):
        self.source = source_contract()
        # physics_dt_s: a coarser physics step for r4 training speed; None keeps R3's .25 ms.
        # Delays stay whole ticks; IMU (208 Hz) and joint (100 Hz) sampling take the first tick at or after
        # their due time, so a coarser step adds up to one tick of sampling jitter.
        if physics_dt_s is not None:
            self.physics_dt_s = float(physics_dt_s)
            self.decimation = round(self.policy_dt_s / self.physics_dt_s)
            if abs(self.decimation * self.physics_dt_s - self.policy_dt_s) > 1e-9:
                raise ValueError("physics step must divide the 20 ms policy period")
        self.contract = json.loads((MODEL / "actuators.json").read_text())
        # c047: a substitute servo's catalog law on the C047 joints (a
        # sensitivity, not the selected model); gain_scale scales their kp/kd.
        c047 = dict(c047 or {})
        gain_scale = c047.pop("gain_scale", 1.)
        self.contract["models"]["c047"].update(c047)
        self.acts = self.contract["actuators"]
        self.names = [a["joint"] for a in self.acts]
        self.imu = next(s for s in json.loads((MODEL / "sensors.json").read_text())["sensors"] if s["name"] == "imu_board")
        self.device, self.dtype, self.num_envs = device, dtype, num_envs
        self.peak, self.backlash = peak, backlash_rad
        # duty: r4 Feetech firmware loop on the C047 joints (dynamics.resolve()["duty"]): a
        # rate-limited internal target, duty = gain x error (less half the backlash) clamped
        # to +-max, torque = stall x (duty - qd/no-load speed) within +-stall. None keeps R3.
        self.duty = duty
        # joint_models: {joint: {"law": {...}, "duty": {...}}} for C047 joints that carry another
        # servo (mixed sizing, dynamics.resolve()["joints"]); law and duty then become per-joint
        # tensors, the other joints keeping the c047 law and duty above. None keeps the scalars.
        self.law = self.contract["models"]["c047"]
        if joint_models:
            names = [a["joint"] for a in self.contract["actuators"]]
            pick = lambda n, part, k, base: (joint_models.get(n) or {}).get(part, {}).get(k, base[k])
            self.law = {k: torch.as_tensor([pick(n, "law", k, self.law) for n in names], device=device, dtype=dtype)
                for k in ("stall_torque_Nm", "no_load_speed_rad_s", "rated_torque_Nm", "idle_current_A", "stall_current_A")}
            if duty:
                self.duty = {k: torch.as_tensor([pick(n, "duty", k, duty) for n in names], device=device, dtype=dtype) for k in duty}
        self.feedback_delay_s = float(feedback_delay_s)
        lag = command_delay_s / self.physics_dt_s
        if abs(lag-round(lag)) > 1e-8 or lag < 0 or feedback_delay_s < 0:
            raise ValueError("Command delay must be nonnegative whole physics ticks; feedback delay nonnegative")
        self.delay_ticks = round(lag)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.physics_armed = False
        self.tensor = lambda v: torch.as_tensor(v, device=device, dtype=dtype)
        self.kp = self.tensor([a["simulation_kp_Nm_rad"] for a in self.acts])
        self.kd = self.tensor([a["simulation_kd_Nm_s_rad"] for a in self.acts])
        # ranges_deg: best-estimate {joint: [lo, hi]} for joints whose source range is
        # unspecified (r4: the hips); otherwise they keep R3's +-.35 rad diagnostic envelope.
        ranges = [a["source_axis"]["range_deg"] or (ranges_deg or {}).get(a["joint"]) for a in self.acts]
        self.lower = self.tensor([math.radians(r[0]) if r is not None else -.35 for r in ranges])
        self.upper = self.tensor([math.radians(r[1]) if r is not None else .35 for r in ranges])
        self.c047 = torch.tensor([a["model"] == "c047" for a in self.acts], device=device)
        self.kp = torch.where(self.c047, self.kp*gain_scale, self.kp)
        self.kd = torch.where(self.c047, self.kd*gain_scale, self.kd)
        self.rail = self.tensor(rail_V).expand(num_envs).clone()
        if torch.any(~torch.isfinite(self.rail) | (self.rail <= 0)):
            raise ValueError("Positive finite rail required")
        self.rotation = self.tensor(self.imu["rotation_link_sensor"])
        band = self.imu["simulation_noise"]["equivalent_bandwidth_Hz"]
        noise = self.imu["source_noise"]
        self.noise_std = self.tensor([noise["accel_density_m_s2_sqrt_Hz"]*math.sqrt(band)]*3 + [noise["gyro_density_rad_s_sqrt_Hz"]*math.sqrt(band)]*3)
        curve = self.contract["models"]["xm335"]["performance_curve"]
        self.neck_torques = self.tensor(curve["torque_Nm"])
        self.neck_speeds = self.tensor(curve["speed_rad_s"])
        self.neck_currents = self.tensor(curve["current_A"])
        self.zero = self.tensor(0.)
        self.idle_neck_current = self.tensor(.017)
        n = num_envs
        self.rows = torch.arange(n, device=device)
        self.tick = torch.zeros(n, dtype=torch.long, device=device)
        self.pending_reset = torch.ones(n, dtype=torch.bool, device=device)
        self.target_history = torch.zeros((n, self.delay_ticks+1, 22), dtype=dtype, device=device)
        self.target = torch.zeros((n,22), dtype=dtype, device=device)
        self.initial_target = self.target.clone()
        self.previous_command = self.target.clone()
        self.torque = self.target.clone()
        self.current = self.target.clone()
        self.saturated = torch.zeros_like(self.target, dtype=torch.bool)
        self.beyond_stop = self.target.clone()
        self.internal_target = self.target.clone()
        self.internal_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.gravity_estimate = torch.zeros((n,3), dtype=dtype, device=device)
        self.sample_gyro = self.gravity_estimate.clone()
        self.accel_bias = self.gravity_estimate.clone()
        self.gyro_bias = self.gravity_estimate.clone()
        self.next_imu = torch.zeros(n, dtype=torch.float64, device=device)
        self.next_joint = self.next_imu.clone()
        self.last_imu_time = self.next_imu.clone()
        self.seq = torch.zeros(n, dtype=torch.long, device=device)
        self.packet_slots = math.ceil(feedback_delay_s/.01)+2
        self.packet_values = torch.zeros((n,self.packet_slots,50), dtype=dtype, device=device)
        self.packet_capture = torch.full((n,self.packet_slots), -float("inf"), dtype=torch.float64, device=device)
        self.packet_imu_capture = self.packet_capture.clone()
        self.last_values = torch.zeros((n,50), dtype=dtype, device=device)
        self.capture_time_s = self.next_imu.clone()
        self.imu_capture_time_s = self.next_imu.clone()
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.pending_reset[ids] = True
        self.tick[ids] = 0
        self.target_history[ids] = 0
        self.target[ids] = 0
        self.previous_command[ids] = 0
        self.torque[ids] = 0
        self.current[ids] = 0
        self.saturated[ids] = False
        self.beyond_stop[ids] = 0
        self.internal_valid[ids] = False  # the firmware target restarts from the measured position
        self.gravity_estimate[ids] = self.tensor([0.,0.,-1.])
        self.sample_gyro[ids] = 0
        self.next_imu[ids] = 0
        self.next_joint[ids] = 0
        self.last_imu_time[ids] = -1
        self.seq[ids] = 0
        self.packet_capture[ids] = -float("inf")
        self.packet_imu_capture[ids] = -float("inf")
        self.packet_values[ids] = 0
        self.last_values[ids] = 0
        self.capture_time_s[ids] = 0
        self.imu_capture_time_s[ids] = 0
        self.valid[ids] = False
        shape = self.accel_bias[ids].shape
        self.accel_bias[ids] = torch.rand(shape, device=self.device, dtype=self.dtype, generator=self.generator)*.1-.05
        self.gyro_bias[ids] = torch.rand(shape, device=self.device, dtype=self.dtype, generator=self.generator)*.006-.003

    def arm_physics_step(self):
        """ActionTerm.apply_actions arms actual decimation-loop writes only.

        MJLab reset/partial-reset writes and sim.forward must not advance this
        state machine. Direct algebra callers use step() without this hook.
        """
        self.physics_armed = True

    def motor(self, target, q, qd):
        error = target-q
        error = error.sign()*(error.abs()-self.backlash/2).clamp(min=0)
        raw = self.kp*error-self.kd*qd
        scale = self.rail[:,None]/12
        spec = self.law
        stall = spec["stall_torque_Nm"]*scale
        speed = spec["no_load_speed_rad_s"]*scale
        cap = stall if self.peak else spec["rated_torque_Nm"]*scale
        if self.duty:
            step = self.duty["rate"]*self.physics_dt_s
            start = torch.where(self.internal_valid[:,None], self.internal_target, q)
            self.internal_target = torch.minimum(torch.maximum(target, start-step), start+step)
            self.internal_valid[:] = True
            duty_error = self.internal_target-q
            duty_error = duty_error.sign()*(duty_error.abs()-self.duty["backlash"]/2).clamp(min=0)
            duty = (self.duty["gain"]*duty_error).clamp(-self.duty["max"], self.duty["max"])
            raw_c = stall*(duty-qd/speed)
            torque_c = raw_c.clamp(-cap, cap)
            saturated_c = ((torque_c-raw_c).abs() > 1e-9) | (duty.abs() >= self.duty["max"]-1e-9)
        else:
            lower = -torch.minimum(cap, (stall*(1+qd/speed)).clamp(min=0))
            upper = torch.minimum(cap, (stall*(1-qd/speed)).clamp(min=0))
            torque_c = raw.clamp(min=lower, max=upper)
            saturated_c = (torque_c-raw).abs() > 1e-9
        current_c = spec["idle_current_A"]+(spec["stall_current_A"]-spec["idle_current_A"])*torque_c.abs()/stall.clamp(min=1e-9)
        neck_scale = self.rail[:,None].clamp(max=12)/12
        # Scaling the query instead of xp preserves each world's voltage curve.
        torques, speeds = self.neck_torques, self.neck_speeds
        curve_limit = interp(qd.abs()/neck_scale, speeds.flip(0), torques.flip(0), left=torques[-1], right=self.zero)*neck_scale
        cap_n = curve_limit if self.peak else curve_limit.clamp(max=.2)
        torque_n = raw.clamp(min=-cap_n, max=cap_n)
        current_n = interp(torque_n.abs()/neck_scale, torques, self.neck_currents, left=self.idle_neck_current, right=self.neck_currents[-1])
        self.torque = torch.where(self.c047, torque_c, torque_n)
        self.current = torch.where(self.c047, current_c, current_n)
        self.saturated = torch.where(self.c047, saturated_c, (torque_n-raw).abs() > 1e-9)
        return self.torque

    def step(self, requested_target, q, qd, accel_sensor, gyro_sensor, *, sensor_noise=None):
        """Read pre-integration native buffers, return torques, then advance clock.

        All positions/velocities shape[N,22]; IMU inputs shape[N,3] in sensor
        frame. sensor_noise is optional already-scaled [N,6] for matched audits.
        Call after reset writes and forward; no warmup physics is inserted.
        """
        target = requested_target.clamp(min=self.lower,max=self.upper)
        self.beyond_stop = (requested_target-target).abs()  # rad the command lies past the stops
        self.initial_target = torch.where(self.pending_reset[:,None],q,self.initial_target)
        self.pending_reset[:] = False
        self.previous_command = target.clone()
        write_slot = self.tick % (self.delay_ticks+1)
        self.target_history[self.rows,write_slot] = target
        read_slot = (self.tick-self.delay_ticks) % (self.delay_ticks+1)
        self.target = torch.where((self.tick < self.delay_ticks)[:,None],self.initial_target,self.target_history[self.rows,read_slot])
        torque = self.motor(self.target,q,qd)
        t = self.tick.to(torch.float64)*self.physics_dt_s
        imu_now = t+1e-10 >= self.next_imu
        noise = (torch.randn((self.num_envs,6), device=self.device,dtype=self.dtype,generator=self.generator)*self.noise_std if sensor_noise is None else sensor_noise)
        accel = accel_sensor+self.accel_bias+noise[:,:3]
        gyro = gyro_sensor+self.gyro_bias+noise[:,3:]
        a_body = accel @ self.rotation.T
        g_body = gyro @ self.rotation.T
        dt = torch.where(self.last_imu_time < 0, 1/208, t-self.last_imu_time).to(self.dtype)
        gravity = self.gravity_estimate-torch.linalg.cross(g_body,self.gravity_estimate)*dt[:,None]
        norm = a_body.norm(dim=1,keepdim=True)
        gravity = torch.where(norm>1e-5,.98*gravity+.02*(-a_body/norm.clamp(min=1e-30)),gravity)
        gravity = gravity/gravity.norm(dim=1,keepdim=True)
        self.gravity_estimate = torch.where(imu_now[:,None],gravity,self.gravity_estimate)
        self.sample_gyro = torch.where(imu_now[:,None],g_body,self.sample_gyro)
        self.last_imu_time = torch.where(imu_now,t,self.last_imu_time)
        self.next_imu += imu_now.to(torch.float64)/208
        joint_now = t+1e-10 >= self.next_joint
        slot = self.seq % self.packet_slots
        packet = torch.cat((torch.round(q/(2*math.pi/4096))*(2*math.pi/4096),qd,self.sample_gyro,self.gravity_estimate),dim=1)
        self.packet_values[self.rows,slot] = torch.where(joint_now[:,None],packet,self.packet_values[self.rows,slot])
        self.packet_capture[self.rows,slot] = torch.where(joint_now,t,self.packet_capture[self.rows,slot])
        self.packet_imu_capture[self.rows,slot] = torch.where(joint_now,self.last_imu_time,self.packet_imu_capture[self.rows,slot])
        self.seq += joint_now
        self.next_joint += joint_now.to(torch.float64)*.01
        due = (self.packet_capture+self.feedback_delay_s <= t[:,None]+1e-10) & torch.isfinite(self.packet_capture)
        latest, selected = torch.where(due,self.packet_capture,-float("inf")).max(dim=1)
        self.valid = torch.isfinite(latest)
        self.last_values = torch.where(self.valid[:,None],self.packet_values[self.rows,selected],self.last_values)
        self.capture_time_s = torch.where(self.valid,latest,self.capture_time_s)
        self.imu_capture_time_s = torch.where(self.valid,self.packet_imu_capture[self.rows,selected],self.imu_capture_time_s)
        self.tick += 1
        return torque

    @property
    def time_s(self):
        return self.tick.to(torch.float64)*self.physics_dt_s

    @property
    def age_s(self):
        return self.time_s-self.capture_time_s

    def observation(self, command):
        """[N,75], with .valid=False until first feedback delivery. No root truth."""
        command = self.tensor(command).expand(self.num_envs,3)
        return torch.cat((self.last_values,command,self.previous_command),dim=1)
