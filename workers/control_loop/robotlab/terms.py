"""Bounded residual actions and proprioception; no simulator world position in policy inputs."""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


def attitude_and_odometry(asset):
    from .config import ORDER, ROBOT_CONFIG
    q, dq = asset.data.joint_pos.torch, asset.data.joint_vel.torch
    g, gyro = asset.data.projected_gravity_b.torch, asset.data.root_ang_vel_b.torch
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    def mean(suffix, values):
        return values[:, [ORDER.index(side + suffix) for side in ("left", "right")]].mean(dim=1)
    h, k, w = [mean(s, q) for s in ("_thigh_joint", "_shin_joint", "_wheel_joint")]
    hd, kd, wd = [mean(s, dq) for s in ("_thigh_joint", "_shin_joint", "_wheel_joint")]
    d = ROBOT_CONFIG["dimensions"]
    p, pd = pitch, gyro[:, 1]
    x = d["wheel_radius_m"] * (w + h + k + p) + d["hip_drop_m"] * torch.sin(p) + d["thigh_m"] * torch.sin(p+h) + d["shin_m"] * torch.sin(p+h+k)
    v = d["wheel_radius_m"] * (wd+hd+kd+pd) + d["hip_drop_m"]*torch.cos(p)*pd + d["thigh_m"]*torch.cos(p+h)*(pd+hd) + d["shin_m"]*torch.cos(p+h+k)*(pd+hd+kd)
    return pitch, gyro, x, v


class ResidualMotion(ActionTerm):
    """Four residuals: two wheel speeds plus symmetric hip/knee position offsets.

    Native ideal PD actuators supply the inner loop. This is a simulation adapter,
    not an implementation of the serial session/arming protocol.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from .config import ORDER, JOINTS, REST
        self._raw = torch.zeros((env.num_envs, 4), device=env.device)
        self.targets = torch.tensor([REST[n] for n in ORDER], device=env.device).repeat(env.num_envs, 1)
        self.previous = self.targets.clone()
        self.reference_x = torch.zeros(env.num_envs, device=env.device)
        self.limit = torch.tensor([j["velocity_limit_rad_s"] for j in JOINTS], device=env.device)
        self.wheels = [i for i,j in enumerate(JOINTS) if j["continuous"]]
        self.positions = [i for i,j in enumerate(JOINTS) if not j["continuous"]]
        self.valid = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    @property
    def action_dim(self): return 4
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self.targets

    def process_actions(self, actions):
        from .config import ORDER, REST, ROBOT_CONFIG, SETTINGS, JOINTS
        self.valid = torch.isfinite(actions).all(dim=1)
        self._raw[:] = torch.nan_to_num(actions).clamp(-1, 1)
        p, gyro, x, v = attitude_and_odometry(self._asset)
        cmd = self._env.command_manager.get_command("motion")
        self.reference_x += cmd[:, 0] * self._env.step_dt
        gains = SETTINGS["balance"]
        speed = gains["pitch_gain_m_s"]*(p-SETTINGS["pitch_trim_rad"]) + gains["pitch_rate_gain_m"]*gyro[:,1] + gains["position_gain_s_inv"]*(x-self.reference_x) + gains["velocity_gain"]*(v-cmd[:,0]) + cmd[:,0]
        radius = ROBOT_CONFIG["dimensions"]["wheel_radius_m"]
        steering = (cmd[:,2]-gyro[:,2]) * SETTINGS["yaw_rate_gain"]
        self.targets[:] = torch.tensor([REST[n] for n in ORDER], device=self._env.device)
        for side, sign in [("left", -1), ("right", 1)]:
            i = ORDER.index(side+"_wheel_joint")
            self.targets[:,i] = speed/radius + sign*steering + self._raw[:, 0 if side=="left" else 1]*SETTINGS["wheel_residual_rad_s"]
            self.targets[:,ORDER.index(side+"_thigh_joint")] += self._raw[:,2]*SETTINGS["leg_residual_rad"]
            self.targets[:,ORDER.index(side+"_shin_joint")] += self._raw[:,3]*SETTINGS["leg_residual_rad"]
        for i,j in enumerate(JOINTS):
            if j["continuous"]:
                self.targets[:,i].clamp_(-self.limit[i],self.limit[i])
            else:
                self.targets[:,i].clamp_(j["limits"][0],j["limits"][1])
                delta=(self.targets[:,i]-self.previous[:,i]).clamp(-self.limit[i]*self._env.step_dt,self.limit[i]*self._env.step_dt)
                self.targets[:,i]=self.previous[:,i]+delta
        self.targets[~self.valid] = self.previous[~self.valid]
        self.targets[~self.valid, self.wheels[0]] = 0
        self.targets[~self.valid, self.wheels[1]] = 0
        self.previous[:] = self.targets

    def apply_actions(self):
        self._asset.set_joint_position_target(self.targets[:,self.positions], joint_ids=self.positions)
        self._asset.set_joint_velocity_target(self.targets[:,self.wheels], joint_ids=self.wheels)

    def reset(self, env_ids=None):
        from .config import ORDER, REST
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.reference_x[ids] = 0
        self.targets[ids] = torch.tensor([REST[n] for n in ORDER], device=self._env.device)
        self.previous[ids] = self.targets[ids]
        self.valid[ids] = True


@configclass
class ResidualMotionCfg(ActionTermCfg):
    class_type = ResidualMotion


def proprioception(env):
    asset = env.scene["robot"]
    term = env.action_manager.get_term("motion")
    p, gyro, x, v = attitude_and_odometry(asset)
    return torch.cat([asset.data.joint_pos.torch, asset.data.joint_vel.torch, asset.data.projected_gravity_b.torch, gyro, env.command_manager.get_command("motion"), (x-term.reference_x)[:,None], v[:,None], term.raw_actions],dim=1)


def tracking_reward(env):
    # True body velocity is an evaluator/reward signal, never a policy input.
    cmd = env.command_manager.get_command("motion")
    asset = env.scene["robot"]
    error = (asset.data.root_lin_vel_b.torch[:,0]-cmd[:,0])**2/0.01 + (asset.data.root_ang_vel_b.torch[:,2]-cmd[:,2])**2/0.25
    return torch.exp(-error)


def upright_reward(env):
    from .config import SETTINGS
    pitch,_,_,_ = attitude_and_odometry(env.scene["robot"])
    return torch.exp(-((pitch-SETTINGS["pitch_trim_rad"])/0.1)**2)


def fallen(env):
    g=env.scene["robot"].data.projected_gravity_b.torch
    return (g[:,2] > -0.96) | (~env.action_manager.get_term("motion").valid)


def overspeed(env):
    term=env.action_manager.get_term("motion")
    return (env.scene["robot"].data.joint_vel.torch.abs() > term.limit*1.25).any(dim=1)
