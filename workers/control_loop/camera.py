"""Native Isaac head-camera check before attaching higher-level learning."""
import argparse,json,os,math
from pathlib import Path
import torch
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--skills',action='store_true');add_launcher_args(p);a=p.parse_args()
a.output.mkdir(exist_ok=False)
from scene import create_target
from robotlab.config import PipEnvCfg,INTERFACE
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim
import cv2,numpy as np
cfg=PipEnvCfg();cfg.seed=11;cfg.scene.num_envs=1;cfg.episode_length_s=25;cfg.sim.render_interval=50
cfg.scene.target=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Target',spawn=sim.UsdFileCfg(usd_path=str(create_target(a.output/'target'))))
cfg.scene.light=AssetBaseCfg(prim_path='/World/Light',spawn=sim.DomeLightCfg(intensity=1500))
if os.environ.get('ROBOTLAB_GAUSSIAN_USD'):
    cfg.scene.capture=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Capture',spawn=sim.UsdFileCfg(usd_path=os.environ['ROBOTLAB_GAUSSIAN_USD']))
# World camera convention: forward +X, up +Z; quaternion is XYZW.
cfg.scene.camera=CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/torso/neck_yaw/head/Camera',offset=CameraCfg.OffsetCfg(pos=tuple(INTERFACE["camera_frame"]["xyz_m"]),rot=(0,0,0,1),convention='world'),spawn=sim.PinholeCameraCfg(focal_length=24,horizontal_aperture=24*2*math.tan(math.radians(INTERFACE["camera_frame"]["optics"]["nominal_vertical_fov_deg"]/2))*320/240,clipping_range=(.02,20)),width=320,height=240,data_types=['rgb'],renderer_cfg=OVRTXRendererCfg(log_file_path=str(a.output/'renderer.log')))
# This renderer consumes sensor poses; the default caches initialization pose.
cfg.scene.camera.update_latest_camera_pose=True
with launch_simulation(cfg,a):
    env=ManagerBasedRLEnv(cfg=cfg);env.reset()
    if os.environ.get('ROBOTLAB_GAUSSIAN_USD'):
        from scene import inspect_gaussian_placement
        placement=inspect_gaussian_placement(sim.get_current_stage())
        (a.output/'gaussian-placement.json').write_text(json.dumps(placement,indent=2))
        if placement['status']!='pass':raise RuntimeError('Native Gaussian placement differs from the declared gym placement')
    for _ in range(3):env.sim.render()
    sensor=env.scene['camera'];sensor.update(env.step_dt,force_recompute=True)
    rgb=sensor.data.output['rgb'].torch[0].cpu().numpy()[:,:,:3]
    cv2.imwrite(str(a.output/'camera.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    k=sensor.data.intrinsic_matrices[0].cpu().numpy()
    from perception_control import detect_target,approach_skill
    estimate=detect_target(rgb,k,.2,0)
    from dataclasses import asdict
    (a.output/'report.json').write_text(json.dumps({'status':'pass' if estimate else 'fail','intrinsics':k.tolist(),'estimate':asdict(estimate) if estimate else None,'skill':asdict(approach_skill(estimate,.02)),'scope':'native camera capture/detection only; learned high-level control not yet exercised'},indent=2))
    if a.skills:
        from skill_loop import run_skills
        if estimate is None:raise RuntimeError('Native camera did not detect target')
        run_skills(env,Path(os.environ.get('ROBOTLAB_MOTOR_CHECKPOINT',str(Path(__file__).parent/'motor.pt'))),a.output/'skills')
    # Reuse the single native camera only AFTER all policy observations/evaluation.
    # The two-camera OVRTX path in v5 returned an aliased/cropped image buffer.
    # This inspection frame is never fed to the policy or its calibration check.
    sensor.set_world_poses_from_view(eyes=torch.tensor([[.55,-.6,.55]],device=env.device),targets=torch.tensor([[0.,0.,.25]],device=env.device))
    for _ in range(3):env.sim.render()
    sensor.update(env.step_dt,force_recompute=True)
    cv2.imwrite(str(a.output/'overview.png'),cv2.cvtColor(sensor.data.output['rgb'].torch[0].cpu().numpy()[:,:,:3],cv2.COLOR_RGB2BGR))
    env.close()
