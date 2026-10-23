"""Make a source-bound Pip arm clip for frozen world-model transfer inspection.

Supported torso, real CAD visuals, bounded PD/pack dynamics. This fixture isolates
one arm; it is not free-base manipulation and does not qualify end-effector control.
"""
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lab.mujoco_learning import PipMotorEnv,BUNDLE,digest
from lab.artifact_store import snapshot_file,plan_migration,apply_migration,verify_manifest


def prepare():
    output=ROOT/'artifacts/world-model-pip-inputs-v1';output.mkdir(exist_ok=False)
    contract=PipMotorEnv();model=mujoco.MjModel.from_xml_path(str(BUNDLE/'model/supported.xml'));data=mujoco.MjData(model)
    qidx=np.array([model.jnt_qposadr[model.joint(n).id] for n in contract.order])
    vidx=np.array([model.jnt_dofadr[model.joint(n).id] for n in contract.order])
    aids=np.array([model.actuator(n+'_motor').id for n in contract.order])
    data.qpos[qidx]=contract.rest;mujoco.mj_forward(model,data)
    target=contract.rest.copy();contract.pack.reset()
    contract.pack._queue[:]=np.stack((target,np.zeros(16),np.zeros(16)),axis=1)
    renderer=mujoco.Renderer(model,height=480,width=640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[0.,.03,.24];camera.distance=.7;camera.azimuth=35;camera.elevation=-10
    poses=[];grippers=[];frames=[];trace=[]
    gripper=model.body('left_gripper').id
    try:
        for frame in range(13):
            while data.time+1e-9<frame/3:
                target[contract.order.index('left_forearm_joint')]=-.4-.25*np.sin(np.pi*data.time/4)
                target[contract.order.index('left_upper_arm_joint')]=-.15*np.sin(np.pi*data.time/4)
                electrical=contract.pack.step(data.qpos[qidx],data.qvel[vidx],target,np.zeros(16))
                data.ctrl[aids]=electrical['applied_effort_nm'];mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            rotation=Rotation.from_matrix(data.xmat[gripper].reshape(3,3)).as_euler('xyz')
            poses.append([*data.xpos[gripper],*rotation])
            grippers.append(float((.45-data.qpos[qidx[contract.order.index('left_gripper_joint')]])/.9))
            renderer.update_scene(data,camera=camera);frames.append(renderer.render())
            trace.append({'time_s':float(data.time),'q_rad':data.qpos[qidx].tolist(),'targets_rad':target.tolist(),
                'end_effector_world_xyz_rpy':poses[-1],'estimated_voltage_v':contract.pack.voltage_v})
        imageio.mimwrite(output/'reference.mp4',frames,fps=3,codec='libx264',quality=8,macro_block_size=1)
        annotation={'state':poses,'continuous_gripper_state':grippers,'videos':['reference.mp4'],
            'task':'Move the left forearm slowly with the torso supported.'}
        # Raw plot actions use the same relative-pose convention as native Cosmos.
        actions=[]
        for i in range(12):
            before=Rotation.from_euler('xyz',poses[i][3:]).as_matrix();after=Rotation.from_euler('xyz',poses[i+1][3:]).as_matrix()
            actions.append([*(before.T@(np.array(poses[i+1][:3])-poses[i][:3])),*Rotation.from_matrix(before.T@after).as_euler('xyz'),grippers[i+1]])
        annotation['action']=actions
        (output/'annotation.json').write_text(json.dumps(annotation))
        (output/'trace.json').write_text(json.dumps(trace))
        snapshot_file(Path(__file__),output/'prepare-source.py')
        manifest={'schema_version':1,'source_kind':'physics','robot':'Pip supported-torso arm fixture',
            'scope':'Zero-shot frozen-world-model transfer input, not an adapted simulator or manipulation dataset.',
            'bundle_manifest_sha256':digest(BUNDLE/'manifest.json'),'model_sha256':digest(BUNDLE/'model/supported.xml'),
            'frames':13,'fps':3,'physics_hz':500,'frame_times_s':[r['time_s'] for r in trace],
            'end_effector':'left_gripper link origin, world metres + XYZ Euler radians; local pose deltas before native scale 20',
            'gripper':'Provisional normalized joint closure (0.45-q)/0.9; actual jaw-width mapping uncalibrated.',
            'camera':'Fixed external observer, not Pip head camera; position/angle fixed for all frames.',
            'files':[{'file':p,'sha256':digest(output/p)} for p in ['reference.mp4','annotation.json']]}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    finally:renderer.close();contract.close()
    migration=plan_migration(ROOT,[output]);apply_migration(ROOT,migration);verify_manifest(ROOT,migration)
    print(output)

if __name__=='__main__':prepare()
