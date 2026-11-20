"""Assess/publish one source-bound task replay, preserving incomplete horizons."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESULT = 'cove50-forward-task-runtime-20261026'


def read(p): return json.loads(Path(p).read_text())


def ref(p):
    p = Path(p).resolve()
    return dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


def write(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def distribution(x):
    a = np.asarray(x)
    if not a.size: return None
    return dict(zip(('min', 'p10', 'p50', 'p90', 'p99', 'max'), np.quantile(a, [0, .1, .5, .9, .99, 1]).tolist()))


def assess(directory):
    report = read(directory / 'report.json')
    exit_record = read(ROOT / 'data/walk-resume-r1-20261026/cpu-whole-exit.json')
    trace = next((directory / 'episodes').glob('*.npz'))
    with np.load(trace) as f: a = {k: f[k].copy() for k in f.files}
    model_ref = report['selected_interface']['selected_model']['rigid_MJCF']
    model_path = ROOT / model_ref['path']
    if ref(model_path)['sha256'] != model_ref['sha256']: raise ValueError('Model identity drift')
    model = mujoco.MjModel.from_xml_path(str(model_path)); data = mujoco.MjData(model)
    t = a['native_time_s']; q = a['qpos']; n = len(t)
    indices = np.flatnonzero(a['source_terminated'])
    first = int(indices[0]) if len(indices) else None
    active = np.arange(n) < first if first is not None else np.ones(n, dtype=bool)
    tilt = np.arccos(np.clip(1 - 2*(q[:, 4]**2 + q[:, 5]**2), -1, 1))
    velocity_error = a['root_velocity_body_m_s'][:, 0] - a['command3'][:, 0]
    yaw_error = a['root_ang_velocity_body_rad_s'][:, 2] - a['command3'][:, 2]
    sole_position = []; sole_speed = []
    sites = [model.site(name).id for name in ('R_sole_contact_frame', 'L_sole_contact_frame')]
    for pos, vel in zip(q, a['qvel']):
        data.qpos[:] = pos; data.qvel[:] = vel; mujoco.mj_forward(model, data)
        sole_position.append(data.site_xpos[sites].copy())
        speeds = []
        for site in sites:
            result = np.zeros(6); mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, result, 0)
            speeds.append(np.linalg.norm(result[3:5]))
        sole_speed.append(speeds)
    sole_speed = np.asarray(sole_speed)
    clipping = abs(a['requested_target22_rad'] - a['accepted_target22_rad'])
    names = report['selected_interface']['physical_order']
    curves = [dict(abs_forward_error_radius_m_s=r,
        fraction_of_pre_termination_controls=float(np.mean(abs(velocity_error[active]) <= r)) if active.any() else None,
        fraction_of_requested_400_controls=float(np.sum((abs(velocity_error) <= r) & active) / 400))
        for r in (.025, .05, .1, .2, .4, .8)]
    stations = []
    for time_s in (.02, .3, .6, .74, 1., 1.5, float(t[-1])):
        k = int(np.argmin(abs(t-time_s)))
        stations.append(dict(time_s=float(t[k]), root_xyz_m=q[k, :3].tolist(), torso_tilt_rad=float(tilt[k]),
            foot_world_z_force_N=a['feet_force_N'][k, :, 2].tolist(),
            distal_forearm_world_z_force_N=a['distal_forearm_force_N'][k, :, 2].tolist(),
            original_physical_termination=bool(a['source_terminated'][k])))
    summary = dict(schema='cove50.forward-task-runtime-assessment.v1', result_id=RESULT,
        source_report=ref(directory / 'report.json'), trace=ref(trace), model=model_ref,
        frozen_launch=ref(ROOT / 'data/walk-resume-r1-20261026/cpu-whole-launch.json'),
        operator_exit=exit_record, episode_completion='operator-truncated',
        original_requested_horizon_s=8., original_requested_controls=400,
        captured_controls=n, captured_duration_s=float(t[-1]), requested_control_coverage_fraction=n/400,
        native_physics_steps=report['native_physics_steps'], interrupted_partial_control_substeps=report['native_physics_steps']-n*80,
        ppo_rounds=0, optimizer_steps=0, new_trained_policies=0, world_slots=1, episodes=1, independent_start_families=1,
        first_original_physical_termination_s=float(t[first]) if first is not None else None,
        first_termination_root_delta_xy_m=(q[first, :2]-a['initial_qpos'][:2]).tolist() if first is not None else None,
        final_root_delta_xy_m=(q[-1, :2]-a['initial_qpos'][:2]).tolist(),
        root_height_m=distribution(q[:, 2]), torso_tilt_rad=distribution(tilt),
        forward_velocity_error_m_s=distribution(velocity_error),
        pre_termination_forward_velocity_error_m_s=distribution(velocity_error[active]),
        yaw_velocity_error_rad_s=distribution(yaw_error),
        command_error_accuracy_curve=curves,
        sensor_input_valid_fraction=float(a['input_sensor_valid'].mean()),
        first_input_sensor_valid=bool(a['input_sensor_valid'][0]),
        capture_age_s=distribution(t-a['sensor_capture_s']),
        native_minus_runtime_clock_s=distribution(t-a['runtime_time_s']),
        contact_force_semantics='Raw net force reported by primary foot/forearm ContactMatch. Negative world-z direction retained; absolute z is a floor-load magnitude proxy, not calibrated FSR or an isolated sole patch.',
        per_joint=[dict(joint=name, abs_torque_Nm=distribution(abs(a['torque22_Nm'][:, i])),
            pre_termination_saturated_fraction=float(a['saturated22'][active, i].mean()) if active.any() else None,
            requested_minus_accepted_abs_rad=distribution(clipping[:, i]),
            requested_target_modified_fraction=float(np.mean(clipping[:, i] > 1e-7))) for i, name in enumerate(names)],
        foot_abs_world_z_load_N_per_source_side=[distribution(abs(a['feet_force_N'][:, i, 2])) for i in range(2)],
        contacting_foot_horizontal_speed_m_s_per_source_side=[distribution(sole_speed[a['feet_found'][:, i] > 0, i]) for i in range(2)],
        reward_terms=[dict(name=name, weighted_rate_per_s=distribution(a['reward_terms'][:, i]))
            for i, name in enumerate(report['rewards'])],
        trajectory_stations=stations,
        observation='The mapped saved actor moved backward and drifted sideways under a +0.30 m/s forward command. At about0.74s a distal forearm touched the floor; the torso descended and remained on the floor through the retained2.44s tail. The CPU wall budget ended the visit; no recovery disturbance or learning was attempted.',
        limitations=['Only122of400 requested controls retained; the8s task is incomplete.',
            'Single nominalR3 seed on CPUWarp; no CUDA throughput, adaptation or hardware inference.',
            'Neutral-axis/sign mapping, changed kinematic chains, delayed sensing, assumed drive curves, contact shape and unknown hip command envelope all remain possible contributors; this visit does not isolate a URDF defect.',
            '50Hz state/effort snapshots may miss4kHz force peaks. Original interrupted-native-state.npz retains the final partial-control state.',
            'The original generic recorder close label learner_budget_censoring means wall-time operator interruption here; this correction follows the frozen exit record.',
            'NativeMJCF visual meshes retain historical reductions; corrected fullURDF visuals are a separate export.'],
        grades=dict(overall='Useful source-matched task/runtime failure exposure; forward gait transfer remains poor and the requested horizon is incomplete.',
            dimensions=[dict(dimension='Task/interface execution',grade=4,confidence='high',trend='new',reason='Original12reward terms, explicit81/24/22 interface and timedR3 boundary executed through upright motion and floor contact; GPU remains unmeasured.'),
                dict(dimension='Forward motion on selected body',grade=1,confidence='high',trend='new',reason='Backwards travel and floor contact under forward command; unchanged actor, zero updates.'),
                dict(dimension='Whole-horizon evidence',grade=1.5,confidence='high',trend='new',reason='122/400controls,2.44/8s retained before180s operator timeout; fallen tail remains visible.'),
                dict(dimension='Recovery evidence',grade=0,confidence='low',trend='new',reason='No controlled mid-episode disturbance or recovery learner was evaluated.')],
            requirement_challenges=['Determine whether inherited gait targets or provisional hip command envelope/body dynamics explain backward travel before recommending mechanical changes.']))
    write(directory / 'assessment.json', summary)
    return summary, a


def render(directory, summary, a):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    model = mujoco.MjModel.from_xml_path(str(ROOT / summary['model']['path'])); data = mujoco.MjData(model)
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith('ape_body_shell__') or name == 'original_profile_head_skin_v23': model.geom_group[i] = 5
    camera = mujoco.MjvCamera(); camera.distance=1.1; camera.azimuth=140; camera.elevation=-18
    option=mujoco.MjvOption(); option.geomgroup[:]=[1,1,0,0,0,0]
    renderer=mujoco.Renderer(model,height=480,width=640)
    output=directory/'operator-truncated-cpu-prefix.mp4';frames=np.arange(0,float(a['native_time_s'][-1]),1/30)
    snapshots=[]
    with imageio.get_writer(output,fps=30,codec='libx264',quality=8,macro_block_size=None) as writer:
        for index,time_s in enumerate(frames):
            k=min(len(a['qpos'])-1,int(np.searchsorted(a['native_time_s'],time_s)))
            data.qpos[:]=a['initial_qpos'] if time_s==0 else a['qpos'][k]
            data.qvel[:]=a['initial_qvel'] if time_s==0 else a['qvel'][k]
            mujoco.mj_forward(model,data);camera.lookat[:]=[data.qpos[0]+.03,data.qpos[1],.22]
            renderer.update_scene(data,camera,scene_option=option)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,53),fill=(14,20,25));draw.text((8,5),'R3 saved actor | CPU native prefix | zero updates',fill='white')
            draw.text((8,22),f't={time_s:.2f}s | forward command +0.30 m/s | shell hidden',fill=(235,197,133))
            draw.text((8,38),'OPERATOR-TRUNCATED: 122/400 controls; planned 8 s',fill=(255,180,150))
            writer.append_data(np.asarray(frame))
            if index in (0,9,21,30,len(frames)-1): snapshots.append(frame.copy())
    renderer.close()
    contact=Image.new('RGB',(640*len(snapshots),480))
    for i,frame in enumerate(snapshots):contact.paste(frame,(640*i,0))
    contact.save(directory/'inspection-frames.png')
    count=0;minimum_std=float('inf')
    with imageio.get_reader(output) as reader:
        for frame in reader:count+=1;minimum_std=min(minimum_std,float(frame.std()))
    record=dict(video=ref(output),trace=summary['trace'],model=summary['model'],renderer=ref(Path(__file__)),
        expected_frames=len(frames),decoded_frames=count,minimum_frame_pixel_std=minimum_std,fps=30,
        native_observed_duration_s=summary['captured_duration_s'],requested_horizon_s=8.,
        operator_truncated=True,method='Later rendering of retained native state prefix, including firstfloor contact and fallen tail. Shell visibility only is changed.')
    write(directory/'video-verification.json',record)
    return record


def publication(directory, summary):
    assets=[dict(id='runtime-'+p.stem,title=p.name,kind='image'if p.suffix=='.png'else'data',path=ref(p)['path'])
        for p in [directory/'assessment.json',directory/'report.json',directory/'contract.json',directory/'episodes/episodes.jsonl',ROOT/summary['trace']['path'],directory/'interrupted-native-state.npz'] if p.exists()]
    video=directory/'video-verification.json';clips=[]
    if video.exists():
        v=read(video);assets.append(dict(id='inspection-frames',title='Upright motion, first forearm contact and fallen tail',kind='image',path=ref(directory/'inspection-frames.png')['path']))
        clips=[dict(id=RESULT+'-cpu-seed135731-prefix',title='Forward command; backward drift and floor contact',role='Operator-truncated CPU native task prefix, later rendered',
            body='SelectedR3 22-axis body',observation=summary['observation'],source_path=v['video']['path'],report_path=ref(directory/'assessment.json')['path'],trace_path=summary['trace']['path'])]
    row=dict(id=RESULT,title='Forward task on R3 · CPU runtime and transfer prefix',workspace='Control',dashboard='learning',area='Forward policy transfer',
        status='operator-truncated',observation=summary['observation'],
        measurements='One CPU world/start/episode;122/400 controls;2.440/8s;9785 native steps;0 optimizer updates. First original physical termination≈0.740s; finalΔx−0.355m.',
        limitations='Operator stopped at180s wall budget. This is incomplete CPU setup/transfer evidence; no GPU run, adaptation, recovery test or qualified walking.',
        body='SelectedR3 native rigid model,22axes, nominal3.8303kg',provenance=[summary['source_report']['path'],summary['frozen_launch']['path']],
        matrix=[dict(condition='CPUWarp nominal12V/20mscommand/10msfeedback,seed135731',requested_controls=400,captured_controls=122,
            requested_duration_s=8.,captured_duration_s=summary['captured_duration_s'],first_source_physical_termination_s=summary['first_original_physical_termination_s'],final_delta_x_m=summary['final_root_delta_xy_m'][0],final_delta_y_m=summary['final_root_delta_xy_m'][1],optimizer_updates=0)],
        grades=summary['grades'],assets=assets,clips=clips,
        sections=[dict(title='Native measurements and incomplete-horizon accounting',payload=dict(kind='analysis',observation=summary['observation'],
            command_error_accuracy_curve=summary['command_error_accuracy_curve'],trajectory_stations=summary['trajectory_stations'],per_joint=summary['per_joint'],limits=summary['limitations']))])
    write(directory/'publication.json',row)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'infra/dashboard/results_hub.py'),'publish',str(directory/'publication.json')],check=True,cwd=ROOT)


def upload(directory):
    sys.path.insert(0, str(ROOT / 'infra' / 'dashboard'))
    spec=importlib.util.spec_from_file_location('walk_runtime_results_hub',ROOT/'infra/dashboard/results_hub.py');hub=importlib.util.module_from_spec(spec);spec.loader.exec_module(hub)
    class ScopedConnection(sqlite3.Connection):
        def execute(self,sql,parameters=(),/):
            if sql=='SELECT * FROM clips WHERE verified_at IS NULL ORDER BY result_id,id':
                return super().execute('SELECT * FROM clips WHERE verified_at IS NULL AND result_id=? ORDER BY id',(RESULT,))
            return super().execute(sql,parameters)
    db=sqlite3.connect(ROOT/'data/results.sqlite',factory=ScopedConnection);db.row_factory=sqlite3.Row;hub.upload(db)
    write(directory/'video-upload-receipt.json',dict(result=RESULT,clips=[dict(r)for r in db.execute('select id,sha256,bytes,bucket_key,verified_at from clips where result_id=?',(RESULT,))]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--render',action='store_true');p.add_argument('--publish',action='store_true');p.add_argument('--upload',action='store_true');args=p.parse_args()
    directory=args.run.resolve();summary,a=assess(directory)
    if args.render:render(directory,summary,a)
    if args.publish:publication(directory,summary)
    if args.upload:upload(directory)
    print(json.dumps(dict(result=RESULT,captured_controls=summary['captured_controls'],captured_duration_s=summary['captured_duration_s'],first_source_termination_s=summary['first_original_physical_termination_s'],optimizer_steps=0)))


if __name__=='__main__':main()
