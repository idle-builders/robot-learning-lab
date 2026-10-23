"""Export retained VLA/motor rollouts through the existing Rerun scene adapter."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def write(run,case,out):
    import rerun as rr
    import rerun.blueprint as rb
    from lab.rerun_scene import log_static_scene,log_body_poses,validate_scene_bundle
    scene=json.loads((out/'scene.json').read_text())
    validate_scene_bundle(scene)
    bodies=json.loads((out/'body-order.json').read_text())
    contract=json.loads((run/'submitted-source/contract.json').read_text())
    result=json.loads((run/'evidence/evaluation.json').read_text())
    dest=run/'evidence'/case
    rows=[json.loads(x) for x in (dest/'trajectory.jsonl').open()]
    decisions=[json.loads(x) for x in (dest/'decisions.jsonl').open()]
    rec=rr.RecordingStream('Pip · learned visual and motor control')
    rec.save(out/'episode.rrd')
    rec.log('about',rr.TextDocument(f"{run.name} / {case}\n\nMotor: {result['whole_body_control']}\n\n"
        f"Physical result: {json.dumps(result['cases'][case])}\n\n"
        'Head RGB and sensor state drive the VLA. Body/object poses here are simulation truth for inspection. '
        'Electrical current is an unmeasured proxy. Video frames combine the head image acquired at the decision time '
        'with an external inspection image after that command; do not interpret both as simultaneous sensor captures.'),static=True)
    rec.log('scene',rr.ViewCoordinates.FLU,static=True)
    log_static_scene(rec,scene)
    video=rr.AssetVideo(path=dest/'replay.mp4')
    rec.log('replay/video',video,static=True)
    stamps=video.read_frame_timestamps_nanos()
    if len(stamps)!=len(decisions):
        raise ValueError('Video frame count does not match recorded policy decisions')
    for stamp,decision in zip(stamps,decisions,strict=True):
        rec.set_time('episode',duration=decision['time_s'])
        rec.log('replay',rr.VideoFrameReference(nanoseconds=int(stamp),video_reference='replay/video'))
        rec.log('instruction',rr.TextDocument(decision['instruction']))
        for index,name in enumerate(contract['action_names']):
            rec.log('vla_commands/'+name+'/predicted',rr.Scalars(decision['predicted_action'][index]))
            rec.log('vla_commands/'+name+'/accepted',rr.Scalars(decision['accepted_action'][index]))
        rec.log('latency_s/vla_or_queue',rr.Scalars(decision['inference_s']))
    order=json.loads((run/'submitted-source/bundle/model/interfaces.json').read_text())['joint_order']
    for row in rows:
        rec.set_time('episode',duration=row['time_s'])
        log_body_poses(rec,{name:{'position_m':pose[:3],'quaternion_xyzw':pose[3:]}
            for name,pose in zip(bodies,row['body_poses_world_xyz_xyzw'],strict=True)})
        rec.log('object_truth/height_m',rr.Scalars(row['block_m'][2]))
        rec.log('motion/pitch_rad',rr.Scalars(row['pitch_rad']))
        rec.log('motion/forward_m_s',rr.Scalars(row['forward_m_s']))
        e=row['electromechanical']
        rec.log('electrical_proxy/current_a',rr.Scalars(e['estimated_current_a']))
        rec.log('electrical_proxy/voltage_v',rr.Scalars(e['bus_voltage_v']))
        for i,name in enumerate(order):
            rec.log('actuators/'+name+'/effort_nm',rr.Scalars(e['applied_effort_nm'][i]))
            rec.log('tracking/'+name+'/target',rr.Scalars(row['targets'][i]))
            rec.log('tracking/'+name+'/actual',rr.Scalars(row['dq'][i] if 'wheel' in name else row['q'][i]))
            if 'motor_policy_action' in row:
                rec.log('motor_actor/'+name,rr.Scalars(row['motor_policy_action'][i]))
        if 'motor_inference_s' in row:
            rec.log('latency_s/motor',rr.Scalars(row['motor_inference_s']))
    rec.send_blueprint(rb.Blueprint(rb.Vertical(rb.TextDocumentView(origin='about'),rb.Horizontal(
        rb.Spatial3DView(origin='scene',eye_controls=rb.EyeControls3D(position=[.9,-1.1,.65],look_target=[.1,0,.2],eye_up=[0,0,1])),
        rb.Vertical(rb.Spatial2DView(origin='replay'),rb.Tabs(rb.TextDocumentView(origin='instruction'),
            rb.TimeSeriesView(origin='motion'),rb.TimeSeriesView(origin='vla_commands'),
            rb.TimeSeriesView(origin='motor_actor'),rb.TimeSeriesView(origin='tracking'),
            rb.TimeSeriesView(origin='actuators'),rb.TimeSeriesView(origin='electrical_proxy'),
            rb.TimeSeriesView(origin='latency_s'),active_tab=0))),row_shares=[.15,.85])))
    rec.flush()

def export(run,case):
    import mujoco
    from lab.rerun_scene import export_scene_bundle
    from lab.artifact_store import snapshot_file,retain_dependency,plan_migration,apply_migration,verify_manifest
    if not (run/'evidence'/case/'trajectory.jsonl').is_file():
        raise ValueError('Select a completed physical rollout')
    out=ROOT/'artifacts'/('learned-control-view-'+uuid.uuid4().hex[:12]);out.mkdir()
    print(out,flush=True)
    scene=run/'submitted-source/scene.xml'
    export_scene_bundle(scene,out/'scene.json')
    model=mujoco.MjModel.from_xml_path(str(scene))
    (out/'body-order.json').write_text(json.dumps([model.body(i).name for i in range(1,model.nbody)]))
    status='fail'
    try:
        with (out/'export.log').open('x') as log:
            subprocess.run([str(ROOT/'.venv-observe/bin/python'),__file__,'write',str(run),case,str(out)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
            subprocess.run([str(ROOT/'.venv-observe/bin/rerun'),'rrd','verify',str(out/'episode.rrd')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        status='pass'
    finally:
        (out/'report.json').write_text(json.dumps({'status':status,'run':str(run),'case':case,
            'scope':'Read-only source-bound scene, learned decisions and simulator telemetry; not task acceptance'},indent=2))
        snapshot_file(Path(__file__),out/'source/tools/write_learned_control_rrd.py')
        snapshot_file(ROOT/'lab/rerun_scene.py',out/'source/lab/rerun_scene.py')
        (out/'dependencies.json').write_text(json.dumps({'rollout':retain_dependency(run)},indent=2))
        plan=plan_migration(ROOT,[out]);apply_migration(ROOT,plan);verify_manifest(ROOT,plan)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['export','write'])
    parser.add_argument('run',type=Path)
    parser.add_argument('case',choices=['nominal_cleanup','lateral_cleanup','hold'])
    parser.add_argument('output',type=Path,nargs='?')
    args=parser.parse_args()
    if args.mode=='write':
        if args.output is None:parser.error('write requires an existing output directory')
        write(args.run,args.case,args.output)
    else:export(args.run,args.case)
