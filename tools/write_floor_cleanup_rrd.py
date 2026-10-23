"""Rerun view of one real cleanup episode, with sensor/estimate/truth separation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rerun as rr
import rerun.blueprint as rb
from lab.rerun_scene import log_static_scene,log_body_poses
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('case');p.add_argument('scene',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
report=json.loads((a.run/'report.json').read_text());scene=json.loads(a.scene.read_text())
rows=[json.loads(s) for s in (a.run/a.case/'trajectory.jsonl').read_text().splitlines()]
# Scene export owns ordered body identities from the same compiled model.
bodies=json.loads((a.scene.parent/'body-order.json').read_text())
rec=rr.RecordingStream('Pip · floor cleanup');rec.save(a.output)
rec.log('about',rr.TextDocument(report['scope']+'\n\nPurple upper arms are a provisional longer-link envelope. Other visible parts reuse retained CAD. Contact collision shapes and all actuator/electrical values remain simulation assumptions. Simulated object truth is used for evaluation and reward, never the controller observation. The PPO policy controls a bounded grasp residual; it does not learn task order.'),static=True)
rec.log('scene',rr.ViewCoordinates.FLU,static=True);log_static_scene(rec,scene)
rec.send_blueprint(rb.Blueprint(rb.Vertical(rb.TextDocumentView(origin='about'),rb.Horizontal(
 rb.Spatial3DView(origin='scene',eye_controls=rb.EyeControls3D(position=[.9,-1.1,.65],look_target=[.1,0,.2],eye_up=[0,0,1])),
 rb.Vertical(rb.Spatial2DView(origin='camera'),rb.Tabs(rb.TextDocumentView(origin='task'),rb.TimeSeriesView(origin='object_height_m'),rb.TimeSeriesView(origin='finger_force_n'),rb.TimeSeriesView(origin='policy'),rb.TimeSeriesView(origin='joint_effort_nm'),rb.TimeSeriesView(origin='joint_tracking_rad'),rb.TimeSeriesView(origin='voltage_v'),rb.TimeSeriesView(origin='estimated_current_a'),active_tab=0))),row_shares=[.13,.87])))
interface=json.loads((a.run/'source/learning/pip-control-v1.json').read_text())
# Order remains the sealed 16-axis interface; morphology changes link length only.
order=json.loads((Path(__file__).resolve().parents[1]/'artifacts/pip-cad-gym-bundle-v4/model/interfaces.json').read_text())['joint_order']
for row in rows:
 if 'image' not in row:continue
 rec.set_time('episode',duration=row['time_s'])
 log_body_poses(rec,{name:{'position_m':pose[:3],'quaternion_xyzw':pose[3:]} for name,pose in zip(bodies,row['body_poses_world_xyz_xyzw'],strict=True)})
 rec.log('camera',rr.EncodedImage(path=a.run/row['image']))
 rec.log('task',rr.TextDocument(f"{a.case} | {row['time_s']:.2f}s | {row['phase']}\nPolicy action: {row['policy_action']}\nRemembered camera estimate: {row['estimate']}\nContact forces (simulator truth): {row['block_contacts_n']}\nObject position (evaluation truth): {row['block_m']}"))
 if row['estimate']:rec.log('scene/perceived_pickup',rr.Points3D([row['estimate']['position_m']],colors=[40,220,250],radii=.005))
 rec.log('object_height_m/truth',rr.Scalars(row['block_m'][2]));rec.log('object_height_m/floor_centre',rr.Scalars(.015))
 for name in ['left_forearm_1','left_gripper_0','floor']:rec.log('finger_force_n/'+name,rr.Scalars(row['block_contacts_n'].get(name,0)))
 rec.log('policy/grip_allowance_nm',rr.Scalars(row['gripper_effort_limit_nm']));rec.log('policy/forearm_offset_rad',rr.Scalars(row['forearm_residual_rad']))
 if row.get('predicted_return') is not None:rec.log('policy/predicted_return',rr.Scalars(row['predicted_return']))
 e=row['electromechanical'];rec.log('voltage_v/model',rr.Scalars(e['bus_voltage_v']));rec.log('estimated_current_a/model',rr.Scalars(e['estimated_current_a']))
 for i,name in enumerate(order):
  rec.log('joint_effort_nm/'+name,rr.Scalars(e['applied_effort_nm'][i]))
  if 'wheel' not in name:
   rec.log('joint_tracking_rad/'+name+'/actual',rr.Scalars(row['q'][i]));rec.log('joint_tracking_rad/'+name+'/target',rr.Scalars(row['targets'][i]))
rec.flush()
