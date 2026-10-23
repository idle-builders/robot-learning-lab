"""One task recording: robot, RGB, learned decisions, estimates and physics truth."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rerun as rr
import rerun.blueprint as rb
from lab.rerun_scene import log_static_scene,log_body_poses
root=Path(sys.argv[1]);data=json.loads((root/'replay.json').read_text())
rec=rr.RecordingStream('Pip · camera-driven interaction');rec.save(root/'episode.rrd')
rec.log('about',rr.TextDocument(data['scope']+'\n\nCamera/encoder/IMU streams are simulated sensor counterparts. Object truth is evaluation only. Voltage/current are model estimates, not measurements. Policy value is expected return, not a physical forecast.'),static=True)
rec.log('scene',rr.ViewCoordinates.FLU,static=True);log_static_scene(rec,json.loads((root/'scene.json').read_text()))
rec.send_blueprint(rb.Blueprint(rb.Vertical(rb.TextDocumentView(origin='about'),rb.Horizontal(
 rb.Spatial3DView(origin='scene',eye_controls=rb.EyeControls3D(position=[1.1,-1.2,.8],look_target=[.35,0,.2],eye_up=[0,0,1])),
 rb.Vertical(rb.Spatial2DView(origin='camera'),rb.Tabs(rb.TimeSeriesView(origin='distance_m'),rb.TimeSeriesView(origin='decision'),
 rb.TimeSeriesView(origin='velocity_m_s'),rb.TimeSeriesView(origin='joint_effort_nm'),rb.TimeSeriesView(origin='joint_angle_rad'),rb.TimeSeriesView(origin='wheel_speed_rad_s'),rb.TimeSeriesView(origin='imu_gyro_rad_s'),rb.TimeSeriesView(origin='voltage_v'),rb.TimeSeriesView(origin='estimated_current_a'),rb.TimeSeriesView(origin='estimated_energy_j'),rb.TimeSeriesView(origin='prediction'),rb.TextDocumentView(origin='task'),active_tab=0))),row_shares=[.12,.88]),rb.SelectionPanel(expanded=False)))
case_offsets={};offset=0.
for row in data['rows']:
 case=row['case']
 if case not in case_offsets:case_offsets[case]=offset
 t=case_offsets[case]+row['time_s'];offset=t+.2
 rec.set_time('episode_sequence',duration=t)
 rec.log('task',rr.TextDocument(f"Case: {case}\nEpisode time: {row['time_s']:.1f} s\nLearned selection: {row['skill']}\nSupervisor intervention: {row['override']}\nDwell at requested distance: {row['dwell_s']:.1f} s\nTask success: {row['success']}\nFall: {row['fallen']} / Overspeed: {row['overspeed']}"))
 bodies={name:{'position_m':pose[:3],'quaternion_xyzw':pose[3:]} for name,pose in zip(row['body_names'],row['body_poses_world_xyz_xyzw'],strict=True)}
 log_body_poses(rec,bodies)
 rec.log('camera',rr.EncodedImage(path=root/row['image']))
 rec.log('distance_m/simulator_truth',rr.Scalars(row['truth_forward_m']))
 rec.log('distance_m/desired',rr.Scalars(.50))
 rec.log('decision/vision_available',rr.Scalars(int(row['estimate'] is not None)))
 if row['estimate']:rec.log('distance_m/camera_estimate',rr.Scalars(row['estimate']['forward_m']))
 for skill,prob in zip(('pause','approach','retreat'),row['action_probabilities']):rec.log('decision/probability_'+skill,rr.Scalars(prob))
 rec.log('prediction/value_expected_return',rr.Scalars(row['predicted_discounted_return']))
 rec.log('velocity_m_s/requested',rr.Scalars(row['command'][0]));rec.log('velocity_m_s/simulated',rr.Scalars(row['forward_m_s']))
 e=row['electromechanical']
 for key,path in [('bus_voltage_v','voltage_v'),('estimated_current_a','estimated_current_a'),('estimated_energy_j','estimated_energy_j')]:rec.log(path+'/model_estimate',rr.Scalars(e[key]))
 for i,axis in enumerate(('x','y','z')):
  if 'imu_gyro_body_rad_s' in row:rec.log('imu_gyro_rad_s/'+axis,rr.Scalars(row['imu_gyro_body_rad_s'][i]))
 for i,j in enumerate(data['interface']['joints']):
  path='wheel_speed_rad_s' if j['continuous'] else 'joint_angle_rad'
  rec.log(path+'/'+j['name']+'/target',rr.Scalars(row['targets'][i]))
  rec.log(path+'/'+j['name']+'/state',rr.Scalars(row['dq'][i] if j['continuous'] else row['q'][i]))
  for kind,key in [('requested','computed_effort_nm'),('applied','applied_effort_nm')]:rec.log('joint_effort_nm/'+j['name']+'/'+kind,rr.Scalars(e[key][i]))
rec.flush()
