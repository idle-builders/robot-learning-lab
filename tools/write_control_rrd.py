"""Write the native-control projection using Rerun and existing scene logger."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rerun as rr
import rerun.blueprint as rb
from lab.rerun_scene import log_static_scene,log_body_poses
root=Path(sys.argv[1]);data=json.loads((root/'replay.json').read_text());scene=json.loads((root/'scene.json').read_text())
rec=rr.RecordingStream('Pip skill gym' if 'skill_id' in data['rows'][0] else 'Pip motor gym');rec.save(root/'episode.rrd')
rec.log('about',rr.TextDocument(data['scope']+'\n\nEvaluation outcome: '+data['outcome']),static=True)
rec.log('scene',rr.ViewCoordinates.FLU,static=True)
log_static_scene(rec,scene)
# Frame the complete retained trajectory, including the robot at the final time.
positions=[row['root_position_world_m'] for row in data['rows']]
center=[(min(p[i] for p in positions)+max(p[i] for p in positions))/2 for i in range(3)]
rec.send_blueprint(rb.Blueprint(rb.Vertical(
 rb.TextDocumentView(origin='about'),
 rb.Horizontal(rb.Spatial3DView(origin='scene',eye_controls=rb.EyeControls3D(
  position=[center[0]+.95,center[1]-1.30,center[2]+.60],look_target=[center[0],center[1],center[2]+.03],eye_up=[0,0,1])),
  rb.Tabs(*([rb.Spatial2DView(origin='camera'),rb.TimeSeriesView(origin='skill')] if 'skill_id' in data['rows'][0] else []),rb.TimeSeriesView(origin='motion'),rb.TimeSeriesView(origin='joint_angle_rad'),rb.TimeSeriesView(origin='wheel_speed_rad_s'),*([rb.TimeSeriesView(origin='supply_voltage_v'),rb.TimeSeriesView(origin='estimated_current_a'),rb.TimeSeriesView(origin='joint_effort_nm'),rb.TimeSeriesView(origin='estimated_energy_j')] if data['rows'][0].get('electromechanical') else []),active_tab=0)),
 row_shares=[.15,.85]),rb.SelectionPanel(expanded=False)))
for row in data['rows']:
 rec.set_time('simulation',duration=row['time_s'])
 log_body_poses(rec,row['bodies'])
 for i,j in enumerate(data['interface']['joints']):
  path='wheel_speed_rad_s' if j['continuous'] else 'joint_angle_rad'
  rec.log(path+'/'+j['name']+'/command',rr.Scalars(row['targets'][i]))
  rec.log(path+'/'+j['name']+'/state',rr.Scalars(row['dq'][i] if j['continuous'] else row['q'][i]))
 if 'velocity' in row:rec.log('motion/forward_m_s',rr.Scalars(row['velocity'][0]))
 rec.log('motion/requested_forward_m_s',rr.Scalars(row['command'][0]))
 if row.get('electromechanical'):
  electrical=row['electromechanical']
  rec.log('supply_voltage_v/shared_bus',rr.Scalars(electrical['bus_voltage_v']))
  rec.log('estimated_current_a/pack',rr.Scalars(electrical['estimated_current_a']))
  rec.log('estimated_energy_j/pack',rr.Scalars(electrical['estimated_energy_j']))
  for i,j in enumerate(data['interface']['joints']):
   rec.log('joint_effort_nm/'+j['name']+'/requested',rr.Scalars(electrical['computed_effort_nm'][i]))
   rec.log('joint_effort_nm/'+j['name']+'/applied',rr.Scalars(electrical['applied_effort_nm'][i]))
 if 'skill_id' in row:
  rec.log('skill/selection_0_hold_1_approach_2_align',rr.Scalars(row['skill_id']))
  rec.log('skill/target_visible',rr.Scalars(int(row['estimate'] is not None)))
  rec.log('skill/task_success',rr.Scalars(int(row['success'])))
  if row.get('image'):
   rec.log('camera',rr.EncodedImage(path=root/row['image']))
rec.flush()
