"""Inspect and rerun the retained native floor-cleanup experiment."""
import hashlib,json,subprocess,uuid
from pathlib import Path
import gradio as gr
from lab.browser_media import browser_video
ROOT=Path(__file__).resolve().parents[1]
SELECTED='floor-learning-2f86cd18caaa'


def runs():
 result={}
 for p in sorted((ROOT/'artifacts').glob('floor-learning-*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
  r=json.loads(p.read_text())
  if r.get('cases'):result[p.parent.name]=r
 return result


def result(name,case):
 if name not in runs():raise ValueError('Choose a retained cleanup run')
 root=ROOT/'artifacts'/name;r=runs()[name]
 if case not in r['cases']:raise ValueError('Choose one of the fixed cases')
 c=r['cases'][case];passed=sum(v['success'] for v in r['cases'].values())
 text=f"**{case}: {'PASS' if c['success'] else 'FAIL'}** · fixed suite: **{passed}/{len(r['cases'])}**. This run: {r['training_steps']} PPO transitions; checkpoint reload matched: {r.get('reload_action_match')}.\n\n"
 text+=f"Lift: **{c['max_lift_m']*1000:.0f} mm** · transport: **{c['max_transport_m']*100:.1f} cm** · simulated duration: **{c['duration_s']:.1f} s**.\n\n"
 text+='The task order, balance controller and posture references are authored and use feedback. **PPO controls a small grasp residual** (gripper effort allowance and forearm angle). The authored reference also succeeds on the nominal case; no learned improvement is established. Vision recognizes this known orange cube and remembers its floor position after it leaves the camera view.\n\n'
 text+='Purple upper arms are **15 mm longer provisional envelopes**. Other visible parts reuse retained CAD. The **1.0 Nm leg limit is an unmeasured hypothesis**. The whole-product workbench still fails actuator rated-load and baseline vision-power coverage checks. Physical hardware is unverified.'
 video=root/case/'replay.mp4'
 files=[str(p) for p in [root/'report.json',root/case/'report.json',root/case/'decisions.json',root/'policy.zip'] if p.exists()]
 return browser_video(video) if video.exists() else None,text,[[k,'pass' if v else 'FAIL'] for k,v in c['checks'].items()],files


def change_run(name):
 choices=list(runs()[name]['cases']);case=choices[0]
 return gr.update(choices=choices,value=case),*result(name,case)


def inspect_3d(name,case):
 if name not in runs() or case not in runs()[name]['cases']:raise ValueError('Select a retained run and case')
 root=ROOT/'artifacts'/name;shared=ROOT/'artifacts/floor-cleanup-observer-v1'
 scene=json.loads((shared/'scene.json').read_text())
 if hashlib.sha256((root/'scene.xml').read_bytes()).hexdigest()!=scene['source_xml_sha256']:
  raise ValueError('This scene differs from the current bound 3D export; export its geometry before viewing.')
 writer=ROOT/'tools/write_floor_cleanup_rrd.py';writer_hash=hashlib.sha256(writer.read_bytes()).hexdigest()
 target=None
 for p in (ROOT/'artifacts').glob('floor-cleanup-view-*/view.json'):
  metadata=json.loads(p.read_text())
  if metadata=={'run':name,'case':case,'writer_sha256':writer_hash}:target=p.parent;break
 if target is None:
  target=ROOT/'artifacts'/('floor-cleanup-view-'+uuid.uuid4().hex[:12]);target.mkdir()
  with (target/'export.log').open('w') as log:
   subprocess.run([str(ROOT/'.venv-observe/bin/python'),str(writer),str(root),case,str(shared/'scene.json'),str(target/'episode.rrd')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
   subprocess.run([str(ROOT/'.venv-observe/bin/rerun'),'rrd','verify',str(target/'episode.rrd')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
  (target/'view.json').write_text(json.dumps({'run':name,'case':case,'writer_sha256':writer_hash},indent=2))
  from lab.artifact_store import snapshot_file,retain_dependency,plan_migration,apply_migration,verify_manifest
  snapshot_file(writer,target/'source/write_floor_cleanup_rrd.py')
  (target/'dependencies.json').write_text(json.dumps([retain_dependency(root),retain_dependency(shared)],indent=2))
  plan=plan_migration(ROOT,[target]);apply_migration(ROOT,plan);verify_manifest(ROOT,plan)
 from lab.experiment_catalog import open_recording
 url=open_recording(target/'episode.rrd')
 return '<iframe title="Floor cleanup in Rerun" src="'+url+'" style="width:100%;height:760px;border:0"></iframe>'


def repeat(name):
 if name not in runs():raise ValueError('Choose a retained checkpoint')
 before=set(runs());command=[str(ROOT/'.venv-learning/bin/python'),str(ROOT/'tools/train_floor_cleanup.py'),'--steps','0','--parent',str(ROOT/'artifacts'/name/'policy.zip')]
 logs=ROOT/'artifacts'/('floor-cleanup-launch-'+uuid.uuid4().hex[:12]);logs.mkdir()
 with (logs/'process.log').open('w') as log:
  proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 (logs/'launch.json').write_text(json.dumps({'command':command,'returncode':proc.returncode},indent=2))
 after=runs();new=next((n for n in after if n not in before),None)
 from lab.artifact_store import plan_migration,apply_migration,verify_manifest
 plan=plan_migration(ROOT,[logs]);apply_migration(ROOT,plan);verify_manifest(ROOT,plan)
 if new is None:raise gr.Error('Runner failed before producing results; inspect '+str(logs))
 return gr.update(choices=list(after),value=new),*change_run(new)


def build_tab():
 available=runs();selected=SELECTED if SELECTED in available else next(iter(available),None)
 with gr.Tab('Cleanup baseline · authored flow'):
  gr.Markdown('## Find → approach → bend and pick up → carry → set down\n**A bounded cleanup simulation now runs.** Play the nominal case to see the robot pick a free object off the floor. Select the shifted cases to inspect robustness failures. Open Rerun to orbit the 3D robot, scrub time and compare perception, policy outputs, contacts, joint effort and electrical estimates.')
  if selected is None:
   gr.Markdown('No completed floor-cleanup experiments.');return
  with gr.Row():
   run=gr.Dropdown(list(available),value=selected,label='Retained experiment');case=gr.Dropdown(list(available[selected]['cases']),value='nominal',label='Evaluation case')
  initial=result(selected,'nominal')
  video=gr.Video(initial[0],label='Actual MuJoCo episode',autoplay=False)
  text=gr.Markdown(initial[1]);checks=gr.Dataframe(initial[2],headers=['Independent check','Result'],interactive=False)
  with gr.Row():
   inspect=gr.Button('Inspect this trial in 3D · Rerun');rerun=gr.Button('Repeat fixed suite · no training')
  native=gr.HTML()
  gr.Markdown('**How to use:** Play the video; use the case menu to see a different reset. In Rerun, drag to orbit and use the timeline to scrub. “Repeat fixed suite” reloads the checkpoint and runs three local trials plus a frictionless negative control using the current source, usually in a few minutes. It uses CPU physics and creates a fresh retained run. Old results are preserved.\n\n**Current boundary:** a 30 mm, 50 g known cube on a flat floor, a nearby marked destination and a narrow starting region. The sideways-shift case fails; general object recognition, learned task planning and hardware transfer remain open. Frozen Cosmos predictions remain in the separate world-model tab and do not drive physics or rewards.')
  files=gr.Files(initial[3],label='Reports, exact policy inputs/actions and checkpoint',interactive=False)
  run.change(change_run,run,[case,video,text,checks,files]);case.change(result,[run,case],[video,text,checks,files]);inspect.click(inspect_3d,[run,case],native);rerun.click(repeat,run,[run,case,video,text,checks,files])
  run.change(lambda: '',outputs=native);case.change(lambda: '',outputs=native)
  with gr.Accordion('Earlier floor-reach evidence and rejected candidates',open=False):
   gr.Markdown('The original arm geometry lifted but dropped the cube. A footed envelope reached lower but retained collisions; feet remain an option. A later lateral-alignment controller also failed and is preserved in the experiment menu. The selected run is the earlier working candidate, not the newest failed attempt. [Floor cleanup notes](/gradio_api/file='+str(ROOT/'docs/design/floor-cleanup.md')+').')
