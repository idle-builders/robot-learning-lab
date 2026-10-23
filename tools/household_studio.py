"""The task comes first: use existing Gradio and Rerun, not a new viewer."""
import json
import gradio as gr
from lab.browser_media import browser_video
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lab.household_catalog import runs,retained,launch
from lab.experiment_catalog import open_replay


def result(name,case='target_moves_closer'):
 path=retained(name);report=runs()[name]
 choices=list(report.get('cases',{}));case=case if case in choices else next(iter(choices),None)
 rows=json.loads((path/case/'trace.json').read_text()) if case else []
 fig=make_subplots(rows=2,cols=2,subplot_titles=['Camera estimate versus physics truth (m)','Learned skill probabilities','Requested versus actual speed (m/s)','Estimated pack current (A)'])
 t=[r['time_s'] for r in rows]
 for key,label in [('truth_forward_m','Simulator truth')]:fig.add_trace(go.Scatter(x=t,y=[r[key] for r in rows],name=label),row=1,col=1)
 fig.add_trace(go.Scatter(x=t,y=[r['estimate']['forward_m'] if r['estimate'] else None for r in rows],name='Camera estimate',connectgaps=False),row=1,col=1)
 fig.add_trace(go.Scatter(x=t,y=[.5]*len(t),name='Desired separation',line={'dash':'dot'}),row=1,col=1)
 for i,label in enumerate(['Pause','Approach','Retreat']):fig.add_trace(go.Scatter(x=t,y=[r['action_probabilities'][i] for r in rows],name=label),row=1,col=2)
 fig.add_trace(go.Scatter(x=t,y=[r['command'][0] for r in rows],name='Requested'),row=2,col=1)
 fig.add_trace(go.Scatter(x=t,y=[r['forward_m_s'] for r in rows],name='Actual simulation'),row=2,col=1)
 fig.add_trace(go.Scatter(x=t,y=[r['electromechanical']['estimated_current_a'] for r in rows],name='Modelled current'),row=2,col=2)
 fig.update_layout(template='plotly_white',height=500,margin=dict(l=30,r=20,t=45,b=25),legend={'orientation':'h'})
 text=f"**Task acceptance: {report['status']}. Execution: {report['execution_status']}.** {report['training_steps']} additional RL steps in this run.\n\n"
 if report.get('training_steps')==0:
  text+='This is a single-scene checkpoint replay; it does not replace the fixed trial-suite result.\n\n'
 text+=f"**{sum(c['success'] for c in report.get('cases',{}).values())}/{len(report.get('cases',{}))} recorded trials passed.**\n\n"
 text+='| Trial | Success | Final separation | Vision error RMS |\n|---|---|---|---|\n'
 for k,v in report.get('cases',{}).items():
  error=v['range_estimate_rmse_m'];text+=f"| {k} | {'pass' if v['success'] else 'fail'} | {v['final_range_m']:.3f} m | {error:.3f} m |\n" if error is not None else f"| {k} | fail | {v['final_range_m']:.3f} m | unavailable |\n"
 text+='\n'+report.get('scope','')+'\n\nRetained attempt: `'+name+'`. Learned decisions are sampled at 5 Hz; the motor policy runs at 50 Hz and physics at 500 Hz. The chart’s probabilities describe the decision made at the start of each 0.2 s interval; measurements describe its end.'
 source=path/'source/lab/household_learning.py'
 if source.exists() and source.read_bytes()!=(__import__('pathlib').Path(__file__).resolve().parents[1]/'lab/household_learning.py').read_bytes():
  text+='\n\n**Historical environment:** this attempt used an earlier task implementation. Its source is retained; current reruns use the revised environment.'
 if report.get('error'):text+='\n\nExecution error: '+report['error']
 files=[str(path/'report.json'),str(path/'policy.zip')]
 if case:files.append(str(path/case/'trace.json'))
 if (path/'episode.rrd').is_file():files.append(str(path/'episode.rrd'))
 video=browser_video(path/case/'replay.mp4') if case and (path/case/'replay.mp4').exists() else None
 return video,text,fig,[p for p in files if __import__('os').path.isfile(p)],gr.update(choices=choices,value=case)


def run_task(parent,case,distance,lateral):
 path=launch(parent,case,distance,lateral)
 return gr.update(choices=list(runs()),value=path.name),*result(path.name,case)


def adapt(parent,steps):
 path=launch(parent,'far',.72,.04,train=True,steps=int(steps))
 return gr.update(choices=list(runs()),value=path.name),*result(path.name)


def inspect(name):
 url=open_replay(name)
 return '<iframe title="Household task: CAD, vision and control" src="'+url+'" style="width:100%;height:740px;border:0"></iframe>'


def build_tab():
 with gr.Tab('Interaction · earlier experiment'):
  gr.Markdown('## Gentle interaction · camera-driven task\nThe robot sees a **marked interaction proxy**, then a learned PPO policy chooses **pause, approach or retreat**. A frozen learned motor policy executes each choice on the actual articulated Pip model. Move the target or hide it and watch the decisions change. This is the first slice of cat play: it does not recognize a cat, circle around it or establish animal comfort.')
  names=list(runs());default=next((n for n in names if runs()[n].get('training_steps',0)>0 and (retained(n)/'episode.rrd').exists()),names[0] if names else None)
  selected=gr.Dropdown(names,value=default,label='Retained trained attempt / replay')
  initial=result(default) if default else (None,'No retained task yet.',None,[],gr.update(choices=[]))
  case=gr.Dropdown(list(runs().get(default,{}).get('cases',{})),value=initial[4].get('value'),label='Recorded trial')
  video=gr.Video(initial[0],label='Task, head camera and controller · synchronized replay',interactive=False)
  summary=gr.Markdown(initial[1]);chart=gr.Plot(initial[2]);files=gr.Files(initial[3],label='Checkpoint, report and Rerun recording',interactive=False)
  with gr.Accordion('Change the scene and run the learned controller',open=True):
   scenario=gr.Radio(['far','near','settle','vision_loss','target_moves_closer'],value='far',label='Trial disturbance')
   with gr.Row():
    distance=gr.Slider(.34,.85,value=.72,step=.01,label='Initial camera-to-target forward distance (m)')
    lateral=gr.Slider(-.12,.12,value=.04,step=.01,label='Target left/right offset (m)')
   execute=gr.Button('Run this task with the selected checkpoint',variant='primary')
   execute.click(run_task,[selected,scenario,distance,lateral],[selected,video,summary,chart,files,case],api_name='run_household_task',concurrency_id='control_jobs')
  gr.Button('Inspect the task in 3D with recorded signals').click(inspect,selected,outputs:=gr.HTML(),api_name='inspect_household_task')
  with gr.Accordion('Adapt the skill policy',open=False):
   steps=gr.Dropdown([256,512,1024,2048,4096],value=1024,label='Additional bounded PPO steps')
   gr.Markdown('Resume the selected policy, then reload it and evaluate the fixed trial set. CPU only; no rental GPU. Every failure remains visible. The motor checkpoint stays frozen.')
   gr.Button('Adapt and evaluate').click(adapt,[selected,steps],[selected,video,summary,chart,files,case],api_name='adapt_household_task',concurrency_id='control_jobs')
  def refresh():
   names=list(runs());name=next((n for n in names if runs()[n].get('training_steps',0)>0 and (retained(n)/'episode.rrd').exists()),names[0])
   return gr.update(choices=names,value=name),*result(name)
  gr.Button('Refresh completed task attempts').click(refresh,outputs=[selected,video,summary,chart,files,case])
  selected.input(result,[selected,case],[video,summary,chart,files,case],api_name='select_household_run')
  case.input(result,[selected,case],[video,summary,chart,files,case],api_name='select_household_case')
  gr.Markdown('**Sim → real:** compare camera-derived range, encoder/IMU state, commands, applied motion and measured supply signals on the same episode clock. Today these are simulated observations and electrical estimates; physical recordings are not run. **Cosmos** remains available in World prediction, but its Pip transfer failed qualification and is not allowed to supply control or rewards.')
