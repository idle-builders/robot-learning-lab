"""Local Gradio workbench for real retained runs and bounded world-model experiments."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gradio as gr
from lab.browser_media import browser_video
import numpy as np
import plotly.graph_objects as go
from workers.world_model.client import access_status, run_prediction, MODES, INPUTS, SOURCES
from workers.world_model.action_loader import change_actions
from lab.experiment_catalog import experiments, retained_run, launch, open_replay

RERUN='http://127.0.0.1:9094/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9881%2Fproxy'
RUNS={
    'Authored motor controller': ROOT/'artifacts/pip-realistic-gym-v4/evidence/control/reference.json',
    'Learned motor controller': ROOT/'artifacts/pip-realistic-gym-v4/evidence/control/evaluation.json',
    'Vision + learned skill selector': ROOT/'artifacts/pip-realistic-gym-v8/evidence/control/camera/skills/trace.json',
}


def access_text():
    result=access_status()
    ready=result['status']=='ready'
    text=('**Ready to request a prediction.** ' if ready else '**Prediction is blocked by model access.** ')+result['reason']
    text+='\n\nUse the retained world-model results below to distinguish completed inference from failed attempts. Pip adaptation and imagined RL remain unqualified; no base-model pretraining runs here.'
    return text,gr.update(interactive=ready)


def control_result(name):
    path=retained_run(name);report=experiments()[name]
    rows=[]
    for case in report.get('cases',{}):
        rows.extend(json.loads((path/(case+'-trace.json')).read_text()))
        break
    fig=go.Figure()
    for key,label in [('forward_m_s','Observed forward speed'),('pitch_rad','Body pitch (rad)')]:
        fig.add_trace(go.Scatter(x=[r['time_s'] for r in rows],y=[r[key] for r in rows],name=label,yaxis='y2' if key=='pitch_rad' else 'y'))
    fig.update_layout(template='plotly_white',height=300,xaxis_title='Time (s)',yaxis_title='m/s',yaxis2={'title':'rad','overlaying':'y','side':'right'})
    text=f"**{report.get('backend')} · {report.get('mode')}**\n\nExecution: **{report.get('execution_status',report['status'])}**. Bounded task criteria: **{report['status']}**. Training steps: {report.get('training_steps',0)}.\n\n"
    if report.get('cases'):
        text+='| Case | Fall / overspeed | Forward error (m/s RMS) | Turn error (rad/s RMS) | Result |\n|---|---|---|---|---|\n'
        for case,c in report['cases'].items():text+=f"| {case} | {c['fall']} / {c['overspeed']} | {c['tracking_rmse_m_s']:.4f} | {str(round(c['yaw_rmse_rad_s'],4)) if 'yaw_rmse_rad_s' in c else 'not recorded'} | {'pass' if c['success'] else 'fail'} |\n"
    if 'validation_l1_after' in report:text+=f"\nHeld-out imitation error: {report['validation_l1_before']:.4f} → {report['validation_l1_after']:.4f}. This is action fit, not proof of task improvement.\n"
    text+='\n'+report.get('scope','')+'\n\nRetained evidence: `artifacts/'+name+'`'
    source=path/'source/mujoco_learning.py'
    if source.exists() and source.read_bytes()!=(ROOT/'lab/mujoco_learning.py').read_bytes():
        text+='\n\n**Historical implementation:** the current environment source differs. Keep this as prior evidence; rerun for current behavior.'
    if report.get('error'):text+='\n\nFailure: '+report['error']
    files=[str(path/x) for x in ['report.json','episode.rrd','policy.zip','pretrained_model/config.json','pretrained_model/model.safetensors'] if (path/x).is_file()]
    return browser_video(path/'replay.mp4') if (path/'replay.mp4').exists() else None,text,fig,files


def run_control(kind,steps,parent,forward,yaw):
    try:
        path=launch(kind,int(steps),parent,forward,yaw)
        return gr.update(choices=list(experiments()),value=path.name),*control_result(path.name)
    except Exception as error:
        raise gr.Error(str(error)) from None


def show_replay(name):
    url=open_replay(name)
    return '<iframe title="Selected Pip experiment" src="'+url+'" style="width:100%;height:700px;border:0"></iframe>'


def motor_comparison():
    # Compare a common retained implementation, even after compatible/new task
    # changes. Never silently mix current and historical dynamics in one table.
    groups={}
    for name,r in experiments().items():
        mode=r.get('mode')
        source=ROOT/'artifacts'/name/'source/mujoco_learning.py'
        if mode not in {'reference','isaac','adapt','act-distill'}:continue
        if set(r.get('cases',{}))!={'hold','forward','reverse','turn'} or not source.exists():continue
        if any('yaw_rmse_rad_s' not in c for c in r['cases'].values()):continue
        chosen=groups.setdefault(source.read_bytes(),{})
        if mode in chosen:continue
        c=list(r['cases'].values())
        chosen[mode]=[mode,r.get('training_steps',0),sum(x['success'] for x in c),
            round(float(np.mean([x['tracking_rmse_m_s'] for x in c])),4),
            round(float(np.mean([x['yaw_rmse_rad_s'] for x in c])),4),
            round(float(np.mean([x['energy_j'] for x in c])),1),name]
    current=(ROOT/'lab/mujoco_learning.py').read_bytes()
    chosen=groups.get(current) or next((g for g in groups.values() if len(g)==4),next(iter(groups.values()),{}))
    return [chosen[k] for k in ['reference','isaac','adapt','act-distill'] if k in chosen]


def world_history(source='bridge'):
    rows=[];videos=[None,None]
    assessments={}
    for assessment in sorted((ROOT/'artifacts').glob('world-model-assessment-*/report.json'),key=lambda p:p.stat().st_mtime):
        assessments.update(json.loads(assessment.read_text()).get('runs',{}))
    for path in sorted((ROOT/'artifacts').glob('world-model-probe-*/evidence/report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        report=json.loads(path.read_text());run=path.parents[1].name
        prediction=path.parent/'prediction'
        detail=json.loads((prediction/'report.json').read_text()) if (prediction/'report.json').exists() else {}
        if detail.get('input_source','bridge')!=source:continue
        rows.append(f"- `{run}`: **{report['status']}**, {report.get('elapsed_s',0):.1f} s worker time; {report.get('training_steps',0)} training steps.")
        if run in assessments:
            assessment=assessments[run]
            rows.append('  Visual review: **'+assessment['visual_status']+'**. '+assessment['note'])
            if assessment.get('review'):
                url='/gradio_api/file='+str(ROOT/'artifacts'/assessment['review']/'comparison.jpg')
                rows.append('  [Compare observed and predicted frames]('+url+').')
        if videos[0] is None and report['status']=='needs_image_review':
            alternative=next((c['mode'] for c in detail.get('cases',[]) if c['mode']!='recorded'),'hold_arm')
            for i,mode in enumerate(['recorded',alternative]):
                matches=list((prediction/mode).glob('*_single_chunk.mp4'))
                if matches:videos[i]=browser_video(matches[0])
    return *videos,'\n'.join(rows) or 'No completed GPU attempts yet.'


def action_plot(mode,source='bridge'):
    data=json.loads((SOURCES[source]/'annotation.json').read_text())
    actions=np.asarray(data['action'],dtype=float)[:12]
    changed=change_actions(actions,mode)
    fig=go.Figure()
    for i,axis in enumerate(['X','Y','Z']):
        fig.add_trace(go.Scatter(x=list(range(len(actions))),y=actions[:,i]*1000,name=axis+' recorded',line={'dash':'dot'}))
        fig.add_trace(go.Scatter(x=list(range(len(actions))),y=changed[:,i]*1000,name=axis+' proposed'))
    fig.update_layout(template='plotly_white',height=320,xaxis_title='Action sample',yaxis_title='Translation increment (mm)',margin=dict(l=45,r=15,t=20,b=40))
    return fig


def compare_runs(names,metric):
    fig=go.Figure();summaries=[]
    key={'Estimated supply voltage':'bus_voltage_v','Estimated pack current':'estimated_current_a','Estimated energy':'estimated_energy_j'}[metric]
    unit={'bus_voltage_v':'V','estimated_current_a':'A','estimated_energy_j':'J'}[key]
    for name in names:
        raw=json.loads(RUNS[name].read_text());rows=raw if isinstance(raw,list) else raw['trace']
        fig.add_trace(go.Scatter(x=[r['time_s'] for r in rows],y=[r['electromechanical'][key] for r in rows],name=name))
        summaries.append(f"- **{name}:** "+('20 s marker approach; no falls; task failed at 46.2 cm against a 45 cm criterion.' if isinstance(raw,list) else f"8 s motor evaluation across 8 environments; {raw['failure_resets']} fall/overspeed resets. Plot shows the recorded sample environment."))
    fig.update_layout(template='plotly_white',height=380,xaxis_title='Episode time (s)',yaxis_title=unit,margin=dict(l=45,r=15,t=20,b=40))
    return fig,'\n'.join(summaries)+'\n\nElectrical values are estimates. The task-selector run has a different command sequence; this overlay is inspection, not a controlled performance ranking.'


def predict(mode,source='bridge'):
    # Recheck permission on every call, even when UI button state is stale.
    status=access_status()
    if status['status']!='ready':
        return None,None,'**Not run.** '+status['reason']
    try:
        output=run_prediction(mode,source=source)
        base=output/'evidence/prediction'
        first=next((base/'recorded').glob('*_single_chunk.mp4'))
        second=next((base/mode).glob('*_single_chunk.mp4'))
        return browser_video(first),browser_video(second),f'**Predictions returned; inspect both videos.** Retained run: `{output.name}`. Frozen weights; zero training. The changed-action prediction has no recorded counterfactual to prove it correct.'
    except Exception as error:
        return None,None,'**Experiment failed or was blocked.** '+str(error)


def select_world_source(source,mode):
    video=ROOT/'artifacts/world-model-input-preview-v1/reference.webm' if source=='bridge' else SOURCES['pip']/'reference.mp4'
    description=('Recorded task: move the spatula behind the pot. This is the model’s example arm, not Pip.' if source=='bridge' else
        'Actual Pip CAD and motor/battery proxy in a supported-torso fixture. One arm moves slowly. Fixed external camera; no grasp/object interaction. The frozen model has not been adapted to this robot.')
    return str(video),description,action_plot(mode,source),*world_history(source)


def build_app():
    with gr.Blocks(title='Pip · Experiment studio') as app:
        gr.Markdown('# Pip · Experiment studio\nTry the tools, inspect the evidence, and learn what each layer actually does.')
        from tools.vla_adaptation_studio import build_tab as build_adaptation_tab
        build_adaptation_tab()
        from tools.agile_studio import build_tab as build_agile_tab
        build_agile_tab()
        from tools.motor_studio import build_tab as build_motor_tab
        build_motor_tab()
        from tools.vla_studio import build_tab as build_vla_tab
        build_vla_tab()
        from tools.cleanup_studio import build_tab as build_cleanup_tab
        build_cleanup_tab()
        from tools.household_studio import build_tab
        build_tab()
        with gr.Tab('Control playground'):
            gr.Markdown('## Run → train → reload → inspect\nThese experiments use **Pip’s actual CAD and free-base MuJoCo model**, with the delayed motor and shared-battery proxy. Small CPU runs finish locally; no GPU rental is needed here. Start by inspecting a saved run, then try a new one.')
            names=list(experiments())
            default=next((name for name in names if (ROOT/'artifacts'/name/'replay.mp4').exists()),names[0] if names else None)
            selected=gr.Dropdown(names,value=default,label='Retained experiment')
            refresh_control=gr.Button('Refresh saved experiments')
            initial=control_result(default) if default else (None,'No runs yet.',None,[])
            with gr.Row():
                control_video=gr.Video(initial[0],label='Actual CAD · first evaluation case',interactive=False)
                control_chart=gr.Plot(initial[2],label='Recorded control signals')
            control_summary=gr.Markdown(initial[1])
            control_files=gr.Files(initial[3],label='Report, replay and checkpoint downloads',interactive=False)
            with gr.Accordion('Run a small experiment',open=True):
                kind=gr.Dropdown(['Reference controller','Frozen Isaac policy in MuJoCo','Adapt motor policy with PPO','Continue PPO checkpoint','LeRobot ACT imitation','Continue ACT checkpoint'],value='Frozen Isaac policy in MuJoCo',label='Experiment')
                gr.Markdown('**Frozen / Reference:** try the speed sliders. **PPO:** adapt the existing Isaac motor weights, then test hold/forward/reverse/turn. **LeRobot:** collect teacher demonstrations, train a tiny state-only ACT policy, reload and test it. “Continue” uses the selected saved run; select the matching checkpoint type. LeRobot is an imitation-learning component, not another physics engine.')
                with gr.Row():
                    forward=gr.Slider(-.06,.06,value=.025,step=.005,label='Requested forward speed (m/s) · frozen/reference only')
                    yaw=gr.Slider(-.15,.15,value=0.,step=.025,label='Requested turn rate (rad/s) · frozen/reference only')
                steps=gr.Number(value=2048,precision=0,label='PPO: 128–8192 steps, multiples of 128 · ACT: 1–500 updates')
                kind.change(lambda k:100 if 'ACT' in k else 2048,kind,steps)
                execute=gr.Button('Run and inspect',variant='primary')
                execute.click(run_control,[kind,steps,selected,forward,yaw],[selected,control_video,control_summary,control_chart,control_files],api_name='run_control',concurrency_limit=1,concurrency_id='control_jobs')
            inspect=gr.Button('Open selected run in the 3D inspector')
            replay=gr.HTML()
            inspect.click(show_replay,selected,replay,api_name='inspect_control')
            selected.change(control_result,selected,[control_video,control_summary,control_chart,control_files],api_name='inspect_saved_control')
            refresh_control.click(lambda:gr.update(choices=list(experiments())),outputs=selected)
            gr.Markdown('The video and 3D replay show the first evaluation case; the report retains all cases. Electrical values are estimates. Passing these short tests is not manipulation, robustness or hardware qualification. [Simple instructions](http://127.0.0.1:9095/gradio_api/file='+str(ROOT/'docs/design/alternative-control-quickstart.md')+').')
        with gr.Tab('World prediction'):
            gr.Markdown('## What would happen if the robot moved differently?\nStart with a recorded observation, change the proposed arm motion, and ask a frozen pretrained world model to predict the next 13 frames. Compare its example arm with a first **Pip transfer test**. Neither is yet a qualified learned RL environment.')
            world_source=gr.Radio([('Upstream example arm','bridge'),('Pip supported arm · transfer test','pip')],value='bridge',label='Observation source')
            with gr.Row():
                with gr.Column():
                    input_video=gr.Video(str(ROOT/'artifacts/world-model-input-preview-v1/reference.webm'),label='Recorded input · observed video',interactive=False,height=290)
                    input_description=gr.Markdown('Recorded task: move the spatula to the far-left corner behind the pot. Watch what actually happened before comparing a prediction.')
                with gr.Column():
                    mode=gr.Radio(choices=[(label,key) for key,label in MODES.items()],value='hold_arm',label='Change the planned movement')
                    plot=gr.Plot(action_plot('hold_arm'),label='Input action comparison')
                    gr.Markdown('The solid lines show proposed translation. Gripper inputs are kept unchanged. This plot uses the dataset’s raw actions; the native model loader separately records its normalized conditioning inputs.')
            status=gr.Markdown('Checking model access…')
            with gr.Row():
                refresh=gr.Button('Check model access')
                generate=gr.Button('Generate and compare predictions',variant='primary',interactive=False)
            gr.Markdown('One request generates the recorded-action prediction and the changed-action prediction. It uses one remote GPU, with a 25-minute processing budget and a 30-minute provider deadline. Setup/download time is included in the experiment; GPU resources are removed afterward. S3 evidence is retained.')
            with gr.Row():
                baseline=gr.Video(label='Predicted · recorded actions',interactive=False,height=290)
                changed=gr.Video(label='Predicted · changed actions',interactive=False,height=290)
            result=gr.Markdown('Retained predictions load below when available. The input video above is an observation, not a world-model prediction. Early attempts showed distorted colors/geometry; check each run’s settings and review before using predictions.')
            history=gr.Markdown()
            history_refresh=gr.Button('Refresh retained predictions and failed attempts')
            history_refresh.click(world_history,world_source,outputs=[baseline,changed,history],api_name='world_history')
            app.load(world_history,world_source,outputs=[baseline,changed,history])
            world_source.change(select_world_source,[world_source,mode],[input_video,input_description,plot,baseline,changed,history],api_name='select_world_source')
            with gr.Accordion('Connect the pretrained model',open=False):
                gr.Markdown('Accept access for [Cosmos Predict2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) and [Cosmos Guardrail](https://huggingface.co/nvidia/Cosmos-Guardrail1), then sign in locally:\n\n```sh\n.venv-experiments/bin/hf auth login\n```\nNever paste a token into this page or chat. Credentials are read locally and excluded from retained experiment files.')
            with gr.Accordion('What to look for',open=False):
                gr.Markdown('Does the arm respond to the changed action? Do objects stay consistent? Does a grasp appear to work only in the prediction? These are visual prediction checks. Contact accuracy, rewards and closed-loop RL still need independent evaluation in physics. Predicting a plausible clip is an early experiment, not a complete learned simulator.')
            mode.change(action_plot,[mode,world_source],plot)
            refresh.click(access_text,outputs=[status,generate],api_name='check_model_access')
            generate.click(predict,[mode,world_source],[baseline,changed,result],api_name='predict',concurrency_limit=1)
            app.load(access_text,outputs=[status,generate])
        with gr.Tab('Compare control runs'):
            gr.Markdown('## Same MuJoCo cases, different policies\nMost recent current-source hold/forward/reverse/turn suite for each policy type. Smaller errors are better. These tiny adaptation runs prove the learning loop; improvement over the authored controller is **not** assumed. PPO counts environment transitions; ACT counts optimizer updates.')
            gr.Markdown('Rows share one retained environment source version. This comparison may predate current task changes; individual runs retain their exact source and drift notices.')
            benchmark=gr.Dataframe(motor_comparison(),headers=['Policy','New training steps','Passed / 4','Mean forward RMS (m/s)','Mean turn RMS (rad/s)','Mean estimated energy (J)','Run'],interactive=False)
            gr.Button('Refresh policy comparison').click(motor_comparison,outputs=benchmark,api_name='motor_comparison')
            gr.Markdown('## Inspect real recorded robot experiments\nChoose which controllers to overlay. These curves come from retained native Isaac runs, not illustrative data.')
            selection=gr.CheckboxGroup(list(RUNS),value=list(RUNS)[:2],label='Recorded controllers')
            metric=gr.Radio(['Estimated supply voltage','Estimated pack current','Estimated energy'],value='Estimated supply voltage',label='Signal')
            initial,description=compare_runs(list(RUNS)[:2],'Estimated supply voltage')
            chart=gr.Plot(initial,label='Recorded telemetry')
            summary=gr.Markdown(description)
            selection.change(compare_runs,[selection,metric],[chart,summary],api_name='compare_runs')
            metric.change(compare_runs,[selection,metric],[chart,summary])
        with gr.Tab('Inspect Pip in Rerun'):
            gr.Markdown('## Actual CAD, camera and motion\nOrbit the robot, scrub the timeline, and select camera or electrical signals. This is the retained failed marker-approach trial. [Open Rerun separately]('+RERUN+').')
            gr.HTML('<iframe title="Pip native control replay" src="'+RERUN+'" style="width:100%;height:760px;border:0;border-radius:12px"></iframe>')
        with gr.Tab('What works today'):
            gr.Markdown('''## Keep the learning experience honest
| Tool | What you can inspect now | What remains |
|---|---|---|
| Isaac + RSL-RL/SB3 | Real motor and skill evaluation recordings | Native Isaac marker approach still fails; floor cleanup currently uses the MuJoCo backend |
| Rerun | Actual CAD, motion, camera and estimated electrical loads | Physical telemetry and calibration |
| MuJoCo + SB3 | Floor cleanup with PPO grasp residual (2/3 fixed cases pass), camera-driven interaction proxy, and motor comparisons | Broader grasp robustness, semantic vision, learned task planning and hardware validation |
| LeRobot ACT | Native demonstration dataset, small state-only imitation, reload and MuJoCo evaluation | Visual/manipulation policy and pretrained VLA adaptation |
| LeRobot GR00T N1.7 | Native pretrained inference on actual robot camera and named joint/IMU inputs; instruction/image sensitivity checks | New-embodiment adaptation, learned whole-body execution and task generalization remain missing |
| Cosmos world model | Action interventions, authenticated GPU request and retained success/failure results | Pip adaptation and imagined RL remain unqualified |
| mjlab / Playground / ManiSkill | Researched alternatives | Additional native adapters not implemented |

**Fetch → cleanup** is the first useful manipulation path. Cat play starts with slow movement, separation and retreat around a proxy; a simulator cannot establish a real cat’s comfort. Base-model pretraining is outside the plan.''')
    return app

if __name__=='__main__':
    # Exit normally so the existing atexit viewer cleanup runs on restart.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    build_app().queue(max_size=2,default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=9095,
        share=False,allowed_paths=[str(ROOT/'artifacts'),str(ROOT/'.artifact-store/objects'),str(ROOT/'docs/design/alternative-control-quickstart.md'),str(ROOT/'docs/design/floor-cleanup.md'),str(ROOT/'experiments/tasks/floor-cleanup-v0.yaml')],
        theme=gr.themes.Soft(primary_hue='teal'),footer_links=[])
