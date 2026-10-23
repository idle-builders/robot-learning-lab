"""Retained native training/evaluation inspection. Viewing starts no GPU job."""
import json
import hashlib
import subprocess
from pathlib import Path
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[1]


def runs():
    return {p.parent.name:p.parent for p in sorted(
        [p for pattern in ['vla-adaptation-*/cleanup.json','vla-evaluation-*/cleanup.json']
         for p in (ROOT/'artifacts').glob(pattern)],key=lambda p:p.parent.stat().st_mtime,reverse=True)}


def inspect(name, case):
    path=runs()[name]
    report_path=path/'evidence/report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {
        'status':'artifact_handoff_failed' if (path/'response.json').exists() else 'cancelled_before_result',
        'training_steps':'unverified in handoff'}
    evaluation=path/'evidence/evaluation.json'
    result=json.loads(evaluation.read_text()) if evaluation.exists() else {'task_success':'not_run'}
    dest=path/'evidence'/case
    plot=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.08,
                       specs=[[{}],[{}],[{'secondary_y':True}]],
                       subplot_titles=['Knee command and encoder','Forward command and motion','Electrical proxy'])
    decisions=dest/'decisions.jsonl'
    prediction_timing = ''
    if decisions.exists():
        rows=[json.loads(line) for line in decisions.open()]
        predictions = [r['inference_s'] for r in rows if r.get('chunk_action_index',0)==0]
        if predictions:
            prediction_timing = f'Fresh predictions: **{len(predictions)}**; mean prediction call: **{1000*sum(predictions)/len(predictions):.0f} ms**. Queued actions are excluded from this latency.\n\n'
        for index,title,panel in [(7,'Left knee command [rad]',1),(14,'Forward command [m/s]',2)]:
            plot.add_trace(go.Scatter(x=[r['time_s'] for r in rows],y=[r['accepted_action'][index] for r in rows],name=title),row=panel,col=1)
    trace=dest/'trajectory.jsonl'
    if trace.exists():
        rows=[json.loads(line) for line in trace.open()]
        times=[r['time_s'] for r in rows]
        plot.add_trace(go.Scatter(x=times,y=[r['q'][7] for r in rows],name='Knee encoder [rad]'),row=1,col=1)
        plot.add_trace(go.Scatter(x=times,y=[r['forward_m_s'] for r in rows],name='Actual forward speed [m/s]'),row=2,col=1)
        plot.add_trace(go.Scatter(x=times,y=[r['electromechanical']['estimated_current_a'] for r in rows],name='Estimated current [A]'),row=3,col=1)
        plot.add_trace(go.Scatter(x=times,y=[r['electromechanical']['bus_voltage_v'] for r in rows],name='Estimated bus voltage [V]'),row=3,col=1,secondary_y=True)
    plot.update_yaxes(title_text='rad',row=1,col=1)
    plot.update_yaxes(title_text='m/s',row=2,col=1)
    plot.update_yaxes(title_text='A',row=3,col=1,secondary_y=False)
    plot.update_yaxes(title_text='V',row=3,col=1,secondary_y=True)
    plot.update_xaxes(title_text='Simulation time [s]',row=3,col=1)
    plot.update_layout(height=630,legend={'orientation':'h','y':-.15})
    scope=('**Experiment: native SmolVLA → LoRA imitation → checkpoint reload → MuJoCo.**\n\n'
           f"Execution: **{report['status']}**. New training updates in this run: **{report.get('training_steps',0)}**. All fixed tasks passed: **{result.get('task_success','not_run')}**.\n\n"
           f"Executed actions per predicted chunk: **{result.get('action_execution_steps','not_run')}**. Evaluation-only runs reuse the submitted adapter without further training.\n\n"
           + prediction_timing +
           'Sensors are sampled at 10 Hz; a fresh observation reaches the policy only when it generates a new chunk. Queued actions use the earlier observation.\n\n'
           'The model receives head RGB, 25 encoder/IMU/effort values and the instruction. It predicts 14 joint references, forward/yaw velocity and an odometry reference error. '
           'The evaluation does not call the authored cleanup teacher or supply task phases/object truth to the model. '
           f"**Motor control:** {result.get('whole_body_control','not_run; no physical evaluation returned')}\n\n"
           '**VLA training data:** four cleanup and two hold demonstrations; supervised imitation, not VLA RL. '
           'An optional PPO motor checkpoint is trained separately and identified in the result. This does not establish qualified whole-body control, broad task generalization or measured sim-to-real. '
           'Simulation pauses during inference; this replay does not establish real-time performance. All three test cases were fixed before training.')
    files=[str(f) for f in [path/'evidence/evaluation.json',path/'evidence/training-summary.json',path/'evidence/models.json',
           path/'submitted-source/contract.json',path/'submitted-source/evaluation-cases.json',path/'submitted-source/motor/contract.json',path/'evidence/plant-preflight.json',path/'evidence/plant-preflight.log',path/'evidence/error.log',path/'response.json',path/'flash.log',path/'evidence/train.log',path/'evidence/evaluate.log',path/'evidence/action-audit.json',dest/'chunks.jsonl',dest/'decisions.jsonl',dest/'trajectory.jsonl',path/'cleanup.json'] if f.exists()]
    files += [str(p) for p in (path/'evidence/training/checkpoints').glob('*/pretrained_model/*') if p.name in {'adapter_model.safetensors','adapter_config.json','config.json','train_config.json'}]
    video=dest/'replay.mp4'
    return str(video) if video.exists() else None,scope,result,plot,files


def inspect_chunk(name, case, time_s):
    dest = runs()[name]/'evidence'/case
    path = dest/'chunks.jsonl'
    plot = go.Figure()
    if not path.exists():
        plot.update_layout(title='Full action chunks were not recorded in this historical run')
        return plot
    chunks = [json.loads(line) for line in path.open()]
    if not chunks:
        plot.update_layout(title='This retained trial produced no action chunks')
        return plot
    selected = [c for c in chunks if c['acquisition_s'] <= time_s + 1e-6]
    chunk = selected[-1] if selected else chunks[0]
    times = [chunk['acquisition_s'] + i/10 for i in range(len(chunk['predicted_actions']))]
    predicted = [a[7] for a in chunk['predicted_actions']]
    plot.add_trace(go.Scatter(x=times,y=predicted,name='Predicted knee commands; not a state forecast'))
    n = chunk['execution_steps']
    plot.add_trace(go.Scatter(x=times[:n],y=predicted[:n],mode='markers',name='Prefix sent through command bounds'))
    trace = [json.loads(line) for line in (dest/'trajectory.jsonl').open()]
    trace = [r for r in trace if times[0] <= r['time_s'] <= times[-1]]
    plot.add_trace(go.Scatter(x=[r['time_s'] for r in trace],y=[r['q'][7] for r in trace],name='Actual encoder, including later replans'))
    plot.update_layout(title=f"Chunk acquired at {times[0]:.1f} s; execute {n}, then replan",height=350,
                       xaxis_title='Simulation time [s]',yaxis_title='Left knee [rad]',legend={'orientation':'h','y':-.25})
    return plot


def inspect_3d(name,case):
    run=runs().get(name)
    if run is None or case not in {'nominal_cleanup','lateral_cleanup','hold'}:
        raise ValueError('Select a retained run and case')
    if not (run/'evidence'/case/'trajectory.jsonl').exists():
        raise ValueError('This run has no physical rollout to inspect')
    writer=ROOT/'tools/write_learned_control_rrd.py'
    writer_hash=hashlib.sha256(writer.read_bytes()).hexdigest()
    candidates=[]
    for path in (ROOT/'artifacts').glob('learned-control-view-*/report.json'):
        report=json.loads(path.read_text())
        source=path.parent/'source/tools/write_learned_control_rrd.py'
        if (report['status']=='pass' and Path(report['run']).absolute()==run.absolute()
            and report['case']==case and source.exists()
            and hashlib.sha256(source.read_bytes()).hexdigest()==writer_hash):
            candidates.append(path.parent)
    if candidates:
        target=candidates[-1]
    else:
        result=subprocess.run([str(ROOT/'.venv-learning/bin/python'),str(writer),'export',str(run),case],
            cwd=ROOT,capture_output=True,text=True,timeout=180)
        if result.returncode:
            raise RuntimeError('Rerun export failed; retained output: '+result.stdout+result.stderr)
        target=Path(result.stdout.splitlines()[0])
    from lab.experiment_catalog import open_recording
    url=open_recording(target/'episode.rrd')
    return '<iframe title="Learned control in Rerun" src="'+url+'" style="width:100%;height:760px;border:0"></iframe>'


def build_tab():
    available=runs()
    with gr.Tab('Learned visual control · trials'):
        gr.Markdown('## Inspect native adaptation and teacher-free physics trials')
        if not available:
            gr.Markdown('The first bounded adaptation is running or has not returned evidence. No learned task success is claimed.')
            return
        first=next(iter(available))
        refresh=gr.Button('Refresh retained runs')
        with gr.Row():
            run=gr.Dropdown(list(available),value=first,label='Retained run')
            case=gr.Dropdown(['nominal_cleanup','lateral_cleanup','hold'],value='nominal_cleanup',label='Fixed evaluation case')
        initial=inspect(first,'nominal_cleanup')
        video=gr.Video(initial[0],label='Actual policy rollout with head camera',interactive=False)
        inspect_button=gr.Button('Inspect learned rollout in 3D · Rerun')
        native=gr.HTML()
        inspect_button.click(inspect_3d,[run,case],native)
        scope=gr.Markdown(initial[1])
        result=gr.JSON(initial[2],label='Independent physical outcomes')
        plot=gr.Plot(initial[3])
        gr.Markdown('### Inspect a predicted action chunk\nThe VLA predicts commands, not the future physical state. Only the marked prefix is executed; later commands are replaced by a new prediction.')
        chunk_time=gr.Slider(0,40,value=0,step=.1,label='Prediction acquisition time [simulation seconds]')
        chunk_plot=gr.Plot(inspect_chunk(first,'nominal_cleanup',0))
        files=gr.Files(initial[4],label='Inputs, predictions, plant telemetry, training and cleanup evidence',interactive=False)
        gr.Markdown('**Use:** select a run and evaluation case, play its actual rollout, then inspect commands and physical outcomes. '
                    'To collect new demonstrations run `.venv-lerobot/bin/python tools/collect_vla.py`. '
                    'To run a new bounded GPU adaptation use `.venv-experiments/bin/python tools/probe_vla.py run --mode adapt --inputs <retained-demonstration-directory>`. '
                    'Viewing this page incurs no GPU usage.')
        refresh.click(lambda: gr.Dropdown(choices=list(runs()),value=next(iter(runs()))), outputs=run).then(inspect,[run,case],[video,scope,result,plot,files])
        for component in [run,case]:
            component.change(inspect,[run,case],[video,scope,result,plot,files])
            component.change(lambda:'',outputs=native)
        for component in [run,case,chunk_time]:
            component.change(inspect_chunk,[run,case,chunk_time],chunk_plot)
