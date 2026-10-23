"""Read-only inspection of native PPO motor learning and retained failures."""
import json
from pathlib import Path
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT=Path(__file__).resolve().parents[1]

def runs():
    return {p.parent.name:p.parent for p in sorted([p for pattern in ['motor-tracker-*/report.json','motor-warmstart-*/report.json'] for p in (ROOT/'artifacts').glob(pattern)],
        key=lambda p:p.parent.stat().st_mtime,reverse=True)}

def inspect(name,stage,case):
    root=runs()[name]
    report=json.loads((root/'report.json').read_text())
    dest=root/stage/case
    path=dest/'trajectory.jsonl'
    rows=[json.loads(x) for x in path.open()] if path.exists() else []
    plot=make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=['Pitch','Wheel command / encoder','Estimated battery current'])
    if rows:
        times=[r['time_s'] for r in rows]
        plot.add_trace(go.Scatter(x=times,y=[r['pitch_rad'] for r in rows],name='Pitch [rad]'),row=1,col=1)
        for key,label in [('targets','Left wheel command'),('dq','Left wheel encoder')]:
            plot.add_trace(go.Scatter(x=times,y=[r[key][8] for r in rows],name=label+' [rad/s]'),row=2,col=1)
        plot.add_trace(go.Scatter(x=times,y=[r['electromechanical']['estimated_current_a'] for r in rows],name='Current proxy [A]'),row=3,col=1)
    plot.update_layout(height=600,legend={'orientation':'h','y':-.18})
    plot.update_xaxes(title_text='Simulation time [s]',row=3,col=1)
    summary=(f"**{report['status']}** · {report.get('actual_transitions',0):,} new RL transitions · "
        f"checkpoint reload match: **{report.get('reload_action_match','not_run')}**.\n\n"
        f"Training method: {report.get('method',report.get('trainer','native PPO'))}.\n\n"
        'PPO owns all 16 actuator commands. Joint targets are conditioned on supplied references; wheel velocity is entirely learned. '
        'There is no analytic balance or gravity feedforward in this path. Joint PD, command guards and electrical dynamics remain defined. '
        'These are motor tests, not cleanup or language-task success. Before means the starting weights of this run, which may be a resumed checkpoint. '
        'Training return is not an acceptance metric; inspect falls and tracking errors below.')
    result=report.get(stage,{}).get(case,{'status':'not_run'})
    files=[str(p) for p in [root/'report.json',root/'policy.zip',root/'contract.json',root/'learning.jsonl',
        root/'evaluation-cases.json',root/'regressor.joblib',dest/'report.json',dest/'trajectory.jsonl',dest/'decisions.jsonl'] if p.exists()]
    video=dest/'replay.mp4'
    return str(video) if video.exists() else None,summary,result,plot,files

def build_tab():
    available=runs()
    with gr.Tab('Learned motor · RL trials'):
        gr.Markdown('## Inspect motor RL, checkpoint continuation and physical failures\nViewing these saved experiments starts no training or cloud worker.')
        if not available:
            gr.Markdown('No retained motor run yet.');return
        first=next(iter(available))
        with gr.Row():
            run=gr.Dropdown(list(available),value=first,label='Motor training run')
            stage=gr.Dropdown(['before','after'],value='after',label='Checkpoint stage')
            case=gr.Dropdown(['hold','forward','bend'],value='hold',label='Motor reference test')
        data=inspect(first,'after','hold')
        video=gr.Video(data[0],label='Actual motor policy rollout',interactive=False)
        scope=gr.Markdown(data[1]);result=gr.JSON(data[2],label='Fixed motor checks')
        plot=gr.Plot(data[3]);files=gr.Files(data[4],label='Training, checkpoint and physical evidence',interactive=False)
        for component in [run,stage,case]:
            component.change(inspect,[run,stage,case],[video,scope,result,plot,files])
        gr.Markdown('**Run locally:** `.venv-learning/bin/python tools/train_motor_tracker.py --steps 32768 --seconds 180 --render`. '
                    'Use `--parent artifacts/MOTOR_RUN/policy.zip` to continue a matching checkpoint. '
                    'The Learned visual control tab shows joined VLA evaluations when supplied with `--motor-checkpoint`.')
