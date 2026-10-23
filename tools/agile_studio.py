"""Expose native AGILE evidence through the existing studio and Rerun viewer."""
import json
from pathlib import Path
import gradio as gr
from lab.experiment_catalog import open_recording

ROOT = Path(__file__).resolve().parents[1]


def replays():
    return {p.parent.name: p.parent for p in sorted((ROOT / 'artifacts').glob('agile-replay-*/report.json'),
            key=lambda p: p.stat().st_mtime, reverse=True) if (p.parent / 'episode.rrd').is_file()}


def inspect(name):
    root = replays()[name]
    report = json.loads((root / 'report.json').read_text())
    native = Path(report['native_episode']).parent
    outcome = (f"First orientation failure: **{report['first_orientation_failure_s']:.2f} s**; task success unqualified. "
               if report.get('first_orientation_failure_s') is not None else
               f"Native recorded success flags: **{report['recorded_is_success']}**. ")
    text = (f"**{report['frames']} recorded frames · {report['last_time_s']:.2f} seconds**\n\n" + outcome +
            "A failed initial hold does not test the later drive, turn or crouch commands.\n\n" + report['scope'])
    files = [str(p) for p in [root / 'report.json', root / 'episode.rrd', native / 'metadata.json',
             Path(report['native_episode']), native.parent / 'report.json', native.parent / 'metrics.json', native.parent / 'reports/index.html'] if p.exists()]
    return text, files, ''


def show(name):
    url = open_recording(replays()[name] / 'episode.rrd')
    return f'<iframe title="Native AGILE CAD replay" src="{url}" style="width:100%;height:750px;border:0"></iframe>'


def build_tab():
    with gr.Tab('Native Isaac · AGILE'):
        gr.Markdown('## Native training → checkpoint → evaluation → CAD replay\n'
                    'The motor policy produces four leg-position actions and two wheel-velocity actions. '
                    'Native AGILE/RSL-RL trains and evaluates it. These are motor tests, not completed cleanup tasks. '
                    'Viewing evidence launches no cloud job.')
        available = replays()
        if not available:
            gr.Markdown('No exported native episode yet.'); return
        first = next(iter(available))
        selected = gr.Dropdown(list(available), value=first, label='Recorded native episode')
        summary, paths, _ = inspect(first)
        details = gr.Markdown(summary)
        with gr.Row():
            refresh = gr.Button('Refresh episodes')
            view = gr.Button('Open CAD replay in Rerun', variant='primary')
        replay = gr.HTML()
        evidence = gr.Files(paths, label='Native trajectory, metrics and replay evidence', interactive=False)
        selected.change(inspect, selected, [details, evidence, replay], api_name='inspect_agile')
        refresh.click(lambda: gr.update(choices=list(replays())), outputs=selected)
        view.click(show, selected, replay, api_name='replay_agile')
        gr.Markdown('**Use Rerun:** drag the simulation timeline, press Space to pause/play, orbit/zoom the CAD, '
                    'then select motion, joint position, joint velocity and action/target plots. '
                    'The MuJoCo comparison also contains torque and electrical estimates. '
                    'Each recording states its actual channels and action semantics; no camera is recorded in these motor trials.')
