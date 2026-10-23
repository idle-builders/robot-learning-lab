"""Inspect native pretrained-policy inputs and outputs using existing Gradio UI."""
import json
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def runs():
    return {p.parent.parent.name: p.parent.parent for p in sorted(
        (ROOT / 'artifacts').glob('vla-probe-*/evidence/inference.json'),
        key=lambda p: p.stat().st_mtime, reverse=True)}


def result(name, case):
    available = runs()
    if name not in available:
        raise ValueError('Choose a retained native VLA probe')
    path = available[name]
    report = json.loads((path / 'evidence/inference.json').read_text())
    observation = json.loads((path / 'submitted-source/observation.json').read_text())
    selected = next(row for row in report['cases'] if row['case'] == case)
    actions = np.asarray(selected['actions'])[0]
    low = np.asarray(observation['engineering_bounds']['action']['min'])
    high = np.asarray(observation['engineering_bounds']['action']['max'])
    limited = np.isclose(actions, low, atol=1e-5) | np.isclose(actions, high, atol=1e-5)
    table = [[joint, *[round(float(x), 5) for x in actions[:, i]], bool(limited[:, i].any())]
             for i, joint in enumerate(observation['action_names'])]
    status = (
        '**Native pretrained GR00T N1.7 inference executed. Robot task control: NOT RUN.**\n\n'
        f"Instruction: **{selected['instruction']}**\n\n"
        f"This call took **{selected['seconds']:.3f} s**. **{int(limited.sum())}/{limited.size}** decoded values touch the provisional engineering bounds. "
        'These are unadapted new-embodiment predictions, not a commanded trajectory. No action was applied to the simulator.\n\n'
        'Inputs: a synchronized simulated head-camera image, joint positions (wheel speeds for continuous wheels), body-frame gravity and gyro. '
        'There is no task-phase label, color-detector estimate or object truth in the policy input. '
        'The image-ablation case supplies black pixels; other cases use the displayed image.\n\n'
        'The native model predicts a 40-step horizon; this probe retains the first four steps. '
        'Engineering normalization bounds are provisional, not demonstration statistics. '
        'Different predictions demonstrate input sensitivity, not correct task understanding.\n\n'
        '**Next missing connection:** robot-specific adaptation and a learned whole-body controller, then fixed multistep task evaluation. '
        'The existing cleanup video remains an authored comparison baseline.'
    )
    files = [str(path / rel) for rel in ['evidence/inference.json', 'evidence/models.json',
              'evidence/policy-config/config.json', 'evidence/versions.log', 'submitted-source/observation.json', 'cleanup.json']]
    return str(path / 'submitted-source/head.png'), status, table, observation, files


def build_tab():
    available = runs()
    with gr.Tab('Pretrained control · integration'):
        gr.Markdown('## Inspect the native model connection\n'
                    'The product goal is to iterate **learned control ↔ humanoid structure/CAD ↔ electronics/PCB** in simulation, '
                    'then improve the same models using physical measurements. Existing designs can change. '
                    'This page shows the current pretrained-policy connection, not completion of that goal.')
        if not available:
            gr.Markdown('No completed native VLA inference probe yet.')
            return
        selected = next(iter(available))
        with gr.Row():
            run = gr.Dropdown(list(available), value=selected, label='Retained GR00T probe')
            case = gr.Dropdown(['pickup', 'hold', 'image_ablation'], value='pickup', label='Same-seed comparison')
        initial = result(selected, 'pickup')
        with gr.Row():
            camera = gr.Image(initial[0], label='Robot head-camera input', interactive=False)
            status = gr.Markdown(initial[1])
        table = gr.Dataframe(initial[2], headers=['Named action and unit', 'Step 0', 'Step 1', 'Step 2', 'Step 3', 'Touches bound'], interactive=False)
        with gr.Accordion('Exact sensor values and provisional action mapping', open=False):
            observation = gr.JSON(initial[3])
        files = gr.Files(initial[4], label='Native outputs, pinned model hashes, processors and cleanup evidence', interactive=False)
        gr.Markdown('**Use:** switch the comparison to inspect the response to a different instruction or a black image. '
                    'The four action columns are predictions, not observed motion. '
                    'This viewer reads retained results and starts no GPU worker.')
        for component in [run, case]:
            component.change(result, [run, case], [camera, status, table, observation, files])
