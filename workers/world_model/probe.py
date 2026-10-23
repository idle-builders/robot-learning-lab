"""Frozen Cosmos example inference, invoked inside the upstream locked environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mode', choices=['hold_arm', 'reverse_translation'], required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
config = json.loads((a.inputs/'config.json').read_text())
request = json.loads((a.inputs/'request.json').read_text())
from huggingface_hub import hf_hub_download
import torch

checkpoint = hf_hub_download(config['model_repo'], config['model_path'], revision=request['model_revision'])
checkpoint_digest = hashlib.sha256()
with open(checkpoint, 'rb') as weights:
    for chunk in iter(lambda: weights.read(8 * 1024 * 1024), b''):
        checkpoint_digest.update(chunk)
report = {'status': 'partial', 'training_steps': 0, 'robot': 'Pip supported-torso fixture; zero-shot transfer' if request.get('source')=='pip' else 'upstream Bridge example; not Pip',
    'input_source':request.get('source','bridge'), 'input_manifest':request.get('input_manifest'),
    'model_revision': request['model_revision'], 'source_commit': config['source_commit'],
    'model_file_sha256': checkpoint_digest.hexdigest(),
    'scope': config['scope'], 'cases': []}
# Input validation happened before submission; rewrite only the relative video location.
data = a.output/'inputs';(data/'annotation').mkdir(parents=True)
annotation = json.loads((a.inputs/'annotation.json').read_text())
annotation['videos'] = ['reference.mp4']
(data/'annotation/0.json').write_text(json.dumps(annotation))
(data/'reference.mp4').write_bytes((a.inputs/'reference.mp4').read_bytes())
try:
    for mode in ['recorded', a.mode]:
        dest = a.output/mode;dest.mkdir()
        # The released example overrides library defaults (guidance 0, 256x320).
        # Use its SHA-verified file rather than accidentally sampling at 640x480/7.
        example = (a.inputs/'upstream-example.json').read_bytes()
        if hashlib.sha256(example).hexdigest() != 'ce457704f9ecb546fecb2704ca7ad343a0deac7f3f9b1d1d7f01f9d1e74dcb65':
            raise ValueError('Upstream example settings changed')
        params = json.loads(example)
        params.update({'name': mode, 'input_root': str(data.absolute()), 'input_json_sub_folder': 'annotation',
            'save_root': str(dest.absolute()), 'start': 0, 'end': 1, 'chunk_size': 12,
            'single_chunk': True, 'num_steps': 35, 'save_fps': 3,
            'action_load_fn': 'action_loader.load_actions'})
        (dest/'request.json').write_text(json.dumps(params, indent=2))
        env = dict(os.environ, ROBOTLAB_ACTION_MODE=mode, ROBOTLAB_ACTION_OUTPUT=str((dest/'actions.json').absolute()))
        started = time.monotonic()
        with (dest/'inference.log').open('x') as log:
            result = subprocess.run([sys.executable, 'examples/action_conditioned.py', '-i', str((dest/'request.json').absolute()),
                '--output-dir', str((dest/'native').absolute()), '--checkpoint-path', checkpoint], env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=600)
        videos = list(dest.glob('*_single_chunk.mp4'))
        report['cases'].append({'mode': mode, 'exit_code': result.returncode, 'elapsed_s': time.monotonic()-started,
            'videos': [str(v.relative_to(a.output)) for v in videos]})
        if result.returncode or len(videos) != 1:
            raise RuntimeError('Native inference failed or expected video is missing')
    report['status'] = 'needs_image_review'
finally:
    report['gpu'] = torch.cuda.get_device_name()
    report['note'] = 'Original recorded video is a dataset observation; alternate-action prediction has no observed counterfactual ground truth.'
    (a.output/'report.json').write_text(json.dumps(report, indent=2))
