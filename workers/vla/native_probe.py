"""Native LeRobot GR00T IO probe, never a claim of an adapted robot controller."""
import argparse
import hashlib
import json
from pathlib import Path
import time

PINS = {
    'nvidia/GR00T-N1.7-3B': '2fc962b973bccdd5d8ce4f67cc63b264d6886495',
    'nvidia/Cosmos-Reason2-2B': '9ce19a195e423419c349abfc86fd07178b230561',
}


def download(output):
    from huggingface_hub import snapshot_download
    records = {}
    for repo, revision in PINS.items():
        path = Path(snapshot_download(repo, revision=revision))
        # Pin the native library's fixed backbone name in this private worker cache.
        # The inference subprocess is offline, so it cannot follow a moving main.
        refs = path.parent.parent / 'refs'
        refs.mkdir(exist_ok=True)
        (refs / 'main').write_text(revision)
        records[repo] = {'revision': revision, 'path': str(path), 'files': {}}
        for f in path.rglob('*'):
            if f.is_file():
                with f.open('rb') as stream:
                    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                records[repo]['files'][str(f.relative_to(path))] = {'sha256': sha, 'bytes': f.stat().st_size}
    (output / 'models.json').write_text(json.dumps(records, indent=2))


def inference(inputs, output):
    import numpy as np
    from PIL import Image
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

    request = json.loads((inputs / 'observation.json').read_text())
    models = json.loads((output / 'models.json').read_text())
    state = torch.tensor(request['state'], dtype=torch.float32)
    rgb = torch.from_numpy(np.asarray(Image.open(inputs / 'head.png').convert('RGB')).copy()).permute(2, 0, 1)
    config = GrootConfig(
        base_model_path=models['nvidia/GR00T-N1.7-3B']['path'],
        embodiment_tag='new_embodiment', device='cuda',
        input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(len(state),)),
                        'observation.images.head': PolicyFeature(type=FeatureType.VISUAL, shape=tuple(rgb.shape))},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(len(request['action_names']),))},
        tune_llm=False, tune_visual=False, tune_projector=False,
        tune_diffusion_model=False, tune_vlln=False, use_flash_attention=False,
        use_relative_actions=False, chunk_size=40, n_action_steps=4, push_to_hub=False,
    )
    config.save_pretrained(output / 'policy-config')
    stats = {key: {k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()}
             for key, values in request['engineering_bounds'].items()}
    pre, post = make_groot_pre_post_processors(config, dataset_stats=stats)
    policy = GrootPolicy(config).to('cuda').eval()
    pre.save_pretrained(output / 'processors')
    post.save_pretrained(output / 'processors')
    rows = []
    for case, instruction, pixels in [
        ('pickup', 'Pick up the orange block from the floor and put it on the green area.', rgb),
        ('hold', 'Stay still and leave the orange block on the floor.', rgb),
        ('image_ablation', 'Pick up the orange block from the floor and put it on the green area.', torch.zeros_like(rgb)),
    ]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        policy.reset()
        started = time.monotonic()
        with torch.inference_mode():
            batch = pre({'observation.state': state.clone(), 'observation.images.head': pixels, 'task': instruction})
            normalized = policy.predict_action_chunk(batch)
            decoded = post(normalized)
        torch.cuda.synchronize()
        if not torch.isfinite(decoded).all():
            raise ValueError('Nonfinite native policy output')
        rows.append({'case': case, 'instruction': instruction, 'seconds': time.monotonic() - started,
                     'normalized_shape': list(normalized.shape), 'normalized': normalized.cpu().float().tolist(),
                     'actions': decoded.cpu().float().tolist()})
    report = {
        'status': 'io_probe_pass', 'task_success': 'not_run', 'training_steps': 0,
        'scope': 'Unadapted NEW_EMBODIMENT native GR00T inference. Engineering bounds are not learned dataset statistics. Outputs were NOT applied to the robot. No task generalization or control success established.',
        'cases': rows, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'language_action_max_difference': float(np.max(np.abs(np.asarray(rows[0]['actions']) - np.asarray(rows[1]['actions'])))),
        'image_action_max_difference': float(np.max(np.abs(np.asarray(rows[0]['actions']) - np.asarray(rows[2]['actions'])))),
    }
    (output / 'inference.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['download', 'inference'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'download':
        download(args.output)
    else:
        inference(args.inputs, args.output)
