"""Download pinned tutorial weights; evaluation is the unmodified LeRobot CLI."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess

PINS = {
    'nvidia/gr00t17-lerobot-libero_10-640': '5ee08ab09fac5c5ef2388a14c882ea825ac861db',
    'nvidia/GR00T-N1.7-3B': '2fc962b973bccdd5d8ce4f67cc63b264d6886495',
    'nvidia/Cosmos-Reason2-2B': '9ce19a195e423419c349abfc86fd07178b230561',
    'lerobot/libero-assets': '0b3ea86be5fe169d0fd036ae63d1070ec09e90f6',
}


def download(output):
    from huggingface_hub import snapshot_download
    records = {}
    for repo, revision in PINS.items():
        if repo == 'lerobot/libero-assets':
            # hf-libero's native asset loader checks this directory first.
            path = Path(snapshot_download(repo, repo_type='dataset', revision=revision,
                local_dir=Path.home()/'.cache/libero/assets'))
        else:
            path = Path(snapshot_download(repo, revision=revision))
            refs = path.parent.parent / 'refs'
            refs.mkdir(exist_ok=True)
            (refs / 'main').write_text(revision)
        files = {}
        for p in sorted(path.rglob('*')):
            if p.is_file():
                with p.open('rb') as f:
                    digest = hashlib.file_digest(f, 'sha256').hexdigest()
                files[str(p.relative_to(path))] = {'sha256': digest, 'bytes': p.stat().st_size}
        records[repo] = {'revision': revision, 'path': str(path), 'files': files}
    (output / 'model-provenance.json').write_text(json.dumps(records, indent=2))
    # Complete upstream's first-import prompt without changing its code or paths.
    subprocess.run([sys.executable, '-c', 'import libero.libero'], input='n\n',
                   text=True, check=True, timeout=30)


def seal_training(output):
    """Keep the native adapter portable, and record its real optimizer step."""
    from safetensors.torch import load_file
    alias = output/'training/checkpoints/last'
    checkpoint = alias.resolve(strict=True)
    if checkpoint.parent != alias.parent.resolve():
        raise ValueError('Unexpected native checkpoint alias')
    (output/'checkpoint-alias.json').write_text(json.dumps({'alias':str(alias.relative_to(output)),
        'target':str(checkpoint.relative_to(output))},indent=2))
    if alias.is_symlink():
        alias.unlink()
    cfg_path = checkpoint/'pretrained_model/adapter_config.json'
    cfg = json.loads(cfg_path.read_text())
    base = 'nvidia/gr00t17-lerobot-libero_10-640'
    if cfg['base_model_name_or_path'] != base:
        raise ValueError('Unexpected PEFT base policy')
    cfg['revision'] = PINS[base]
    cfg_path.write_text(json.dumps(cfg, indent=2))
    tensors = load_file(str(checkpoint/'pretrained_model/adapter_model.safetensors'))
    b = [v for k,v in tensors.items() if 'lora_B' in k]
    import torch
    changed = bool(b) and all(torch.isfinite(v).all().item() for v in tensors.values()) and any(torch.count_nonzero(v).item() for v in b)
    step = json.loads((checkpoint/'training_state/training_step.json').read_text())['step']
    summary = {'method':'native LeRobot PEFT LoRA imitation', 'training_steps':step,
        'rl_steps':0, 'adapter_finite_and_changed_from_zero_B':changed,
        'checkpoint':str(checkpoint.relative_to(output)), 'base_revision':PINS[base]}
    (output/'training-summary.json').write_text(json.dumps(summary,indent=2))
    if step != 20 or not changed:
        raise ValueError('Native adapter training evidence is incomplete')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'seal':
        seal_training(Path(sys.argv[1]))
    else:
        download(Path(sys.argv[1]))
