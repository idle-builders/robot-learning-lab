"""Bounded supervisor for the official LeRobot HIL-SERL actor/learner tutorial."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import difflib
import importlib.util

PINS = {
    'lerobot/config_examples': ('dataset', '4803f5f379568ddc77763f5a4dd7cd56973d6a10'),
    'lilkm/pick_cube_franka_panda_30': ('dataset', 'f80a680a117d0c8573ab453e3fd9e792e308dfa4'),
    'lerobot/resnet10': ('model', '60e4365989619db6365ba92b1169d5dcc143d165'),
}


def main(output):
    from huggingface_hub import snapshot_download, hf_hub_download
    records = {}
    paths = {}
    for repo, (kind, revision) in PINS.items():
        if repo == 'lerobot/config_examples':
            path = Path(hf_hub_download(repo, 'rl/gym_hil/train_config.json', repo_type=kind, revision=revision))
            (output/'upstream-train-config.json').write_bytes(path.read_bytes())
            files = [path]
        else:
            path = Path(snapshot_download(repo, repo_type=kind, revision=revision))
            files = [p for p in path.rglob('*') if p.is_file()]
            refs = path.parent.parent/'refs'
            refs.mkdir(exist_ok=True)
            (refs/'main').write_text(revision)
        paths[repo] = path
        records[repo] = {'revision': revision, 'type': kind, 'files': {
            str(p.relative_to(path) if path.is_dir() else p.name): {
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
            for p in files}}
    (output/'model-provenance.json').write_text(json.dumps(records, indent=2))
    cfg = json.loads((output/'upstream-train-config.json').read_text())
    cfg.update(output_dir=str(output/'training'), job_name='tutorial-smoke',
               batch_size=32, num_workers=0, save_freq=20)
    cfg['wandb']['enable'] = False
    cfg['dataset'].update(root=str(paths['lilkm/pick_cube_franka_panda_30']), video_backend='pyav')
    cfg['algorithm']['use_torch_compile'] = False
    cfg['policy'].update(online_steps=200, online_buffer_capacity=1000, offline_buffer_capacity=1000)
    # Same native wrapped task, without the physical gamepad/passive GUI wrapper.
    cfg['env']['task'] = 'PandaPickCube-v0'
    # gym_hil owns reset and truncates at 100 steps. The optional processor's
    # 150-step limit is redundant; its Any-typed physical pose breaks draccus.
    cfg['env']['processor']['reset'] = None
    config = output/'train-config.json'
    config.write_text(json.dumps(cfg, indent=2))
    before = json.loads((output/'upstream-train-config.json').read_text())
    def diff(a, b, path=''):
        changes = {}
        for k in a.keys() | b.keys():
            key = path+'.'+k if path else k
            if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
                changes.update(diff(a[k], b[k], key))
            elif a.get(k) != b.get(k):
                changes[key] = {'upstream': a.get(k), 'run': b.get(k)}
        return changes
    (output/'config-diff.json').write_text(json.dumps(diff(before, cfg), indent=2))
    # LeRobot 0.6.1's shared dataset factory hardcodes return_uint8=True;
    # HIL-SERL's online observations and replay writer require float [0,1].
    # Use the same public native dataset loader with its float option in this
    # isolated worker. Keep the trainer/algorithm/environment implementations.
    learner_source = Path(importlib.util.find_spec('lerobot.rl.learner').origin)
    original = learner_source.read_text()
    old = '        offline_dataset = make_dataset(cfg)'
    new = '''        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id, root=cfg.dataset.root,
            episodes=cfg.dataset.episodes, revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend, return_uint8=False,
        )'''
    if original.count(old) != 1:
        raise ValueError('Upstream RL dataset loader changed; review compatibility fix')
    corrected = original.replace(old, new)
    (output/'native-learner-before.py').write_text(original)
    (output/'native-learner-after.py').write_text(corrected)
    (output/'native-compatibility.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), corrected.splitlines(True), fromfile='lerobot/rl/learner.py',
        tofile='lerobot/rl/learner.py')))
    learner_source.write_text(corrected)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    commands = {role: [sys.executable, '-m', 'lerobot.rl.'+role, '--config_path', str(config)]
                for role in ['learner', 'actor']}
    # Both native CLIs reject an existing output_dir when resume=false.
    commands['actor'].append('--output_dir='+str(output/'actor-runtime'))
    (output/'native-commands.json').write_text(json.dumps(commands, indent=2))
    processes = {}
    logs = {}
    completed = False
    try:
        for role, command in commands.items():
            logs[role] = (output/(role+'.log')).open('x')
            processes[role] = subprocess.Popen(command, env=env, stdout=logs[role],
                stderr=subprocess.STDOUT)
        deadline = time.monotonic()+240
        while time.monotonic() < deadline:
            learner_log = (output/'learner.log').read_text(errors='replace')
            if 'Resume training' in learner_log and (output/'training/checkpoints/last').is_symlink():
                completed = True
                break
            for role, proc in processes.items():
                if proc.poll() is not None:
                    if role == 'actor' and proc.returncode == 0:
                        continue  # Bounded actor may finish while learner saves.
                    raise RuntimeError(f'Native {role} exited before a completed checkpoint: {proc.returncode}')
            time.sleep(.2)
        if not completed:
            raise TimeoutError('Native actor/learner did not complete a checkpoint within 240 seconds')
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in processes.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs.values():
            log.close()
        alias = output/'training/checkpoints/last'
        if alias.is_symlink():
            checkpoint = alias.resolve(strict=True)
            if checkpoint.parent != alias.parent.resolve():
                raise ValueError('Unexpected checkpoint alias')
            (output/'checkpoint-alias.json').write_text(json.dumps({
                'alias': str(alias.relative_to(output)), 'target': str(checkpoint.relative_to(output))}))
            alias.unlink()
    step = json.loads((checkpoint/'training_state/training_step.json').read_text())
    if step['step'] < 20:
        raise ValueError('Incomplete native RL update evidence')
    (output/'rl-summary.json').write_text(json.dumps({
        'status': 'native_rl_checkpoint_saved', 'optimizer_steps': step['step'],
        'interaction_step_at_checkpoint': step['interaction_step'],
        'task_success': 'not_evaluated', 'checkpoint': str(checkpoint.relative_to(output)),
        'scope': 'Native SAC tutorial smoke test; not connected to GR00T or claimed task competence.'}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
