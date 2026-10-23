"""Native LeRobot/PEFT fine-tuning and teacher-free robot evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PINS = {'lerobot/smolvla_base':'d9f33c94a60fb382c90dea2164c96845bd955e28',
        'HuggingFaceTB/SmolVLM2-500M-Video-Instruct':'7b375e1b73b11138ff12fe22c8f2822d8fe03467'}


def download(inputs, output):
    from huggingface_hub import snapshot_download
    records = {}
    for repo, revision in PINS.items():
        path = Path(snapshot_download(repo, revision=revision))
        refs = path.parent.parent/'refs'
        refs.mkdir(exist_ok=True)
        (refs/'main').write_text(revision)
        files = {}
        for f in path.rglob('*'):
            if f.is_file():
                with f.open('rb') as stream:
                    digest = hashlib.file_digest(stream,'sha256').hexdigest()
                files[str(f.relative_to(path))] = {'sha256':digest,'bytes':f.stat().st_size}
        records[repo] = {'revision':revision,'files':files}
    (output/'models.json').write_text(json.dumps(records,indent=2))


def prepare_plant(inputs):
    sys.path.insert(0,str(inputs))
    artifacts = inputs/'artifacts'
    artifacts.mkdir(exist_ok=True)
    alias = artifacts/'pip-cad-gym-bundle-v4'
    if not alias.exists():
        alias.symlink_to(inputs/'bundle', target_is_directory=True)


def robot_from_inputs(inputs):
    from lab.vla_control import VisualRobot
    motor = inputs/'motor/policy.zip'
    return VisualRobot(inputs/'scene.xml', motor_checkpoint=motor if motor.exists() else None,
        motor_contract=json.loads((motor.parent/'contract.json').read_text()) if motor.exists() else None,
        motor_plant_reference=motor.parent/'plant-reference.npz' if (motor.parent/'plant-reference.npz').exists() else None)


def preflight(inputs, output):
    import numpy as np
    from PIL import Image
    prepare_plant(inputs)
    robot = robot_from_inputs(inputs)
    try:
        image = robot.image()
        if image.shape != (180,320,3) or np.std(image) < 1:
            raise ValueError('Invalid native head-camera render')
        Image.fromarray(image).save(output/'preflight-head.png')
        (output/'plant-preflight.json').write_text(json.dumps({'status':'pass','head_shape':list(image.shape),
            'state_shape':list(robot.state().shape),'physics':'MuJoCo 3.13.0','renderer':os.environ.get('MUJOCO_GL'),
            'motor_control':robot.contract()['low_level'],
            'motor_plant_validation':robot.learned_motor.validation if robot.learned_motor else None},indent=2))
    finally:
        robot.close()


def seal(inputs, output):
    # LeRobot's convenience alias contains no unique weights. Retain its exact
    # meaning as metadata and keep the numbered checkpoint once, without relaxing
    # the shared transport's prohibition on output directory symlinks.
    alias = output/'training/checkpoints/last'
    if alias.is_symlink():
        target = alias.resolve(strict=True)
        if target.parent != alias.parent.resolve() or not target.is_dir():
            raise ValueError('Unexpected native checkpoint alias target')
        record = {'alias':str(alias.relative_to(output)), 'target':str(target.relative_to(output.resolve()))}
        (output/'checkpoint-alias.json').write_text(json.dumps(record,indent=2))
        alias.unlink()


def train(inputs, output):
    from lerobot.configs.default import DatasetConfig, PeftConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.scripts.lerobot_train import train as native_train
    policy = SmolVLAConfig.from_pretrained('lerobot/smolvla_base', revision=PINS['lerobot/smolvla_base'])
    policy.pretrained_path = 'lerobot/smolvla_base'
    policy.pretrained_revision = PINS['lerobot/smolvla_base']
    policy.input_features = {}
    policy.output_features = {}
    policy.device = 'cuda'
    policy.push_to_hub = False
    policy.n_action_steps = 1
    policy.optimizer_lr = 1e-3
    policy.scheduler_warmup_steps = 10
    policy.scheduler_decay_steps = 300
    policy.scheduler_decay_lr = 1e-4
    # Keep native pretrained image processing, expert and flow-matching horizon.
    report = json.loads((inputs/'dataset-report.json').read_text())
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/pip-visual-control',root=str(inputs/'dataset'),
                              episodes=report['training_episodes'], video_backend='pyav'),
                              policy=policy, output_dir=output/'training',steps=300,batch_size=8,num_workers=0,
                              env_eval_freq=0,log_freq=25,save_freq=0,
                              peft=PeftConfig(method_type='LORA',r=16,lora_alpha=16),seed=712)
    # Native trainer handles preprocessing, optimizer, scheduler, AMP and checkpoints.
    start = time.monotonic()
    native_train(cfg)
    # Record the base revision in the native PEFT config as well as the run pins.
    for path in (output/'training/checkpoints').glob('*/pretrained_model/adapter_config.json'):
        config = json.loads(path.read_text())
        config['revision'] = PINS['lerobot/smolvla_base']
        path.write_text(json.dumps(config, indent=2))
    (output/'training-summary.json').write_text(json.dumps({'method':'native LeRobot SmolVLA LoRA imitation',
        'steps':300,'wall_s':time.monotonic()-start,'episodes':report['training_episodes'],
        'rl_steps':0,'scope':'Small adaptation smoke experiment; task success measured separately.'},indent=2))


def load_checkpoint(inputs, output, checkpoint=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    checkpoint = checkpoint or output/'training/checkpoints/last/pretrained_model'
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = str(checkpoint)
    cfg.pretrained_revision = None
    cfg.device = 'cuda'
    meta = LeRobotDatasetMetadata('local/pip-visual-control',root=inputs/'dataset')
    policy = make_policy(cfg,ds_meta=meta).eval()
    pre, post = make_pre_post_processors(cfg,pretrained_path=checkpoint)
    return policy, pre, post


def audit(inputs, output, checkpoint=None, action_steps=1):
    """In-sample action diagnosis, explicitly not a held-out success measure."""
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    policy, pre, post = load_checkpoint(inputs, output, checkpoint)
    dataset = LeRobotDataset('local/pip-visual-control',root=inputs/'dataset')
    contract = json.loads((inputs/'contract.json').read_text())
    scale = np.asarray(contract['action_max']) - np.asarray(contract['action_min'])
    rows = []
    errors = []
    for index in np.linspace(0, len(dataset)-1, 48, dtype=int):
        sample = dataset[int(index)]
        observation = {k: sample[k] for k in ['observation.state','observation.images.head','task']}
        policy.reset()
        torch.manual_seed(713)
        with torch.inference_mode():
            chunk = post(policy.predict_action_chunk(pre(dict(observation)))).cpu().numpy()[0]
        policy.reset()
        torch.manual_seed(713)
        with torch.inference_mode():
            selected = post(policy.select_action(pre(dict(observation)))).cpu().numpy().reshape(-1)
        error = np.abs(chunk[0] - sample['action'].numpy())
        errors.append(error)
        rows.append({'index':int(index),'episode':int(sample['episode_index']),
                     'instruction':sample['task'], 'reference_action':sample['action'].tolist(),
                     'predicted_chunk':chunk.tolist(),
                     'native_select_equivalence_max_error':float(np.max(np.abs(chunk[0]-selected)))})
    (output/'action-audit.json').write_text(json.dumps({'scope':'48 in-sample demonstration observations; not task evaluation',
        'action_names':contract['action_names'], 'mean_absolute_error':np.mean(errors,axis=0).tolist(),
        'mean_error_fraction_of_engineering_range':np.mean(np.asarray(errors)/scale,axis=0).tolist(),
        'native_select_equivalence_max_error':max(r['native_select_equivalence_max_error'] for r in rows),
        'samples':rows},indent=2))
    if max(r['native_select_equivalence_max_error'] for r in rows) > 1e-6:
        raise RuntimeError('Public chunk path differs from native single-action selection')


def evaluate(inputs, output, checkpoint=None, action_steps=1):
    import numpy as np
    import torch
    import imageio.v2 as iio
    from PIL import Image, ImageDraw
    prepare_plant(inputs)
    from lab.vla_control import VisualRobot, evaluate as evaluate_outcome
    policy, pre, post = load_checkpoint(inputs, output, checkpoint)
    if action_steps not in (1,5):
        raise ValueError('Bounded action execution horizon must be 1 or 5')
    robot = robot_from_inputs(inputs)
    cases = json.loads((inputs/'evaluation-cases.json').read_text())
    results = {}
    try:
        for case in cases:
            dest = output/case['name']
            dest.mkdir()
            robot.reset(case['x'],case['y'])
            policy.reset()
            torch.manual_seed(713)
            rows = []
            latencies = []
            prediction_latencies = []
            with (dest/'decisions.jsonl').open('w') as decisions, (dest/'chunks.jsonl').open('w') as chunks, iio.get_writer(dest/'replay.mp4', fps=10, macro_block_size=1) as movie:
                for step in range(400 if case['task']=='cleanup' else 80):
                    state, rgb = robot.state(), robot.image()
                    observation = {'observation.state':torch.from_numpy(state),
                                   'observation.images.head':torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,
                                   'task':case['instruction']}
                    start = time.monotonic()
                    chunk_index = step % action_steps
                    if chunk_index == 0:
                        with torch.inference_mode():
                            chunk = post(policy.predict_action_chunk(pre(observation))).cpu().numpy()[0]
                        chunks.write(json.dumps({'acquisition_s':float(robot.motor.data.time),
                            'execution_steps':action_steps,'predicted_actions':chunk.tolist()})+'\n')
                    action = chunk[chunk_index]
                    torch.cuda.synchronize()
                    latency = time.monotonic()-start
                    latencies.append(latency)
                    if chunk_index == 0:
                        prediction_latencies.append(latency)
                    acquired = float(robot.motor.data.time)
                    new_rows, accepted = robot.step(action)
                    rows.extend(new_rows)
                    decisions.write(json.dumps({'time_s':acquired,'state':state.tolist(),'instruction':case['instruction'],
                        'predicted_action':action.tolist(),'accepted_action':accepted.tolist(),'inference_s':latency,
                        'chunk_action_index':chunk_index,'policy_observation_age_s':chunk_index/10})+'\n')
                    # Inspection camera is never a model input; only head RGB above is.
                    canvas = Image.new('RGB',(960,440),'#102027')
                    canvas.paste(Image.fromarray(robot.motor.render()),(0,0))
                    canvas.paste(Image.fromarray(rgb),(640,0))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((650,195),f'SmolVLA / LoRA / {case["task"]}',fill='white')
                    draw.text((650,220),f'Policy/queue {latency:.3f}s (sim paused)',fill='white')
                    draw.text((650,245),f'Execute {action_steps} / 50 predicted steps',fill='white')
                    draw.text((15,395),('Learned VLA + PPO motor tracker; defined servo/electrical proxy.'
                        if robot.learned_motor is not None else
                        'Learned visual actions; defined balance + electrical servo. No task phase controller.'),fill='white')
                    movie.append_data(np.asarray(canvas))
                    if abs(rows[-1]['pitch_rad'])>.28:
                        break
            result = evaluate_outcome(rows,initial=(case['x'],case['y'],.015),task=case['task'])
            result.update(duration_s=float(robot.motor.data.time),mean_inference_s=float(np.mean(prediction_latencies)),
                          mean_policy_or_queue_s=float(np.mean(latencies)),prediction_calls=len(prediction_latencies),
                          max_inference_s=float(max(prediction_latencies)),instruction=case['instruction'])
            results[case['name']] = result
            (dest/'report.json').write_text(json.dumps(result,indent=2))
            (dest/'trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            print(case['name'],result,flush=True)
    finally:
        robot.close()
        (output/'evaluation.json').write_text(json.dumps({'cases':results,'checkpoint_reloaded':True,
            'teacher_in_evaluation':False,'whole_body_control':robot.contract()['low_level'],
            'motor_checkpoint_sha256':hashlib.sha256((inputs/'motor/policy.zip').read_bytes()).hexdigest() if (inputs/'motor/policy.zip').exists() else None,
            'action_execution_steps':action_steps,
            'task_success':bool(len(results)==len(cases) and all(r['success'] for r in results.values())),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['download','preflight','train','evaluate','audit','seal'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--action-steps',type=int,choices=[1,5],default=1)
    args=parser.parse_args()
    # No custom optimizer, trainer or dataset format.
    kwargs = {'checkpoint':args.checkpoint,'action_steps':args.action_steps} if args.stage in {'evaluate','audit'} else {}
    globals()[args.stage](args.inputs,args.output,**kwargs)
