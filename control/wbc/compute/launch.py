"""WBC GPU pilots: bundle, one submit, sparse status, verified restore.

Run with .venv-runpod from the repository root. There is no monitor: status
is one explicit request at least five minutes after the previous one, and a
submitted job is never resubmitted. The endpoint is the existing
Terraform-owned scale-to-zero trial endpoint (see docs/operations/runpod-infrastructure.md).
--pilot selects the plan: the native teacher pilot (T-e7vm69, 2026-11-01),
the GEAR-SONIC warm-start pilots, the native teacher's continuation
(2026-11-02) or its continuation with R3 reward widths (2026-11-03).

    .venv-runpod/bin/python control/wbc/compute/launch.py --pilot P bundle --work DIR
    .venv-runpod/bin/python .../launch.py --pilot P submit --work DIR
    .venv-runpod/bin/python .../launch.py --pilot P status
    infra/lab storage run cove50-wbc-P -- .venv-runpod/bin/python .../launch.py --pilot P restore --evidence '{evidence}' --work '{work}'
"""
import argparse
import datetime
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import uuid

HERE = Path(__file__).resolve().parent
WBC = HERE.parent
ROOT = WBC.parents[1]
MODEL = ROOT / 'artifacts/robot/model/timestep'
PROVIDER = 'control/common/compute/launch.py'  # shared s3()/call() helpers
MOTION = 'forward-walk'
SONIC_SOURCE = ROOT / '.cache/sonic-base'  # sonic.py fetch --dir (pinned, hash-verified)
_COMMON = dict(motion=f'control/wbc/motions/{MOTION}.npz',
    stage_script='control/wbc/compute/stage.py',
    bootstrap='control/wbc/compute/bootstrap.sh',
    benchmark_steps=10, eval_worlds=16, seed=1, ttl_safety_s=120, reserve_s=300, train_join_s=600,
    minimum_train_s=900)
PILOTS = {
    'teacher-pilot-20261101': dict(_COMMON, schema='cove50.wbc.teacher-pilot-plan.v1', idle_task='T-e7vm69',
        benchmark_worlds=[1024, 2048, 4096], push_step=100, arms=['native'], zero_shot=False,
        hard_deadline_s=5400, timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=900),
        provider_policy=dict(executionTimeout=6_000_000, ttl=14_400_000)),
    # GEAR-SONIC warm start vs the same network and recipe from random weights,
    # matched in PPO iterations; 80 GB GPU (4096 R3 worlds ran out of 24 GB).
    'sonic-pilot-20261102': dict(_COMMON, schema='cove50.wbc.sonic-pilot-plan.v1', idle_task='T-e7vm69',
        sonic='control/wbc/sonic-base',
        benchmark_worlds=[4096, 8192], push_step=50, arms=['sonic-warm', 'sonic-random'], zero_shot=True,
        hard_deadline_s=8000, timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=900),
        provider_policy=dict(executionTimeout=8_300_000, ttl=14_400_000)),
    # Cancelled after the benchmark above: the H100 gave 3.3k transitions/s at
    # 4,096-8,192 worlds (no faster than the 4090 at 2,048), leaving ~27 PPO
    # iterations per arm. One warm-start arm on the 4090 pool instead.
    'sonic-finetune-20261102': dict(_COMMON, schema='cove50.wbc.sonic-pilot-plan.v1', idle_task='T-dfmjb9',
        sonic='control/wbc/sonic-base',
        benchmark_worlds=[2048], push_step=50, arms=['sonic-warm'], zero_shot=True,
        hard_deadline_s=8000, timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=900),
        provider_policy=dict(executionTimeout=8_300_000, ttl=14_400_000)),
    # The run above went silent 0-100 min into training (progress unchanged,
    # stage timeout passed; cancelled at 10:47 UTC). Same fine-tune with
    # heartbeats, a stall stop and mid-run checkpoint uploads; training wall
    # matched to the native scratch teacher r1 (2,905 s, same GPU and worlds).
    'sonic-finetune-r2-20261102': dict(_COMMON, schema='cove50.wbc.sonic-pilot-plan.v2', idle_task='T-dfmjb9',
        sonic='control/wbc/sonic-base',
        benchmark_worlds=[2048], push_step=50, arms=['sonic-warm'], zero_shot=True, train_s=2905,
        heartbeat_s=300, stall_s=900, hard_deadline_s=6000, timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=900),
        provider_policy=dict(executionTimeout=6_300_000, ttl=14_400_000)),
    # r2 never started (empty account balance; cancelled while queued). On the
    # 48/80 GB pools R3 physics runs no faster than on the 4090, so the extra
    # card goes into worlds per iteration (most within 5% of the best
    # throughput), mini-batches scaled to the 2,048-world sample count per
    # gradient step, and 64 evaluation worlds (4 per push direction and speed).
    'sonic-finetune-r3-20261102': dict(_COMMON, schema='cove50.wbc.sonic-pilot-plan.v2', idle_task='T-dfmjb9',
        sonic='control/wbc/sonic-base',
        benchmark_worlds=[4096, 6144, 8192], push_step=50, arms=['sonic-warm'], zero_shot=True, eval_worlds=64,
        minibatch_samples=12288, train_s=4500, heartbeat_s=300, stall_s=900, hard_deadline_s=7800,
        timeouts_s=dict(bootstrap=1200, benchmark=1200, evaluate=1200),
        provider_policy=dict(executionTimeout=8_100_000, ttl=14_400_000)),
    # r3 pulled SONIC away from its start under the release recipe. r4 runs
    # the trust-region recipe (fixed noise, no entropy bonus, frozen reference
    # encoder, KL-adapted decoder rate) on the 4090 at 2,048 worlds like the
    # native teacher r1, then the same network from random layers (release FSQ
    # levels) for the same PPO rounds, inside the endpoint's 140-minute
    # deadline. r3 holds the zero-shot evaluation on the same 64 world draws.
    'sonic-trust-region-r4-20261102': dict(_COMMON, schema='cove50.wbc.sonic-pilot-plan.v2', idle_task='T-dfmjb9',
        sonic='control/wbc/sonic-base', recipe='trust-region',
        benchmark_worlds=[2048], push_step=50, arms=['sonic-warm', 'sonic-random'], zero_shot=False, eval_worlds=64,
        train_join_s=300, heartbeat_s=300, stall_s=900, hard_deadline_s=8000,
        timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=600),
        provider_policy=dict(executionTimeout=8_300_000, ttl=14_400_000)),
    # r4 found the G1 weights a worse start for R3 than random layers. Continue
    # the native scratch teacher r1 (still improving at its 45-minute budget)
    # with its own actor, critic, noise and optimizer for the rest of the
    # endpoint's 140-minute deadline; r1 and the continued teacher are both
    # evaluated on r4's 64 worlds with pushes at 1.0 s.
    'teacher-continue-20261102': dict(_COMMON, schema='cove50.wbc.teacher-pilot-plan.v2', idle_task='T-ykjzwy',
        init_from='artifacts/runs/cove50-wbc-teacher-pilot-5373d7868388/evidence/train/teacher.pt',
        benchmark_worlds=[2048], push_step=50, arms=['native'], zero_shot=True, eval_worlds=64,
        train_join_s=300, heartbeat_s=300, stall_s=900, hard_deadline_s=8000,
        timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=600),
        provider_policy=dict(executionTimeout=8_300_000, ttl=14_400_000)),
    # The 4090 pool had no stock; the run above was cancelled while queued.
    # Same continuation on the 48/80 GB pools, spending the card on worlds as
    # in sonic-finetune-r3: the most worlds within 5% of the best throughput
    # and 12,288 samples per gradient step (r1's mini-batch size).
    'teacher-continue-r2-20261102': dict(_COMMON, schema='cove50.wbc.teacher-pilot-plan.v2', idle_task='T-ykjzwy',
        init_from='artifacts/runs/cove50-wbc-teacher-pilot-5373d7868388/evidence/train/teacher.pt',
        benchmark_worlds=[4096, 6144], push_step=50, arms=['native'], zero_shot=True, eval_worlds=64,
        minibatch_samples=12288, train_join_s=300, heartbeat_s=300, stall_s=900, hard_deadline_s=8000,
        timeouts_s=dict(bootstrap=1200, benchmark=1200, evaluate=600),
        provider_policy=dict(executionTimeout=8_300_000, ttl=14_400_000)),
    # The continued teacher stands through the clip and pushes but turns in
    # place instead of walking. Continue it with the tracking reward widths
    # scaled to R3 (tracking.REWARD_STD['r3']) on the 4090 pool at 2,048
    # worlds, keeping r1's 12,288 samples per gradient step.
    'teacher-widths-r3-20261103': dict(_COMMON, schema='cove50.wbc.teacher-pilot-plan.v2', idle_task='T-3bnsnk',
        init_from='artifacts/runs/cove50-wbc-teacher-continue-r2-20261102-639a28b2684b/evidence/train-native/teacher.pt',
        reward_widths='r3', benchmark_worlds=[2048], push_step=50, arms=['native'], zero_shot=True, eval_worlds=64,
        minibatch_samples=12288, train_join_s=300, heartbeat_s=300, stall_s=900, hard_deadline_s=8000,
        timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=600),
        provider_policy=dict(executionTimeout=8_300_000, ttl=14_400_000)),
}
# The run above stopped after bootstrap on a single storage read timeout; the
# worker now retries storage writes after its claim. Same plan.
PILOTS['teacher-widths-r3-retry-20261103'] = PILOTS['teacher-widths-r3-20261103']
# The retry steps in the clip's rhythm with short strides and was still
# improving at its deadline. Continue it unchanged for another deadline.
PILOTS['teacher-widths-r3-continue-20261103'] = dict(PILOTS['teacher-widths-r3-20261103'], idle_task='T-pba4zk',
    init_from='artifacts/runs/cove50-wbc-teacher-widths-r3-retry-20261103-a5fb7c6f2fea/evidence/train-native/teacher.pt')
# T-gtxyrn direction checks: the continued teacher, the same 4,500 s of
# training each, side by side on a temporary 4-worker endpoint. The control
# changes nothing; the others change one thing each.
_CHECK = dict(PILOTS['teacher-widths-r3-20261103'], idle_task='T-gtxyrn', train_s=4500,
    init_from='artifacts/runs/cove50-wbc-teacher-widths-r3-continue-20261103-24d03dbe7e6f/evidence/train-native/teacher.pt')
PILOTS.update({
    'check-control-20261103': _CHECK,
    # The walk replayed at 0.6x, the speed the continued teacher settled at;
    # its 13.3 s episodes need a longer evaluation stage.
    'check-slow-clip-20261103': dict(_CHECK, motion=f'control/wbc/motions/{MOTION}-speed60.npz',
        timeouts_s=dict(_CHECK['timeouts_s'], evaluate=1200)),
    'check-peak-torque-20261103': dict(_CHECK, peak_torque=True),  # C047 stall curve instead of the rated cap
    'check-sts3250-20261103': dict(_CHECK, actuator='sts3250'),  # tracking.ACTUATORS
})
# Its worker downloaded the CUDA wheels at 2.5 MB/s and hit the 1,200 s
# bootstrap timeout before training; the other three bootstrapped in ~3 min.
PILOTS['check-sts3250-retry-20261103'] = PILOTS['check-sts3250-20261103']
# T-wy4arf: both stronger-torque checks were still rising at 4,500 s.
# Continue each from its own teacher with its own plant for the endpoint's
# deadline (no fixed train_s), stopping early once the last four full chunks
# (~20 min) average less than 1% above the four before (stage.py
# --plateau-window). evaluate-start re-evaluates the check's trained teacher.
for _name, _run in (('peak-torque', 'cove50-wbc-check-peak-torque-20261103-1d9942fb9a5b'),
                    ('sts3250', 'cove50-wbc-check-sts3250-retry-20261103-bd48817511e9')):
    PILOTS[f'check-{_name}-continue-20261104'] = dict(
        {k: v for k, v in PILOTS[f'check-{_name}-20261103'].items() if k != 'train_s'}, idle_task='T-wy4arf',
        init_from=f'artifacts/runs/{_run}/evidence/train-native/teacher.pt', plateau=dict(window=4, gain=0.01))
# Both ran to the deadline with the last window still 2.5% (peak) and 3.7%
# (STS3250) above the one before; resume each from its continued teacher.
for _name, _run in (('peak-torque', 'cove50-wbc-check-peak-torque-continue-20261104-e75d26f62572'),
                    ('sts3250', 'cove50-wbc-check-sts3250-continue-20261104-f1780a5373a8')):
    PILOTS[f'check-{_name}-continue-r2-20261104'] = dict(PILOTS[f'check-{_name}-continue-20261104'],
        init_from=f'artifacts/runs/{_run}/evidence/train-native/teacher.pt')
# Session 2: STS3250 hit the plateau rule at chunk 15 (0.96%); peak torque ran
# to the deadline with its last window still 1.16% up, so only it continues.
PILOTS['check-peak-torque-continue-r3-20261104'] = dict(PILOTS['check-peak-torque-continue-20261104'],
    init_from='artifacts/runs/cove50-wbc-check-peak-torque-continue-r2-20261104-212391d7eac1/evidence/train-native/teacher.pt')
# T-t68e54: the r4 best-estimate actuator dynamics (robot/model/dynamics:
# Feetech firmware duty loop, rotor inertia, Coulomb friction, backlash) with each servo's full
# torque-speed curve. evaluate-start shows how its starting teacher copes with the realistic plant;
# training then continues to the plateau rule or the deadline. The ST3025 starts from the
# STS3250 teacher, the nearest plant of the earlier checks.
_R4 = dict({k: v for k, v in PILOTS['check-peak-torque-continue-20261104'].items() if k not in ('init_from', 'actuator')},
    idle_task='T-t68e54', dynamics='r4', peak_torque=True, plateau=dict(window=4, gain=0.01))
_PEAK_FINAL = 'artifacts/runs/cove50-wbc-check-peak-torque-continue-r3-20261104-8063a49316f1/evidence/train-native/teacher.pt'
_STS_FINAL = 'artifacts/runs/cove50-wbc-check-sts3250-continue-r2-20261104-518ea6de8cb0/evidence/train-native/teacher.pt'
PILOTS.update({
    'r4-c047-20261105': dict(_R4, init_from=_PEAK_FINAL),
    'r4-sts3250-20261105': dict(_R4, actuator='sts3250', init_from=_STS_FINAL),
    'r4-st3025-20261105': dict(_R4, actuator='st3025', init_from=_STS_FINAL),
})
# T-t68e54 iteration 2: the agility tape (agility/tape.py: 41 motions chained in 254 s with 1 s blends)
# on the r4 plant, from each actuator's r4 walk teacher. evaluate-start shows how a single-walk teacher
# copes with the other motions; both evaluations cover one segment per family from its first frame
# (32 worlds, nominal and pushed, teacher only, so training keeps most of the deadline).
TAPE_SEGMENTS = ['forward-walk', 'walk-stop', 'deep-squat-x2', 'alternating-arms-x2', 'chest-beat-2hz', 'overhead-wave-2hz',
    'arm-display-1hz', 'knee-bounce-1.5hz', 'side-sway-1.5hz', 'quick-crouch-0.35s', 'hop-1cm', 'turn-30dps', 'march-0.5s']
_TAPE = dict(_R4, motion='control/wbc/motions/agility-tape.npz', eval_worlds=32,
    eval_modes=['teacher'], eval_segments=TAPE_SEGMENTS, timeouts_s=dict(_R4['timeouts_s'], evaluate=1500))
TAPE_FROM = {a: f'artifacts/runs/{r}/evidence/train-native/teacher.pt' for a, r in (  # the r4 walk teachers
    ('c047', 'cove50-wbc-r4-c047-20261105-281cff696f00'), ('sts3250', 'cove50-wbc-r4-sts3250-20261105-75fa0c868d2b'),
    ('st3025', 'cove50-wbc-r4-st3025-20261105-c7effd680db1'))}
for _a, _t in TAPE_FROM.items():
    PILOTS[f'tape-{_a}-20261105'] = dict(_TAPE, init_from=_t, **({} if _a == 'c047' else dict(actuator=_a)))
# The ST3025 tape job's evaluate-start needed 1,502 s against its 1,500 s stage limit (the evaluation is
# physics-bound: about 9 s of wall per simulated second at 0.25 ms) and stopped before training. Retry without
# evaluate-start (the C047 and STS3250 jobs give the walk-teacher baseline) and with room for the evaluation.
PILOTS['tape-st3025-retry-20261105'] = dict(PILOTS['tape-st3025-20261105'], zero_shot=False,
    timeouts_s=dict(PILOTS['tape-st3025-20261105']['timeouts_s'], evaluate=2400))
# T-t68e54 iteration 3. The tape teachers survived most segments by doing less: the R3 boundary
# clamps commands to the joint stops, and the six hip axes (unspecified in the selected CAD) to a
# +-0.35 rad diagnostic envelope, so deep squats, crouches, hops and bounces (hip pitch to -82 deg)
# were unreachable and both teachers pushed hip, elbow and neck commands far past the stops where
# every value acts the same. r4 revision 2 gives best-estimate hip ranges; each job keeps its tape
# teacher's features, critic and normalizers, zeroes the action head (every joint starts at its
# default target) and adds a small penalty per rad commanded past a stop so commands stay live.
TAPE2_FROM = {a: f'artifacts/runs/{r}/evidence/train-native/teacher.pt' for a, r in (
    ('c047', 'cove50-wbc-tape-c047-20261105-0bbe68125297'), ('sts3250', 'cove50-wbc-tape-sts3250-20261105-dca2cd082ee8'),
    ('st3025', 'cove50-wbc-tape-st3025-retry-20261105-836da92ea6f0'))}
_TAPE2 = dict(_TAPE, zero_shot=False, stop_weight=0.5, reset_actor_output=True,
    timeouts_s=dict(_TAPE['timeouts_s'], evaluate=1800))
for _a, _t in TAPE2_FROM.items():
    PILOTS[f'tape2-{_a}-20261105'] = dict(_TAPE2, init_from=_t, **({} if _a == 'c047' else dict(actuator=_a)))
# T-t68e54 iteration 4. Tape2 freed the arms and neck (no parked commands; elbows follow) but the legs stay
# straight through squats and crouches. (a) The mixed servo set from agility/allocation.py (ST3025 on hip
# pitch, hip roll, knee, ankle pitch; STS3250 elsewhere) on the tape2 recipe: an A/B against tape2-st3025
# with only the plant changed. (b) The same plant from the tape2 ST3025 teacher with a joint-angle tracking
# reward (tracking.joint_position_tracking), probing whether the legs then follow the squats. The ST3025
# evaluation needed more than 1,800 s, so both get 2,700 s.
_MIXED = 'st3025-pitch-roll-sts3250'
PILOTS['tape2-mixed-20261105'] = dict(_TAPE2, init_from=TAPE2_FROM['st3025'], actuator=_MIXED,
    timeouts_s=dict(_TAPE2['timeouts_s'], evaluate=2700))
PILOTS['tape3-mixed-joint-20261105'] = dict(_TAPE2, actuator=_MIXED, reset_actor_output=False, joint_weight=1.0,
    init_from='artifacts/runs/cove50-wbc-tape2-st3025-20261105-9e9504ee174e/evidence/train-native/teacher.pt',
    timeouts_s=dict(_TAPE2['timeouts_s'], evaluate=2700))
# Agility step 1: a training signal that pays for performing the legs. The feasible tape
# (agility/feasible-selection.json: 26 of 41 motions, 179 s; the 13 evaluation segments are all kept), actions
# as offsets on the reference joint angles (a zero action performs the motion), the joint-angle reward, a 5 cm
# torso-height termination (below the 6.8 cm deep squat; end effectors keep 10 cm), and half the adaptive starts
# drawn by per-bin joint-angle error (tracking.ErrorAwareMotionCommand). From the tape2 ST3025 teacher's features
# with the action head reset, on the mixed servo set. No plateau stop: shifting adaptive sampling moves the return.
PILOTS['signal-mixed-20261106'] = dict(_TAPE2, actuator=_MIXED, reset_actor_output=True, residual_action=True,
    joint_weight=1.0, anchor_tolerance=0.05, error_sampling=0.5, plateau=None, idle_task='T-kemqtv',
    init_from='artifacts/runs/cove50-wbc-tape2-st3025-20261105-9e9504ee174e/evidence/train-native/teacher.pt',
    timeouts_s=dict(_TAPE2['timeouts_s'], evaluate=2700))


def _record_dir(pilot):
    """Process records of a pilot (bundle, submission, restore receipts): .bookkeeping/ (docs/operations/
    git-and-bookkeeping.md); pilots before 2026-11-12 keep theirs in data/."""
    old = ROOT / f'data/cove50-wbc-{pilot}'
    return old if old.exists() else ROOT / f'.bookkeeping/cove50-wbc-{pilot}'


def _restored_teacher(pilot):
    """The trained teacher of a restored pilot (data/cove50-wbc-PILOT/restored.json), or None before its restore."""
    receipt = _record_dir(pilot) / 'restored.json'
    return f"{json.loads(receipt.read_text())['evidence']}/evidence/train-native/teacher.pt" if receipt.exists() else None


# Agility step 3: the step-1 signal trained an order of magnitude longer at the step-2 physics step
# (2 ms: servo step responses within 0.03 rad and squat torso height within 2 mm of 0.25 ms, about 7x faster on
# CPU). long1 first re-evaluates the step-1 teacher at 2 ms (evaluate-start: the closed-loop check of the step
# change against its own 0.25 ms evaluation), benchmarks 2,048-4,096 worlds, then trains to the deadline;
# long2 continues from long1's teacher.
_LONG = dict(PILOTS['signal-mixed-20261106'], reset_actor_output=False, physics_dt=0.002, benchmark_worlds=[2048, 3072, 4096],
    idle_task='T-87gbsn', timeouts_s=dict(PILOTS['signal-mixed-20261106']['timeouts_s'], evaluate=900))
if _restored_teacher('signal-mixed-20261106'):
    PILOTS['long1-mixed-20261106'] = dict(_LONG, zero_shot=True, init_from=_restored_teacher('signal-mixed-20261106'))
if _restored_teacher('long1-mixed-20261106'):
    PILOTS['long2-mixed-20261106'] = dict(_LONG, zero_shot=False, init_from=_restored_teacher('long1-mixed-20261106'))
# Agility step 4: characterful references. The tape adds the 15 feasible moves of the ape
# catalogue (agility/library.py --catalogue ape; agility/feasible-selection-ape.json) to the 26 feasible
# agility moves: 41 motions, 270 s, built with tape.py --motions LIBRARY APE_LIBRARY and motions.py --tape in
# .cache/work/wbc-ape. evaluate-start runs the step-3 teacher on the new moves before it has seen them (held
# out), then it trains on the combined tape and is evaluated again on the same segments.
APE_SEGMENTS = ['ape-crouch-bounce-2hz', 'excited-beat-bounce-2hz', 'side-rock-reach-0.5hz', 'crouch-scan-0.5hz',
    'knee-bounce-2hz', 'quick-crouch-0.27s', 'deep-squat-x3', 'chest-beat-2.5hz', 'overhead-wave-2.5hz',
    'bilateral-arms-x3', 'forward-walk-x1.1', 'forward-walk', 'walk-stop', 'turn-30dps', 'deep-squat-x2', 'hop-1cm']
_APE = dict(_LONG, zero_shot=True, eval_segments=APE_SEGMENTS, idle_task='T-2smahq', benchmark_worlds=[2048])  # 3,072 and 4,096 worlds failed GPU allocation in the step-2 benchmark (possibly memory the benchmark kept)
if _restored_teacher('long2-mixed-20261106'):
    PILOTS['ape-mixed-20261106'] = dict(_APE, init_from=_restored_teacher('long2-mixed-20261106'))
# Agility step 5: the servo comparison on the resulting controller. Each servo set continues the
# step-4 teacher on the same tape with the same fixed training wall (train_s), so the arms differ only in the
# plant. evaluate-start shows how the mixed-trained teacher does on each plant before adapting.
if _restored_teacher('ape-mixed-20261106'):
    for _a in ('c047', 'sts3250', 'st3025', _MIXED):
        PILOTS[f'servo-{_a}-20261106'] = dict(_APE, actuator=_a, train_s=3600, idle_task='T-jhfh6e',
            init_from=_restored_teacher('ape-mixed-20261106'))
# Agility step 4, side shuffle: five side-shuffle tempos (0.8-0.33 s per step, 30 mm steps; all kept by the
# step-14 selection rule, cove50-agility-allocation-shuffle-r4-20261106) join the tape: 46 motions, 319 s, built in
# .cache/work/wbc-shuffle. evaluate-start runs the step-4 teacher on them before it has seen them; the other
# segments check what the added motions cost elsewhere.
SHUFFLE_SEGMENTS = ['side-shuffle-0.8s', 'side-shuffle-0.6s', 'side-shuffle-0.5s', 'side-shuffle-0.4s',
    'side-shuffle-0.33s', 'forward-walk', 'turn-30dps', 'side-rock-reach-0.5hz', 'ape-crouch-bounce-2hz',
    'chest-beat-2.5hz', 'hop-1cm']
if _restored_teacher('ape-mixed-20261106'):
    PILOTS['shuffle-mixed-20261106'] = dict(_APE, eval_segments=SHUFFLE_SEGMENTS, train_s=3600,
        init_from=_restored_teacher('ape-mixed-20261106'))
# Agility step 6: a 24 V brushless knee (CubeMars AK45-10 V3) on the mixed set. With it the demand check
# (cove50-agility-allocation-knee-20261106) admits the 0.4 and 0.33 s marches, the 0.2 s crouch and the x4 deep
# squat; the tape is the 46-motion shuffle tape plus those four (50 motions, 339 s, .cache/work/wbc-knee).
# Both runs continue the step-4 teacher for the same training wall; the second adds a narrow knee stance term.
KNEE_SEGMENTS = ['hop-1cm', 'march-0.5s', 'march-0.4s', 'march-0.33s', 'quick-crouch-0.2s', 'deep-squat-x4',
    'side-shuffle-0.33s', 'side-shuffle-0.5s', 'forward-walk', 'turn-30dps', 'ape-crouch-bounce-2hz',
    'excited-beat-bounce-2hz', 'side-rock-reach-0.5hz', 'chest-beat-2.5hz', 'deep-squat-x2', 'knee-bounce-2hz']
_KNEE = dict(_APE, actuator='st3025-pitch-roll-sts3250-ak45knee', eval_segments=KNEE_SEGMENTS, train_s=3600,
    idle_task='T-p8cye8', init_from=_restored_teacher('ape-mixed-20261106'))
if _restored_teacher('ape-mixed-20261106'):
    PILOTS['knee-ak45-20261106'] = _KNEE
    PILOTS['knee-ak45-stance-20261106'] = dict(_KNEE, knee_weight=0.5)
    # One shared 12 V rail: the ST3025 knee through a 1:1.5 speed-up stage (cove50-agility-allocation-knee-geared-20261106),
    # with the stance term, against knee-ak45-stance-20261106 on the same tape and budget.
    PILOTS['knee-geared-stance-20261106'] = dict(_KNEE, actuator='st3025-pitch-roll-sts3250-geared-knee', knee_weight=0.5)
# Reward set v2 (tracking.REWARD_SET_V2): adaptive tracking widths plus joint-velocity and foot tracking. Each run
# continues its own plant's latest teacher for the same training wall, so evaluate-start is the v1-trained teacher
# on the same segments: the 12 V mixed set on the 46-motion shuffle tape (.cache/work/wbc-shuffle), and the 24 V
# AK45 knee on the 50-motion tape (.cache/work/wbc-knee), which tests whether that knee adds anything once the
# reward asks for speed. The first 13 segments are shared.
REWARD_SEGMENTS = ['forward-walk', 'turn-30dps', 'hop-1cm', 'march-0.5s', 'side-shuffle-0.5s', 'side-shuffle-0.33s',
    'ape-crouch-bounce-2hz', 'excited-beat-bounce-2hz', 'side-rock-reach-0.5hz', 'chest-beat-2.5hz', 'deep-squat-x2',
    'knee-bounce-2hz', 'quick-crouch-0.27s']
if _restored_teacher('shuffle-mixed-20261106'):
    PILOTS['reward-v2-mixed-20261106'] = dict(_APE, reward_set='v2', eval_segments=REWARD_SEGMENTS, train_s=3600,
        idle_task='T-b26dyp', init_from=_restored_teacher('shuffle-mixed-20261106'))
if _restored_teacher('knee-ak45-20261106'):
    PILOTS['reward-v2-ak45-20261106'] = dict(_KNEE, reward_set='v2', train_s=3600, idle_task='T-b26dyp',
        eval_segments=REWARD_SEGMENTS + ['march-0.4s', 'march-0.33s', 'quick-crouch-0.2s'],
        init_from=_restored_teacher('knee-ak45-20261106'))
# Agility loop, cycle 1 (step 9): reward set v3 (tracking.REWARD_SET_V3), each from the v2 mixed teacher for the
# same training wall. A: the 12 V mixed plant (reward effect); B: the same with half the command and feedback delay
# (latency effect); C: the 24 V brushless qdd-damiao-gim plant (5.5 kg; power effect) on its own 59-motion tape
# (agility/feasible-selection-qdd-damiao.json, .cache/work/wbc-qdd), evaluated also on the dynamic moves it carries.
LOOP_DYNAMIC = ['march-0.4s', 'march-0.33s', 'knee-bounce-3hz', 'quick-crouch-0.2s', 'side-rock-reach-1hz', 'arm-display-2hz']
if _restored_teacher('reward-v2-mixed-20261106'):
    _LOOP = dict(_APE, reward_set='v3', eval_segments=REWARD_SEGMENTS, train_s=3600, idle_task='T-27cyrn',
        init_from=_restored_teacher('reward-v2-mixed-20261106'))
    PILOTS['loop1-v3-mixed-20261106'] = _LOOP
    PILOTS['loop1-v3-mixed-fast-20261106'] = dict(_LOOP, command_delay=0.01, feedback_delay=0.005)
    PILOTS['loop1-v3-damiao-20261106'] = dict(_LOOP, actuator='qdd-damiao-gim', eval_segments=REWARD_SEGMENTS + LOOP_DYNAMIC)
# Agility loop, cycle 2 (step 10): feet 1.3x longer and wider (tracking.scale_feet; motions regenerated with
# COVE50_FEET_SCALE=1.3,1.3). The demand check (cove50-agility-allocation-feet-20261106) keeps 64 motions on the 24 V
# qdd-damiao-gim plant (agility/feasible-selection-qdd-damiao-feet.json, .cache/work/wbc-qdd-damiao-feet; the 1.5, 2
# and 4 cm hops and the 3.5 Hz bounce join) and 50 on the 12 V mixed set (feasible-selection-mixed-feet.json,
# .cache/work/wbc-mixed-feet). Each continues its plant's cycle-1 teacher; evaluate-start is that teacher on the
# longer feet before it adapts.
FEET_DYNAMIC = ['hop-1.5cm', 'hop-2cm', 'hop-4cm', 'knee-bounce-3.5hz']
if _restored_teacher('loop1-v3-damiao-20261106'):
    # 6,000 s: the cycle-1 Damiao teacher spent part of its hour recovering from the v2 teacher's oscillation on this
    # low-inertia plant (evaluate-start speed ratios x2-35), so its gait moves may not have converged.
    PILOTS['loop2-damiao-feet-20261106'] = dict(PILOTS['loop1-v3-damiao-20261106'], feet_scale=[1.3, 1.3], train_s=6000,
        eval_segments=REWARD_SEGMENTS + LOOP_DYNAMIC + FEET_DYNAMIC, init_from=_restored_teacher('loop1-v3-damiao-20261106'))
if _restored_teacher('loop1-v3-mixed-20261106'):
    PILOTS['loop2-mixed-feet-20261106'] = dict(_LOOP, feet_scale=[1.3, 1.3],
        eval_segments=REWARD_SEGMENTS + ['knee-bounce-3hz', 'quick-crouch-0.2s'], init_from=_restored_teacher('loop1-v3-mixed-20261106'))
    # Reward timing: mjlab's rewards compare the robot after the physics step with the reference before it advances,
    # paying for a one-frame (20 ms) lag; every trained policy settled there, also with half the plant delay
    # (loop1-v3-mixed-fast). reference_lead=1 aligns the command and every reward with the robot state's time. Same
    # plant, tape and budget as loop1-v3-mixed, from its teacher.
    PILOTS['loop2-mixed-lead-20261106'] = dict(_LOOP, reference_lead=1, init_from=_restored_teacher('loop1-v3-mixed-20261106'))
# Agility loop, cycle 3: with the lead the policy is on time (lag 0-6 ms), but the 12 V plant's peak knee speed is still
# 0.78-0.80 of the reference. A reference preview (tracking.reference_preview: angles 1, 2 and 4 frames ahead, in actor
# and critic) lets the policy command ahead of its servos. From the lead teacher, widened by zero input columns
# (stage.widened), same tape and budget.
if _restored_teacher('loop2-mixed-lead-20261106'):
    PILOTS['loop3-mixed-preview-20261106'] = dict(_LOOP, reference_lead=1, preview_steps=[1, 2, 4],
        init_from=_restored_teacher('loop2-mixed-lead-20261106'))
# Cycle 3 on the 24 V feet plant: after 5,439 s the feet run hops 16-18 mm on the 2-4 cm references (4-7 lift-offs)
# but less high than its zero-shot start, and its planted moves lift a foot 16-18 mm. Reward v4 pays for the torso's
# height and vertical speed (its contrast is mostly on the hop), with the now-default reference lead. 1 h from the
# feet teacher.
if _restored_teacher('loop2-damiao-feet-20261106'):
    PILOTS['loop3-damiao-feet-v4-20261106'] = dict(PILOTS['loop2-damiao-feet-20261106'], reward_set='v4', reference_lead=1,
        train_s=3600, init_from=_restored_teacher('loop2-damiao-feet-20261106'))
# The ape body. Cancelled after 42 min: loop2-ape-body-damiao-20261106, arms 1.7x (the user: too long; the ape should
# lean forward on all fours, and the arm inertia would need bigger arm actuators or make the arms slow). Replacement:
# reach as arms 1.4x, taken mostly in the upper arm (1.6x; forearm 1.27x), which costs the least shoulder and elbow
# inertia for that reach, plus a 30 mm, 10 g knuckle fist (tracking.add_fists); feet 1.3x; 24 V plant. Knuckle stance
# 68 deg lean, knuckles 30 mm ahead of the shoulders (library.KNUCKLE). The force check
# (cove50-agility-allocation-ape-body-r3-20261106) keeps 81 of 86 motions with the same arm motors, including every
# knuckle move and the fast arm swings the 1.7x arm lost. Tape .cache/work/wbc-ape-body-r3-damiao. Reward v3k adds
# knuckle contact. From the cycle-1 Damiao teacher, so evaluate-start is that teacher on the new body.
KNUCKLE_SEGMENTS = ['knuckle-drop-0.6s', 'knuckle-rock-1.5hz', 'knuckle-sway-1hz', 'knuckle-bounce-2hz', 'knuckle-walk-1s']
if _restored_teacher('loop1-v3-damiao-20261106'):
    PILOTS['loop2-ape-body-damiao-20261106'] = dict(PILOTS['loop1-v3-damiao-20261106'], feet_scale=[1.3, 1.3], arm_scale=1.7,
        reward_set='v3k', train_s=6000, eval_segments=REWARD_SEGMENTS + KNUCKLE_SEGMENTS[:4] + ['hop-2cm', 'march-0.4s'],
        init_from=_restored_teacher('loop1-v3-damiao-20261106'))  # cancelled; kept so its record resolves
    PILOTS['loop2-ape-body-r3-damiao-20261106'] = dict(PILOTS['loop1-v3-damiao-20261106'], feet_scale=[1.3, 1.3],
        arm_scale=[1.6, 1.27], fist=[0.03, 0.01], reward_set='v3k', train_s=6000,
        eval_segments=REWARD_SEGMENTS + KNUCKLE_SEGMENTS + ['hop-2cm', 'march-0.4s', 'overhead-wave-3hz', 'arm-display-2hz'],
        init_from=_restored_teacher('loop1-v3-damiao-20261106'))
# Cycle 3 on the ape body: the knuckle walk steps in place (1.6 of 7.5 cm) because only the fixed-width global root
# term pays for travel. Both runs continue the ape-body teacher for 1 h with the reference lead: A adds the adaptive
# global root term (v3rk), B also v4's torso terms (v5k). A against loop2-ape-body-r3 shows the root term with the lead;
# B against A the torso terms on this body.
if _restored_teacher('loop2-ape-body-r3-damiao-20261106'):
    _APE3 = dict(PILOTS['loop2-ape-body-r3-damiao-20261106'], reference_lead=1, train_s=3600,
        init_from=_restored_teacher('loop2-ape-body-r3-damiao-20261106'))
    PILOTS['loop3-ape-root-20261106'] = dict(_APE3, reward_set='v3rk')
    PILOTS['loop3-ape-v5-20261106'] = dict(_APE3, reward_set='v5k')
# Cycle 4, the long runs: the v5 ape teacher (full hop flight, closest knuckle moves) on the r4 tape (knuckle walk feet
# lift 12 mm and stride 30 mm; swing term on at 5 mm; .cache/work/wbc-ape-body-r4-damiao), reference lead, about
# 6,600 s. v5k as tested, and v6k with the global root weight doubled for the walk drift; they differ only there.
if _restored_teacher('loop3-ape-v5-20261106'):
    _APE4 = dict(PILOTS['loop2-ape-body-r3-damiao-20261106'], reference_lead=1, train_s=6600,
        init_from=_restored_teacher('loop3-ape-v5-20261106'))
    PILOTS['loop4-ape-v5-20261106'] = dict(_APE4, reward_set='v5k')
    PILOTS['loop4-ape-v6-20261106'] = dict(_APE4, reward_set='v6k')
# Target playful-ape-t1 (infra/config/targets.json): the 12 V mixed set gets the full evaluation on the long-arm body that the
# 24 V set already had (loop3-ape-root, same recipe: reward v3rk, reference lead). Tape: the 59 target motions only
# (.cache/work/wbc-t1-mixed). From the 12 V lead teacher. No zero-shot stage (it cannot change the servo decision).
# Training stops at 5,400 s or earlier once return flattens. Evaluation: the target segments shared with
# loop3-ape-root, plus the 0.4 s shuffle and the 0.8 s knuckle walk.
T1_SEGMENTS = [s for s in REWARD_SEGMENTS if s not in ('hop-1cm', 'side-shuffle-0.33s')] + ['side-shuffle-0.4s'] \
    + KNUCKLE_SEGMENTS + ['knuckle-walk-0.8s']
if _restored_teacher('loop2-mixed-lead-20261106'):
    PILOTS['t1-mixed-longarm-20261108'] = dict(PILOTS['loop2-mixed-lead-20261106'], feet_scale=[1.3, 1.3],
        arm_scale=[1.6, 1.27], fist=[0.03, 0.01], reward_set='v3rk', train_s=5400, plateau=dict(window=4, gain=0.01),
        zero_shot=False, eval_segments=T1_SEGMENTS, idle_task='T-gk76wr', target='playful-ape-t1',
        init_from=_restored_teacher('loop2-mixed-lead-20261106'),
        question='Does the 12 V mixed servo set perform the target on the long-arm body, and how hot do hip roll and '
            'hip pitch run under a trained controller?',
        decision='Which servo set to build: 12 V mixed (ST3025/STS3250) or 24 V qdd-damiao-gim.')
# The locked three-tier set (robot/model/motor-map/motor-map.json): ST3025 on hip pitch, knee and
# ankle pitch; STS3250 on hip roll and the arms; C047 on hip yaw and ankle roll. Against t1-mixed-longarm, hip roll
# moves from the ST3025 to the STS3250 (it ran 1.05-1.18x rated current there in the shuffles, march and turn) and
# hip yaw and ankle roll move to the C047. Continues that run's teacher on the same body, tape and recipe.
if _restored_teacher('t1-mixed-longarm-20261108'):
    PILOTS['t1-tier3-longarm-20261108'] = dict(PILOTS['t1-mixed-longarm-20261108'], actuator='tier3-st3025-sts3250-c047',
        init_from=_restored_teacher('t1-mixed-longarm-20261108'), idle_task='T-ddeqx2',
        question='How do tracking, contacts and per-joint current look on the target with the locked three-tier servo set?',
        decision='Confirms the per-joint motor map before bench parts are bought, or moves a joint to another tier.')
# Score of the three-tier teacher across all motions, no training (evaluate_only). Tape (.cache/work/wbc-validation,
# agility/tape.py --shuffle 7 --blend-s 0.5 --windows 4): the 59 target motions in an order and with blends it never
# trained on, 35 never-trained motions (library.py --catalogue validation: in-between tempos, mirror images, arm moves
# on leg moves) and 14 windows of four chained target motions. Nominal only: the 0.1-0.2 m/s pushes changed no measure
# in the last two runs. 16 worlds: without corruption or pushes the worlds differ only by their reset draws.
_VAL = ROOT / '.cache/work/wbc-validation/motions/motions.json'  # the held-out validation tape (library.py --catalogue validation)
if _restored_teacher('t1-tier3-longarm-20261108'):
    PILOTS['t1-tier3-validation-20261108'] = dict(PILOTS['t1-tier3-longarm-20261108'], evaluate_only=True, zero_shot=False,
        init_from=_restored_teacher('t1-tier3-longarm-20261108'), eval_worlds=16, eval_conditions=['nominal'],
        eval_segments=[s['name'] for s in json.loads(_VAL.read_text())['motions'][0]['segments']] if _VAL.exists() else [],
        timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=3600), hard_deadline_s=5400,
        provider_policy=dict(executionTimeout=5_700_000, ttl=14_400_000), idle_task='T-xjjwrs',
        question='How well does the three-tier WBC move across all 59 target motions, and how much worse on 35 motions and '
            '14 chained sequences it never trained on?',
        decision='Whether training needs more varied motions, orders and blends to be agile on motions it has not seen, '
            'before any further reward, servo or body work.')
# More realism and variety in training (owner, 2026-11-09: "add more realism and diversity to the training scenarios
# to make the controller more robust"). Evaluation sets, all from the held-out validation tape (.cache/work/wbc-validation):
# the 35 never-trained motions; on the plane also the 59 target motions and five chained windows.
_VAL_SEGS = [s['name'] for s in json.loads(_VAL.read_text())['motions'][0]['segments']] if _VAL.exists() else []
_NEVER = [n for n in _VAL_SEGS if not n.startswith('window-') and n not in json.loads((ROOT / 'infra/config/targets.json').read_text())['targets'][0]['motions']]
_FLAT_EVAL = [n for n in _VAL_SEGS if not n.startswith('window-')] + [f'window-{k}' for k in range(1, 6)]
# Baseline: the three-tier teacher on the obstacle world, never-trained motions only (no training).
if _restored_teacher('t1-tier3-longarm-20261108'):
    PILOTS['t1-tier3-world-eval-20261109'] = dict(PILOTS['t1-tier3-validation-20261108'], terrain='parcour', eval_segments=_NEVER,
        timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=2400), hard_deadline_s=3600,
        provider_policy=dict(executionTimeout=3_900_000, ttl=14_400_000), idle_task='T-9nn8az',
        question='How does the three-tier WBC, trained on a plane, move on the obstacle world (rough ground, slopes, steps, clutter, '
            'slippery to grippy floors, a ball, a door, a carried toy) on 35 motions it never trained on?',
        decision='The baseline the varied-training trials are compared against; whether the obstacle world is worth training on.',
        cheaper_check='CPU obstacle-world smoke: 15 of 16 short episodes upright, ground followed within 1-2 mm, ball, door and '
            'per-world floor grip in place. The plane validation (0.75 GPU-h) scored the same 35 motions on flat ground.',
        value='Shows how much the plane-trained WBC loses on home floors and with objects, over all 35 never-trained motions; '
            'both trials are judged against it.')
# Trials, each from the three-tier teacher with one change against it, 2,500 s of training at most (plateau stop):
# A: the varied tape (.cache/work/wbc-t2-diverse: target motions with references clipped 0.02 rad inside the joint stops,
#    55 mirror images and 10 arm-leg combinations not held out, shuffled, blends drawn from 0.5-1 s), on the plane.
# B: the same tape on the obstacle world. A against the baseline shows the data; B against A shows the world.
_T2 = dict(PILOTS.get('t1-tier3-longarm-20261108', {}), init_from=_restored_teacher('t1-tier3-longarm-20261108'), train_s=2500,
    eval_motion='control/wbc/motions/eval/agility-tape.npz',
    eval_source='.cache/work/wbc-validation/motions', eval_segments=_FLAT_EVAL, eval_terrain='parcour',
    eval_terrain_segments=_NEVER, eval_worlds=16, eval_conditions=['nominal'], eval_modes=['teacher'],
    timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=2400), hard_deadline_s=8600,
    provider_policy=dict(executionTimeout=8_900_000, ttl=14_400_000), idle_task='T-9nn8az')
if _restored_teacher('t1-tier3-longarm-20261108'):
    PILOTS['t2-varied-motions-20261109'] = dict(_T2,
        question='Does training on stop-clipped references plus mirror images, new arm-leg combinations, shuffled orders and '
            'varied blends make the WBC track never-trained motions, combinations and chained sequences better?',
        decision='Whether varied training data becomes the default tape before more reward, servo or body work.',
        cheaper_check='Plane validation (0.75 GPU-h): never-trained tempos track like trained ones, arm-leg combinations are '
            'worse (leg 0.031 rad), and 20 of 94 references cross joint stops. The tape was built and checked on CPU '
            '(124 motions, none held out).',
        value='One change, the training data, measured over all 94 motions and chained windows; it sets the default tape.')
# t2-varied-world-20261109 (trial B on the first obstacle world) was cancelled after 205 s on the worker: its heightfield
# tiles gave the mesh feet contacts with downward normals and one or two support points in mujoco_warp, so a 3 mm
# heightfield toppled the controller more often than 3 cm box steps. Its receipt is in data/cove50-wbc-t2-varied-world-20261109.
# The world is now built from boxes (parcour.py). t1-tier3-world2-eval-20261109 scored the baseline on it with every
# episode starting on the flat spawn square at the tile centre (data/cove50-wbc-t1-tier3-world2-eval-20261109; its first
# run, receipt *-unpaired-1, landed on random tiles). 31 of the 35 motions stay in place, so the robot barely met the
# obstacles, and t2-varied-world2-20261109 was cancelled after 63 s for that reason. Now each episode starts anywhere
# within 1 m of its tile centre, on the ground there (tracking.ErrorAwareMotionCommand._spawn), and each evaluation puts
# its 20 worlds on fixed tiles and fixed start points, two tiles per type (parcour.place_eval_worlds), so every run on
# it compares world by world: B, and the baseline and A scored again.
_A_TEACHER = 'artifacts/runs/cove50-wbc-t2-varied-motions-20261109-63932dd6f5ea/evidence/train-native/teacher.pt'
_WORLD3 = dict(terrain='parcour', eval_segments=_NEVER, eval_worlds=20, timeouts_s=dict(bootstrap=1200, benchmark=900, evaluate=2400),
    hard_deadline_s=3600, provider_policy=dict(executionTimeout=3_900_000, ttl=14_400_000), idle_task='T-9nn8az')
if _restored_teacher('t1-tier3-longarm-20261108'):
    PILOTS['t1-tier3-world3-eval-20261109'] = dict(PILOTS['t1-tier3-validation-20261108'], **_WORLD3,
        question='How does the plane-trained three-tier WBC move when each episode starts on rough ground, slopes, steps, mats, '
            'among furniture or clutter, on 35 motions it never trained on?',
        decision='The baseline trial B is compared against, world by world; whether the obstacle world plays as its sizes suggest.',
        cheaper_check='CPU: box-built tiles give upward contact normals; start points land on the planned tiles with the feet '
            '22-33 mm above the ground; the reference height holds through the ground sensor lag after a reset; one training '
            'iteration on the world runs with finite rewards. The same teacher on the centre spawn squares fell in 15 of 700.',
        value='Without it trial B has no baseline on this world; at 0.25 GPU-h it is also the cheapest check of the start points.')
    PILOTS['t2-varied-world3-20261109'] = dict(_T2, terrain='parcour', eval_worlds=20,
        question='Does training the varied tape on the obstacle world, starting anywhere on its tiles, keep plane tracking and '
            'make the WBC more robust on uneven ground, slopes, steps, clutter, slippery floors and with a ball and a door?',
        decision='Whether the obstacle world becomes the default training world.',
        cheaper_check='CPU checks of the world and its start points, a CPU training iteration on it, and the baseline scored on '
            'it first (t1-tier3-world3-eval-20261109).',
        value='One change against trial A, the world, over all 99 plane motions and the 35 never-trained motions on fixed tiles '
            'and start points; it sets the default training world.')
    # Trial B again, starting at the teacher's reward widths (init_widths). The first B made 154 PPO updates in its
    # 2,500 s against trial A's 760 (the world simulates 3,106 transitions/s against the plane's 11,043) and stopped with
    # its widths about twice the teacher's, so it tracked looser on every motion, plane included, and left its question
    # open. Same world, start points, tape and plane evaluation; up to 5,000 s of training with the plateau stop. The
    # obstacle-world evaluation runs as its own job afterwards (the same 20 fixed tiles), so one job stays within 2.5 GPU-h.
    PILOTS['t2-varied-world3-widths-20261109'] = dict(PILOTS['t2-varied-world3-20261109'], init_widths=True, train_s=5000, eval_terrain=None,
        question='Trained on the obstacle world as long as trial A on the plane, does the varied tape keep plane tracking '
            'across all motions and make the WBC more robust on the world?',
        decision='Whether the obstacle world becomes the default training world.',
        cheaper_check='Trial B (t2-varied-world3-20261109): 154 updates, widths twice the teacher\'s; the cause found from its '
            'train report and benchmark on CPU, and the width start checked on CPU.',
        value='Answers trial B\'s question with a comparable training length (about 320 updates at the teacher\'s widths); '
            'it sets the default training world.')
if _restored_teacher('t2-varied-world3-widths-20261109'):
    PILOTS['t2-varied-world3-widths-world3-eval-20261109'] = dict(PILOTS['t1-tier3-validation-20261108'], **_WORLD3,
        init_from=_restored_teacher('t2-varied-world3-widths-20261109'),
        question='How does the rerun of trial B (trained on the obstacle world) move on that world, from the same 20 fixed tiles and start points?',
        decision='Completes the rerun of trial B: whether the obstacle world becomes the default training world.',
        cheaper_check='The rerun\'s plane evaluation, run in its training job.',
        value='The world half of the paired comparison against today\'s controller and trial A, at about 0.25 GPU-h.')
if (ROOT / _A_TEACHER).exists():
    PILOTS['t2-varied-motions-world3-eval-20261109'] = dict(PILOTS['t1-tier3-validation-20261108'], **_WORLD3, init_from=_A_TEACHER,
        question='How does trial A (varied motions, trained on the plane) move on the obstacle world with spread start points?',
        decision='Separates what the varied motions and what the world did in trial B: B against A on the same start points.',
        cheaper_check='Trial A on the plane and the baseline on this world (t1-tier3-world3-eval-20261109).',
        value='Completes the paired comparison on the world at 0.25 GPU-h.')
# Slope-aware references (T-9nn8az, 2026-11-12): on inclined tiles each reference foot turns with the ground face under
# it and the ankle joints take that turn (tracking.ErrorAwareMotionCommand._slope_adjust). A short GPU smoke before any
# training on it (owner, 2026-11-12: no smoke tests on the laptop CPU): trial A's teacher on the same 20 fixed worlds and
# 35 never-trained motions, once with slope references and once on the flat tape, in one job, so the pair differs only
# in the references. It also exercises the new code on the worker before the matched training run (T-vj9x22).
if (ROOT / _A_TEACHER).exists():
    PILOTS['slope-refs-smoke-20261112'] = dict(PILOTS['t2-varied-motions-world3-eval-20261109'], eval_flat_references=True,
        idle_task='T-9nn8az',
        question='With slope-aware references, does trial A\'s teacher fall less and drift less on the hill and hollow tiles '
            'than with the flat tape, across the 35 never-trained motions, with the other tiles unchanged?',
        decision='Whether slope-aware references go into the matched training run on the obstacle world (T-vj9x22), or '
            'hills need a different fix (a terrain command mode) first.',
        cheaper_check='Reading the existing world evaluations: on the two hill worlds A fell in 39 of 70 episodes and drifted '
            '6-7 cm median against 1-2 cm elsewhere, with the leg-joint reward paying for level ankles; the slope map was '
            'checked from the terrain geometry alone (hills 4.4-8.6 deg, level tiles 0).',
        value='About 0.4 GPU-h; a clean A/B on all 35 never-trained motions and 20 fixed worlds, and a worker run of the '
            'new code before the 4 GPU-h training is asked for.')
# Matched run on the obstacle world (T-vj9x22, 2026-11-12). Trial A trained the varied tape on the plane for 760 PPO
# updates from the three-tier teacher. Here the same start, tape and recipe train on the obstacle world, slope-aware
# references on (tracking slope_references), for exactly 760 updates: matched on PPO updates, not wall time (M-qvrepk).
# No plateau stop (it counts time chunks). At the world's 16.3 s per update (rerun of B) that is about 3.4 GPU-h, over
# the 2.5 GPU-h per-job limit, so it runs as two training halves (train_only) and two evaluation jobs. The first half
# starts like A (widths from init_std, as A did); the second continues the first's teacher at its own last widths
# (init_widths), the remaining updates to 760. Evaluations match A's: 99 plane motions and windows on 16 worlds, and the
# 35 never-trained motions on the 20 fixed obstacle worlds.
MATCHED_UPDATES = 760  # trial A's PPO updates (t2-varied-motions-20261109, plateau stop)
if _restored_teacher('t1-tier3-longarm-20261108'):
    PILOTS['world-matched-1-20261112'] = dict(PILOTS['t2-varied-motions-20261109'], terrain='parcour', plateau=None,
        iterations=400, train_only=True, train_s=7400, eval_terrain=None, idle_task='T-vj9x22',
        question='Trained on the obstacle world (slope-aware references) for as many PPO updates as trial A on the plane, '
            'does the WBC track all motions as well as A and fall and drift less on the world, per tile type?',
        decision='Whether the obstacle world joins the default training worlds (M-9rahw2 keeps the plane today).',
        cheaper_check='Rerun of trial B (166 updates, flat references): legs as A, arms looser, hills still falling; the '
            'slope-reference smoke (slope-refs-smoke-20261112) on A\'s teacher; the 760-update match computed from A\'s report.',
        value='First half (400 of 760 updates, about 2.1 GPU-h) of the matched comparison the owner approved (about 4.9 GPU-h in all).')
if _restored_teacher('world-matched-1-20261112'):
    _done = json.loads((ROOT / Path(_restored_teacher('world-matched-1-20261112')).parent / 'report.json').read_text())['native']['ppo_rounds']
    PILOTS['world-matched-2-20261112'] = dict(PILOTS['world-matched-1-20261112'], iterations=MATCHED_UPDATES - _done,
        init_from=_restored_teacher('world-matched-1-20261112'), init_widths=True,
        value=f'Second half ({MATCHED_UPDATES - _done} updates to reach {MATCHED_UPDATES}) of the approved matched comparison.')
if _restored_teacher('world-matched-2-20261112'):
    _C = _restored_teacher('world-matched-2-20261112')
    PILOTS['world-matched-world-eval-20261112'] = dict(PILOTS['t1-tier3-validation-20261108'], **dict(_WORLD3, idle_task='T-vj9x22'),
        init_from=_C, question='How does the matched world-trained WBC move on the 20 fixed obstacle worlds, 35 never-trained motions?',
        decision='Whether the obstacle world joins the default training worlds.',
        cheaper_check='Its two training halves; A on the same worlds with slope references (slope-refs-smoke-20261112).',
        value='The world half of the matched comparison, about 0.25 GPU-h.')
    PILOTS['world-matched-plane-eval-20261112'] = dict(PILOTS['t1-tier3-validation-20261108'], init_from=_C, eval_segments=_FLAT_EVAL,
        eval_modes=['teacher'], idle_task='T-vj9x22',
        question='Does the matched world-trained WBC track the 99 plane motions and five chained windows as well as trial A?',
        decision='Whether the obstacle world joins the default training worlds.',
        cheaper_check='Its two training halves and its world evaluation.',
        value='The plane half of the matched comparison, about 0.5 GPU-h.')
# t1-tier3-armbin-20261108 (a reward change aimed at the fast arm beats) was cancelled after 984 s on the worker: the
# owner scores agility across all motions, and single motions are irrelevant (AGENTS.md). Its receipt is in
# data/cove50-wbc-t1-tier3-armbin-20261108; the plan and its code are removed.
PILOT = PLAN = RECORD = None  # set by main() from --pilot


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def provider():
    spec = importlib.util.spec_from_file_location('wbc_provider_helpers', ROOT / PROVIDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def members(motions):
    """Capsule path -> local source. Only what tracking.py, sonic.py, stage.py and the R3 boundary read."""
    rel = lambda p: p.relative_to(ROOT).as_posix()
    files = [WBC / 'tracking.py', WBC / 'parcour.py', WBC / 'sonic.py', WBC / 'references.json', HERE / 'stage.py', HERE / 'bootstrap.sh',
        HERE / 'requirements-worker.txt', ROOT / 'robot/model/current-model.json',
        *(ROOT / 'control/common/plant' / n for n in ('__init__.py', 'boundary.py', 'mjlab_adapter.py', 'dynamics.py')),
        ROOT / 'robot/model/dynamics/actuator-dynamics.json',
        *(MODEL / n for n in ('cove50-rigid.xml', 'actuators.json', 'sensors.json', 'timing.json')),
        *sorted((MODEL / 'meshes').iterdir())]
    out = {rel(p): p for p in files}
    for name in (Path(PLAN['motion']).name, 'motions.json'):
        out[f'control/wbc/motions/{name}'] = Path(motions) / name
    if PLAN.get('eval_motion'):  # a separate evaluation tape (held-out motions), from its own built folder
        for name in (Path(PLAN['eval_motion']).name, 'motions.json'):
            out[str(Path(PLAN['eval_motion']).parent / name)] = ROOT / PLAN['eval_source'] / name
    if PLAN.get('sonic'):
        for name in ('model_encoder.onnx', 'model_decoder.onnx'):
            out[f"{PLAN['sonic']}/{name}"] = SONIC_SOURCE / name
    if PLAN.get('init_from'):  # restore the retained run first (infra.storage.blob_store restore)
        out[PLAN['init_from']] = ROOT / PLAN['init_from']
        if PLAN.get('init_widths'):  # its train report carries the reward widths the run starts at
            out[str(Path(PLAN['init_from']).parent / 'report.json')] = ROOT / Path(PLAN['init_from']).parent / 'report.json'
    return out


def bundle(work):
    """Deterministic gzip tar (sorted, zero mtimes) plus capsule-manifest.json."""
    work = Path(work)
    motions = work / 'motions'
    if not (motions / Path(PLAN['motion']).name).exists():
        raise FileNotFoundError(f'Convert motions first: motions.py --output {motions} --only {MOTION} [--speed]')
    files = members(motions)
    manifest = dict(schema='cove50.wbc.capsule.v1', plan=PLAN,
        files=[dict(path=k, sha256=sha(v), bytes=v.stat().st_size) for k, v in sorted(files.items())])
    raw = json.dumps(manifest, indent=1).encode()
    archive = work / 'input.tar.gz'
    with archive.open('wb') as out, gzip.GzipFile(fileobj=out, mode='wb', mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tf:
        info = tarfile.TarInfo('capsule-manifest.json')
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
        for name, path in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = path.stat().st_size, 0o644
            with path.open('rb') as stream:
                tf.addfile(info, stream)
    receipt = dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
        members=len(files) + 1, manifest_sha256=hashlib.sha256(raw).hexdigest(), plan=PLAN)
    write(RECORD / 'bundle.json', receipt)
    return {k: receipt[k] for k in ('sha256', 'bytes', 'members')}


def submit(work):
    receipt = RECORD / 'submission.json'
    if receipt.exists():
        raise FileExistsError('A submission exists; never resubmit automatically')
    expected = read(RECORD / 'bundle.json')
    archive = Path(work) / 'input.tar.gz'
    if (sha(archive), archive.stat().st_size) != (expected['sha256'], expected['bytes']) or expected['plan'] != PLAN:
        raise ValueError('Bundle or plan changed since bundle.json')
    sys.path.insert(0, str(ROOT))
    from infra.storage import spend_guard  # writes down why this job is worth running, with notes; never refuses
    tape = read(Path(work) / 'motions' / 'motions.json')
    # The reflection looks at the motions a job trains on. An evaluation-only job trains on none; the never-trained
    # motions it scores serve the owner's measure of agility across all motions (AGENTS.md).
    motions = [] if PLAN.get('evaluate_only') else [s['name'] for m in tape['motions'] for s in m.get('segments') or [m]]
    spend_guard.reflect(PILOT, PLAN['provider_policy']['executionTimeout'] / 3.6e6, PLAN.get('target'),
        PLAN.get('question'), PLAN.get('decision'), motions, PLAN.get('cheaper_check'), PLAN.get('value'))
    base = provider()
    client, bucket, endpoint = base.s3()
    if not endpoint:
        raise SystemExit('The Terraform trial endpoint is not provisioned')
    sys.path.insert(0, str(ROOT))
    from infra.compute.serverless.s3_artifacts import CONFIG, profile_auth
    from runpod_flash.runtime.serialization import serialize_kwargs
    prefix = f"robot-learning-lab/v1/cove50-wbc-{PILOT}/{uuid.uuid4().hex}"
    now = time.time()
    record = dict(endpoint=endpoint, bucket=bucket, input_key=prefix + '/input.tar.gz', output_prefix=prefix,
        bundle_sha256=expected['sha256'], bundle_bytes=expected['bytes'], plan=PLAN,
        submitted_utc=datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(),
        absolute_expiry_utc=datetime.datetime.fromtimestamp(now + PLAN['provider_policy']['ttl'] / 1000, datetime.timezone.utc).isoformat(),
        state='input_upload_pending')
    write(receipt, record)  # an ambiguous outcome stays visible; no retry
    with archive.open('rb') as stream:
        client.put_object(Bucket=bucket, Key=record['input_key'], Body=stream)
    response = client.get_object(Bucket=bucket, Key=record['input_key'])
    try:
        stored = hashlib.file_digest(response['Body'], 'sha256').hexdigest()
    finally:
        response['Body'].close()
    if stored != expected['sha256']:
        raise ValueError('Uploaded capsule readback differs')
    if (CONFIG['endpoint'].rstrip('/'), CONFIG['bucket']) != (client.meta.endpoint_url.rstrip('/'), bucket):
        raise ValueError('Transport configuration differs from Terraform output')
    code = (HERE / 'worker.py').read_text().split("\n\nif __name__ == '__main__':")[0]
    data = dict(storage=dict(CONFIG), storage_auth=profile_auth(), plan=PLAN, input_key=record['input_key'],
        output_prefix=prefix, bundle_sha256=expected['sha256'], bundle_bytes=expected['bytes'],
        absolute_expiry_utc=record['absolute_expiry_utc'])
    record.update(state='request_pending', worker_sha256=hashlib.sha256(code.encode()).hexdigest())
    write(receipt, record)
    payload = dict(function_name='run_wbc_pilot', function_code=code, kwargs=serialize_kwargs({'data': data}),
        serialization_format='cloudpickle', dependencies=['boto3==1.43.98'], system_dependencies=[],
        accelerate_downloads=False)
    accepted = base.call(endpoint, 'run', {'input': payload, 'policy': PLAN['provider_policy']})
    record.update(state='accepted', job_id=accepted['id'], last_query_epoch=time.time(), queries=0)
    write(receipt, record)
    return {k: record[k] for k in ('endpoint', 'job_id', 'submitted_utc', 'absolute_expiry_utc')}


def status():
    record = read(RECORD / 'submission.json')
    if record.get('provider_status') in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
        raise PermissionError('Terminal status already recorded; no further query')
    if time.time() - record['last_query_epoch'] < 300:
        raise PermissionError('Status queries must be at least five minutes apart')
    if record['queries'] >= 12:
        raise PermissionError('Twelve status queries used; inspect manually')
    base = provider()
    record.update(last_query_epoch=time.time(), queries=record['queries'] + 1)
    write(RECORD / 'submission.json', record)
    result = base.call(record['endpoint'], 'status/' + record['job_id'])
    client, bucket, _ = base.s3()
    try:
        response = client.get_object(Bucket=bucket, Key=record['output_prefix'] + '/progress.json')
        with response['Body'] as body:
            progress = json.loads(body.read())
    except Exception as exc:
        progress = dict(unavailable=type(exc).__name__)
    record['provider_status'] = result.get('status')
    write(RECORD / 'submission.json', record)
    stages = [dict(name=s['name'], status=s['status'], wall_s=round(s.get('wall_s', 0)),
        **({k: s[k] for k in ('elapsed_s', 'ram_available_gb', 'disk_free_gb', 'gpu', 'heartbeat_error') if k in s} if s['status'] == 'running' else {}),
        **({'last_log_line': s['log_tail'].strip().splitlines()[-1][-200:]} if s['status'] == 'running' and s.get('log_tail', '').strip() else {}),
        **({'partial_bytes': s['partial']['bytes']} if 'partial' in s else {})) for s in progress.get('stages', [])]
    return dict(provider=result.get('status'), worker=progress.get('status', progress.get('state')),
        stages=stages, output=result.get('output'), error=result.get('error'))


def restore(evidence, work):
    record = read(RECORD / 'submission.json')
    evidence, work = Path(evidence), Path(work)
    base = provider()
    client, bucket, _ = base.s3()
    # A cancelled job never writes completion.json; its closed packs are listed in progress.json.
    name = 'progress.json' if record.get('state') == 'cancelled_by_operator' else 'completion.json'
    response = client.get_object(Bucket=bucket, Key=record['output_prefix'] + '/' + name)
    with response['Body'] as body:
        raw = body.read()
    (evidence / name).write_bytes(raw)
    completion = json.loads(raw)
    completion.setdefault('status', 'cancelled')
    restored, lost = [], []
    for pack in completion['packs']:
        archive = work / (pack['stage'] + '.tar')
        try:
            client.download_file(bucket, pack['key'], str(archive))
        except Exception as exc:  # a pack deleted from transport before restore (recorded in submission.json)
            if not record.get('transport_deleted_early') or pack['key'].split('/')[-1] not in {o.split('/')[-1] for o in record['transport_deleted_early']['objects']}:
                raise
            lost.append(dict(stage=pack['stage'], reason=f'{type(exc).__name__}: deleted from transport before restore'))
            continue
        if (sha(archive), archive.stat().st_size) != (pack['sha256'], pack['bytes']):
            raise ValueError('Pack differs: ' + pack['stage'])
        expected = {r['path']: r for r in pack['files']}
        with tarfile.open(archive) as tf:
            if {m.name for m in tf.getmembers()} != set(expected):
                raise ValueError('Pack members differ: ' + pack['stage'])
            tf.extractall(evidence, filter='data')
        for name, row in expected.items():
            if sha(evidence / name) != row['sha256']:
                raise ValueError('Restored file differs: ' + name)
        restored.extend(expected)
        archive.unlink()
    # An incomplete training stage leaves only its newest mid-run checkpoint.
    if completion['status'] != 'completed':
        for row in completion.get('report', completion).get('stages', []):
            if 'partial' in row:
                target = evidence / 'evidence' / row['name'] / 'partial.pt'
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, row['partial']['key'], str(target))
                if (sha(target), target.stat().st_size) != (row['partial']['sha256'], row['partial']['bytes']):
                    raise ValueError('Partial checkpoint differs: ' + row['name'])
                restored.append(str(target.relative_to(evidence)))
    index = dict(schema=f'cove50.wbc.{PILOT}-index.v1', job_id=record['job_id'], status=completion['status'],
        submission={k: record[k] for k in ('endpoint', 'job_id', 'output_prefix', 'bundle_sha256', 'submitted_utc')},
        plan=record['plan'], files=sorted(restored), **({'lost_packs': lost} if lost else {}))
    write(evidence / 'restored-index.json', index)
    write(RECORD / 'restored.json', dict(evidence=str(evidence.relative_to(ROOT)), status=completion['status'], files=len(restored),
        **({'lost_packs': lost} if lost else {})))
    return dict(status=completion['status'], files=len(restored))


def main():
    global PILOT, PLAN, RECORD
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pilot', choices=sorted(PILOTS), required=True)
    parser.add_argument('action', choices=('bundle', 'submit', 'status', 'restore'))
    parser.add_argument('--work', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    PILOT, PLAN = args.pilot, PILOTS[args.pilot]
    RECORD = _record_dir(PILOT)
    if args.action == 'bundle':
        result = bundle(args.work)
    elif args.action == 'submit':
        result = submit(args.work)
    elif args.action == 'status':
        result = status()
    else:
        result = restore(args.evidence, args.work)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
