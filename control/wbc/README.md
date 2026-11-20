# Cove-50 whole-body controller (WBC)

Idle root `T-xjjwrs`. The WBC is the bridge layer between slower upstream
sources (skill policies, VLAs, teleop, world-model planners) and the servos'
internal position PID. It runs on the robot, faster than the sources above it,
and turns their targets into agile, feasible servo commands.

**The score is agility across all motions** (owner, 2026-11-08; `AGENTS.md`).
Every motion on the tapes is a made-up stand-in, and the final set is not
known. Any single motion, or single family, is irrelevant on its own. The
per-motion values in the steps below are evidence of where motion degrades,
not goals. No run or change is planned around one.

## Step 1: native tracking port (`T-ge3q2k`)

`tracking.py` runs the upstream MJLab BeyondMimic tracking task
(`mjlab/tasks/tracking` in `.venv-control`) on the selected R3 body through the
walk-resume R3 boundary (`control/common/plant`): 0.25 ms
physics, 50 Hz control, 20 ms command and 10 ms feedback delay. Native rewards,
adaptive sampling and PPO are unchanged. Differences from the G1 config:

- actor reads the delayed R3 feedback packet (joints, gyro, gravity estimate)
  and has no root position/linear velocity; `motion_anchor_ori_b` still uses
  true torso orientation;
- critic reads true root velocity from the entity (R3 has no velocimeter);
- narrowed ranges in `RANGES`, each with its source;
- soft joint-limit factor 1.0, because straight R3 knees and elbows sit on
  their hard 0 rad stop;
- action scale 0.5 rad (native base value).

`references.json` pins the first references: published R3 skill episodes that
already run on this body. Convert them, or run the smoke:

```sh
.venv-control/bin/python control/wbc/motions.py --output .cache/wbc-motions
infra/lab storage run cove50-wbc-port-smoke -- .venv-control/bin/python \
  control/wbc/smoke.py --scratch '{work}' --output '{evidence}'
```

Traces and the R3 visual meshes may be in private storage; restore with
`.venv/bin/python -B -m infra.storage.blob_store restore PATH` first.

## Step 2: GPU teacher pilot (`T-e7vm69`)

`compute/` holds the bounded serverless job (`launch.py --pilot
teacher-pilot-20261101`): native scratch actor, 2,048 worlds on an RTX 4090.
Dashboard row `cove50-wbc-teacher-pilot-20261101-r1`.

## Step 3: GEAR-SONIC warm start (`T-dfmjb9`)

`sonic.py` starts the WBC from NVIDIA GEAR-SONIC (GR00T-WholeBodyControl,
default release, NVIDIA Open Model License) instead of random weights:

- the g1 encoder (10 future frames 0.1 s apart -> 64-D FSQ token) and the
  10.2 M-parameter decoder are rebuilt in PyTorch from the pinned ONNX files
  (`check-onnx` compares them; `g1-check` runs the released weights on G1 in
  MuJoCo against SONIC's sample walk);
- `MAP` maps 20 R3 joints onto G1 by joint axis and side. R3 `R_*` links are
  on the robot's anatomical left (model faces +x, `frames.json`), so they map
  to G1 `left_*`; G1's elbow zero is ~90 deg bent (offset 1.278 rad). `map`
  re-derives and checks it. The neck gets new zero-initialized outputs; G1
  waist and wrists are fed as neutral;
- R3 state is presented in SONIC's G1 observation layout; the measured
  feedback packet follows the actuator-contract order, not the entity order;
- actions stay in G1 units (`default + 0.25 effort/kp * a`), converted to R3
  targets per joint.

```sh
.venv-control/bin/python control/wbc/sonic.py fetch
.venv-control/bin/python control/wbc/sonic.py check-onnx
infra/lab storage run sonic-g1-check -- .venv-control/bin/python \
  control/wbc/sonic.py g1-check --out '{evidence}'
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot sonic-trust-region-r4-20261102 bundle --work .cache/work/sonic-pilot
```

`launch.py` keeps every pilot's plan. The H100 pilot (`sonic-pilot-20261102`)
was cancelled after its benchmark: R3 physics at 0.25 ms is latency-bound, so
4,096-8,192 worlds on the H100 gave no more transitions/s than 2,048 on the
4090. The first 4090 fine-tune (`sonic-finetune-20261102`) went silent during
training and was cancelled. `sonic-finetune-r2-20261102` repeats it with stage
heartbeats (log tail, host/GPU state, newest checkpoint), a stop when a stage's
log stops growing, and a training wall matched to the r1 scratch teacher.
`status` shows the heartbeat; `restore` recovers the newest checkpoint from an
incomplete run. r2 was cancelled before it started (empty account).

`sonic-finetune-r3-20261102` ran on an H100 and used the card for worlds:
the benchmark takes the most worlds within 5% of the best throughput (6,144;
8,192 ran out of memory), `--minibatch-samples` keeps 12,288 samples per
gradient step, and evaluation uses 64 worlds. The rebuilt SONIC walks G1 for
40 s, but on R3 it falls at about 0.5 s zero-shot (servo playback: 0.86 s),
and 105 PPO rounds made it worse (about 0.2 s, joints 0.82 rad off): the
entropy bonus raised the 0.05 std threefold, the surrogate stayed near +0.25
and the critic diverged in spikes. Dashboard record:
`cove50-wbc-sonic-finetune-r3-20261102`. Clips regenerate with
`compute/report.py --arm 'SONIC fine-tuned=evaluate-sonic-warm' --arm
'SONIC zero-shot=evaluate-zero-shot'` from the restored run and the bundled
motion.

`sonic-trust-region-r4-20261102` (4090, 2,048 worlds) reruns the comparison
with `--recipe trust-region`: std fixed at 0.05, no entropy bonus, reference
encoder frozen, decoder rate adapted per mini-batch to KL 0.01 from 1e-7. It
then trains the same network from random layers (`sonic-random`, with the
release FSQ levels kept) for the same 192 PPO rounds. The warm start stays at
its zero-shot fall (about 0.5 s, joints 0.40 rad off). Random init learns a
quiet stance: 3 of 64 worlds stand for 8 s, most tip backward at about 0.9 s,
and its training episodes were still lengthening at the end. Dashboard record:
`cove50-wbc-sonic-trust-region-r4-20261102`. Clips regenerate with
`compute/report.py --arm 'Same net, random init=evaluate-sonic-random' --arm
'SONIC warm start=evaluate-sonic-warm'`. Arms evaluate in separate processes,
so per-world start velocities differ slightly between them.

## Step 4: continue the native teacher (`T-ykjzwy`)

`launch.py` pilots with `init_from` bundle a retained `teacher.pt` (restore its
run first) and pass `stage.py train --init-checkpoint`. That restores the
actor, critic, exploration noise, normalizers and optimizer, and resumes the
adaptive rate where the teacher left it. The run first evaluates the starting
checkpoint (`evaluate-start`) on the same 64 worlds.

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-teacher-pilot-5373d7868388
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot teacher-continue-r2-20261102 bundle --work .cache/work/teacher-continue
```

`teacher-continue-20261102` (4090) was cancelled while queued with no 4090
stock. `teacher-continue-r2-20261102` ran on the large-GPU pool, which placed it
on an RTX PRO 6000 MIG 2g.48gb slice: 4,096 worlds at 2.7k transitions/s, below
the 4090's 3.7k. 174 more PPO rounds (17.1 M transitions) took r1 from 6 of 64
worlds clear for 8 s to all 64, nominal and pushed, and it turns with the clip.
It does not walk: the root stays within about 0.12 m of its start while the
clip travels 1.54 m, and the feet barely lift. The tracking reward widths are
still mjlab's G1 values (root and body position std 0.3 m, body velocity 1.0
m/s), so standing costs little at R3 scale. Dashboard record:
`cove50-wbc-teacher-continue-r2-20261102`. Clips regenerate with
`compute/report.py --arm 'r1 start=evaluate-start' --arm 'Continued
teacher=evaluate-native'` (r1 picks the worlds; the continued teacher has no
spread to pick from).

The evaluation's `final_root_xy_error_m` used to compare the last trace row,
where the motion command has already wrapped to the clip start. `stage.py` and
`report.py` now compare the row before it and also report root travel. The r1,
r3 and r4 records keep the old value as `final_root_xy_to_clip_start_m_p50`.

## Step 5: R3 reward widths (`T-3bnsnk`)

`tracking.REWARD_STD` holds the tracking reward widths. `g1` is mjlab's
default. `r3` scales the length and linear-speed widths by `HEIGHT_RATIO`: root
and body position 0.3 to 0.12 m, body velocity 1.0 to 0.4 m/s. Angle and
angular-velocity widths are unchanged. Select it with `stage.py
--reward-widths r3`, or `reward_widths='r3'` in a `launch.py` pilot.
`env_cfg` raises if upstream changes a G1 width it overrides.

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-teacher-continue-r2-20261102-639a28b2684b
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot teacher-widths-r3-retry-20261103 bundle --work .cache/work/teacher-widths
```

`teacher-widths-r3-20261103` stopped after bootstrap. One 90 s storage read
timed out, and the worker's storage client had a single attempt. After its
conditional claim the worker now retries every write (5 attempts, standard
mode). `teacher-widths-r3-retry-20261103` ran the same plan on a 4090:
2,048 worlds, 440 more PPO rounds from r2 (21.6 M transitions). The teacher
now steps in the clip's rhythm, about 12 lifts per foot like the reference,
and turns with it. Its strides are short: 0.35 m along the clip in 8 s against
1.53 m. 4 of 64 nominal worlds fall early (1.1-1.5 s), where r2 fell in none.
Training was still improving at the end. Dashboard record:
`cove50-wbc-teacher-widths-r3-20261103`. Clips regenerate with
`compute/report.py --motion .cache/work/teacher-widths/motions/forward-walk.npz
--arm 'R3 widths=evaluate-native' --arm 'r2 start=evaluate-start'`. The R3
widths arm picks the worlds. `motions.py --output .cache/work/teacher-widths/motions
--only forward-walk` rewrites the motion byte for byte.

`report.py` adds foot lifts and heading change to each matrix row. A lift is a
rise of more than 1 cm above the foot's own 10th-percentile height.

## Step 6: continue the R3-width teacher (`T-pba4zk`)

`teacher-widths-r3-continue-20261103` continues the retry teacher unchanged:
R3 widths, adaptive sampling, 2,048 worlds on the 4090.

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-teacher-widths-r3-retry-20261103-a5fb7c6f2fea
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/robot/model/timestep/meshes
.venv-control/bin/python control/wbc/motions.py \
  --output .cache/work/teacher-widths-continue/motions --only forward-walk
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot teacher-widths-r3-continue-20261103 bundle --work .cache/work/teacher-widths-continue
```

The run made 428 more PPO rounds (21.0 M transitions). The teacher covers 0.85 m
along the clip in 8 s against 1.54 m (start 0.36 m), keeps the step rhythm and
turn, and no nominal world falls. One pushed world falls (0.2 m/s). Saturation
rose to 10.3%, with both ankle pitch servos at their limit about a third of
the time. The head is held on the neck's -0.52 rad stop throughout. Training
episodes flattened near the 400-step clip. Dashboard record:
`cove50-wbc-teacher-widths-r3-continue-20261103`. Clips regenerate with
`compute/report.py --motion .cache/work/teacher-widths-continue/motions/forward-walk.npz
--arm 'Continued=evaluate-native' --arm 'Retry start=evaluate-start'`.

`report.py` now breaks survival ties by the larger final path error, so when
every world stays up the severe clip is the worst tracker. Pushed "best" is
the longest-clear world with the strongest push, and a fourth clip shows the
earliest pushed fall. Earlier records picked the first index among tied
worlds. Their published clips are the retained copies.

## Step 7: direction checks, slow clip, torque cap and stronger servo (`T-gtxyrn`)

Three short checks and an unchanged control, each resuming the step 6 teacher
(iteration 1206) for 4,500 s at 2,048 worlds, so gains from extra training are
not credited to a change:

- `check-control-20261103`: nothing changed.
- `check-slow-clip-20261103`: `motions.py --speed 0.6` resamples the walk in
  time (same path and steps, 13.3 s); evaluation runs the full clip.
- `check-peak-torque-20261103`: `stage.py --peak-torque`, the C047 catalog
  stall curve up to 2.94 N m instead of the 0.98 N m rated cap.
- `check-sts3250-retry-20261103` (`--actuator sts3250`): `tracking.ACTUATORS`
  puts the Feetech STS3250 catalog law on the C047 joints (4.9 N m stall, cap
  at an assumed stall/3, 7.87 rad/s, gains scaled with stall torque) and
  `heavier_servos` adds its extra 19.5 g per servo at each joint anchor in the
  case body. The CAD is unchanged. The first submission
  (`check-sts3250-20261103`) lost its worker to the 1,200 s bootstrap timeout.

`infra/compute/runpod/walking-trial.tf` takes `walking_trial_workers_max` (1 to 4) so
a reviewed batch of independent jobs can run side by side; return it to 1.

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-teacher-widths-r3-continue-20261103-24d03dbe7e6f
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/robot/model/timestep/meshes
.venv-control/bin/python control/wbc/motions.py \
  --output .cache/work/wbc-checks/normal/motions --only forward-walk
.venv-control/bin/python control/wbc/motions.py \
  --output .cache/work/wbc-checks/slow/motions --only forward-walk --speed 0.6
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot check-peak-torque-20261103 bundle --work .cache/work/wbc-checks/peak-torque
```

(each pilot bundles from a work folder holding its motion under `motions/`).

Findings on one clip and one seed: the control walked 77% of the clip's
distance, with 3 of 64 falling in an early lunge, and its reward flattened.
The slow clip made the teacher stand almost in place: a time-stretched walk
keeps the step length but lengthens single support 1.67 times, so it is not a
slower walk. Peak torque and the STS3250 both removed every fall and kept
reward rising for the whole run. Peak torque walked 25% faster than the clip
and used above-rated torque 5% of the time. The STS3250 kept the closest step
timing and path, at 7% leg saturation. Dashboard records:
`cove50-wbc-check-{control,slow-clip,peak-torque,sts3250-retry}-20261103`.
Clips regenerate with `compute/report.py --motion <that arm's motion> --arm
'<Arm> trained=evaluate-native' --arm 'Start=evaluate-start'`.

## Step 8: continue the stronger-torque checks until return flattens (`T-wy4arf`)

`check-peak-torque-continue-20261104` and `check-sts3250-continue-20261104`
resume the step 7 peak-torque and STS3250 teachers, each with its own plant.
There is no fixed training wall: a session trains for the rest of the
endpoint's deadline, about 5,400 s. `stage.py --plateau-window 4
--plateau-gain 0.01` stops it earlier once the mean training return of the last
four full chunks (about 5 min each) is less than 1% above the four before.
The two resume warm-up chunks and short tail chunks are left out. This is a
compute stop, not a grade. Replayed on the step 7 logs, the rule stops the
control at chunk 14 (its return had flattened). The two stronger-torque arms
were still 11–13% up per window at the end. The train report records the
per-iteration returns, each chunk's mean and window gain, and why training
stopped. `evaluate-start` re-evaluates the step 7 teacher on the same 64
worlds. A session that ends at the deadline still rising continues from its
own `teacher.pt` while the balance keeps a margin for a full job.

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-check-peak-torque-20261103-1d9942fb9a5b/evidence/train-native
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-check-sts3250-retry-20261103-bd48817511e9/evidence/train-native
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/robot/model/timestep/meshes
.venv-control/bin/python control/wbc/motions.py \
  --output .cache/work/wbc-continue/peak/motions --only forward-walk
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot check-peak-torque-continue-20261104 bundle --work .cache/work/wbc-continue/peak
```

Session 1 (`cove50-wbc-check-{peak-torque,sts3250}-continue-20261104`, about 98
min of training each) ran to the deadline. The rule did not fire: the last
window was still 2.5% (peak) and 3.7% (STS3250) above the one before. Peak
torque: joint error 0.153 to 0.119 rad, step timing correlation 0.88 to 0.94,
final distance to the clip path 0.38 to 0.31 m, leg saturation 12% to 6.6%.
One of 64 worlds (21) now lunges forward and falls at 0.56 s, unrecovered; the
rated-cap control also fell from that start. STS3250: no falls in 128, path
0.35 to 0.27 m with the sideways drift gone (0.25 to 0.03 m), leg saturation
7.6% to 3.9%. Both still run about 20% ahead of the clip's pace. Session 2
(`check-*-continue-r2-20261104`) resumes each from its session 1 teacher:

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-check-peak-torque-continue-20261104-e75d26f62572/evidence/train-native
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/runs/cove50-wbc-check-sts3250-continue-20261104-f1780a5373a8/evidence/train-native
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot check-peak-torque-continue-r2-20261104 bundle --work .cache/work/wbc-continue/peak
```

Session 2 re-evaluated the session 1 peak-torque teacher on the same 64 worlds
and all 64 stayed up, world 21 included: its session 1 fall was run-to-run
contact spread at a marginal start (the rated-cap control also fell there).
STS3250 hit the plateau rule at chunk 15 (0.96%) after 75 min. It ended 0.235
m from the clip's path, turned with the clip (63.7 against 63.2 degrees) and
had legs saturated 3.1% of the time, with no falls. Joint error barely moved
(0.164 to 0.161 rad). Peak torque ran to the deadline with its last window
1.16% up. Path 0.31 to 0.23 m, joint error 0.119 to 0.107 rad, timing
correlation 0.96, above rated torque 3.7% to 2.1%, no falls.
`check-peak-torque-continue-r3-20261104` continues only the peak-torque arm.

Session 3 stopped on the plateau rule at chunk 13 (0.97%) after 64 min. Both
arms have now flattened under this recipe. Peak torque, after 4.3 h of training
beyond step 7: path 0.38 to 0.20 m, joint error 0.153 to 0.096 rad, the clip's
12 and 12 steps at timing correlation 0.96, push dip under 1 cm, no falls. Leg
saturation fell from 12% to 5.5% and samples above rated torque from 5.4% to
2.4%. The left hip pitch carries more of the load (saturated 37% of the time),
the walk is still about 15% fast, and the heading is 57 against the clip's 63
degrees. STS3250, after 2.9 h: path 0.35 to 0.235 m, heading on the clip, legs
saturated 3.1%, but joint error flat near 0.16 rad. Five jobs cost 11.75 USD.
Runpod's status route returns 404 about 30 min after a job finishes. Check
`completion.json` under the job's `output_prefix` in transport storage, which
is what `restore` reads.

## Step 9: agility motion set and servo demand (`T-hw56vg`, local CPU)

Before more GPU training, `agility/` asks what an agile, playful Cove-50 needs
from its servos, independent of any learned controller.

- `agility/library.py` builds a fixed motion set at two to four tempos each:
  the six retained R3 skill rollouts; the learned forward walk replayed 1.25x
  and 1.5x (same stride, quicker cadence); the four non-stepping rollouts
  time-compressed 2x and 4x; authored ape moves with planted feet (chest beat,
  overhead wave, arm display swing, knee bounce, side sway, quick crouch,
  three flat-footed hops with ballistic flight); and ZMP preview-control
  stepping (turn in place, stomping march) with a heel lift before each swing.
- `agility/body.py` runs whole-body inverse dynamics on the R3 model with the
  floor forces at the sole corners chosen by non-negative least squares in a
  0.7 friction pyramid. The base must be carried by the floor; among the ways
  to do that the least rated-normalised joint torque is taken. Whatever the
  floor cannot carry is a residual outside force on the torso: it marks a
  reference this body cannot do as given. The demand is a lower bound for a
  perfect controller.
- `agility/demand.py` compares each joint's demand with the C047 rated cap,
  the C047 stall curve and the STS3250 catalog (on the 4.22 kg body), with and
  without reflected rotor inertia. The R3 model has none; BAM (Rhoban)
  identified the 1:345 STS3215 at 0.022-0.026 kg m2, used here as 0.024 on
  all 20 C047 joints (an assumption for the STS3250). 'Use' is demanded over
  available torque at that joint speed; above 1 the servo cannot supply it.
- `agility/render.py` renders each family's tempos side by side as kinematic
  reference playback.

```sh
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/robot/model/timestep/meshes
# restore the six retained traces named in references.json the same way, then:
A=control/wbc/agility
infra/lab storage run cove50-agility-demand -- sh -c ".venv-control/bin/python -B $A/library.py --output \"\$LAB_WORK_DIR/motions\" && cp \"\$LAB_WORK_DIR/motions/motions.json\" \"\$LAB_EVIDENCE_DIR/\" && .venv-control/bin/python -B $A/demand.py --motions \"\$LAB_WORK_DIR/motions\" --work \"\$LAB_WORK_DIR\" --evidence \"\$LAB_EVIDENCE_DIR\" && .venv-control/bin/python -B $A/render.py --motions \"\$LAB_WORK_DIR/motions\" --output \"\$LAB_EVIDENCE_DIR/clips\""
```

A flat-footed walk generator was tried and dropped: on 0.2 m legs with +-35
degree ankles it could not make 7 cm steps at 0.2 m/s or faster, and its 0.1
m/s walk needed more joint speed than the learned 0.2 m/s walk (an artifact
of its swing, not of the body). Walking speed therefore comes from the
learned walk replayed faster.

Run `cove50-agility-demand-233683dfc811` (local CPU, 11 min; 41 motions x 4
body conditions; record `cove50-wbc-agility-demand-20261105` with ten
reference clips). Checked against the torques the trained policies applied,
the arm demand is 0.89-1.05 of it and the leg demand 0.51-0.81, a lower bound
(the policies also push the feet apart in double support).

- Slow and postural moves fit the C047: squats and rises up to 4x, arm raises,
  a 1 Hz chest beat or wave, slow sway.
- Rotor inertia changes the picture most. BAM's figure is 8x the knee's link
  inertia and 200x the shoulder yaw's. With it, the learned 0.2 m/s walk is
  beyond the C047 stall curve at some leg joint in 41% of frames (4.5% beyond
  rated without it): its knee and ankle accelerate at about 105 rad/s2 (p95),
  2.5 N m for the rotor alone. The training model has armature 0.
- The STS3250 catalog law, with the same inertia and mass, covers that walk
  (1%) and reaches about one tempo step further in every family: 2 Hz chest
  beat and wave, 1 Hz arm display, 1.5 Hz knee bounce, 0.35 s crouch.
- Arm play is limited by speed (C047 4.7 rad/s), leg play by torque and
  inertia at hip pitch, knee and ankle pitch (hip roll for swaying).
- Even a 1 cm flat-footed hop needs 15 rad/s and is beyond both servos.

## Step 10: r4 best-estimate actuator dynamics and a third servo (`T-t68e54`)

`robot/model/dynamics/` holds best estimates for every
actuator value not yet measured, each sourced and bounded:
- reflected rotor inertia;
- Coulomb friction;
- the Feetech firmware duty loop on a rate-limited internal target (BAM's
  model);
- backlash;
- the catalog torque-speed line;
- the 3S3P pack rail (11.1 V).

The R3 body is unchanged and R3 stays the default plant.
`tracking.env_cfg(dynamics='r4', peak=True, actuator=...)` selects r4.
`tracking.ACTUATORS` gains the Waveshare ST3025, a 12 V brushless serial-bus
servo. BAM identified its rotor at 0.0046 kg m2 against about 0.024 for the
1:345 Feetech gear. It is faster and stronger (3.9 N m, 8.9 rad/s) but weighs
124 g, putting the robot at 5.21 kg on all 20 joints.

The demand check v2 (`agility/demand.py`, record
`cove50-wbc-agility-demand-v2-20261105`) shows the average share of
agility-set frames beyond the stall curve:

| C047 | STS3250 | ST3025 |
|---|---|---|
| 17% | 8% | 3% |

Rotor inertia moves the 1:345 servos' result by up to 42 points; the ST3025
moves by a few points at most across its bounds.

Pilots `r4-{c047,sts3250,st3025}-20261105` ran on three 4090 workers at 12 V
(before the rail was added). Each continued its r3 teacher on the r4 plant,
the ST3025 from the STS3250 teacher.

| Starting teacher on r4 (falls, nominal/pushed, of 64) | r4 teacher after training |
|---|---|
| C047: 2/3, walks 1.18 m | 0/0, joints 0.092 rad, 0.23 m from the clip's end; hottest servo 128% of rated current, mean 9.5 A (4.5 A on r3) |
| STS3250: 8/11 | 1/1 (one marginal start), 0.14 m; 116%, 12.5 A |
| ST3025: 0/0 | 0/0, the clip's 12 and 12 steps, 0.13 m; 104%, 9.3 A; still improving at the deadline |

Records `cove50-wbc-r4-{c047,sts3250,st3025}-20261105`. On the realistic plant
every servo's hottest joint runs at or above rated current while walking.

```sh
.venv-control/bin/python -B robot/model/dynamics/build.py
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot r4-st3025-20261105 bundle --work .cache/work/wbc-r4/st3025
```

## Step 11: the agility tape, multi-motion WBC training

`agility/tape.py` chains the 41 agility motions into one 254 s reference at 50
Hz. Each motion is placed where the previous one ended, with 1 s min-jerk
blends between them. `motions.py --tape` converts it to the native motion
file. `stage.py evaluate --segments motions.json --segment-names ...` evaluates
chosen motions from their first frame inside the tape. Pilots
`tape-{c047,sts3250,st3025}-20261105` start from the r4 walk teachers on the
r4 plant at 11.1 V, with 13 evaluation segments (one per family, 32 worlds).

The walk-only teachers follow only the walk: on every other segment all 32
worlds reach a native condition within 0.5-1.4 s, mostly hands or torso more
than 10 cm off the reference. Evaluation is physics-bound at about 9 s of wall
time per simulated second. The ST3025 job's evaluate-start needed 1,502 s
against its 1,500 s limit, so `tape-st3025-retry-20261105` runs without it.

```sh
A=control/wbc
.venv-control/bin/python -B $A/agility/library.py --output .cache/work/wbc-tape/library
.venv-control/bin/python -B $A/agility/tape.py --motions .cache/work/wbc-tape/library --output .cache/work/wbc-tape/tape
.venv-control/bin/python -B $A/motions.py --output .cache/work/wbc-tape/c047/motions --tape .cache/work/wbc-tape/tape
```

After 65-72 min on the tape, each servo's teacher stays upright through 10 of
the 13 segments. But it does so mostly by standing:
- **Vertical motions:** the deep squat, quick crouch, hop and knee bounce reach
  a fifth or less of their torso travel.
- **Arm gestures:** the elbows stay straight, so the chest beat and overhead
  wave end in falls and arm display flags the forearms.

Records: `cove50-wbc-tape-{c047,sts3250,st3025}-20261105`.

Two causes sit in the plant and training interface, not in the servos:
1. **Hip cap.** The six hip axes have no range in the selected CAD, and R3 caps
   their commands at a ±0.35 rad diagnostic envelope. The tape needs hip pitch
   down to −82°.
2. **Commands parked past stops.** The walk teachers command elbows, neck pitch
   and one hip pitch far past their stops. There the boundary clamps every
   value to the same target, as servo firmware angle limits do, so nothing
   pulls them back. On the walk this was invisible, because the elbows sit on
   their 0 stop.

## Step 12: hip ranges, live commands (tape2)

r4 revision 2 (`robot/model/dynamics/README.md`) adds
best-estimate hip ranges: pitch −110..30°, roll ±35°, yaw ±45°. Pilots
`tape2-{c047,sts3250,st3025}-20261105` continue each tape teacher on that plant
with three changes:
- the action head is reset (`stage.py --reset-actor-output`), which keeps the
  features, critic and normalizers and starts every joint at its default
  target;
- a 0.5 per rad penalty applies to commands past a stop (`--stop-weight`,
  `tracking.target_beyond_stop`);
- there is no evaluate-start.

The same 13 segments are evaluated.

```sh
.venv-runpod/bin/python control/wbc/compute/launch.py \
  --pilot tape2-c047-20261105 bundle --work .cache/work/wbc-tape2/c047
```

Results (records `cove50-wbc-tape2-{c047,sts3250,st3025}-20261105`). Commands
now stay inside the joint ranges and the elbows follow the reference. The C047
and STS3250 stay up on 12 of 13 segments, nominal and pushed. The overhead wave
is still lost: the C047 stays upright with its arms at shoulder height, the
STS3250 falls. The ST3025 now holds the turn but loses the arm display (forearm
height, torso upright) and the overhead wave. Its evaluation hit the 1,800 s
limit one condition short. The legs stay straight on all three. Torso travel is
0.5 of 6.8 cm in the deep squat and 0.6–0.7 cm in the crouch, hop and bounce,
and knee commands stay near 0 even when a crouch starts bent. That is a learned
preference: the upstream terms score body positions, not joint angles. Returns
peaked mid-run and eased as adaptive sampling moved to the hard segments.

## Step 13: mixed servo sizing and a joint-angle reward

`agility/allocation.py` (record `cove50-wbc-servo-allocation-20261105`) puts a
servo per joint group through the demand check. The runtime now takes such an
allocation: `dynamics.ALLOCATIONS` gives each joint its own law, duty loop,
rotor inertia, friction and mass. `tracking.ACTUATORS` and `--actuator` name
the allocation `st3025-pitch-roll-sts3250`: the ST3025 on hip pitch, hip roll,
knee and ankle pitch, the STS3250 elsewhere, 4.62 kg. The R3 audit is unchanged.
Two pilots:
- `tape2-mixed-20261105`: the tape2 recipe from the tape1 ST3025 teacher. It is
  an A/B against `tape2-st3025` with only the plant changed.
- `tape3-mixed-joint-20261105`: the same plant from the tape2 ST3025 teacher,
  plus `tracking.joint_position_tracking` (`--joint-weight 1.0`, std 0.5 rad on
  the mean squared joint-angle error). It probes whether the legs then follow
  squats and crouches.

Both pilots get a 2,700 s evaluation limit.

Results (records `cove50-wbc-tape2-mixed-20261105`,
`cove50-wbc-tape3-mixed-joint-20261105`):
- The mixed set behaves like the all-ST3025 robot on every segment.
- It has more thermal headroom: the hottest servo in the walk runs at 0.63×
  rated RMS current against 0.97×.
- The joint-angle reward holds the arm display with 0 of 32 worlds flagged
  (32 of 32 before), and leg error drops 10–20%.
- The squats still stay standing: 0.5 of 6.8 cm.

## Step 14: a training signal that pays for the legs (`T-kemqtv`)

`signal-mixed-20261106` changes five things at once, starting from the tape2
ST3025 teacher with the action head reset, on the mixed set:
- The feasible tape. `agility/feasible-selection.json` keeps 26 of 41 motions
  (179 s). It drops a motion when, on the mixed set, more than 10% of its
  frames are beyond stall or its base residual p95 exceeds 5% of the weight.
- Residual actions (`--residual-action`): target = reference + 0.5 × action,
  so a zero action performs the reference.
- The joint-angle reward.
- A 5 cm torso-height termination (`--anchor-tolerance`); end effectors keep
  10 cm.
- Half of the adaptive starts drawn by per-bin joint RMS error
  (`--error-sampling`, `tracking.ErrorAwareMotionCommand`).

There is no plateau stop.

Results (record `cove50-wbc-signal-mixed-20261106`): 16.6 M transitions and 337
PPO rounds.
- No episode leaves the reference: 0 of 832, pushed ones included. The push
  is a single 0.1–0.2 m/s kick at 1 s, so it says little about recovery.
- Squats, crouches and bounces are performed: the deep squat travels 6.2 of
  6.8 cm.
- Joint error is 1.8–5.6× lower, depending on the segment: legs 0.034–0.071
  rad, arms 0.045–0.087 rad.
- Steps lift the foot 17–19 mm (2–4 mm before). Lift is the foot body
  origin's peak above its own 5th-percentile height, not the sole.
- The walk covers 1.05 of 1.54 m (0.10 m before).
- The 1 cm hop has no flight phase.
- Hip roll reaches 1.0–1.07× rated current in the march and turn.
- Return was still rising at the deadline.

Tape3 (the joint reward alone) did not squat. That points at the residual
action as the main cause, but this run does not separate the five changes.

## Step 15: physics step (`T-fj8xtc`)

The runtime takes `physics_dt_s` (`BatchedRuntime`, `R3ActuatorCfg`,
`robot_cfg`, `tracking.env_cfg(physics_dt=...)`, `stage.py --physics-dt`). The
default stays 0.25 ms, and the R3 audit is unchanged. Delays remain whole
control ticks.

`agility/physics_step.py` (CPU MuJoCo with the same boundary) runs two probes
on C047, ST3025 and the mixed set: step responses of the knee, hip pitch,
shoulder pitch and elbow, and the deep-squat-x2 joint angles commanded open
loop on the floor. Against 0.25 ms:

| Step | Joint RMS diff (max) | Squat torso height | CPU speed-up |
|---|---|---|---|
| 1 ms | 0.003–0.004 rad (0.012) | within 0.5–2.2 mm | 3.6–4× |
| 2 ms | 0.008–0.010 rad (0.029) | within 0.6–1.0 mm | 7.2–7.6× |

At 2 ms, rise is 2–6 ms earlier and overshoot stays within 0.004 rad. The
open-loop squat falls at every step; the torso height comparison runs through
the fall. Its largest joint difference is 0.033–0.036 rad at 2 ms (0.055 rad
for the mixed set at 1 ms).

Closed loop (record `cove50-wbc-physics-step-20261106`): step 16's first job
re-ran the step-14 teacher, trained at 0.25 ms, at 2 ms on its 13 segments.
Against its 0.25 ms evaluation:
- joint error is within 0.0013 rad on every segment;
- squat travel is 6.2 cm both ways;
- the walk covers 1.046 m against 1.05 m;
- current is within about 1%;
- 0 of 832 worlds leave the reference at either step.

On the GPU, sustained training at 2,048 worlds ran about 18,000 transitions/s
at 2 ms (step 16) against 3,928 at 0.25 ms (step 14). That is 4.6×, below the
CPU probe's 7×; the per-step cost outside physics was not profiled. The
10-step random-action benchmark shows a lower 3.1× (11,758 against 3,836).
Why it is lower is not tested: a short run's overhead, or random actions
causing more contacts and resets. 3,072 and 4,096 worlds failed GPU allocation
in the benchmark on the 24 GB card. That may be memory the benchmark kept from
the 2,048-world run (3,072 failed on a 0.49 GB allocation), not a hard limit.
2 ms is adopted from step 16 on.

```sh
.venv-control/bin/python control/wbc/agility/physics_step.py \
  --tape .cache/work/wbc-signal/motions --evidence DIR
```

## Step 16: longer training runs (`T-87gbsn`)

`long1-mixed-20261106` and `long2-mixed-20261106` train the step-14 recipe at
2 ms, each for a full job (about 8,000 s deadline). long1 starts from the
step-14 teacher and long2 from long1's. Nothing else changes. Since step 14:
216 M transitions, 88,020 optimizer updates, 3 jobs and about 5.1 GPU-hours
(roughly $5.7 at the listed flex rate). The step-14 teacher itself started
from earlier tape teachers, so the lineage holds more than this.

| | Step 14 | long1 | long2 |
|---|---|---|---|
| Transitions in the job | 16.6 M | 100.9 M | 98.8 M |
| Return at the end | rising | 53.7 | 56.3 |
| Forward walk, travel of 1.54 m | 1.05 m | 1.57 m | 1.54 m |
| Walk, final position error | 66 cm | 4.7 cm | 11.9 cm |
| Deep squat, depth of 6.8 cm | 6.2 cm | 5.1 cm | 5.6 cm |
| Hop crouch, of 5.2 cm | 4.7 cm | 4.0 cm | 4.1 cm |
| Leg joint error (walk / squat) | 0.066 / 0.054 rad | 0.074 / 0.070 | 0.060 / 0.063 |
| Arm joint error (chest beat / walk) | 0.087 / 0.063 rad | 0.049 / 0.048 | 0.072 / 0.075 |
| Walk foot lift, right/left | 28/17 mm | 25/21 | 22/21 |

The return rose from 42 to 52 in long1's first 23 M transitions and then only
by 2 over the next 78 M; long2 added 5%. Budget fixed the walk (distance and
even steps) and halved arm error in long1. After that, the motion trades: in
long2 the legs improve while fast arm moves get worse, and the walk drifts
more. The hop still has no flight phase. 0 of 832 evaluation episodes leave
the reference in every job. Hip roll stays at about rated current in the
march, turn and side sway (0.99–1.10×).

Reading: the return flattens about 25 M transitions into long1, about 42 M
since step 14. That is read from the return under error-driven sampling, which
moves with what the sampler picks; the per-segment evidence only places the
flattening between 16.6 M and 117.5 M transitions. More budget re-balanced arm
against leg tracking rather than adding agility. A hypothesis, not tested
here: the reward balance between arms and legs, and the hop's demand, limit
agility more than training length does. The hop is kept on the tape at 9% of
frames beyond stall on the mixed set; step 9 found even a 1 cm flat-footed hop
needs 15 rad/s, beyond both servos. Each comparison is between end-of-job
checkpoints of one seed, so part of the arm–leg see-saw may be variance
between checkpoints.
Records: `cove50-wbc-long1-mixed-20261106`, `cove50-wbc-long2-mixed-20261106`.

## Step 17: characterful references (`T-2smahq`)

`agility/library.py --catalogue ape` adds 29 motions:
- Tempos between each agility family's kept and excluded step-14 tempos:
  walk x1.1; deep squat, deep rise and both arm sets x3; 2.5 Hz chest beat and
  wave; 3 Hz knee bounce; 0.27 s crouch; 1.5 cm hop; 0.4 s march; 1.5 Hz arm
  display; 2 Hz side sway. The 2 Hz knee bounce lies between the trained 1.5
  and 2.5 Hz.
- Five ape-character moves at several tempos:
  - a low crouch bounce: 30 mm below stand, 12 mm bounces, arms hanging forward;
  - an excited bounce with alternating chest beats;
  - a side rock with the opposite arm reaching overhead;
  - a crouch that scans the torso left and right;
  - a side shuffle: three 30 mm sideways steps, the foot on the side of travel
    stepping out and the other closing up, then three back, at 0.8, 0.6, 0.5,
    0.4 and 0.33 s per step (`gait()`, as for the turn and march). 40 mm
    steps leave the stepping leg nearly straight, and the leg IK does not
    converge at walking height.

`agility/feasible-selection-ape.json` applies the step-14 selection rule
(frames beyond stall above 10%, or base residual p95 above 5% of weight) to
the 29 motions on the mixed set. The demand comes from inverse dynamics of each
reference (`cove50-agility-allocation-ape-r2-20261106`, and
`cove50-agility-allocation-shuffle-r4-20261106` for the shuffle). 20 motions
are kept and 9 excluded. The rule reads the leg or arm group's share of frames
beyond stall; the joint that limits each excluded motion:
- the 1.5 cm hop and the 0.4 s march: legs 10.4% and 15.5%, mostly the knee;
- the 0.75 and 1 Hz side rock: arms 14.1% and 23.6%, and the 1.5 Hz arm
  display: arms 17.1%, all mostly shoulder pitch;
- the 3 Hz knee bounce: a 7% base residual (legs 4.6%, hip pitch);
- the x3 alternating arms, x3 deep rise and 2 Hz side sway: the floor cannot
  carry them (6–17% residual).
All five shuffle tempos are kept: legs 0.2–0.4% at 0.8–0.5 s, 4.4% at 0.4 s and
8.8% at 0.33 s (mostly the knee), residual 3.3–3.6%. Hip roll carries the
shuffle at 0.86–1.07× rated RMS current, the most of any joint. The turn has no
tempo above 60 deg/s, so its fastest feasible tempo is not known.

`tape.py --motions LIBRARY APE_LIBRARY` chains the 26 step-14 motions and
the 15 kept ones into a 41-motion, 270 s tape.

`ape-mixed-20261106` first evaluates the step-3 teacher (long2) on 16
segments. 11 of them are not on its tape: six faster tempos of trained
families, the 2 Hz knee bounce (between two trained tempos) and the four ape
moves.
It then trains for one job on the 41-motion tape and is evaluated again on the
same segments. The job used 96.8 M transitions, 39,380 optimizer updates and
6,589 s of worker wall (about $2).

| | Before (step-3 teacher) | After ape training |
|---|---|---|
| Episodes leaving the reference | 0 of 1,024 | 0 of 1,024 |
| Walk, final position error | 12.0 cm | 2.4 cm |
| Walk x1.1, travel of 1.53 m / final error | 1.47 m / 15.2 cm | 1.57 m / 5.4 cm |
| Ape crouch bounce, torso travel of 4.2 cm | 3.4 cm | 3.4 cm |
| Excited bounce, torso travel of 2.0 cm | 2.0 cm | 1.7 cm |
| Arm error, chest beat 2.5 Hz / excited bounce | 0.079 / 0.074 rad | 0.064 / 0.065 |
| Leg error, side rock / crouch scan | 0.056 / 0.056 rad | 0.075 / 0.070 |
| Knee offset from reference, side rock R/L | −0.08 / −0.06 rad | −0.18 / −0.10 |
| Hip roll current over rated, side rock / turn | 1.10 / 1.05 | 0.69 / 1.03 |

Before training, the controller already performs the new moves at about its
level on trained moves. That suggests the faster tempos and the ape moves lie
within what it had learned. Training on them:
- sharpens fast arm moves;
- closes the walk drift;
- straightens the right knee further while hip-roll current falls. Its median
  offset from the reference over the 16 segments grows from 0.07 to 0.11 rad.
  The change is largest where the reference legs barely move: chest beat
  0.02 → 0.14 rad, side rock 0.08 → 0.18. The left knee's median stays near
  0.06 only because its changes cancel: it straightens on standing moves
  (chest beat 0.00 → 0.08) and bends more on the walks and turn
  (0.09 → 0.05). The cause is not tested.

A bent-knee stance is part of the ape look. Our untested reading is that the
next change lies in what the reward asks of the stance (a knee-flexion term,
or a lower torso reference) more than in more motions. The hop still has no flight phase.
Record: `cove50-wbc-ape-mixed-20261106`.

### The side shuffle (`shuffle-mixed-20261106`)

The shuffle tempos join the tape: 46 motions, 319 s. The step-4 teacher is
evaluated on the five shuffles before it has seen them, then trained for 1 h
(64.2 M transitions, 26,120 optimizer updates, about $1.4). In both
evaluations, none of the 704 episodes leaves the reference.

Sideways travel is measured along the reference's own sideways axis; every
world starts at −47.7° heading, so world y would show only 0.67 of it.

| Step time | Sideways travel, before | After 1 h | Reference |
|---|---|---|---|
| 0.8 s | 5.2 cm | 8.6 cm | 12.3 cm |
| 0.5 s | 5.5 cm | 10.1 cm | 11.9 cm |
| 0.33 s | 6.1 cm | 10.9 cm | 11.2 cm |

Before training it already steps: each foot lifts six times at every tempo,
as in the reference, but it covers under half the path. The zero-shot run is
the step-4 teacher, which had not seen the move. Afterwards it covers 70–97%
of the path and ends 1.4–4.7 cm behind its start. Feet lift 10–13 mm against the
reference's 20 mm swing.
- **Joint load:** hip roll carries the shuffle at 0.95–1.08× rated, the most
  of any joint.
- **Other moves:** the extra hour also tightens the legs. Chest-beat leg error
  goes from 0.060 to 0.046 rad, and the right knee's offset on the shuffle
  shrinks from 0.14 to 0.09 rad.

Record: `cove50-wbc-shuffle-mixed-20261106`.

### The acted-video route (documented, not run)

The next source of character is acted video of a person moving like an ape:
1. Phone video, one person, whole body in view, camera still or slowly moving.
2. GVHMR (zju3dv/GVHMR) recovers world-grounded SMPL-X motion per frame,
   with gravity up and a floor.
3. GMR (YanjieZe/GMR) retargets SMPL-X to a robot by inverse kinematics.
   Cove-50 needs its own IK configuration: which human bodies map to which
   Cove-50 bodies, with scale and offsets for the shorter legs and longer
   arms.
4. The result is written as a library motion (an `npz` with the same fields
   as `library.py`). It is screened by `agility/demand.py` and the selection
   rule like every authored motion, then added to the tape.

It needs the user's capture, plus permission to download the GVHMR and GMR
checkpoints and the SMPL-X body model (registration and licence at the MPI
site).

## Step 18: servo comparison on the controller that moves (`T-jhfh6e`)

`servo-{c047,sts3250,st3025,st3025-pitch-roll-sts3250}-20261106` each continue
the step-4 teacher for 3,600 s of training on their own plant, on the 41-motion
ape tape. Each is evaluated before and after on the same 16 segments × nominal
and pushed × 32 worlds. Totals: 64–70 M transitions per set, four policies,
about 4.9 GPU-hours (roughly $5.5).

| | C047 | STS3250 | ST3025 | Mixed |
|---|---|---|---|---|
| Robot mass | 3.83 kg | 4.22 kg | 5.21 kg | 4.62 kg |
| Episodes leaving the reference, before / after adapting | 11 / 0 of 1,024 | 2 / 0 | 0 / 0 | 0 / 0 |
| Walk travel of 1.54 m (final error) | 1.74 m (22 cm) | 1.98 m (46 cm) | 1.54 m (3.3 cm) | 1.54 m (1.5 cm) |
| Hop crouch of 5.2 cm | 5.0 cm | 3.9 cm | 4.8 cm | 4.5 cm |
| Hottest leg joint on the walk (× rated) | ankle pitch 1.43 | hip and ankle pitch 0.90 | hip roll 0.71 | ankle pitch 0.70 |
| Hip roll in the turn (× rated) | 1.23 | 0.82 | 1.17 | 1.06 |
| Knee and hip pitch in the hop (× rated) | 1.23 / 1.28 | 1.11 / 1.12 | 0.67 / 0.76 | 0.68 / 0.67 |

Before adapting, the mixed-trained controller fails on the C047 in pushed deep
squats and hops and falls 0.6 m short on the walk. One hour fixes the falls,
but the C047's four loaded leg joints then run 1.1–1.4× rated on the walk (up
to 1.5× on the faster walk), and its
right foot stops lifting in the turn. The STS3250 and C047 overshoot the walk
after the hour; that may still be settling. All-ST3025 and the mixed set track
alike and stay below rated except hip roll in the turn. Hip roll is the hottest
leg joint in the turn on every set; over all segments it is the joint closest to
its rating on the ST3025 and the mixed set, while the C047's ankle pitch and the
STS3250's hip pitch and knee run hotter. For the legs, the mixed set gives up nothing measurable
against all-ST3025 at 0.59 kg less.

The bench priorities and buy list are updated in
`robot/model/dynamics/README.md`:
- ST3025 hip-roll heat first;
- then the ST3025's torque–speed line and firmware loop, because the demand
  check of the reference motions puts knee speed as the ceiling for faster
  moves (Step 19);
- then the STS3250 rotor inertia for the fast arm moves.

Record: `cove50-wbc-servo-compare-20261106`.

## Step 19: knee options and a knee stance term (`T-p8cye8`)

**Demand check first.** `agility/allocation.py` runs inverse dynamics of the reference motions. It screens 23
knee-heavy moves with a stronger knee on the mixed set:
`cove50-agility-allocation-knee-20261106`, `-knee12v-` and `-knee-geared-`. The figures are leg frames beyond
stall.

| Move | Mixed (ST3025 knee) | AK45-10 knee (24 V) | ST3025 + 1:1.5 stage (12 V) |
|---|---|---|---|
| March, 0.4 s per step | 15.5% | 2.0% | 3.5% |
| March, 0.33 s per step | 20.9% | 5.8% | 9.4% |
| Side shuffle, 0.33 s per step | 8.8% | 0% | 2.2% |
| Hop, 1 cm | 9.0% | 3.5% | 8.5% |

Other knee candidates:
- the AK60-6 adds nothing over the AK45-10 and weighs 0.24 kg more;
- the 12 V ST3215-HS (too weak) and RSBL85-12 (too slow) are worse than the ST3025.

After the knee, hip pitch and the floor limit the hops. Sidecar revisions 3–3.2 hold the sourced figures. The
stage is an assumed design, not a product.

**Three runs** continue the step-4 teacher for 3,600 s on a 50-motion tape: the 46 shuffle-tape motions plus
the 0.4 and 0.33 s marches, the 0.2 s crouch and the x4 squat. Each is evaluated on 16 segments × nominal and
pushed × 32 worlds. The runs:
- `knee-ak45-20261106`: the AK45 knee alone;
- `knee-ak45-stance-20261106`: plus `knee_tracking` (weight 0.5, 0.1 rad width);
- `knee-geared-stance-20261106`: the geared ST3025 knee with the same term.

Per run: 63–70 M transitions and about 1.3 GPU-hours.

| After 1 h | AK45 | AK45 + stance | Geared + stance |
|---|---|---|---|
| Knee offset from reference, typical | 0.08–0.25 rad straighter | within 0.03 rad | within 0.03 rad |
| Deep squat torso travel, of 6.8 cm | 5.4 cm | 6.9 cm | 6.8 cm |
| March 0.4 s foot lift R/L, reference swing 40 mm | 27 / 33 mm | 42 / 39 mm | 41 / 42 mm |
| March 0.4 s knee speed p99 R/L, reference 13.8/14.6 rad/s | 6.5 / 8.4 rad/s | 6.6 / 8.3 rad/s | 9.7 / 11.4 rad/s |
| Hop: frames with both feet up (median of 32 worlds) | 0 | 0 | 2 |
| Episodes leaving the reference, before / after | 0 / 0 of 1,024 | 0 / 0 | 2 / 3 (pushed hops, after) |

The stance term fixed what the widths had left free:
- the knees hold the reference's bend;
- the squat reaches full depth, and the crouch bounce full travel;
- march and shuffle feet reach their reference swing height.

The term started at 44% of its weight and climbed to 92%. In the same runs the old joint- and body-position terms
sat at 98–99%, and body linear velocity at 92–95%. Fast arm moves track a little worse with it: chest beat 0.058 to 0.068–0.072 rad.

The knee-drive comparison is open at one seed per run:
- **Geared knee:** runs its knee faster (0.4 s march 9.7/11.4 against 6.6/8.3 rad/s) and attempts the hop's
  flight, at 1.05× rated. 3 of 32 pushed hops trip the foot-height condition, coinciding with those attempts.
- **Caveat on the geared run:** its knee position gain was inverted to 2.25× too stiff (sidecar revision 3.2,
  corrected in 3.3), which may explain part of its speed.
- **AK45 knee:** left the reference in none of 1,024 episodes and loads its knee less in the hop. The demand
  check also favours it (hop 3.5% against 8.5%).

Without the stance term the AK45 controller uses its faster knee only partly: the left knee reaches 10.8 rad/s
on the 0.33 s march, above the ST3025's rail-scaled 8.2, against the reference's 16.9. Neither drive reaches
the reference's knee speed under a reward that does not ask for it. The marches reach full foot height anyway.
The v2 reward runs (step 20) test whether a faster knee adds value once speed is asked for.

One seed per run. Hip roll stays at 1.0–1.2× rated in marches, turns and shuffles. Reward design is now a
recorded loop: `.agents/skills/ape-control/references/reward-design.md`.
Record: `cove50-wbc-knee-20261106`.

## Step 20: reward set v2 (`T-b26dyp`)

`reward_set='v2'` (`tracking.REWARD_SET_V2`) changes the joint, body-position and body-velocity terms to
shrink-only widths that follow the population's error, as in PBHC. It also adds worst-joint, joint-velocity and
foot-position terms. Two runs continue their plant's latest v1 teacher for 3,600 s:
- `reward-v2-mixed-20261106`: the 12 V mixed set on the 46-motion tape;
- `reward-v2-ak45-20261106`: the 24 V AK45 knee on the 50-motion tape.

Each is evaluated before and after on 13 shared segments (16 for the AK45 knee), at 32 worlds per condition.
Together: 121 M transitions, about 2.5 GPU-hours.

The widths reached their floors early in training:

| Term | Width at the end (mixed) |
|---|---|
| Body position | 2 cm |
| Body velocity | 0.1 m/s |
| Joint position | 0.05 rad |
| Worst joint | 0.09 rad |
| Joint velocity | 0.68 rad/s |
| Feet | 1.1 cm |

The terms then sat at 37–73% of their weight, against v1's 92–99%.

| 12 V mixed set | Before (v1 teacher) | After 1 h of v2 |
|---|---|---|
| Leg joint error | 0.044–0.088 rad | 0.019–0.050 rad |
| Arm joint error, walk / chest beat | 0.055 / 0.059 rad | 0.013 / 0.072 rad |
| Knee offset, most moves | 0.05–0.16 rad straighter | within 0.01 rad |
| Deep squat, of 6.8 cm | 5.7 cm | 7.0 cm |
| March 0.5 s foot lift R/L, swing 40 mm | 31 / 27 mm | 45 / 43 mm |
| Right knee speed p99, 0.5 s march (reference 10.9) | 7.9 rad/s | 8.4 rad/s |
| Hop, frames with both feet up (median of 32) | 0 | 1 |
| Chest beat, foot lift R/L (reference planted) | 4 / 2 mm | 9 / 4 mm |
| Walk, final position error | 3.4 cm | 8.7 cm |
| Episodes leaving the reference | 0 of 832 | 0 of 832 |

What the widths had left free is now paid for. The legs track twice as closely, and the knees keep the
reference bend without a knee term. The squat, crouch bounce and march swing come back in full.

Two gaps remain:
- **Joint speed** is still short of the reference: knee 8.4 against 10.7–13.2 rad/s. The velocity term averages
  22 joints, so one joint's deficit is diluted.
- **On planted moves a foot now lifts**: chest beat 9 mm, excited bounce 10 mm with 4 frames of both feet up.
  Chest-beat arms track worse.

The walk ends further off the reference line.

The 24 V AK45 plant improves just as much but does not move its knee faster: knee 8.8 against 16.8 rad/s in the
0.33 s march, and no hop flight. Under this reward the 24 V knee adds no visible value.

Next reward change, one at a time:
- a worst-joint or leg velocity term;
- a contact-phase match (PBHC), which keeps planted feet down and pays for flight.

Record: `cove50-wbc-reward-v2-20261106`.

## Step 21: agility loop, cycle 1 (`T-27cyrn`)

The loop: state a hypothesis, screen it cheaply on the CPU, train only what the screen ranks high, look at the
motion, repeat. Every comparison below continues the same reward-v2 mixed teacher for one hour.

| Run | Changed | Transitions | Optimizer steps |
|---|---|---|---|
| `loop1-v3-mixed-20261106` | reward set v3 (per-joint terms, speed-weighted leg velocity, foot contact, swing height) | 58.0 M | 23,580 |
| `loop1-v3-mixed-fast-20261106` | v3, plus half the command and feedback delay (10 / 5 ms) | 58.4 M | 23,760 |
| `loop1-v3-damiao-20261106` | v3 on the 24 V `qdd-damiao-gim` plant (5.5 kg), on its 59-motion tape | 55.7 M | 22,660 |

Each run has 2,048 worlds and is evaluated on 13 shared segments (19 for Damiao) at 32 worlds per condition. The
agility lens `agility/agility_metrics.py` adds, per segment:
- peak joint speed against the reference;
- knee lag, refined between 50 Hz frames;
- foot contact agreement, swing ratio and frames with both feet up;
- torso vertical speed.

**Reward v3 against v2 (12 V mixed).**
- **Hop:** the knee now follows the hop crouch. Knee offset goes from −0.09 to −0.02/−0.04 rad. Frames with both
  feet up rise from 1 to 3.5 of the reference's 13, and leg error falls from 0.048 to 0.038 rad.
- **Planted feet:** they lift about half as much. The chest beat goes from 8.7 to 4.7 mm and the excited bounce from
  10.2 to 5.1 mm.
- **Unchanged:** peak knee speed is still short (march 8.4 and hop 9.2 rad/s, against references of 10.9 and 13.2),
  and chest-beat arm error got worse (0.072 → 0.094 rad).
- **Saturation at the end:** body position, body orientation and foot contact sit at 91–95% of their weight. The
  joint terms sit at 47–75%, so most of the remaining gradient is on them.

**Half the delay, then the lag comes back.**
- **Before retraining:** the v2 teacher on the faster plant lags 9–13 ms on the walk, hop and march (20 ms on the
  nominal plant).
- **After one hour:** it lags about 20 ms again.
- **The cause is the reward's timing.** mjlab computes the rewards after the physics step and before the reference
  advances, so it compares the robot at t + 20 ms with the reference at t. It pays for being exactly one control
  frame late, and every trained policy settles there.
- **The fix:** `reference_lead=1` (`tracking.ErrorAwareMotionCommand`) makes the actor's command and every reward
  term use the reference for the time the robot state represents. The evaluator logs the reference at that true
  time.
- **Zero-shot check:** with the lead, the v3 teacher's hop error over the first 30 steps falls from 0.030 to
  0.023 rad (CPU).

**24 V Damiao against 12 V mixed**, after the same hour:
- **Before training:** the v2 teacher, tuned to high-inertia hobby servos, shakes on the low-inertia plant. Peak
  joint speeds are 2–35× the reference and feet leave the floor in a third of the frames. One hour settles it, and
  1 of 1,216 episodes leaves the reference.
- **Hop:** the robot leaves the floor. The table is per foot (right / left), median of 32 episodes.

  | Hop 1 cm | Peak foot lift R / L | Lift-offs R / L | Both feet up (of 13) | Knee speed p99 (ref 13.2) | Knee lag |
  |---|---|---|---|---|---|
  | 12 V mixed, v3 | 4.6 / 3.5 mm | 0 / 0 | 3.5 | 9.2 rad/s | 25 ms |
  | 24 V Damiao, v3 | 13.1 / 6.8 mm | 5 / 1.5 | 5 | 11.4 rad/s | 12 ms |

- **Gait moves** (march, turn, shuffle): foot lift and leg error are the same as 12 V, at lower peak knee speed
  (march 5.5 against 8.4 rad/s). That may be smoother motion rather than slower motion; the 12 V p99 includes servo
  jitter. The run may also not have converged after its bad start.
- **Chest beat:** arms track better (0.066 against 0.094 rad), but a planted foot lifts 9.6 mm.

**Design levers screened on the CPU.** These use the force check (`agility/allocation.py`), with no training.
- **Parallel springs at knee, hip and ankle pitch:** RMS leg torque falls 18–36% (less heat in the crouched stance),
  but peaks are unchanged.
- **Ankle servo moved up to the knee:** no gain. The servos' own rotor inertia dominates, not link mass.
- **Feet 1.3× longer and wider** (sole 86 × 60 → 112 × 78 mm; `tracking.scale_feet`, `COVE50_FEET_SCALE`): with the
  motions regenerated (`cove50-agility-allocation-feet-20261106`), the 24 V set keeps 64 motions, adding the 1.5, 2
  and 4 cm hops and the 3.5 Hz bounce. The 12 V set keeps 50.
- **Reference preview** (`tracking.reference_preview`, `preview_steps`): the actor and critic see the reference 1, 2
  and 4 frames ahead. A retained teacher widens by zero input columns (`stage.widened`), and its outputs are
  unchanged. Ready, not yet trained.
- **Torso height and vertical-speed terms** (reward set v4): with the v2 teacher, torso height error is 1.2–1.7 mm
  RMS on bounces, squats and the walk, but 7.8 mm on the hop, so these terms would pay mostly for take-off. Ready,
  not yet trained.
- **Long arms and knuckle support** (the user accepts the look where it adds agility):
  - **Today's arms** reach 0.22 m from shoulder pitch to knuckle. A bent-knee knuckle stance would need 131° of hip
    flexion against the 110° limit.
  - **Arms 1.7×** (`tracking.scale_arms`; upper arm and forearm stretched, structure mass and second moments scaled,
    the C047 case kept at the elbow) give a 64° lean with hips about 97° flexed. Mass is 4.04 kg (+0.21 kg) and the
    arm's inertia about shoulder pitch rises from 35 to 118 kg cm².
  - **New pieces:** knuckle contacts in the force check (`body.knuckle_points`), arm IK in the library
    (`Body.hands`), a knuckle catalogue (drop at 0.8, 0.6 and 0.45 s, rock, sway and bounce) and a knuckle-contact
    term (reward set v3k).
  - **Force check** (`cove50-agility-allocation-knuckle-20261106`): all 11 knuckle moves stay within the motor lines,
    with at most 3.6% residual on both plants.
  - **Cost** (`cove50-agility-allocation-arms-20261106`): on 24 V, the 3 Hz overhead wave and the 2 Hz arm display
    drop out, and shoulder RMS about doubles. The 12 V set keeps all its moves.
  - **Ape-body tape** (`cove50-agility-allocation-ape-body-20261106`): the 24 V body keeps 74 motions
    (`agility/feasible-selection-ape-body-damiao.json`, 462 s).
  - **Corrected to about 1.4×.** The user judged 1.7× too long: on all fours the ape should lean forward, and the
    arm inertia would need bigger arm actuators or make the arms slow.
    - At 1.4× reach no knuckle stance fits today's joint ranges. The 35° ankle stops the hips at about 0.19 m and
      hip flexion stops at 110°, so the knuckles end 16–33 mm short of the floor.
    - Taking the same reach mostly in the upper arm costs the least inertia: shoulder 68–72 kg cm² against 74 for
      both segments and 80 for the forearm alone, and the elbow keeps today's inertia.
    - The chosen body (`arm_scale=[1.6, 1.27]`, `fist=[0.03, 0.01]`) has an upper arm 1.6× and a forearm 1.27×, plus
      a 30 mm, 10 g knuckle fist (`tracking.add_fists`, `COVE50_FIST`). It reaches 0.265 m, weighs 3.97 kg and has a
      shoulder inertia of 78 kg cm² (today 35, 1.7× arms 118).
    - Knuckle stance (`library.KNUCKLE`): 68° lean, hips about 102° flexed, knuckles 30 mm ahead of the shoulders.
      That puts about 55% of the way from heels to knuckles; knuckles directly under the shoulders put 90% toward the
      hands. The knuckle walk lifts the feet 8 mm, because in this crouch a 15 mm lift is reachable only by flipping
      the hip roll.
    - Force check (`cove50-agility-allocation-ape-body-r3-20261106`): the 24 V set keeps 81 of 86 motions with the
      same arm motors. That includes all 16 knuckle moves and the two arm swings the 1.7× arm lost, with shoulder RMS
      0.43–0.82 of rated. The 12 V set keeps 66.

**Cycle 2**, one change each from a cycle-1 teacher:
- `loop2-mixed-lead-20261106`: reference lead, 1 h;
- `loop2-damiao-feet-20261106`: feet 1.3×, 64 motions, 6,000 s;
- `loop2-ape-body-damiao-20261106`: 1.7× arms, feet and knuckles, reward v3k, 74 motions; cancelled after 42 minutes
  when the user rejected 1.7×;
- `loop2-ape-body-r3-damiao-20261106`: the corrected body (upper arm 1.6×, forearm 1.27×, fist) with knuckle walks,
  reward v3k, 81 motions, 6,000 s.

Records: Idle `T-27cyrn`.

## Step 22: agility loop, cycle 2 (`T-27cyrn`)

**Reference lead** (`loop2-mixed-lead-20261106`). This is the 12 V mixed plant with `reference_lead=1`, from the
cycle-1 v3 teacher, on the same tape for the same hour. It is compared with `loop1-v3-mixed-20261106`; nothing left
the reference in either run (0 of 832 episodes).

| | v3, upstream timing | v3, reference lead |
|---|---|---|
| Knee lag, walk / hop / march / 0.33 s shuffle | 22 / 25 / 19 / 25 ms | 1 / 6 / −2 / 5 ms |
| Leg joint error, walk / hop / march / 0.33 s shuffle | 0.026 / 0.038 / 0.020 / 0.023 rad | 0.017 / 0.029 / 0.014 / 0.015 rad |
| Walk final position error | 9.8 cm | 6.0 cm |
| Excited bounce: planted-foot lift; frames with both feet up (reference 0) | 5.1 mm; 3.5 | 3.6 mm; 0 |
| Chest beat: planted-foot lift; frames with both feet up (reference 0) | 4.7 / 2.6 mm; 0 | 6.0 / 8.5 mm; 18.5 |
| Peak knee speed, march / hop (reference 10.9 / 13.2) | 8.4 / 9.2 rad/s | 8.3 / 9.3 rad/s |

- **Timing:** with the lead the policy moves on time. Most of the leg error it removes was timing error.
- **Regression:** the chest beat now lifts its feet.
- **Unchanged:** the 12 V plant's peak knee speed.
- **Decision:** the lead becomes the default for later runs (Idle decision on `T-27cyrn`).

**Feet 1.3× on the 24 V plant** (`loop2-damiao-feet-20261106`). The cycle-1 Damiao teacher trained 5,439 s on the
64-motion feet tape, which adds 1.5, 2 and 4 cm hops. No episode left the reference (0 of 1,472).
- **Before training on them:** the cycle-1 teacher already hops the new references, with more frames with both feet
  up than the reference on the 4 cm hop (35.5 against 27).
- **After training, the hops are real but capped.**

  | Hop | Peak foot lift R / L | Lift-offs R / L | Both feet up, robot / reference | Torso travel, robot / reference |
  |---|---|---|---|---|
  | 2 cm | 15.8 / 15.9 mm | 4 / 4.5 | 10.5 / 19 | 5.2 / 5.1 cm |
  | 4 cm | 17.2 / 18.2 mm | 6 / 7 | 17 / 27 | 5.2 / 7.0 cm |

- **Fewer flight frames than at the start:** the 1.5 cm hop went from 11 to 5 frames, the 2 cm from 16 to 10.5.
  Training trades flight for safety. The smallest hop got smaller: peak foot lift 13.1 / 6.8 mm in cycle 1 against
  5.2 / 5.7 mm now.
- **Planted moves lift a foot:** 18 mm in the chest beat and 16 mm in the excited bounce, with 10.5 and 13 frames of
  both feet up against a planted reference.
- **Gait:** walk drift is 17.9 cm against 28.1 cm in cycle 1. The march and shuffle track as before.

Next: reward v4 (torso height and vertical speed) with the reference lead, from this teacher
(`loop3-damiao-feet-v4-20261106`).

**Reference preview** (`loop3-mixed-preview-20261106`). The 12 V lead teacher with the reference 1, 2 and 4 frames
ahead added to the actor and critic (widened by zero columns), trained 3,319 s more on the same tape. Against the
lead run:
- leg error is 5–20% lower (hop 0.029 → 0.023 rad);
- the chest beat's both-feet-up frames fall from 18.5 to 4;
- walk drift is worse (6.0 → 10.7 cm);
- peak knee speed is unchanged (march ×0.78, hop ×0.78).

The extra hour of training is not separated from the preview. On the 12 V plant neither reward, timing nor
anticipation raised peak knee speed. That points to the servo itself; the 24 V plant reaches the hop's knee speed.

**The ape body** (`loop2-ape-body-r3-damiao-20261106`). The 24 V plant with feet 1.3×, upper arm 1.6×, forearm 1.27×
and a 30 mm fist, trained 5,438 s on the 81-motion tape with reward v3k, from the cycle-1 Damiao teacher. No episode
left the reference (0 of 1,408). Knuckle measures (`knuckle_stats`, medians of 8 worlds, nominal):

| Knuckle move | Contact agreement | Knuckle down when the reference's is | Torso height error | Leg error | Travel, robot / reference |
|---|---|---|---|---|---|
| Drop 0.6 s, before training | 0.98 | 0.98 | 4.4 mm | 0.031 rad | 8.6 / 0 cm (drift) |
| Drop 0.6 s, after | 0.97 | 0.97 | 2.1 mm | 0.020 rad | 1.5 / 0 cm |
| Rock, sway, bounce, after | 0.99 | 0.99 | 2.0–2.3 mm | 0.019–0.020 rad | about 1.5 / 0 cm |
| Walk 1 s cycle, after | 0.96 | 0.95 | 2.1 mm | 0.021 rad | 1.6 / 7.5 cm |

- **Before training:** the teacher had never seen a knuckle move, yet put its knuckles down where the reference does.
  Training removed the drift and halved the torso error.
- **The knuckle walk steps in place.** Its limbs follow the sequence, but the body travels a fifth of the reference.
- **Hops are better than on the feet-only body.** On the 2 cm hop: peak foot lift 21.9 / 14.7 mm, 7 / 6 lift-offs, 13.5
  of 19 frames with both feet up, and knee speed 17.2 against the reference's 16.5 rad/s.
- **Arms:** the 2 Hz arm display tracks a little worse (arm error 0.044 against 0.035 rad), and its shoulder pitch runs
  at 1.14× rated RMS.
- **Still open:** the chest beat still lifts a foot (9–13 mm), and walk drift stays at 19 cm.

Why the walk does not travel: every body term is relative to the robot's own torso, so only the global root term pays
for travel, and its fixed 11.4 cm width scores stepping in place at 0.87 against 0.94–0.98 for still moves. Reward v5
puts that term on a shrink-only adaptive width (11.4 → 3 cm floor); at 3 cm the same traces score 0.36 against
0.77–0.80.

**Reward v4 with the reference lead on the feet plant** (`loop3-damiao-feet-v4-20261106`). The feet teacher trained
3,322 s more, against `loop2-damiao-feet-20261106`. No episode left the reference in either (0 of 1,472).

| | Feet, v3 | Feet, v4 with lead |
|---|---|---|
| Both feet up, hop 1 / 1.5 / 2 / 4 cm (reference 13 / 15 / 19 / 27) | 3 / 5 / 10.5 / 17 | 6 / 11 / 19.5 / 21.5 |
| Lift-offs on the 2 cm hop, R / L | 4 / 4.5 | 7 / 7 |
| Leg / arm joint error on the hops | 0.057–0.065 / 0.037–0.041 rad | 0.090–0.095 / 0.071–0.073 rad |
| Torso travel on the 4 cm hop (reference 7.0) | 5.2 cm | 5.1 cm |
| Excited bounce: planted-foot lift; frames with both feet up | 16.2 / 8.4 mm; 13 | 3.1 / 1.4 mm; 0 |
| Chest beat: planted-foot lift; frames with both feet up | 18.0 / 9.2 mm; 10.5 | 13.7 / 8.2 mm; 11 |
| Walk final position error | 17.9 cm | 22.5 cm |

- **Hops:** the robot now spends about as much time in the air as the reference on the 2 cm hop. Its hopping pose
  tracks worse, and the torso still does not rise as high.
- **Planted moves:** the excited bounce keeps its feet down.
- **One run:** the lead and v4 changed together here. The ape-body pair below separates v4's torso terms.

**Adaptive global root on the ape body** (`loop3-ape-root-20261106`). The ape-body teacher with the reference lead
and reward v3rk (v3k plus the adaptive global root term), 1 h, against `loop2-ape-body-r3-damiao-20261106`. No episode
left the reference (0 of 1,408).
- **Knuckle moves track more closely.** Torso height error is 1.2–1.8 mm (was 2.0–2.3), arm error 0.026–0.029 rad
  (was 0.031–0.035), and contact agreement 0.99.
- **Other moves:**
  - the 2 Hz arm display's arm error fell from 0.044 to 0.028 rad;
  - the excited bounce's frames with both feet up fell from 14 to 2;
  - chest-beat foot lift fell from 9–13 to 7–10 mm;
  - walk drift fell from 19.1 to 16.9 cm.
- **The knuckle walk still does not travel:** 1.4 of 7.5 cm. Per limb, the right foot and hand move forward 0.3 cm and
  the left ones 3–4 cm, against the reference's 7.5; the right foot lifts about 3 mm.
- **The adaptive root width settled at 5.1 cm**, held up by the forward walk's drift.
- **The cause was the reference.** Its feet lifted only 8 mm, which is exactly the swing term's 8 mm threshold and
  only 3 mm above the contact band, so no term saw the swing.

Fix:
- the knuckle walk now lifts the feet 12 mm and strides 30 mm (`cove50-agility-allocation-knuckle-r4-20261106`: at
  most 1.9% residual, hips 0.74–0.80 rated);
- the swing term now turns on at 5 mm;
- the ape-body tape is rebuilt as `.cache/work/wbc-ape-body-r4-damiao`.

A 15 mm foot lift is not reachable in this crouch: the IK flips the hip roll, and higher hips leave the knuckles short
of the floor, even with a 40 mm fist.

**Full v5 on the ape body** (`loop3-ape-v5-20261106`, v5k: root plus torso terms) against the root-only run. Same
teacher, lead, tape and hour; no episode left the reference in either (0 of 1,408).

| | Root only (v3rk) | Root and torso (v5k) |
|---|---|---|
| Hop 2 cm: frames with both feet up (reference 19); leg error | 12; 0.055 rad | 19; 0.045 rad |
| Hop 1 cm: frames with both feet up (reference 13); leg error | 7.5; 0.058 rad | 9; 0.042 rad |
| Knuckle moves: torso height error | 1.2–1.8 mm | 0.8–1.5 mm |
| Knuckle walk travel (reference 7.5) | 1.4 cm | 2.6 cm |
| Forward walk final position error | 16.9 cm | 33.8 cm |

- **On this body the torso terms help:** full hop flight with closer leg tracking, unlike on the feet body, plus closer
  knuckle moves.
- **But the forward walk drifts twice as far.** The feet body showed the same direction (17.9 → 22.5 cm).

**Cycle 4, the long runs.** Both start from the v5 teacher on the r4 tape with the lead, for about 6,600 s:
- `loop4-ape-v5-20261106`: v5k as tested;
- `loop4-ape-v6-20261106`: v6k, which doubles the global root weight (0.5 → 1.0) against the walk drift.

They differ only in that weight.

## Step 23: review and reset (2026-11-07)

`PLAN.md` reviews steps 1–22. It lists the benchmark that was actually chased (a self-built tempo ladder of 86
motions with yes/no selection gates), what each line of work cost and how much it added, and every motion, metric and
gate that was called agility. It then sets out the plan.

- **The target:** the owner-approved `playful-ape-t1` in `infra/config/targets.json`. It has 59 motions: lively standing
  play and gait, plus 16 all-fours moves on the long-arm body. Hops and the fastest tempos are stress tests only.
- **The guards:** `infra/storage/spend_guard.py` writes down, before every GPU job, why it is worth running, with notes on
  anything off that target (a reflection, never a refusal); `infra/storage/score_guard.py` refuses yes/no results at publish time.
- **The first result on the target:** `agility/target_check.py` (run
  `cove50-target-check-playful-ape-t1-317acd9906a0`). The 12 V mixed set carries almost all of it; hip roll is its
  one weak joint. The 24 V set costs about the same, weighs 0.9 kg more and runs no cooler.
