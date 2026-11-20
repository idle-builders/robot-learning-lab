# Selected R3 forward-policy retry

Idle `T-gz6n4d`. The selected pointer is resolved from
`robot/model/current-model.json`. No mechanical, motor, mass or travel
change is made here. Historical Direct Frame checkpoints retain their source
body and claims.

`policy/adapter.py` keeps all 72 original observation fields and 21 raw action
outputs, including both removed gripper histories. It maps 19 common axes by
the existing reviewed name/sign table; R/L source labels follow the proper CAD
rotation. Three new heads command neck pitch and both ankle rolls. Nine new
observations expose their positions, velocities and previous raw outputs.
Added input columns and mean heads initialize to zero, preserving the old
21-output actor exactly at initialization. These neutral-axis mappings do not
preserve complete poses through the different serial joint chains.

The authoritative task is the retained original forward run's resolved
`artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml`, rather than
the subsequently changed task factory alone. `task/config.py` reconstructs its
12 reward terms, numeric parameters/weights, command distribution, reset noise
and action-rate curriculum from data without executing YAML Python object tags.
Selectors map pelvis to torso, soles to the current sole frames and old distal
hand contact to current distal forearm contact. Source ankle-target clips are
retained before the independent R3 runtime clipping. Action-rate cost still
covers exactly the original 21 raw channels. Joint-limit and torque costs use
the current body's actual joints/motors. Old-body mass/friction/effort/PD
randomizers are not ported into a different motor model silently; this first
run uses nominal R3 dynamics and the saved reset variation.

The source BC checkpoint SHA is
`0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f`.
Its critic is tensor-identical to the original PPO forward checkpoint
`artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt` (SHA
`1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2`).
Warm-start loads both the saved actor and value network, including both learned
normalizers. The critic keeps its original 87-feature prefix and appends nine
features with zero initial first-layer weights. Its privileged input uses native
joint/gyro/gravity truth, the velocity of the original virtual IMU point 65 mm
above the root, raw action history, commands and the original foot feature
order. Actor observations still use the delayed sensor packet. Signed log1p
contact forces retain their original units/transform.

The initial cross-body seed uses fresh Adam moments with the inherited effective
learning rate `3.375e-5`, betas `(0.9,0.999)` and epsilon `1e-8`; the saved YAML's
initial `0.001` is provenance rather than the continuation rate. The original
adaptive KL schedule remains active. Source curriculum counter 20400 is explicit
(initial action-rate cost -0.4); new PPO completed-round count starts at zero.
The BC's 800 optimizer steps and inherited `iter=844` remain provenance.
Compatible R3 checkpoints restore actor, value, normalizers, Adam moments,
effective learning rate and curriculum. Earlier actor-only pilot checkpoints
lack this contract and are retained as historical plumbing evidence.

R3 physics runs at 0.25 ms with 80 native substeps per 20 ms policy action.
`runtime/` owns the exact assumed motor/command queue/delayed sensor boundary.
The task copies source R3 solver/friction-cone/gravity settings into MJLab's
scene parent explicitly. The first 20 ms of each reset holds the mapped home
while waiting for a real delayed sensor packet; that interval and its sampled
raw output remain in the episode and transition accounting. Initial missing
packets are retained as invalid data, never replaced with instantaneous truth.

## Commands

Local tensor preparation (zero physics or optimizer steps):

```sh
.venv-control/bin/python control/skills/walker/policy/prepare.py
```

Local interface smoke (two control transitions, no fit):

```sh
.venv-control/bin/python control/skills/walker/task/run.py \
  --mode smoke --device cpu --worlds 1 --steps 2 \
  --out artifacts/cove50-walk-resume/local-interface-NEW
```

On the lead-verified GPU worker, replay the unadapted mapped actor for the full
eight-second horizon, including motion after the original physical termination
conditions become true:

```sh
python control/skills/walker/task/run.py \
  --mode replay --device cuda:0 --worlds 1 --seed 136101 --velocity-x .30 \
  --out artifacts/cove50-walk-resume/r3-unadapted-seed136101
```

Measure whole-task capacity and wall time before choosing training world count.
The bounded worker recipe is
`control/common/compute/recipe.json`: 256/512/1024-world
capacity with full episode retention and zero optimizer updates; paired fresh
eight-second before/after starts 136101 and 136201; measured allocation of
200–400 PPO rounds in 100-round continuation chunks. Its 3420-second local
deadline reserves final comparisons and durable, hash-verified stage uploads
inside the lead's 3600-second provider expiry. Actual first-chunk time adjusts
the remaining budget. Planned allocation and actual completed counts stay
separate, including partial candidates and interrupted native stages.

Direct commands for one measured 512-world chunk and its replay are:

```sh
python control/skills/walker/task/run.py \
  --mode train --device cuda:0 --worlds 512 --updates 100 --seed 136200 \
  --out artifacts/cove50-walk-resume/r3-ppo-chunk100
python control/skills/walker/task/run.py \
  --mode replay --device cuda:0 --worlds 1 --seed 136101 --velocity-x .30 \
  --checkpoint artifacts/cove50-walk-resume/r3-ppo-chunk100/candidate.pt \
  --out artifacts/cove50-walk-resume/r3-adapted-seed136101
```

Use `--checkpoint` for a compatible R3 continuation. `--source-checkpoint`
accepts an original 72/87 actor/value pair and performs the initial expansion;
omitting both selects the saved BC actor with inherited critic. `--fresh` uses
new random actor and value networks with this same task/body. Do not infer walking
quality from this launcher or static weight parity. A two-round pilot is not
an assertion of adequate training. Retain each full/terminated/censored episode
and compare whole fresh starts, severe trajectories, continuous displacement,
velocity errors, contact, slip and motor effort. `task/record.py` stores every
control transition at 50 Hz, including terminal transitions before resets; it
does not claim 4 kHz full-state logging.

Interface/provenance audits are in
`data/forward-urdf-retry-20261026/critic-transfer/`. They include random nonzero
actor/value prefix parity, native privileged feature reconstruction, the source
IMU velocity identity, a two-control CPUWarp smoke, portable input closure and a
synthetic Adam checkpoint round-trip. The synthetic fixture performed zero
optimizer steps; its arbitrary restored counters are serialization evidence.

The 26 October R2 native run completed 200 PPO rounds on 512 worlds, with
2,457,600 control transitions, 196,608,000 physics steps and 4,000 optimizer
updates. Both actor and value weights/normalizers adapted; the second chunk
restored Adam moments and curriculum state. The final checkpoint is
`artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/candidate.pt`
(SHA256 `467b12a17a71f06d951294b3d9185503ea78c6edda431ae7e7870aa648fc02bd`).
The two paired eight-second starts change from backward falls at 1.42/0.88 s
to repeated supported steps, moving 1.329/1.277 m forward. Forward RMSE is
0.133/0.126 m/s against the 0.30 m/s command, with 0.433/0.575 m lateral drift.
See `data/forward-urdf-retry-20261026/critic-transfer/paired-200-r2.json` for
graded gait/contact/motor interpretation, limitations and exact trace hashes.
This is short flat-floor walking evidence; perturbation recovery, longer
exposure and physical transfer remain unmeasured.

Use a fresh `--out` directory each time. Native reports/traces and original
failures are retained evidence. Mapped seeds and rendered media are disposable
derived outputs once unused; retain exact source checkpoint/configuration,
generation command and verified regeneration record. Publish reviewed motion
through `infra/dashboard/results_hub.py`, with selected videos uploaded to the existing
private bucket; do not create per-run presentation pages.
