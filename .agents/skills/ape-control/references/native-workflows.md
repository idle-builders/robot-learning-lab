# Native ape control entrypoints

For work on the selected assembled robot, first resolve
`robot/model/current-model.json` using the
[selected-model reference](../../ape-body-transfer/references/selected-model.md).
The current package is `cove50_description`; `rigid_URDF` and `shell_hidden_URDF`
select the corrected exports. Native dynamics use the separately referenced
MJCF and actuator/sensor/timing contracts. The saved-policy commands below
retain their historical source bodies; a new body override requires explicit
observation/action/joint adaptation and matched replay.

Paths below are relative to the repository root. These are existing interfaces inspected on 6 October 2026, not a new training framework.

## Runtime boundary

`control/common/mjlab_ape/pyproject.toml` requires Python 3.12, `mjlab==1.3.0`, `mjlab-microduck` and PyYAML. The entrypoint `ape_walk = ape_walk.task` registers tasks with MJLab. The package metadata alone does not pin the Pollen checkout or Robomimic; recover the chosen run's dependency/source record and resolved configs before reproducing it. Existing setup/replay wrappers under `experiments/compact_morphology/ape_design/` (commit 6fbb278e4) are references for worker layout, not authority to allocate a pod or replay their hard-coded paths blindly.

The native evaluators instantiate `device="cuda:0"`. They need a compatible NVIDIA runtime, installed `ape_walk` package and the relevant upstream dependencies; the local Mac `.venv`, `.venv-learning`, and `.venv-microduck` inspected for this skill do **not** contain MJLab or Robomimic. Do not promise that selecting one of those interpreters makes a native replay local. Prefer inspecting source for CLI arguments when top-level imports prevent `--help`; do not install GPU packages or start remote compute just to validate a skill.

Selected checkpoints may be symlinks into the machine-local SHA-256 store. Use `infra.storage.artifact_store.validate_file` plus the registry hash before loading, with the repository's `.venv/bin/python` (Python 3.12). Older system Python lacks the `hashlib.file_digest` used by the validator. `skills/ape/README.md` contains the field-based validation command; `docs/design/artifact-storage.md` describes current archival and restore paths. The former Runpod volume IDs in the registry are provenance, not live recovery locations. The separate `skills/ape/boot_cad_mass_forward.json` policy is local only until a verified remote copy is made.

## Saved locomotion replay

In a prepared GPU environment, with a new `artifacts/control-replay-<run>` directory, the historical 20-axis forward actor mapped onto the 22-axis body uses:

```sh
python control/common/mjlab_ape/evaluate.py \
  artifacts/policies/checkpoints/forward_walk_stop.pt \
  --task Mjlab-Velocity-Flat-Ape-Elbows --legacy-policy-on-elbow \
  --velocity-x 0.35 --velocity-y 0 --angular-z 0 \
  --worlds 1 --seed 142 --stop-at-s 4 \
  --out artifacts/control-replay-<run>/forward.json \
  --video artifacts/control-replay-<run>/forward.mp4 \
  --state-out artifacts/control-replay-<run>/forward.npz
```

Replace `<run>` before execution. Set all three commands when testing a single axis; otherwise omitted values are derived from task command ranges. The evaluator rejects an all-zero initial command; stopping is exercised with `--stop-at-s` in `(0, 8)`. `--stop-policy-checkpoint` requires that stop flag and the legacy mapping. A second actor is not automatically a valid handoff.

`--body-xml`, `--physics-model candidate`, `--physics-dt`, `--motor-rail-v` and `--command-delay-ms` are transfer/sensitivity controls. `--body-xml` and candidate physics are mutually exclusive; candidate delay and motor behavior are assumptions unless measured. Route such changes through `ape-body-transfer`. Do not apply `--legacy-policy-on-elbow` to a native 22-axis actor.

### Native 22-axis CAD-v8 checkpoint

For registry id `cad_v8_forward_walk`, use its explicit fields: `local_checkpoint` is `artifacts/policies/checkpoints/cad_casedynamics_forward_499.pt`, its body override is `control/common/ape_cad/ape_spaced_casedynamics.xml`, and its training config `local_path` fields are `skills/ape/configs/cad_casedynamics_forward/env.yaml` and `skills/ape/configs/cad_casedynamics_forward/agent.yaml`. Verify each against its registry hash and inspect those configs before replay. The evaluator loads the registered task's current environment/RL config rather than automatically loading the shelf YAML files, so resolve any consequential mismatch explicitly.

This source-verified command uses the registered native 22-axis task and the exact body override, with **no legacy adapter**:

```sh
python control/common/mjlab_ape/evaluate.py \
  artifacts/policies/checkpoints/cad_casedynamics_forward_499.pt \
  --task Mjlab-Velocity-Flat-Ape-Elbows-CadV8 \
  --body-xml control/common/ape_cad/ape_spaced_casedynamics.xml \
  --velocity-x 0.35 --velocity-y 0 --angular-z 0 \
  --worlds 1 --seed 142 --physics-dt 0.002 \
  --out artifacts/control-replay-<run>/cad-v8.json \
  --video artifacts/control-replay-<run>/cad-v8.mp4 \
  --state-out artifacts/control-replay-<run>/cad-v8.npz
```

Replace `<run>` and prepare fresh output paths. This remains the historical small-body **case-envelope scaffold**, not the selected 50 cm assembly. Its warm-start ancestry includes a mapped 20-axis actor, but the retained actor itself is 22-axis.

At this skill's audit, this entry's `evidence` path `docs/research/ape-physics-retrain-20260930.md` was missing. Report that evidence as unavailable; hashing the checkpoint or quoting the registry's 8/8 claim does not independently verify retraining performance. The existing `docs/research/mechanical-design.md` supports the earlier unadapted v6/v7/v8 comparison and need for retraining, not the later retrained 8/8 claim. Use it for that baseline, recover the missing retraining provenance or run a fresh native evaluation, and keep those evidence levels distinct.

## Training and step replay

The following parsers call existing upstream trainers; inspect the selected task/config before running. Defaults can be substantial GPU jobs.

| Operation | Existing entrypoint and relevant arguments |
| --- | --- |
| Train on an explicit exported body | `train_boot.py --body-xml <body.xml> --worlds <N> --updates <N> --run-name <name>`; optional `--resume-run <run> --resume-checkpoint <file>` |
| Train a foot/step tracker | `train_foot_transition.py --worlds <N> --updates <N>`; select the actual curriculum with `--residual`, `--right`, or `--step-side left/right` as appropriate; inspect combinations in code |
| Train an absolute-action distilled step | `train_step_distill.py --side left/right --worlds <N> --updates <N> --bc-epochs <N>`; optional resume arguments |
| Replay distilled step | `evaluate_foot_transition.py <checkpoint> --distilled-side left/right --seed <N> --out <json> --video <mp4> --qpos-out <npz>` |
| Replay reference step | Same evaluator with `--step-side left/right`; this is a different task/interface from `--distilled-side` |

All filenames in the table are under `control/common/mjlab_ape/`. The distilled actor still takes a phase clock and was taught a fixed landing distance; exporting it without motion-file inputs does not establish variable-goal stepping. `train_boot.py` sets 2 ms physics and decimation 10 (50 Hz control), with PPO save intervals of 50 updates. A five-update smoke checks plumbing, not learning.

## Demonstration learning and uninterrupted composition

`collect_jaw_demos.py`, `collect_place_demos.py`, `collect_place_from_acquire.py` and `collect_place_dagger.py` collect distinct reset/handoff cohorts; read the selected collector and corresponding research note before reusing a dataset. The last collector accepts `--source`, `--checkpoint`, `--out`, `--summary`, `--first-seed` and `--episodes`. Its teacher remains time-indexed.

`train_jaw_bc.py --dataset <hdf5> --output <new-dir> --epochs <N> --save-every <N> --append-phase --feedforward --deterministic` trains the retained style of phase-conditioned feedforward MSE BC. Add `--acquire` or `--place` for the lesson and match its collected task data. Those flags name/configure the experiment; they do not turn incompatible data into a new skill. Native rollout inside the trainer is disabled: separate physical evaluation is essential.

For the retained narrow composed pair, use a prepared GPU environment and fresh output paths:

```sh
python control/common/mjlab_ape/evaluate_acquire_place.py \
  --acquire-checkpoint artifacts/policies/checkpoints/open_jaw_acquire.pt \
  --place-checkpoint artifacts/policies/checkpoints/place_after_acquire.pt \
  --first-seed 1142 --episodes 5 --nominal-friction --full-body-contact \
  --out artifacts/control-replay-<run>/handoff.json \
  --video artifacts/control-replay-<run>/handoff.mp4 \
  --state-out artifacts/control-replay-<run>/handoff.npz
```

This is a six-second fixed-target historical-body replay with a switch at three seconds, not autonomous pickup from an arbitrary position. `--nominal-friction` deliberately removes startup friction draws; preserve/report that distinction. `--goal-x`, `--goal-y`, `--goal-scale`, `--goal-xy-only` and `--explicit-goal` need a checkpoint trained with the matching goal contract. A CLI flag does not make a fixed-target actor goal-conditioned.

## Local contact audit

The following help command was actually run successfully with the root `.venv`:

```sh
.venv/bin/python control/common/mjlab_ape/audit_self_contacts.py --help
```

It accepts `--motion <npz> --model <xml> [--model <other.xml>] --out <json>`. The NPZ must contain `qpos` and `dt`; pose width must equal each compiled model's `nq`, and joint order must correspond. It calls MuJoCo forward on saved poses. That measures penetration and joint-range excursions, **not** fresh contact dynamics or policy stability. The compiled-model collision audit supersedes old code comments claiming the historical `CollisionCfg` erased every limb: on audited v3/v5 imports the flag change restored only pelvis contact. Inspect actual compiled masks when interpreting a new body.
