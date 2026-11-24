---
name: ape-control
description: Train, replay, evaluate or compose the ape robot's low-level MJLab PPO and Robomimic policies. Use for locomotion, balance, arms, steps, grasp/place and policy data interfaces; route perception and physical-body changes to their own ape skills.
---

# Ape control learning

## Owner rule: agility across all motions

The score is agility across all motions (owner, 2026-11-08; `AGENTS.md`). The target and tape motions are made-up stand-ins, and the final set is not known. Any single motion, or single family, is irrelevant on its own. Never plan, tune or judge a reward, a run or a controller around one. Read results as distributions over all motions, including motions not trained on.

## Owner rule: no simulation or smoke tests on the local CPU

Never run training, evaluation rollouts, physics smoke tests or simulation probes on the laptop CPU (owner, 2026-11-12). Run them on the GPU; when a smoke test is needed, make it a short GPU job with its spend reflection. Locally, write code, read evidence, and analyse or render traces that already exist. Older "local CPU" screens in the run history are provenance only.

## User rule: no background Runpod watchdogs

Never create, install, enable, or relaunch background Runpod watchdogs or provider-polling services. This forbids launchd, cron, systemd, detached shell polling loops, and persistent retry timers. Historical watchdog scripts are evidence only; do not run or reinstall them. Use provider-side expiry for bounded compute and explicit, bounded foreground checks for status. Training and file-transfer processes are not watchdogs.

Stop provider requests on HTTP 429 / `rate_limited`. Honor `Retry-After` or the stated reset window; a daily-window limit must not trigger automatic retries. Do not change networks, credentials, accounts, or API routes to evade a rate limit. Stop querying after a job is terminal, deleted, or retired. Do not run multiple monitors for one resource. Inspect `launchctl list` and live processes during cleanup; deleting a script alone does not unload a service.

For the selected assembled robot, resolve `robot/model/current-model.json` and read the [current URDF and native-model reference](../ape-body-transfer/references/selected-model.md). It identifies `cove50_description`, the corrected motor visuals, shell-free view and exact actuator/sensor contracts. Historical policy body overrides remain their original sources.

Work from the `robot-learning-lab` root identified by `AGENTS.md`. The target is the **50 cm ape**. Most retained policies were trained on earlier roughly 35 cm bodies; a checkpoint name or successful old replay establishes no transfer to the selected 50 cm assembly. `skills/ape/` is a **robot-policy shelf**, separate from these Codex workflow skills.

Use `ape-experiments` for the active objective and Idle experiment planning, `ape-body-transfer` when changing morphology, actuators or contacts, and `ape-perception` for sensed targets and belief health. Use `train-arena-groot` only for the explicit historical Berkeley/G1 reference paths. Do not start from GR00T merely because the request mentions robot learning.

Use the existing private Infomaniak blob archive for retained large completed rollouts, full episode traces, datasets and checkpoint bundles; keep the current learner/replay inputs and compact registry/config/hash records local. Read the [retention and restore guide](../../../docs/design/artifact-storage.md) before finalizing large outputs or reusing cold runs. Restore the exact checkpoint/body/mesh/data dependencies before replay or worker export, verify recovery before eviction, and keep generated bulk out of Git. Local `.artifact-store` deduplication alone does not archive to blob storage.

## Evaluate without pass/fail

Follow the [evaluation principles](../../../docs/design/evaluation-principles.md): collect native-unit measurements with conditions and distributions, describe in words how the motion behaves and where it degrades, and grade each relevant dimension (for example motion quality, contact and support, motor duty, robustness across conditions, recovery, interface completeness) from 0 to 5 for distance to a viable skill, with confidence and trend. When evidence shows a gap, ask whether the policy or the requirement (a reference value, gate or duty limit) is wrong, record both, and put proposed requirement changes to the user. The run history below quotes frozen gates, pass counts and GO/NO-GO reviews; they are provenance to re-read as evidence, not decisions.

## Reward design is a design loop

Most APE0 WBC setbacks so far were reward failures: wide widths that make the remaining error free, joint means
that hide one straight knee, no joint-velocity or foot term, and terminations that make flight risky. When a policy
survives but avoids what its reference asks for, suspect the reward before the body, actuators, simulator or
budget. Do not recommend hardware changes until a reward that asks for the behaviour shows the hardware is the limit.

Before every paid fit:
- read each term's logged fraction of its weight (above about 0.9 means no gradient);
- push degraded copies of the reference through the reward (straighter knees, lower feet, half speed, no flight);
- check each width against the current error;
- check that every quality of the reference has a term that sees it;
- price regularisers and terminations against the dynamics wanted.

Then iterate one change per run, logging per-term values, adaptive widths and native-unit motion measurements.
Keep the change only when the motion moves toward the reference. The checks, the loop, the terms in `tracking.py`
and the primary sources are in [reward design](references/reward-design.md).

## Select a supported starting point

Read `control/skills/registry.json` (or `control/wbc/registry.json`), `control/skills/profiles.json` and the selected entry's evidence, body override and resolved training configuration. `docs/research/low-level-control.md` explains the main failures; it describes the historical body. For commands and runtime requirements, read [references/native-workflows.md](references/native-workflows.md). Read [references/interfaces-and-gates.md](references/interfaces-and-gates.md) for demonstrations, command changes, or skill composition.

Before fitting a basic skill, also audit current-body saved artifacts by checkpoint SHA and **entry state**. A learned actor that succeeds from a raised-foot reset may already be reusable as a hold component yet provide no upright-to-release transition; [the Direct Frame support audit](../../../artifacts/ape-low-level-training-20261015/support-foot-release-next/INDEPENDENT-CHALLENGE.md) found both exact-hash 2.98 s hold actors locally. Reuse an existing component rather than retrain it. The later V19 upright foot-release pilot used 49,152 PPO transitions and passed 0/4 fresh clean releases; its reference only shifted load and its abrupt learned lift produced falls. V19 already exposed side and time to the actor. A later [static bridge-shape screen](../../../artifacts/ape-low-level-training-20261015/support-bridge-static-shape/assessment.json) passed 0/6 combined diagnostic stations per side under local MuJoCo, with no native visits or updates. The subsequent [four-visit assisted native feasibility screen](../../../artifacts/ape-low-level-training-20261015/support-assist-feasibility/report.json) achieved 0/2 assisted half-second clean releases with zero optimizer updates; one side never armed its load latch, and the other lifted briefly but fell. That assisted path was stopped at its frozen historical gate. A future support cycle needs a changed native-feasible release path before more paid updates. Rotate compute across missing primitives instead of repeatedly polishing one failed branch.

When choosing a learner or diagnosing failure, read [learning methods and failure diagnosis](references/learning-methods-and-failure-diagnosis.md). It separates interface/body/data failures from hypotheses about BC, PPO, distillation, recurrent state and newer action models.

For APE0 low level training, train basic movements, object contact and short handoffs; a later planner consumes their observable outcomes. The exact Direct Frame source remains a valid **experimental training embodiment** even when mechanical selection or physical transfer is unresolved. Do not use a CAD decision as a stop condition for source-labelled control learning.

When changing the simulation world, prioritize motor/contact credibility, attained-state lessons, mid-episode recovery and useful full-training throughput. Keep the mechanics-only control lane fast; use calibrated rendering for image-dependent skills. Record raw motion, force and timing with a written behavior review, not a combined score.

Before another fresh low-level fit, pin a released checkpoint and copy its runnable observation/action, control, physics and visual contracts; bring the exact Cove body into that runner or prove an adapter with traces. Compare its **zero-update** Cove motion with the saved modular controller and model-based feedback. Use body-specific sources for necessary differences and identify unknowns explicitly. Train only a measured residual gap; do not train a reference G1 policy merely to benchmark Cove.

For large parallel evaluation, keep one source record per full episode and aggregate unit-bearing command errors, duration, contact, slip, motor effort and recovery into distributions and accuracy curves by condition. Pair active-time error accuracy with requested-horizon coverage so early falls do not disappear; distinguish planned-horizon recovery censoring from terminal falls and infrastructure aborts. Count world slots, episodes, independent starts, transitions, optimizer updates and checkpoints separately. Use words for motion quality and engineering judgment.

The campaign lead coordinates skill agents through Idle and asks independent agents to challenge stalled methods; the lead does not run each skill fit. Describe each candidate qualitatively from native whole-motion evidence: what it attempted, what it accomplished, how it adapted after a mid-episode disturbance, where it degraded, and whether a human would consider it a useful next baseline. A fresh-start variation is not a recovery test. Keep distance, timing, contact, motor force and falls as raw evidence and safety observations, not a pass/fail score or numeric cutoff; grade dimensions on the distance-to-viable scale instead. Preserve historical frozen gates as provenance without letting them discard useful rough behavior. Never describe parallel simulator worlds as separately trained policies, or transitions as optimizer updates. See progress and recovery review.

For rail variation in this DC motor model, change continuous torque, stall torque and no-load speed together per world, then compute rated-duty rewards from each world's effective force limit. A generic effort-only scale leaves the 10 V torque-speed case uncovered. The corrected stop packet has independent static GO for unscored smoke only; measure full-task world throughput and native prefix parity before PPO, then verify the reloaded candidate on disjoint whole starts against the frozen baseline.

For new primitive training or an environment change, read [low level environment and learning loop](references/low-level-training-environment.md). Before fitting, freeze the method and validate complete demonstrations or a documented demonstration-free curriculum, deployable observation, reward/physical checks and fresh evaluation families: body/channel identity, real entry transitions, physical outcomes versus reward/value, deployable observations, demonstrations, effective variation, GPU throughput and cost. Step, free-object acquisition and handoffs need their own physical entry and outcome evidence. A consequential new environment change needs another source-matched, stage-specific review before broader paid fitting.

Prefer the existing upstream mechanism that matches the lesson:

- Velocity, posture and arm control: MJLab with Pollen's task recipe and RSL-RL PPO.
- Reference foot transitions and steps: MJLab/BeyondMimic tracking; distinguish residual reference tracking from the distilled absolute-action step actors.
- Narrow acquire/place lessons: upstream Robomimic BC, with fresh MJLab rollouts. DAgger-style collection can add teacher labels on the learner's visited states, but a time-only teacher cannot correct an off-path object.
- Arena, Isaac Lab, Mimic, LeRobot and GR00T remain useful upstream references where their interfaces fit. They are not replacements for a presently working small-body baseline.

Authored trajectories may supply an explicitly labeled teacher or baseline. A scripted replay is not a learned policy; imitation learning is not PPO. More epochs or optimizer updates do not resolve an unobservable command, impossible contact geometry, wrong reset, or mismatched interface.

## Run the smallest discriminating lesson

1. Check the actual post-reset physical state and compare the retained policy with an appropriate zero-action or teacher baseline. Define the observed success and failure before optimizing: displacement, swing/contact, possession, release, settling, support, effort, or another outcome relevant to the task.
2. Change the consequential variable being investigated. Use the existing native trainer and record body, task, data and checkpoint hashes, command distribution, reset distribution, physics/control timestep, seed and resolved config. Runpod is temporary compute; keep code and compact reviewable records local, with retained bulk in the existing private blob archive. Use the Runpod skill for provisioning and cleanup.
3. In a fresh process, reload the saved checkpoint and run the native closed-loop evaluator. Include reserved starts and the failure case motivating the change; inspect the recording together with contact, effort and outcome traces. A repeated seed, training reward, finite loss, exported ONNX or a pose-only replay is insufficient evidence of improved control.
4. Retain compact native reports, traces and decisive failure clips for the comparison, then publish one Control workspace row in the shared dashboard. Archive only the strongest replayable checkpoint for a skill/body/interface, with its predecessor baseline, exact body/config/data provenance and supported entry/goal envelope. Keep inferior candidates' scores and decisive failure clips; discard their bulk checkpoints and raw frame arrays after review. Use existing `lab.artifact_store` helpers rather than recursively copying prior artifact trees; read registry `local_checkpoint` and config `local_path` fields rather than deriving filenames.

For recovery, use the existing private Hugging Face ape archive for retained small bundles and Infomaniak S3 for selected large files when needed. The 10 GB Runpod volume is worker transport, not a policy archive. Read [artifact storage](../../../docs/design/artifact-storage.md) before moving or removing retained artifacts, and verify a fresh download and SHA-256 before deleting the last local copy. Never send Infomaniak project credentials to a worker.

For each low level skill, follow the linked environment and learning loop: validate whole native teacher episodes, compare imitation closed loop, test reward shortcuts and critic fit to reward returns while scoring physical success separately, measure useful worlds per GPU, and report variation by actual parameter slice. Keep the installed Python **and OS-library** closure and exact disjoint CUDA constructor/reset/step evidence in the run manifest before originals or optimizer updates. If video or camera input is required, attest generic GLVND `libEGL.so.1` and NVIDIA vendor libraries on Linux, set `MUJOCO_GL=egl` before MuJoCo import, then prove the exact headless graphics and encode/decode path before an original; physics-only CUDA success is insufficient. Once that exact workload has run cleanly, spend the next cycle on fresh learned closed-loop outcomes, not another equivalent setup probe. Retain imperfect but reproducible policy improvements as experimental baselines; higher success fractions and motor margins are later robustness goals, not reasons to avoid short learning cycles. After two uninformative short cycles on one hypothesis, obtain an independent diagnosis and consult primary research/upstream code before another paid tuning branch.

For seated recovery, do not score support from the feet flags alone: a fallen body can touch the floor while both sole flags return true. The old pelvis-to-pelvis contact sensor missed floor–pelvis contact; [V6 verified a corrected floor-to-nonsole-body sensor](../../../artifacts/ape-cove50-stand-recovery/contact-sensor-parity-v6/report.json). Check whole-episode upright/height, forbidden floor contact, slip and motor duty against the full reward and termination rule. V4 showed that 5 cm absolute root drift mostly measured foot-locked stand-up motion; the revised V5 physical gate still failed on two genuine falls. A hidden internal phase or time-only script is not a deployable corrective imitation teacher.

Present a reviewed training run through the shared [ape-inspector](../ape-inspector/SKILL.md): publish one Control result row with native-unit measurements, qualitative motion judgment and per-dimension distance-to-viable grades in `data/results.sqlite`, upload selected videos to the existing private bucket with full SHA-256 read-back, and share the stable dashboard link when asked. Keep the exact native report, trace, checkpoint and source hashes as compact evidence. Do not generate a per-run landing page or routine research note, or open a new browser tab or Codex panel for each rollout.

Running this workflow is the POC test. Avoid broad test scaffolding and hardware-success claims; make the measured limitation and next useful experiment explicit.
