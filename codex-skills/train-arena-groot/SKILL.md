---
name: train-arena-groot
description: Reproduce or extend the pinned G1 and Berkeley reference workflows using Isaac Lab Arena, Mimic, LeRobot and GR00T in robot-learning-lab. Use for those named reference robots or an explicit Arena/GR00T integration; use ape-control for the current ape MJLab and Robomimic workflows.
---

# Train Arena GR00T

This is the retained upstream reference path. For the current 50 cm ape objective, start with [ape-experiments](../ape-experiments/SKILL.md) and [ape-control](../ape-control/SKILL.md). G1/Berkeley successes are not evidence for the ape. The constraints below apply to this reference workflow; they do not prohibit the explicitly labeled reference trackers and scripted teaching data retained in the separate ape experiments.

Use the active goal and idle tasks alongside the proven reference path before changing a task, dataset, model or robot. Locate the repository root by finding `AGENTS.md`; run every command from that root.

Use the existing private Infomaniak blob archive for useful large completed datasets, recordings and native reference outputs; keep active inputs and compact pins/configs/manifests/hashes local, with retained small policy bundles in their existing private archive. Read the [retention and restore guide](../../docs/design/artifact-storage.md) before closing a large run or preparing native replay/worker exports. Restore required paths/dependencies, verify recovery before eviction, and keep generated bulk out of Git. Runpod storage is bounded transport, and Infomaniak project credentials stay local.

For Berkeley Humanoid Lite playroom work, read `docs/research/compact-playroom-plan-20260925.md` and [references/berkeley-playroom.md](references/berkeley-playroom.md). The G1 files below describe the archived G1 workflow; use them only when the task concerns G1.

For G1, read these records before acting:

- The active goal and idle tasks for the current stage and permitted progression.
- `docs/design/arena-demonstration-learning.md` for what has actually executed.
- `workers/arena/pins.json` for immutable upstream revisions.
- `docs/design/arena-teleoperation.md` when the request involves collecting new human data.

## Choose the operation

- For Berkeley ball, hand-button, or Cosmos-to-GR00T composition, use [references/berkeley-playroom.md](references/berkeley-playroom.md).
- For the already supported box-to-bin reference continuation, checkpoint creation, or candidate re-evaluation, read [references/reference-training.md](references/reference-training.md).
- For a new task or freshly collected human demonstrations, read [references/new-task-data.md](references/new-task-data.md). One near-box desktop-demonstration loop has run end to end with a native success; do not infer the same for later tasks.
- For result classification, cleanup, inspection, or a failure, read [references/evidence-and-failures.md](references/evidence-and-failures.md).

## Preserve these boundaries

- Use Arena, Isaac Lab, Isaac Lab Mimic, Isaac-GR00T, LeRobot format, HOMIE and their native scripts. Do not create a replacement trainer, task controller or authored trajectory.
- Change one consequential component per experiment. Keep a known working checkpoint and native replay as comparisons.
- Call demonstration fine-tuning **imitation learning**. Call a run RL only when a reward-driven optimizer actually updates a policy. Keep human demonstrations, Mimic augmentation and autonomous rollouts distinct.
- A finite loss or a saved checkpoint is not task success. Require a fresh process to load the persisted checkpoint and run the native closed-loop evaluator with native metrics and an inspected video.
- For released-policy G1 recordings, inspect the raw action width before LeRobot conversion. The i4h converter's default teleop mapping covers 23-D keyboard actions, but released-policy rollouts can store 50-D actions: the first 43 are joint targets, indices 43–45 are navigation, 46 is base height, and 47–49 are torso commands. Configure the converter's existing `teleop_sources` for those fields and compare converted values with the raw HDF5; a silent zero-filled navigation column invalidates locomotion training. See `experiments/g1_multitask/i4h-50d-teleop-source.patch`.
- When mixing task datasets from a task-specific base, check whether GR00T uses that base's saved action statistics or the mixed dataset statistics. Record the choice and inspect nonconstant navigation ranges; do not infer a useful motor policy from training loss.
- A known task/seed success is not improvement, repeatability or unseen-task generalization. State exactly what was evaluated.
- Keep every attempt's native evidence in a separate run directory. Retain failures, source seals, manifests, hashes and dependency links. Never overwrite or recursively copy prior artifact trees. Publish reviewed G1 and Berkeley outcomes to the References workspace of the shared [ape-inspector](../ape-inspector/SKILL.md); attach selected videos through the existing private bucket. Do not make a per-run HTML page or open a new browser tab or Codex panel for each result.
- Use one bounded GPU worker at a time, zero idle minimum workers and explicit deadlines. A client timeout is not cancellation; inspect the existing job before submitting another.
- Never print credentials. Use the existing Hugging Face login, `runpodctl` authentication and `runpod` S3 profile.
- The old G1/Berkeley Runpod volumes have been deleted. Use the 10 GB Runpod volume only for bounded worker evidence and datasets; recover selected large historical references from Infomaniak S3 with the verified manifests in [the storage plan](../../docs/operations/storage-retention-plan-20261013.md). Keep Infomaniak project credentials local, never in the GPU worker. Check Hugging Face quota before a new large checkpoint upload.

## Evidence for a completed cycle

A completed training cycle collects all of these: validated demonstration provenance; native optimizer steps with finite metrics; a complete SHA-256 checkpoint manifest in its actual storage destination; fresh-process checkpoint reload; native closed-loop outcome metrics; an inspectable synchronized recording; baseline comparison; and verified provider cleanup. Mark a missing item `not_run`, or describe how it broke, rather than filling the gap with configuration or playback evidence. Then describe the outcome in words and grade each dimension 0–5 for distance to viable, with confidence and trend, per the [evaluation principles](../../docs/design/evaluation-principles.md). When the result misses the task's native predicate, ask whether the policy, the data or the predicate is wrong, and propose requirement changes to the user.
