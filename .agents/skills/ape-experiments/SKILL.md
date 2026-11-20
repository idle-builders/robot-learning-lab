---
name: ape-experiments
description: Plan and assess robotics experiments in robot-learning-lab using its learning materials, evidence and Idle roadmap toward the 50 cm ape. Use for choosing the next experiment, interpreting a result, or turning a new learning into work; route policy, perception and body-transfer execution to their focused skills.
---

# Ape experiments

## User rule: no background Runpod watchdogs

Never create, install, enable, or relaunch background Runpod watchdogs or provider-polling services. This forbids launchd, cron, systemd, detached shell polling loops, and persistent retry timers. Historical watchdog scripts are evidence only; do not run or reinstall them. Use provider-side expiry for bounded compute and explicit, bounded foreground checks for status. Training and file-transfer processes are not watchdogs.

Stop provider requests on HTTP 429 / `rate_limited`. Honor `Retry-After` or the stated reset window; a daily-window limit must not trigger automatic retries. Do not change networks, credentials, accounts, or API routes to evade a rate limit. Stop querying after a job is terminal, deleted, or retired. Do not run multiple monitors for one resource. Inspect `launchctl list` and live processes during cleanup; deleting a script alone does not unload a service.

For the selected assembled robot, resolve `robot/model/current-model.json` and read the [current URDF and native-model reference](../ape-body-transfer/references/selected-model.md). It identifies `cove50_description`, the corrected motor visuals, shell-free view and exact actuator/sensor contracts. Historical policy body overrides remain their original sources.

Locate `robot-learning-lab` by its `AGENTS.md`; work there. Read the active goal and relevant Idle tree state before choosing work. Use the installed idle skill/tools with an explicit `by`; search existing tasks and decisions before adding work. A planning request does not itself authorize running all experiments or provisioning compute.

Plan storage as part of the experiment: keep active inputs and compact provenance local, and move useful large completed evidence to the existing private Infomaniak blob archive with verified recovery before eviction. Read the [retention and restore guide](../../../docs/design/artifact-storage.md) before closing large runs or reusing cold inputs. Local deduplication is not cloud archival; restore required paths/dependencies before native enumeration or export, and keep generated bulk out of Git.

The target is the **50 cm ape**. Historical 35 cm/22-axis policies and G1/Berkeley/Microduck/Microban results are references. Check the selected body's actual axes: current 21-axis intent and 19/14-axis comparisons are not a frozen embodiment with graded physical evidence. Keep control learning ahead of CAD/electronics, and derive mechanical questions from a task failure or capability need.

For the control-first campaign, the Direct Frame source named by the mechanical pointer is an experimental training body; unresolved mechanical selection does not block low level learning on it. Keep trained simulation outcomes distinct from later physical evidence. Structure Idle work as parallel primitive tracks with only genuine prerequisites, and keep high level planning outside the skill-training scope.

For simulation-world decisions, compare the exact Cove world with a working upstream humanoid recipe and trace motor/contact, perturbation recovery, camera geometry and *useful* sustained experience per GPU hour. Do not equate photorealism, raw transitions/s or thousands of parallel worlds with improved low level control. New development reviews use directly measured facts, qualitative judgment in words and per-dimension 0–5 distance-to-viable grades ([evaluation principles](../../../docs/design/evaluation-principles.md)); historical binary gates remain provenance.
For a new perception/localization world, include a bounded run of the relevant existing consumer through its actual observation adapter. Grade rendering, consumer behavior and metric/physical support separately; a visual reconstruction can support RGB experiments while still lacking scale, depth, a floor frame or collisions for the full route. Use [the captured-worlds README](../../../control/perception/worlds/README.md); retain unavailable continuations rather than treating reconstruction quality as downstream validation.
For a reuse experiment, reproduce a released controller's **inference** in its pinned native environment, then bring Cove into that model's actual contract and compare zero-update motion. Copy sourced details rather than designing a generic gym by assumption. Disclose robot-specific differences instead of copying another body's motor or mass values. Put no new G1 training ahead of this comparison.
For thousands of parallel rollouts, report continuous tracking-error accuracy curves and native-unit distributions by condition and independent family, with direct event counts and reviewed source clips. Do not convert world slots or repeated resets into independent policy counts, or bury short falls in a frame-weighted average.

Before a paid low level fit, use [ape-control's environment readiness and learning loop](../ape-control/references/low-level-training-environment.md). Require complete demonstrations or an explicitly reviewed demonstration-free curriculum, deployable actor observation, physical/reward checks and task-specific intended-world stress before the next fit. Record each track's permitted next stage separately: feasibility data, short learned pilot or larger fit. A new task interface or widened fit needs a fresh source-matched stage review. Do not present an environment review or a training plan as proof that a policy works.

## Find the right evidence and workflow

Read only the records relevant to the question:

For a learning-material claim or a new evaluation design, read [evidence and evaluation](references/evidence-and-evaluation.md). Knowledge belongs in reusable guidance, while an unproven improvement for this ape belongs in a falsifiable Idle experiment.

- Policy training, replay, contact manipulation or composition: [ape-control](../ape-control/SKILL.md); `control/skills/registry.json` and `control/wbc/registry.json` are the **robot policy shelf**, distinct from these Codex instructions.
- Image-grounded teaching, finding, belief, gaze or sensor experiments: paused. Results are in [perception and sensing](../../../docs/research/perception-and-sensing.md) and [high-level behaviour](../../../docs/research/high-level-behaviour.md); the code and the former ape-perception skill are in commit `6fbb278e4` (`experiments/ape_high_level/`, `.agents/skills/ape-perception/`).
- Geometry, joint allocation, actuator/contact assumptions or transfer to a changed body: [ape-body-transfer](../ape-body-transfer/SKILL.md).
- G1/Berkeley Arena/GR00T reference work was dropped (see [landscape and references](../../../docs/research/landscape-and-references.md)); its skill and tools are in commit `6fbb278e4`.
- Runpod provisioning or cleanup: use the installed runpod skill. Develop locally; use the 10 GB Runpod volume only for bounded worker transfer and retrieve results. A stopped request is not proof a remote job stopped. The deleted 320 GB and 235 GB volumes are historical provenance.

## Turn a question into a discriminating experiment

State the capability being improved, the current failure evidence, and one hypothesis. Choose the cheapest comparison that can change a decision. Examples: unchanged policy on old/new body; same images with different identity gate; synchronous versus delayed actions with the same policy. Avoid a new framework when existing upstream trainers and repository probes cover the question.

When a learned policy survives but avoids what its reference or task asks for, the first hypothesis is the reward, not the body, actuators or budget. Run the saturation, contrast, width, coverage, regulariser and termination checks of [ape-control's reward design](../ape-control/references/reward-design.md) before proposing a hardware change or a longer run. Iterate the reward one measured change at a time. A hardware buy or design change is justified only after a reward that asks for the behaviour shows the current hardware cannot deliver it.

Before running, write into the Idle task's plan:

- Baseline and changed factor; exact body, checkpoint, controller and data identities.
- What the actor actually observes and emits: frames, units, timing, normalization, history/phase, privileged training inputs and deployable inputs.
- Training/development cases versus held-out episodes/scenes; independent physical or interaction outcome, failure/abstention measures and a negative control where it discriminates.
- A bounded run and stopping condition; the evidence that would lead you to continue, change direction or retain the baseline (a judgement, not an automatic cutoff). Choose reference values for this task, not from a generic prep document, and record where each comes from so it can be challenged. Declare the regression size you would consider meaningful and the sampling before comparison; a nonsignificant adverse result is not proof of no regression.
- Valid demonstration provenance and the teacher's physical success from actual reset/attained states; reward terms with physical meaning and shortcut checks; per-world parameter intervals with source; world-count/throughput benchmark and attributable compute budget.

For camera or video work on a Linux worker, include OS graphics libraries in the frozen runtime inventory as well as Python distributions. On Ubuntu 24.04, attest generic `libEGL.so.1` from `libegl1` and `libOpenGL.so.0` from `libopengl0`, NVIDIA vendor libraries, `MUJOCO_GL=egl` before import, and a live NVIDIA context. Then decode a nonblank MP4 from the exact task before opening an original. A provider graphics flag or Python `pip check` cannot establish this path alone.

Organize each skill into short smoke, pilot and fresh replay cycles; smoke tests are short GPU jobs, never local CPU simulation (owner, 2026-11-12). Track development and frozen independent evidence separately; grade progress against prospective, skill-specific reference values rather than demanding that every case succeed. After two uninformative discriminating cycles, get a fresh adversarial diagnosis and check primary papers/upstream code before spending on another variant. Record the resulting hypothesis change in Idle. Preparation and evaluation for unrelated skills can run concurrently; on a single GPU, sequence training jobs to avoid contention.

For learned skills, specify supported entry states, command range, outcome and stop/recovery conditions. A planner's completion text or a policy timeout cannot verify possession, placement, stopping or object identity. Compose on states actually reached by the predecessor, without resetting physics or silently changing control coordinates.

## Classify results before recording them

Distinguish a proposal, static/source inspection, offline replay, fresh closed-loop simulation, and physical measurement. Label scripted references, BC/imitation, reward-updated RL and memory edits accurately. A saved checkpoint, model load, plausible video, detector box, scalar reward or low loss is not the intended task outcome.

Then grade each relevant dimension (for example motion quality, contact and support, motor duty, robustness across conditions, recovery, interface completeness) from 0 to 5 for distance to viable, with confidence and trend, using the reporting template in the [evaluation principles](../../../docs/design/evaluation-principles.md). Do not average the grades. When a result falls short of a reference value, ask whether the solution or the requirement is wrong (where the number came from, what it buys, what it costs), record both, and propose any requirement change to the user instead of relaxing or enforcing it silently.

Keep source/config/model hashes, seed and split, exact evaluated body, observation/action meanings, metrics and inspectable failed as well as successful episodes. Use existing `lab.artifact_store` and run manifests rather than a new artifact system. Retain the selected replayable policy, baseline and compact decisive evidence; expire unselected weights, caches and redundant raw arrays after review. Archive small policies in the existing private Hugging Face archive or selected large files to the existing Infomaniak S3 archive with a SHA-256 verified restore; follow [artifact storage](../../../docs/design/artifact-storage.md) and the [storage lifecycle](../../../docs/operations/storage-lifecycle.md). Inspect native metrics alongside synchronized video/traces. If a required check cannot run, report `not_run` with its missing prerequisite; do not substitute a different kind of evidence.

The [ape-inspector](../ape-inspector/SKILL.md) routes policy, perception, motion-control and planning results to `http://127.0.0.1:8805/`, and CAD/engineering studies to the separate section-based design dashboard at `http://127.0.0.1:8805/design/`, with selected videos in the existing private bucket. Publish compact records through that flow; keep authoritative native reports, traces and source hashes as evidence. Extend a reusable widget when a display pattern recurs. Do not ask agents to make per-run web pages or open a new tab or Codex panel for each result.

Use task feedback for what changed the next decision. Update existing unfinished goals instead of multiplying duplicate task lists; retain completed historical evidence and active claims. New exploration ideas outside accepted scope start `proposed`. A current roadmap link is useful; copying the whole live task backend into docs creates drift.

## Incorporate another learning material

The five HTML files in `docs/learning materials/` (local only: Git-ignored, not in the public repository) are preparation notes, with substantial text inside JavaScript arrays. Extract their visible text and structured learning content without executing embedded code. Preserve the claim's source and section. Turn relevant principles into a concrete ape failure/question; verify original papers or official pinned code before relying on changing model capabilities, vendor specifications or numerical rates. Do not elevate company claims, broad “replacement” narratives or fleet-scale architecture into requirements for this POC.
