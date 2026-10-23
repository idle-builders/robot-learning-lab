---
name: ape-body-transfer
description: Evaluate how changed ape geometry, joint allocation, mass, actuator or contact assumptions affect learned behavior in robot-learning-lab. Use for 35-to-50 cm transfer, CAD-to-physics comparisons and sim-to-real measurement plans; not general appearance design or PCB authoring.
---

# Ape body transfer

Work from the repository root, its active goal and relevant Idle state. The 50 cm ape is the target; retained 35 cm policies are baselines. `robot/ape-current-design.json` is the selected mechanical candidate and records why its older physics body is not yet matched. Read [references/body-contract.md](references/body-contract.md) and the task's selected body/manifest before running anything. Use [ape-control](../ape-control/SKILL.md) for native policy execution.

For scale, contact, actuator or system-identification claims, read [dynamics and identification](references/dynamics-and-identification.md). It distinguishes physical knowledge from unmeasured ape parameters and preserves the decision to drop the ToddlerBot-shaped exterior.

Move retained large completed CAD/simulation outputs and full transfer traces to the existing private Infomaniak blob archive; keep current body XML, its mesh dependencies and compact source/config/hash records local. Follow the [retention and restore guide](../../docs/design/artifact-storage.md): restore both bodies and their dependencies before paired replay, native CAD work or export, and verify recovery before eviction. Local `.artifact-store` deduplication is not a remote archive; keep generated bulk out of Git.

## Match the experiment to the physical question

1. Identify which layer changed: visual mesh, collision shape, joint frame/axis, mass/inertia, transmission, actuator law, sensing or control timing. Record selected geometry and all input hashes. A visual mesh export alone does not change physical contacts or inertia. Visual-only changes can still alter camera observations, occlusion and image-driven policy behavior; physical parity does not imply perception parity.
2. Reconcile joint names, limits, neutral/reset offsets, observation layout and actuator ordering with the saved policy. The historical 22-axis body, current 21-axis intent, and 19/14-axis comparisons differ. Never drop a tensor column and assume a transferred controller. Explicitly map surviving joints and report new/missing channels; do not replay through an incompatible interface, adapt with fresh evidence instead.
3. Compare the same saved actor on baseline and changed body with matched conditions. Compare initial observations/actions to expose interface differences. Include travel, drift/turn/stop, falls, foot slip, effort saturation and relevant hand/body contacts. Retarget object/reset placement when geometry changes; a toy at the old hand position tests a broken scene, not grasp transfer.
4. Use a bounded native continuation only after the paired comparison identifies a useful learning question. Keep an unadapted comparison on the changed body and evaluate fresh held-out starts. Change geometry, dynamics or curriculum one consequential factor at a time where practical.
5. Feed an observed limitation back into one mechanical coupon or measurement, rather than polishing the whole shell. Match task motion, output support, purchased-part size, cable/service access and positive clearance to that coupon. A sampled pose screen is not continuous clearance or structural proof.

## Physical assumptions remain visible

Keep catalog rated/stall/no-load points separate from identified torque-speed, impedance and sustained thermal duty. The generic DC motor approximation and rail/delay sweeps in `src/ape_walk/physics/candidate.py` are sensitivity models, not measurements. Account for transmission ratio/efficiency, current/voltage limits and load-dependent speed before describing margin. Servo torque saturation on a simulated walk is evidence of a constraint, not a safe demand curve.

Choose randomization ranges from measured uncertainty when available; otherwise label and vary assumptions explicitly. Identify effects that observations cannot disambiguate before fitting a large parameter set. Actual hardware/system-ID work requires the selected parts and measurement setup; a prepared bench protocol cannot close the sim-to-real task.

## Useful existing tools and limits

- `experiments/compact_morphology/mjlab_ape/audit_self_contacts.py --motion <trace.npz> --model <body.xml> --out <new-report.json>` reads `qpos` and `dt`. Use a local Python with NumPy and MuJoCo. It checks native geometry and joint ranges **offline**; it does not rerun a controller or prove dynamics/printed-shell clearance. Matching `nq` alone is not a joint-order check.
- `experiments/compact_morphology/mjlab_ape/evaluate.py` performs native policy replay; select its body, timestep, command and output arguments using the control skill. GPU/MJLab availability must be checked separately from local MuJoCo.
- `experiments/compact_morphology/ape_cad/build.py` is the historical, fixed-layout case/battery envelope builder. Inspect its hard-coded `MODEL`, `OUT` and trace paths before reuse; it is not a general 50 cm exporter or safe `--help` probe.
- `tools/build_cad_gym_bundle.py` attaches CAD **visuals** to a sealed legacy dynamics bundle and checks physical parity. Do not cite its success as new ape mass/contact integration.

Grade the transfer per dimension (for example kinematics and range, fit, drive, mass, behaviour on the changed body) on the 0–5 distance-to-viable scale with confidence and trend; see [evaluation principles](../../docs/design/evaluation-principles.md). When a gap appears, ask whether the body or the requirement is wrong (mass budget, axis count, clearance, rated duty), record both, and propose requirement changes to the user rather than relaxing or enforcing them silently.

Retain old/new identities, measured outcomes and failures as compact native run evidence, then publish learned behavior or policy-transfer evidence with `dashboard: "learning"` in the Control workspace. Register mechanical/CAD measurements against the exact one-variant study/version in the [design dashboard](../../docs/operations/design-dashboard.md), using `design.evidence` and its complete source assets. Do not create a per-run presentation page or open a new tab. Report which level the evidence covers (behavior on one model, a geometric envelope, or a physical measurement) and grade each at its own level; one level's grade does not carry over to another.

Use the shared [ape-inspector](../ape-inspector/SKILL.md) to compare saved CAD, video and reports in the same dashboard tab. Publish a complete-scene GLB as a model asset where possible. For recurring articulated or synchronized views, extend the dashboard widget; existing unique interactive pages remain available in Archive.
