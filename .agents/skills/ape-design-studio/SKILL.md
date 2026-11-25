---
name: ape-design-studio
description: Run or contribute to an APE0 joint design studio (hip, knee, ankle/foot, shoulder, elbow, neck) — three competing designers, graded evidence cards, user look reviews, requirement challenges. Use for any joint mechanism design, round, or review on the classic body.
---

# Ape joint design studio

For the selected assembled robot, resolve `robot/model/current-model.json` and read the [current URDF and native-model reference](../ape-body-transfer/references/selected-model.md). It identifies `cove50_description`, the corrected motor visuals, shell-free view and exact actuator/sensor contracts. Historical policy body overrides remain their original sources.

Work locally in `robot-learning-lab`.

Keep active studio outputs and exact working CAD inputs local; move useful large completed CAD, simulation traces and selected media to the existing private Infomaniak blob archive. Read the [retention and restore guide](../../../docs/design/artifact-storage.md) before closing a round or reopening cold evidence. Retain compact cards/configs/source hashes and versioned asset references locally, verify recovery before eviction, and restore required native assets/dependencies before reuse. Do not accumulate duplicate renders or put generated bulk in Git; local deduplication alone does not upload it.

## Owner rule: agility across all motions

The score is agility across all motions (owner, 2026-11-08; `AGENTS.md`). The target and tape motions are made-up stand-ins, and the final set is not known. Any single motion, or single family, is irrelevant on its own. Never plan, tune or judge a joint requirement, a part or a design around one. Read results as distributions over all motions, including motions not trained on.

## Read first
- **[Studio process](../../../docs/operations/design-studio-process.md).**
- **[Evaluation principles](../../../docs/design/evaluation-principles.md).** No pass/fail. Grade each
  dimension on the 0–5 distance-to-viable scale, with confidence and trend. For every gap, ask whether the
  solution or the requirement is wrong.
- **Joint cross-check and inventory** (`git checkout history/design-fitkit`, `studies/cove50-engineering-competition/inventory/`).
  Prior designs per joint form a parts and failure-mode library; each earlier design has a `design/<study>` tag.

## Find the work
Each joint has **one Idle task covering all its variants**:

| Joint | Idle task |
|---|---|
| Hip | T-bzedjy |
| Knee | T-rtp2rb |
| Ankle/foot | T-z5wca9 |
| Shoulder | T-vthwrk |
| Elbow | T-mf7gyn |
| Neck | T-xt5qkc |

Read the task's plan, metadata and feedback before starting. Do not create per-variant tasks.

## Tools
The sheet-metal studios' tools are in history (`git checkout history/design-fitkit`, folder `studies/cove50-engineering-competition/fitkit/`); bring back what a new studio needs into the part's folder under `robot/`:
- **`check.py`:** metric collection; findings beyond reference points, grouped by dimension.
- **`render.py --opaque --cover-colour`:** looks, which the user judges.
- **`requirements.json`:** reference points, not limits.
- **`DESIGN-CHECKLIST.md`:** an evidence list. Grade every item.

The reference body is `classic`. Soft ivory covers are allowed on every joint.

## Design kit: how parts look and are built

Before designing or redesigning a shell or a connector, read the [design kit](../../../docs/design/design-kit.md) (owner, 2026-11-24). It holds the owner's aesthetic and robustness rules, so the owner does not have to explain them again:
- **Shells:** calm, continuous parametric surfaces that look closed. Each is one surface per limb or region (superellipse sections on C2 splines, fitted to requirement point sets), cut into parts at analytic seams about the joint axes. No bulges, plates, pods or dark gaps. Template: `robot/body/arm/arm_shell.py`.
- **Connectors:** U / H / I / Y shapes that grip every actuator from two sides (horn and idler, 2 + 2 screws), with flat members, arcs about the axis, rounded corners, pressable 304 sleeves, 1.5-2 mm between moving bodies and never a slit under 1 mm. Template: `robot/body/arm/shoulder_clevis.py`.
- **Code:** build on `robot/body/design_kit/` (`shell.Sweep`, the seam cutters, `connector` pieces) instead of copying functions.
- **Candidates:** report the measurements the kit lists for each kind of part, and show the owner, next to the renders, every place where the candidate departs from the kit and why. The kit is a starting point, not a gate. When the owner approves or rejects a new pattern, add it to the kit.

## Rules that matter
- Three genuinely different solutions per joint (lenses A Packager / B Mass Centraliser / C Mechanism
  Inventor). Borrow pieces, not architectures.
- Never "call" or reject a design on a threshold. Write a graded card with an overall judgement and the
  next most valuable step. The user decides when to stop investing in a variant.
- Show visible adaptations (bulges, recesses, seams, covers) to the user as renders before steering by
  them.
- Honest mass: name every alloy, and list all unmodelled hardware in `unmodelled_hardware`.
- Write the round card to disk early. Sessions get interrupted.
- Keep one study folder and one card per variant, named by content (`robot/<area>/<part>/`, e.g. `robot/body/leg/`), and revise them in place: a new round is a new commit, not a `-r2`, `-v16` or dated folder. Publish with `source` naming that folder; the publisher takes the commit as the dashboard version. Competitor handoffs, reviews, freezes and publication checks are process records: write them under `.bookkeeping/<task-id>/` ([git and bookkeeping](../../../docs/operations/git-and-bookkeeping.md)).

## Hardware starts at the BOM

For any part, screw, print or purchase question start from the generated BOM and assembly order, not from a manifest or an old study folder: [robot/body/bom/README.md](../../../robot/body/bom/README.md) (`BOM.md`, `ASSEMBLY.md`, `findings.json` in `.cache/work/bom/` after `.venv/bin/python -B robot/body/bom/build.py`). The BOM tool knows no part: your builder's definition, `<builder>.bom.yaml` next to its script, says what it generates from what, which parts it owns and what each is (line, make / buy, spec, supplier or `unknown`), its assembly steps and its open defects. When your work adds, changes, renames or removes a part, screw or builder step:
- edit only your builder's `.bom.yaml` (a new builder gets a new one next to it); never edit files under `robot/body/bom/` for a design change;
- a part's check settings (print setup `process`, `fem`, `expected_hardware`, `fit_classes`, `covers`, `heat_source`, ...) live on its entry in that same file; the checks and the BOM both read it, and every check finding names the file;
- rerun `build.py`, the assembly check and the bom check (commands in the BOM README) and work down the findings that name your file; report the counts that moved;
- draft steps with `robot/body/checks/assembly_seed.py propose` and hardware specs with `robot/body/standard-hardware/propose_specs.py`; you decide what goes in.

## Publish a part's design study

Use the dedicated [robot design dashboard](http://127.0.0.1:8805/design/), never the learning table for mechanical CAD. Read the [design registration and evidence contract](../../../docs/operations/design-dashboard.md). A study of body parts is a `study` block in the owning body definition (`robot/**/<builder>.bom.yaml`, schema `robot/body/bom/schema.json`), published by `robot/body/bom/publish_body.py`; its sections come from its parts' `section`. Each study is exactly one variant with a stable content-named `study_id`. Git holds the versions: commit, then publish; the previous version stays in the study's version menu under its commit. Different mechanisms become separate studies in the same body section. The current version is the default. Register each alternative's neutral GLB, other configurations, STEP, named-part/joint manifest and renders as assets and `cad` references. Include real joint axes, pivots, units, ranges and couplings for dynamic inspection; do not substitute independent sliders for a constrained linkage.

Attach this version's measurements, simulated results and physical trials in `design.evidence`: native-unit values, conditions, observation, limitations and registered report/configuration/full-trace source references. A `design_evidence` section still belongs to this one variant/version; cross-variant source comparisons remain in Archive. Register simulation videos in `clips`, reference their IDs from evidence, and use the existing private upload workflow. Preserve failed and early-ending trajectories as evidence. Keep names and first-view findings short; the dashboard exposes detailed evidence, source files and human reviews on demand. A materially new variant gets a new study; an iteration is a new commit of the same study, and old reviews stay with the version they were written on. Users may save one preferred study per body section; do not infer that preference or whole-body selection.
