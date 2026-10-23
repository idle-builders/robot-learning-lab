---
name: ape-design-studio
description: Run or contribute to a Cove-50 joint design studio (hip, knee, ankle/foot, shoulder, elbow, neck) — three competing designers, graded evidence cards, user look reviews, requirement challenges. Use for any joint mechanism design, round, or review on the classic body.
---

# Ape joint design studio

Work locally in `robot-learning-lab`.

Keep active studio outputs and exact working CAD inputs local; move useful large completed CAD, simulation traces and selected media to the existing private Infomaniak blob archive. Read the [retention and restore guide](../../docs/design/artifact-storage.md) before closing a round or reopening cold evidence. Retain compact cards/configs/source hashes and versioned asset references locally, verify recovery before eviction, and restore required native assets/dependencies before reuse. Do not accumulate duplicate renders or put generated bulk in Git; local deduplication alone does not upload it.

## Read first
- **[Studio process](../../studies/cove50-engineering-competition/STUDIO-PROCESS.md).**
- **[Evaluation principles](../../docs/design/evaluation-principles.md).** No pass/fail. Grade each
  dimension on the 0–5 distance-to-viable scale, with confidence and trend. For every gap, ask whether the
  solution or the requirement is wrong.
- **[Joint cross-check and inventory](../../studies/cove50-engineering-competition/inventory/README.md).**
  Prior designs per joint form a parts and failure-mode library.

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
`studies/cove50-engineering-competition/fitkit/` holds the tools:
- **`check.py`:** metric collection; findings beyond reference points, grouped by dimension.
- **`render.py --opaque --cover-colour`:** looks, which the user judges.
- **`requirements.json`:** reference points, not limits.
- **`DESIGN-CHECKLIST.md`:** an evidence list. Grade every item.

The reference body is `classic`. Soft ivory covers are allowed on every joint.

## Rules that matter
- Three genuinely different solutions per joint (lenses A Packager / B Mass Centraliser / C Mechanism
  Inventor). Borrow pieces, not architectures.
- Never "call" or reject a design on a threshold. Write a graded card with an overall judgement and the
  next most valuable step. The user decides when to stop investing in a variant.
- Show visible adaptations (bulges, recesses, seams, covers) to the user as renders before steering by
  them.
- Honest mass: name every alloy, and list all unmodelled hardware in `unmodelled_hardware`.
- Write the round card to disk early. Sessions get interrupted.

## Publish a part's design study

Use the dedicated [robot design dashboard](http://127.0.0.1:8805/design/), never the learning table for mechanical CAD. Read the [design registration and evidence contract](../../docs/operations/design-dashboard.md). Start with `tools/results_hub.py design-template --robot-section Knee` (use the actual part). Each study is exactly one variant: use stable `study_id`, explicit `robot_sections`, a `version`, a unique result `id`, `dashboard: "design"`, and one `design` object. Register iterations as versions of that variant; different mechanisms become separate studies in the same body section. The latest version is the default. Register each alternative's neutral GLB, other configurations, STEP, named-part/joint manifest and renders as assets and `cad` references. Include real joint axes, pivots, units, ranges and couplings for dynamic inspection; do not substitute independent sliders for a constrained linkage.

Attach this version's measurements, simulated results and physical trials in `design.evidence`: native-unit values, conditions, observation, limitations and registered report/configuration/full-trace source references. A `design_evidence` section still belongs to this one variant/version; cross-variant source comparisons remain in Archive. Register simulation videos in `clips`, reference their IDs from evidence, and use the existing private upload workflow. Preserve failed and early-ending trajectories as evidence. Keep names and first-view findings short; the dashboard exposes detailed evidence, source files and human reviews on demand. A materially new variant gets a new study; an iteration gets a new version/result/design ID so old reviews remain tied to the original CAD. Users may save one preferred study per body section; do not infer that preference or whole-body selection.
