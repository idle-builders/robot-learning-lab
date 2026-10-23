---
name: ape-perception
description: Develop and evaluate the 50 cm ape's perception, scene memory, teach-and-find interaction, social grounding, and sensor evidence in robot-learning-lab. Use for image-grounded decisions and their interface to motion; use ape-control for physical policy training and ape-body-transfer for body changes.
---

# Ape perception and interaction

Make the ape's next look, question, memory update or motion request follow observable evidence. Work in the local `robot-learning-lab` checkout; paths below are repository-relative. Track the experiment in Idle and follow the active objective using `ape-experiments`. The target is the 50 cm ape; earlier 35 cm colored-cube perception is a plumbing baseline, not evidence of ordinary-object recognition.

Store retained large completed scene datasets, sensor recordings and replay arrays in the existing private Infomaniak blob archive; keep active captures, current dataset inputs and compact annotations/calibration/split/hash records local. Read the [retention and restore guide](../../docs/design/artifact-storage.md) before sealing large outputs or enumerating cold datasets: restore the required paths first so missing files cannot silently change a split. Verify recovery before local eviction; keep generated bulk out of Git and avoid accumulating duplicate frames or render variants.

Read `docs/research/ape-high-level-experience-map-20261005.md` for the owner experience, then only the current evidence relevant to the question:

- Teaching, finding, correction, sensing: `docs/research/ape-sensor-decision-20261006.md`.
- Person/hand/pointing or speech association: `docs/research/ape-social-grounding-20261006.md`.
- Ego-motion, remembered locations, tracking loss: `docs/research/ape-belief-motion-20261006.md`.
- Running an existing probe: [entrypoints and limits](references/entrypoints.md).
- Camera/pose adapters, moving memory or asynchronous sensing: [frames, time and belief](references/frames-time-and-belief.md), including transform direction, sample age and heuristic-versus-calibrated uncertainty.

These reports are dated evidence. Inspect the current implementation and newer results before describing a capability as present.

## Choose the discriminating experiment

Begin with the behavior being decided: for example, when the ape may claim it found a taught object, whether a second view helps, or when a location becomes unusable after motion. Compare on the same episodes and retain the baseline. Do not replace a failed owner-facing question with a detector benchmark that cannot answer it.

Define the attempt, success, failure and uncertainty response before running. Split at scene, session, object instance or trajectory level as appropriate; adjacent video frames are correlated. Include target-absent, lookalike, occlusion, wrong teaching/correction and missing-signal cases when they affect the claim. Measure wrong confident answers, useful answers, unnecessary questions and time to resolution together. A model that always asks has not solved find. Grade each relevant dimension (for example identity accuracy, wrong confident claims, question burden, pose health, latency) from 0 to 5 for distance to viable, with confidence and trend, per the [evaluation principles](../../docs/design/evaluation-principles.md); no single pass/fail rate. When a gap appears, ask whether the perception solution or the requirement (a claim threshold, a latency budget, an owner-interaction expectation) is wrong, record both, and propose requirement changes to the user.

Use `ape-experiments` for broader prioritization, evidence records and comparisons. Reuse the released image models and existing probes before inventing a new perception framework. Claims that a newer model replaces an older method in the preparation HTML are research leads; verify relevant upstream source, compatibility and the actual experiment before adopting one.

## Keep the observation and belief boundary visible

For a decision, retain the source image/crop, capture time, transform direction and units, camera intrinsics/extrinsics, head/base pose source and health, model/config version, association history, and any owner label/correction. Use the pose at image capture, not inference completion. Record latency and sample age separately. Known simulator pose, published dataset pose, estimated pose and physical sensor readings are different evidence.

Keep **seen now**, **last seen**, **ambiguous** and **not currently located** distinguishable. A remembered item may support a question while being too stale or uncertain for a metric motion target. Missing camera/pose health makes a coordinate unavailable; recovering a frame alone does not prove relocalization. Require fresh sighting and restored pose evidence before coordinate-based approach/reach resumes. A supplied health flag only tests response to that flag, not automatic fault diagnosis.

Keep object truth, simulator segmentation and published evaluation annotations out of actor decisions unless the experiment explicitly studies an oracle upper bound. Label evaluator-selected crops/clicks and calibrated planes as supplied inputs. Ground-truth scoring must not silently choose the candidate, memory update or gaze action.

## Teach, correct, find, then verify

A first owner teaching turn should show one object, supply a name, highlight the proposed crop, and obtain confirmation. A hand detection, finger ray or top retrieval score is not a unique object label. Keep heard cue, visible person, visible hand and associated speaker as separate hypotheses. Test unfamiliar people/backgrounds and ambiguous input before treating social acknowledgements as grounded.

On correction, preserve what was contradicted and the source of the new label; evaluate replacement versus retained exemplars rather than always accumulating them. Search fresh scenes without reteaching. Reobserve the candidate to gain discriminating evidence; geometric consistency between two views can repeat the same identity error. When evidence remains weak, leave the item unidentified and offer optional candidate review/show-again. Evaluate the burden of that interaction rather than automatically presenting the best distractor every time.

Distinguish an owner-taught memory update, a collected demonstration, a trained model update and an authored acknowledgement. Claim learned improvement only after the relevant update changes behavior on fresh evidence.

## Connect sensing to the real body deliberately

The active-search rig has a fixed base, exact camera kinematics and a provisional head-pitch joint. The current 50 cm draft has a yaw-only head. Read the current body contract through `ape-body-transfer` before mapping its gaze actions onto the robot. Existing aperture/paw screens are geometric counterfactuals; they do not establish camera packaging, walking localization or final grasp visibility.

Let sensor candidates earn their place on paired behavior trials: head pitch versus fixed views, RGB versus near-hand depth, telemetry versus contact sensing. Motor current is not calibrated fingertip force, and simulator jaw contacts are evaluator labels until a deployable channel supplies them. Keep person-height attention, floor search, final approach and possession/release as distinct observability questions.

For motion requests, pass the target frame, timestamp, pose health, uncertainty and validity envelope to a bounded skill. Check entry conditions and observe completion before claiming approach, possession or placement. Use `ape-control` for the actual controller, handoff history, contact and recovery evidence. Give gaze capture and expressive motion one coordinated owner so a cute movement cannot invalidate the evidence being gathered. Measure end-to-end timing before allowing a slow image or language model to influence continuous motion.

When training an image-grounded low level reach, check [the control environment's observation checks](../ape-control/references/low-level-training-environment.md): the actor receives only captured camera-derived goals and available proprioception, with age and missing-signal behavior tested. Simulator object pose may be a teacher, critic or evaluator input; a policy whose actor receives the exact `ee_to_cube` vector is an oracle arm-control baseline, regardless of its camera evaluation page.

## Deliver evidence at the level actually run

Save annotated views, selected and unselected candidates, beliefs, decisions, interventions, actual outcomes and source/model/data versions as compact native evidence. Publish reviewed learning results as Perception rows in the shared results table, annotated images as `assets`, and selected videos to the existing private bucket; extend the single dashboard when a new display recurs. Do not generate a per-run HTML gallery or open a new browser tab or Codex panel for a routine result. Report per-condition counts, limitations and dimension grades. Offline replay can compare hypotheses on recorded views; image-driven simulated looking establishes that bounded loop in its synthetic scene; uninterrupted physics, real owner interaction and physical sensing each require separate evidence. Neither a successful import nor an impressive video counts as evidence for those later levels.
