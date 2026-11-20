# Policy interfaces and evidence checkpoints

Here and in this file's name, "gates" means checkpoints that collect evidence, not pass/fail decisions; grade what they collect per the [evaluation principles](../../../../docs/design/evaluation-principles.md).

## What must match before a policy is reused

Resolve the specific registry entry and its optional body override. Read its exact observation/action construction in `control/common/mjlab_ape/src/ape_walk/` and saved config rather than assuming all ape policies share a contract.

- Preserve joint names/order, action width, absolute versus normalized/residual targets, scales and reset offsets. The legacy gait mapping is 20 → 22 axes by joint name, not positional padding. Distilled steps emit absolute joint targets; a reference tracker learns corrections around a time-indexed motion.
- Preserve observation ordering, relative-joint defaults, previous-action history, command features, normalization, phase and recurrent-state handling. A model accepting the same tensor width can still interpret it incorrectly.
- Preserve frame conventions, units and sample/control clocks. The nominal ape loop often uses 2 ms physics and 20 ms actions; read the actual task before applying that assumption.
- Distinguish actor input from training/evaluator truth. Simulator object positions and contacts can define a research task and score outcomes; they do not establish that onboard perception or sensors can supply those values.
- Record the supported entry state and goal range. Raised-foot holds, preclosed-jaw lifts, open-aperture acquisition and place-from-acquire are different skills. Validate the actual post-`env.reset()` pose, not only the pre-reset scene.

The exported distilled single-step actor recorded in `docs/research/low-level-control.md` uses `obs[1,76] → actions[1,22]`. It still needs phase and knows one taught displacement. This is evidence for that export only; inspect other checkpoint contracts individually.

## The existing acquire/place handoff

`evaluate_acquire_place.py` preserves robot and cube physics while switching at three seconds. It changes the target, replaces reset-relative joint/action offsets for `right_shoulder_pitch`, `right_elbow_pitch` and `right_gripper`, resets action history, and recomputes the actor observation without advancing history before the first place action. Both loaded policies start their episode state once at episode initialization; phase restarts at the stage boundary. Removing one of these steps can break a seemingly compatible pair.

The place base actor observation is 72-D in the goal-scaling path; cube-to-goal values occupy indices 47:50, or 47:49 for xy only, before phase is appended. `train_jaw_bc.py` applies that same transform to data. `--explicit-goal` instead passes a separate `target_xy` goal; do not silently mix these representations. This is an implementation-specific fragile interface, not a general schema to propagate into unrelated tasks.

The collector `collect_place_from_acquire.py` supplies demonstrations from states the saved acquire actor actually produces. `collect_place_dagger.py` adds labels on states visited by a place actor. Preserve source/derived dataset hashes, seed cohorts, initial state and teacher identity. A teacher that emits a timed path cannot label a useful recovery for every slipped or displaced cube.

Before composing new skills, define the incoming body's state, goal representation, observable success/failure, timeout, stop behavior, and outgoing support/contact conditions. Exercise the uninterrupted transition, including a failed predecessor, rather than only two reset-based successes. An LLM may select a supported operation; it does not certify the entry conditions or physical outcome.

For separately trained actors, arbitrate **physical joint targets**, not raw normalized actions. Reconstruct each actor's target from its saved scale and offset, then convert to the one live action term with its actual joint-travel clip. Preserve the successor's trained base pose in channels it did not control; an otherwise successful squat actor destabilized a both-arm actor by continuing to change leg targets during arm raise. Log both actor observations, commands, full target vectors, applied filter state, foot support and contact at the switch. The [Direct Frame squat→arms packet](../../../../artifacts/ape-cove50-low-level-handoffs/squat-arms-cuda-v1/report.json) records the local failures and four corrected CUDA development passes. A fixed arbitration layer is a bounded composition, not evidence that an end-to-end combined actor was learned.

## Evidence checkpoints grounded in failures already seen

| Lesson | Observe and report | Shortcut that is not evidence of the skill |
| --- | --- | --- |
| Walk/turn/stop | Signed displacement or angle, command tracking, touchdowns, slip, stopping drift, torso stability, torque saturation | Reward gained while barely moving; asymmetric turn called bidirectional precision; completion time alone |
| Foot lift or step | Start state, actual loss of foot contact, swing clearance, planted-foot movement, landing and settling | Foot rocking on an edge; shifting the planted foot to satisfy relative separation; reset already placing the foot forward |
| Acquire/hold | Free-object rise, sustained two-jaw contact, support and object retention | Hand approaching the object; cube already held at reset counted as acquisition; brief lift counted as sustained possession |
| Place/handoff | Possession at switch, destination error, released jaw contact, hand withdrawal, tail drift and both-foot support | Robot motion or policy stage flag treated as a released object at the goal |
| Command/goal change | Observable command feature, nonoverlapping evaluation goals, held-out starts and response magnitude | Out-of-distribution command; overlapping target tolerances hiding a command-insensitive policy |

The historical six-second handoff gate used these reference values: two-jaw hold at or above 4 cm at the switch, at least 0.5 s of clear release, final xy error at most 15 mm, tail drift at most 2 mm and both feet grounded throughout. Report measured values against them and grade; they are a historical task reference, not universal hardware requirements. If one looks wrong for the question, record why and propose a change to the user. The fixed-center pair's narrow 14/15 result does not establish reach, transport or arbitrary destinations.

Keep body/proxy self-contact and effort beside success. The historical stand actor lost to zero action on support and torque; nominal gait/turn success often coexisted with saturation. Source-rated torque limits, unmeasured impedance and a neat mesh do not establish thermal duty or print clearance. Use `ape-body-transfer` for a measured physical model and paired dynamics replay.
