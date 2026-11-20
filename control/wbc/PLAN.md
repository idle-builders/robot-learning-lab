# Cove-50 WBC and servo selection: review and plan (2026-11-07)

## 1. The intent

The owner's goal is a 50 cm ape robot that moves in an agile, playful way, like a little monkey or a baby ape, on a
hobby budget. From the root task (Idle `T-xjjwrs`, interview of 2026-11-01):

- The WBC is the bridge from slower sources (skills, VLAs, teleop) to the servos, and it runs on the robot.
- It must follow their commands quickly and recover, and turn imperfect references into feasible, lively motion.
- It should answer four questions for the electronics:
  - which control rate agility needs;
  - whether it should output gains or torque limits;
  - whether the onboard ROCK 4D can run it;
  - which servos to build with and test on a bench.

**The score is agility across all motions (owner, 2026-11-08).** This is a whole-body controller on an agile ape
design. The target's motions are made-up stand-ins, and the final set is not known. Any single motion, or single
family, is irrelevant on its own. No run, reward change, servo choice or body change is planned around one; it is
judged by distributions over all motions, including motions the controller has not trained on.
On 2026-11-08 a reward change aimed at the fast arm beats was stopped for this reason, after about 0.27 GPU-h.

## 2. What I chased this week, and what it cost

**The benchmark I actually used.** No target was ever declared. I built a reference tape of my own, 41 motions and
later 86, with tempo ladders made to run past the hardware limits. I then judged runs by how closely they tracked it:
- joint error;
- peak knee speed against the reference;
- frames with both feet in the air;
- foot lift and walk drift.

For the servos I required every tape motion to stay within 10% of frames past the motor line, which let one or two
extreme motions set the price. None of the root task's own questions (rate, latency, upstream commands, onboard
compute, gains or torque limits, skills on top) received a single GPU-hour.

**Everything I called agility.** 86 motions in 20 families, with every tempo:

| Family | Tempos | Now |
|---|---|---|
| Forward walk (learned 0.2 m/s walk, replayed faster) | ×1, ×1.1, ×1.25, ×1.5; walk-stop | ×1, ×1.1, stop on target |
| Turn in place | 30, 60 °/s | target |
| March in place | 0.5, 0.4, 0.33 s per step | 0.5 s target |
| Side shuffle | 0.8, 0.6, 0.5, 0.4, 0.33 s per step | 0.8–0.4 s target |
| Deep squat / deep rise (time-compressed rollouts) | ×1, ×2, ×3, ×4 | squat ×1–×3, rise ×1–×2 target |
| Quick crouch | 0.6, 0.35, 0.27, 0.2 s | 0.6–0.27 s target |
| Knee bounce | 1.5, 2, 2.5, 3, 3.5 Hz | 1.5–2.5 Hz target |
| Hop, flat-footed with flight | 1, 1.5, 2, 4 cm | stress test only |
| Side sway | 0.75, 1.5, 2, 2.5 Hz | 0.75, 1.5 Hz target |
| Bilateral / alternating arm raises | ×1–×4 | ×1, ×2 target |
| Chest beat | 1, 2, 2.5, 3 Hz | 1–2.5 Hz target |
| Overhead wave | 1, 2, 2.5, 3 Hz | 1–2.5 Hz target |
| Arm display swing | 0.5, 1, 1.5, 2 Hz | 0.5, 1 Hz target |
| Ape crouch bounce, excited beat bounce | 1.5, 2, 2.5 Hz | target |
| Side rock with overhead reach | 0.5, 0.75, 1 Hz | 0.5 Hz target |
| Crouch scan | 0.5 Hz | target |
| Knuckle drop, rock, sway, bounce, walk | 0.45–0.8 s, 1–2.5 Hz, 0.8–2 s per cycle | all 16 target |

**Metrics I collected:**
- joint RMS error (legs, arms) and knee offset;
- peak joint speed over the reference speed, and knee lag (cross-correlation, sub-frame);
- foot contact agreement, swing ratio, frames with both feet up;
- foot lifts and peak lift height, torso travel and vertical speed;
- walk travel, final position error and heading;
- knuckle contact agreement;
- servo saturation share, current over rated and torque above rated;
- episodes that met a termination condition;
- training return, transitions and optimizer updates.

**Yes/no gates I used. All are retired.**
- Tape selection: a motion was dropped when more than 10% of frames were past the motor line or the base residual
  p95 was above 5% of weight (7% on 24 V).
- Category tiers: "tape" required at most 10% beyond, "dynamic" at most 2%, core RMS at most rated and any motion at
  most 1.6× rated, all on every motion.
- "0 of 832 episodes left the reference" was used as a headline.

**Each line of work, graded 0–5 on its value to the intent.** GPU-hours are upper estimates from job plans (about
119 in all, roughly $130–150).

| Line of work | GPU-h | What it produced | Value |
|---|---|---|---|
| Tracking port and first teacher | 1.9 | The working training stack everything stands on | 4.0 |
| GEAR-SONIC warm start | 9.5 | G1-scale model does not transfer to 50 cm (falls at 0.5 s); H100 no faster, physics is latency-bound. Cost inflated by a silent worker and a recipe confound | 2.0 |
| Walk teacher, reward widths | 10.6 | G1 widths let standing pay; scaled widths made stepping appear | 3.0 |
| Your direction checks (slow clip, torque cap, STS3250) | 21.5 | A slowed clip is not a slower walk; torque headroom removes falls. Three extra sessions went to the C047, a servo now out | 2.5 |
| r4 best-estimate dynamics, three servos | 7.9 | Rotor inertia dominates the 1:345 servos; every servo near rated current on the walk | 4.5 |
| Multi-motion tapes 1–3 | 23.8 | Multi-motion training works, but the GPU found interface bugs (hip cap, commands past stops) that a CPU check would have caught | 2.0 |
| Residual actions, 2 ms physics, long runs | 7.9 | Legs follow squats; 2 ms physics gives 4.6× throughput at the same result; return flattens after 25–40 M transitions | 4.5 |
| Ape references and side shuffle | 4.1 | Character moves performed zero-shot at near-trained level | 3.5 |
| Servo comparison, 12 V | 5.7 | Mixed ST3025/STS3250 tracks like all-ST3025 at 0.59 kg less; C047 runs hot; hip roll is the hottest joint | 4.5 |
| Knee options (24 V AK45, geared) | 4.2 | Chased my 0.33–0.4 s marches and hops; the geared knee had an inverted gain. One keeper: the knee stance term | 1.5 |
| Reward v2 | 2.8 | Adaptive widths: legs track twice as closely, knees hold their bend | 4.0 |
| 12 V reward v3, reference lead, preview | 5.7 | The reward paid for a 20 ms lag; the fix cut leg error 25–35%. Hop crouch and planted feet fixed | 4.5 |
| 24 V feet and torso terms | 4.9 | Hops with flight on a plant and motion outside the target | 1.0 |
| 24 V long-arm body | 8.5 | Knuckle moves track (contact agreement 0.96–0.99, torso 1–2 mm), and found two reference/reward flaws. But on 24 V, with time spent on hop flight and drift | 2.5 |

About 35–40 GPU-hours (roughly a third) went to lines of low value. The larger loss is where I aimed: most runs
optimised my invented benchmark rather than the intent's questions. Free CPU work, the force checks and catalogue,
was useful as a screen. Its yes/no category method and its unsimulated proposals are retracted.

**Lessons carried forward:**
- The reward lead: the policy is on time.
- Residual actions, the knee stance term and adaptive widths.
- 2 ms physics; training long only up to the flattening point.
- Best-estimate rotor inertia.
- The 12 V mixed set; hip roll as the joint to watch.
- The knuckle body and moves.
- Workers spend 2–15 minutes of paid GPU time downloading Python wheels on every job (v6 died of it).

## 3. The target

`infra/config/targets.json`, `playful-ape-t1`, owner-approved on 2026-11-07, has 59 motions:
- the 43 lively motions in the table above;
- the 16 all-fours moves, on the long-arm body (upper arm ×1.6, forearm ×1.27, 30 mm fist, feet ×1.3).

Hops and the fastest tempos stay stress tests. They can be evaluated (on the GPU) and are never trained toward.

The target's "how well" grades are continuous 0–5, each with a stated reference scale:
- timing (knee lag);
- leg and arm tracking;
- planted contact;
- step and knuckle contact agreement;
- travel;
- heat over the whole tape;
- push response;
- tracking of target motions in an unseen order.

## 4. Plan

Each step states its question, the decision it can change, its cost and the cheaper check that comes first. Simulation,
training, evaluation and smoke tests run on the GPU only, never on the laptop CPU (owner, 2026-11-12); a smoke test is a short GPU job. Each GPU job writes its spend reflection first. There is no weekly cap; I keep the total reasonable and report it.

### Servo selection: mixed, brick-shaped, about three tiers

Settled by the owner (2026-11-08):
- Mixed motors, all brick-shaped like the C047, so the current CAD needs only small changes.
- From a full survey of brick-shaped actuators, about three tiers are picked to fill the 22 joints at the right
  performance and a good price.
- Picking the tiers is an engineering task: it compares datasheets and offers against the per-joint requirements in
  `studies/cove50-actuators/joint-requirements-playful-ape-t1.json`. It is not further WBC training or
  simulation.
- Whole-set comparisons (24 V, all one servo) are closed.

Inputs: `robot/actuators/brick-survey/` (204 part variants, 282 offers, being checked in the
browser) and the requirements file.

Today's two categories, the starting point:
- **Leg:** ST3025 (40 × 20 × 40 mm) on hip pitch, hip roll, knee and ankle pitch, 8 servos.
- **Light:** STS3250 (the C047 case) on the shoulders, elbow, hip yaw and ankle roll, 12 servos.

1. **Force check on the target** (done, CPU, free). See `robot/actuators/category-round/README.md`.
   - The leg category carries the target; hip roll is its hottest joint, at 0.9–1.1× rated in side shuffles and the
     turn.
   - The STS3250 is too slow for the leg joints, through its rotor inertia, but carries the light category.
2. **Trained run of the mixed set on the long-arm body and the target tape** (running,
   `t1-mixed-longarm-20261108`, at most 2.31 GPU-h).
   - It measures, per joint category, the trained current and heat, tracking, contacts and knuckle moves.
   - Decision: whether any joint needs its own category or motor. Hip roll is the candidate.
3. **Per-category deals** (subagent, read-only): best offers and package deals for each category, plus fitting
   alternatives labelled as not simulated (`studies/cove50-actuators/deals/`).
4. **Bench (your purchase decision).**
   - Parts: 2× ST3025 and 1× STS3250, about $228, plus a 12 V supply if none is on hand.
   - Measurements:
     - ST3025 hip-roll heating at the trained shuffle and turn duty over several minutes;
     - its torque–speed line and firmware loop;
     - the STS3250 rotor inertia.
   - Decision: keep the ST3025 on hip roll, or give hip roll its own motor. The data also re-centres the r4 model.
5. **Hip-roll levers**, only if steps 2 and 4 show the heat matters: a stronger fitting hip-roll motor (simulated
   first), or stance and sole width or a parallel spring, screened by the CPU force check before any CAD.

### The WBC as a bridge

5. **Cheap infrastructure** (no GPU).
   - A worker image or volume cache with the wheels preinstalled, saving 2–15 minutes of paid bootstrap per job.
   - A target scorer that grades every "how well" measure per motion from evaluation traces, through
     `infra.storage.score_guard`.
6. **Unseen order and sparse commands** (short GPU evaluation of the step 2 policy).
   - Target motions in a new order with new blends, and upstream targets arriving at 5–10 Hz with brief dropouts.
   - Decision: whether training needs randomized order and command-rate augmentation (then one GPU job).
7. **Rate and latency** (short GPU evaluation first).
   - Evaluate the step 2 policy with 0–40 ms command delay.
   - Train a 100 Hz variant only if quality falls clearly within the latency the bus can reach.
   - Feeds the electronics rate requirement.
8. **Onboard compute** (inference timing only, on a labelled proxy).
   - Export the actor to ONNX and time it inside a loop on a labelled proxy, then on the ROCK 4D when available.
   - Feeds the compute decision.
9. **Masked multi-mode commands and a deployable student**, after steps 6–8 show what the interface needs. This is
   the HOVER-style input the root plan asked for.
10. **Later, each only on evidence:**
    - torque-limit or gain outputs (if the bench shows heat or torque matters);
    - skills relearned on top of the WBC;
    - an ape style prior from video (needs your permission for downloads).

**Stopped:**
- the agility loop (`T-27cyrn`);
- the yes/no category round (`T-z7k5kz`);
- 24 V feet, hop and knee lines;
- the 86-motion ladder as a goal.

**Expected GPU for this cycle:** step 2 now, then at most one or two jobs if steps 6–7 call for them. That is about
2–6 GPU-h, $2–7.

## 5. Guards in code

- **`infra/storage/spend_guard.py`.** A reflection, not a gate (owner, 2026-11-09: "an incentive to reflect if this training
  is useful and valuable"). Before every GPU job, `reflect()` prints and records in `data/spend-ledger.jsonl`:
  - the question, the target it serves, the decision it could change, the cheaper check that came first and its value;
  - the GPU hours it reserves, and the hours reserved in the last 7 days.

  It adds notes on anything worth a second look and never refuses: uncommitted limits or targets, an unapproved
  target, held-out validation motions or off-target motions on the training tape, a missing field, or more than the
  owner's per-job limit (2.5 GPU-h). Acting on a note is my judgment in line with `AGENTS.md`; spending beyond the
  limit still means asking the owner first. The Runpod `call()` in
  `control/common/compute/launch.py` matches each `run` request to its reflection and
  requires a provider execution timeout, the bound that stops a job billing after the work ends.
- **`infra/storage/score_guard.py`.** `infra/dashboard/results_hub.py publish` refuses a record that contains any of:
  - booleans outside configuration;
  - verdict keys (pass, fail, ok, keep, exclude, covered, feasible and the like) or verdict strings;
  - grades outside 0–5.

  `python -m infra.storage.score_guard FILE` checks any report. `grade()` gives the continuous 0–5 grades.
