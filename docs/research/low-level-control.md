# Low-level control

Status 2026-11-20. Training and simulation are paused until the body design is finished (owner, 2026-11-17). The current control approach is the learned whole-body controller (WBC) in `control/wbc/`. Its README holds the numbered steps 1-23 with their commands, `control/wbc/PLAN.md` the 2026-11-07 review and plan. Results are rows in `data/results.sqlite` (learning dashboard, `infra/dashboard/results_hub.py`). The reward checks to run before a paid job are in `.agents/skills/ape-control/references/reward-design.md`.

The score is agility across all motions. Results are distributions over every evaluated motion and over motions the controller never trained on. The motions on the target (`playful-ape-t1` in `infra/config/targets.json`, 59 motions) are made-up stand-ins. A single motion's value below only shows where motion degrades.

## 1. Where the WBC stands (2026-11-12)

**Plant.** The R3 model (`robot/model/current-model.json`, 22 axes, 3.83 kg as built) with the r4 best-estimate actuator dynamics (`robot/model/dynamics/`: rotor inertia, Coulomb friction, the Feetech firmware duty loop, backlash, torque-speed line, 3S pack rail) and the tier-3 motor map (`robot/model/motor-map/`). The long-arm body (upper arm 1.6x, forearm 1.27x, 30 mm fist, feet 1.3x) is applied at training time and is not in the exported URDF. 2 ms physics, 50 Hz control, 20 ms command and 10 ms feedback delay. The model was built on 2026-10-26, before the printed FDM body was selected (`robot/design.json`, 2026-11-02 to 11-05) and before the motor refit. Nothing here is a physical-transfer result.

**Controller.** Upstream MJLab BeyondMimic tracking, PPO, 2,048 worlds on one RTX 4090, residual actions (target = reference + 0.5 x action), adaptive reward widths, a one-frame reference lead. No policy is promoted in `control/wbc/registry.json`. Today's controller is the three-tier teacher (`cove50-wbc-t1-tier3-longarm-20261108`). The default training setup is that teacher trained on the varied tape on the plane ("trial A", `cove50-wbc-t2-varied-motions-20261109`, Idle decision `M-9rahw2`).

**Tracking across all motions (plane, nominal).** RMS joint error per motion, median (p10-p90), with the number of motions:

| Motion set | Controller | Legs, rad | Arms, rad |
|---|---|---|---|
| 59 target motions | today's | 0.0195 (0.0158-0.0268) | 0.0278 (0.0161-0.0684) |
| 35 never-trained (25 in-between tempos, 4 mirror images, 6 arm moves on leg moves) | today's | 0.0196 (0.0161-0.0309) | 0.030 (0.0195-0.0689) |
| 6 never-trained arm moves on leg moves | today's | 0.0306 (0.0214-0.0391) | 0.043 (0.0335-0.1209) |
| 14 windows of four target motions in a new order, 0.5 s blends | today's | 0.0228 (0.0148-0.0257) | 0.0299 (0.0246-0.0966) |
| 99 evaluated motions | trial A | 0.020 (0.0175-0.032) | 0.0294 |

- Never-trained tempos and mirror images track like trained motions. Arm moves layered on leg moves are the gap: walking while waving ended 21.8 cm off its path. Trial A's varied tape (124 motions: references clipped to the joint stops, mirror images, arm-leg combinations, shuffled orders and blends) brings that to 8.1 cm and the combinations to legs 0.026 / arms 0.038 rad. Mirror images of the held-out motions are about 11% worse, because the tape holds mirrors of other motions.
- The weakest motions were every chest beat, at every tempo (arm error 0.13-0.19 rad). Their references asked shoulder yaw for 0.21 rad past its 60 deg stop on 11-15% of frames; 20 of 94 references passed a stop somewhere. This was a reference fault. References are now clipped to the stops when the tape is built.
- Fall-condition events: 0 of 944 (target), 0 of 560 (never-trained), 0 of 224 (chained), 0 of 1,584 (trial A). The push in these evaluations is one 0.1-0.2 m/s kick at 1 s and changes little, so push response is not measured.

**Obstacle world (20 worlds: two tiles each of rough and rolling ground, stairs, box grid, mats and sills, rooms, clutter, hollows, two hills at 7.7 and 8.6 deg; 35 never-trained motions; no terrain observation).**

| Controller | Falls on hills, of 70 | Falls elsewhere, of 630 | Legs, rad | Arms, rad | Travelling motions, end off path |
|---|---|---|---|---|---|
| Today's, flat references | 39 | 23 | 0.0215 | 0.0305 | 10.5 cm (3.5 cm on the plane) |
| Trial A, slope-aware references | 27 | 16 | 0.0211 | 0.0305 | 9.05 cm |
| C: trained on the world, 760 PPO updates as A | 11 | 8 | 0.0253 | 0.0488 | 5.0 cm |

- Slope-aware references turn each reference foot with the ground under it and put the turn into the ankles. Hill drift fell from a median 23.7 cm (p90 76) to 7.7 cm (p90 43). They are the default on the world (`cove50-wbc-slope-refs-smoke-20261112`).
- World training (C, `cove50-wbc-world-matched-20261112`) is a trade: more robust on uneven ground, looser everywhere. On the plane its legs track 1.20x and its arms 1.50x trial A's (worse on 88-89 of 99 motions). Its arm widths ended only 10-30% wider than A's, so widths do not explain the arm loss. One reading, not tested: on uneven ground the arms are recruited for balance. The plane stays the default training world; C is kept as the robust candidate.
- The world simulates 3.6x slower. The plateau stop works on fixed time chunks, so an unmatched world run got 166 updates against the plane's 760 (`cove50-wbc-t2-varied-world3-widths-20261109`). Compare world and plane runs on matched updates.

**Joint load on the three-tier set** (today's controller, 18 target motions, RMS current over rated). Hip roll on the STS3250 peaks at 0.98x in the turn (1.18x on the ST3025) and runs 0.63x over the tape. Hip yaw 0.41x (peak 0.78x), ankle roll 0.53x (peak 0.78x), hip pitch 0.54x, knee 0.38x, ankle pitch 0.47x. Shoulder yaw reaches 1.06x in the 2.5 Hz chest beat. Hip roll stays the hottest leg joint on every servo set (today's controller: 0.60x on the plane, 0.70x on the obstacle world). The tier pick itself is in `robot/actuators/tier-pick/`.

## 2. What the WBC work taught (2026-11-01 to 11-12)

Reward and reference (detail and checks in `reward-design.md`):

- **Upstream G1 reward widths let standing pay at 50 cm.** With root and body position widths of 0.3 m the teacher stood through a 1.54 m walk clip (root within 0.12 m). Scaling length widths by the height ratio (0.12 m, 0.4 m/s) made it step.
- **A residual action was the change that made the legs follow.** Five changes at once (feasible tape, residual actions, joint-angle term, 5 cm torso termination, error-driven sampling) took the deep squat from 0.5 to 6.2 of 6.8 cm torso travel and cut joint error 1.8-5.6x. The joint-angle term alone did not squat. The run did not separate the five.
- **Wide fixed widths leave errors free.** A 22-joint mean at 0.5 rad let both knees sit 0.08-0.25 rad straight. A knee term (0.1 rad) fixed it; shrink-only adaptive widths that follow the population error (PBHC) then fixed it without a knee term and halved leg error (legs 0.019-0.050 rad, squat full depth, march swing 45 / 43 of 40 mm).
- **The upstream reward paid for being 20 ms late.** mjlab compares the robot at t + 20 ms with the reference at t, and every policy settled one frame late, also with half the plant delay. A one-frame reference lead put the policy on time (knee lag 19-25 ms to -2-6 ms) and cut leg error by a quarter to a third. It is the default.
- **Travel needs a global term.** Every body term is torso-relative, so a fixed 11.4 cm global root width scored stepping in place at 0.87. An adaptive root width (11.4 cm to a 3 cm floor) and a swing threshold below the smallest reference swing fixed a knuckle walk that stepped in place.
- **Torso height and vertical-speed terms bring hop flight back** (the 2 cm hop: 19 of the reference's 19 frames with both feet up on the long-arm body) but doubled forward-walk drift (16.9 to 33.8 cm). Without them, training traded flight for safety.
- **References must be feasible and physically consistent.** Commands parked past joint stops get no restoring signal, so the policy leaves them there (elbows, neck). A hip cap of +-0.35 rad in the plant blocked squats. References past the stops ask for motion the robot cannot make. Clip references to the stops; check plant ranges on the CPU before a GPU job.
- **A time-stretched clip is not a slower walk.** Step length stays and single support lasts 1.67x longer; the teacher stood almost in place.

Training:

- **Return flattens after about 25-40 M transitions.** Past that, more budget re-balanced arm against leg tracking (two long runs, 0.2 B transitions) rather than adding agility. A plateau stop on the training return ends a job; it is a compute stop, not a grade.
- **2 ms physics matches 0.25 ms** (joint error within 0.0013 rad on every segment, same squat travel and current within 1%) and trains 4.6x faster (about 18,000 transitions/s at 2,048 worlds on a 4090).
- **A continued teacher restarts its adaptive widths** unless the plan sets `init_widths`; the first ~300 of 760 rounds of trial A re-tightened them.
- **Character moves are within reach zero-shot.** Untrained faster tempos, ape moves and the side shuffle ran at about trained-motion level before training on them; the knuckle moves put the knuckles down where the reference does before ever seeing one.
- **A G1-scale warm start does not transfer.** GEAR-SONIC, rebuilt from its ONNX release and mapped onto R3, falls at about 0.5 s zero-shot; 105 PPO rounds made it worse. R3 physics is latency-bound: 4,096-8,192 worlds on an H100 gave no more transitions/s than 2,048 on the 4090.
- **Paid bootstrap is a real cost.** Workers spent 2-15 min of GPU time downloading wheels per job; one job died of it.

Cost: about 119 GPU-h (upper estimate from job plans, roughly $130-150) through 2026-11-07, of which `PLAN.md` grades 35-40 GPU-h as low value (an invented 86-motion tempo ladder, 24 V hop and knee lines). Reservations from 2026-11-08 to 11-12 add about 38 GPU-h as an upper bound (`data/spend-ledger.jsonl`).

## 3. Servo and body findings that feed the design

All from simulation with catalogue best estimates; no bench measurement exists.

- **Rotor inertia dominates the 1:345 hobby servos.** BAM's identified 0.024 kg m2 reflected inertia is 8x the knee link's and 200x the shoulder yaw's. With it, the learned 0.2 m/s walk is beyond the C047 stall curve at some leg joint in 41% of frames; without it, 4.5% of frames are beyond rated. The training model before r4 had armature 0.
- **Trained comparison on the 12 V servo sets** (one seed each, 3,600 s from one teacher, 16 segments): the ST3025 / STS3250 mixed set tracked like all-ST3025 at 0.59 kg less. The C047 legs ran 1.1-1.4x rated on the walk (up to 1.5x faster). Hip roll is the hottest leg joint in the turn on every set (0.82-1.23x). On the 12 V plant neither reward, timing nor reference preview raised peak knee speed (about 0.8 of the reference); the 24 V plant reached it. Under the v2 reward a 24 V AK45 knee showed no visible gain.
- **CPU force-check levers** (inverse dynamics, no training): parallel springs at knee, hip and ankle pitch cut RMS leg torque 18-36% with unchanged peaks. Moving the ankle servo up to the knee gives nothing, because rotor inertia dominates, not link mass. Feet 1.3x (sole 86 x 60 to 112 x 78 mm) bring the 1.5-4 cm hops and a 3.5 Hz bounce within the 24 V motor lines. Even a 1 cm flat-footed hop needs about 15 rad/s, beyond the 12 V servos; hops are stress tests, never targets.
- **Long arms and knuckle stance.** Today's arm reaches 0.22 m; a bent-knee knuckle stance would need 131 deg hip flexion against the 110 deg stop. The chosen long-arm body (upper arm 1.6x, forearm 1.27x, 30 mm fist) reaches 0.265 m at 3.97 kg and 78 kg cm2 shoulder inertia (35 today); its knuckle stance leans 68 deg with hips about 102 deg flexed. The 35 deg ankle stop limits how low the hips go. The knuckle walk can lift a foot about 12 mm; 15 mm flips the hip roll. Trained knuckle moves track with contact agreement 0.96-0.99 and 1-2 mm torso height error.
- **Earlier body, single support** (Direct Frame, 2026-10-10): a straight lateral weight shift needed 2.21 N m of stance hip roll (1.41x the 1.57 N m rated) and the lateral COM sat 8.1-9.0 cm from the stance sole centre against a 3.5 cm sole half-width. Swapping the ankle drive moved the limit to hip roll instead of removing it. Sole width, stance width and hip-roll torque belong together in the design.

## 4. Before the WBC (2026-09-26 to 10-30), condensed

Superseded code was removed on 2026-11-12. The R3 per-skill policies of 2026-10-26 to 10-30 remain in `control/skills/` (`profiles.json`: forward walk, walk-to-stop, bilateral and independent arms, supported transitions) until the WBC supersedes them. Most earlier results ran on the Direct Frame (a 21-axis, 3.354 kg simulation body, rejected 2026-10-10) and a 35 cm body. What they showed, as a graded reading:

| Capability | Reading on those bodies | Main limit |
|---|---|---|
| Stand, shallow squat, rise from a crouch | Rough posture components | No push or fall recovery measured |
| Forward walk, turn | 0.76-1.26 m and 1.4-1.5 rad in 4 s | A leg at rated torque in 90-100% of moving frames |
| Reverse and sideways | Under 50 mm in 4 s after three PPO attempts (2.6 M transitions) | A velocity reward that pays at zero speed: the policy parked |
| Single-foot release | Not obtained in seven authored and learned families | Stance hip roll above rated; no teacher on the body |
| Reach to fingertip contact | 3.4-5.1 s contact with a simulator-truth goal | Live camera goal moves 12-20 mm and breaks contact |
| Two-pad pinch | Not obtained (0.135 mm closure margin on a 20 mm cube) | No attained pregrasp state |
| Walk, stop, squat, arms with no reset | Four saved actors behind a router | A distilled shared network failed every fresh sequence |

## 5. Learning lessons that carry to any body

1. **Reward and success diverge.** A support reward rose to 3.28 with 0 of 4 fresh supports; failed residuals out-scored their passing references. Compare matched successful, parked, falling and slipping traces, not the terminal flag.
2. **Hops, parks and shuffles are attractors.** A footfall count (24 alternating touchdowns, 20 mm travel) is not travel.
3. **Commands and observations must be in distribution.** A forward-only command normaliser put a reverse command 10.6 standard deviations out; a missing heading error turned a squat into a spin.
4. **Behaviour cloning copies the route, not the correction.** Offline action errors of 0.001-0.01 gave 0 of 4 closed-loop successes. A teacher must be shown to recover from learner-visited states before its labels count as corrective.
5. **Continuations help only when the changed factor addresses the failure.** Three cardinal-direction continuations bought nothing in the missing directions.
6. **Interface faults look like learning failures.** Zero-filled action channels, one-step-shifted demonstration rows, wrong history inputs, axis maps that moved 11 indices. Check first-step observation, action and state parity before spending.
7. **The environment can break silently.** A table left at the origin while cubes moved (0 of 64 touches), startup-only randomisation, a curriculum that never advanced. Record actual draws and effective contact.
8. **Throughput hides early falls.** A 4,096-world sweep at 16,166 steps/s had a mean episode of 1.04 s.
9. **Keep simulator truth out of the actor.** A touch teacher that read exact cube coordinates and contact did not survive an RGB cue.

## 6. Measurement contract

- One source record per episode: body, world, weights, initial state, seed family, command and disturbance schedule, requested horizon, lifetime, termination, native series, clip. Count world slots, episodes, independent start families, transitions, optimizer updates and distinct checkpoints separately.
- Accuracy is continuous command and tracking error in native units, with requested-horizon coverage beside it, so a short-lived controller cannot look good on its first second. Early falls stay in the cohort.
- Report median, p10, p90 (and p99 or extrema where tails matter) with n, per condition and over all motions; uncertainty is by start family or training seed. Every WBC comparison above is one seed.
- Pick clips from median and tail ranges and every severe event; describe what the controller attempted, how it adapted and where it broke down. No composite score.

## 7. Open questions

Ordered by how much an answer changes a decision. The first four are the questions the WBC must answer for the electronics (root task `T-xjjwrs`).

1. **Control rate and latency.** Evaluate today's controller at 0-40 ms command delay; train a 100 Hz variant only if quality falls within the latency the bus can reach. Not run.
2. **Onboard compute.** Export the actor to ONNX and time it in a loop on the ROCK 4D. Not run.
3. **Sparse upstream commands.** Targets at 5-10 Hz with brief dropouts, and masked multi-mode commands (HOVER-style). Not run; the new-order chained windows above are the only related result.
4. **Hip-roll heat.** Hip roll is the hottest leg joint on every set. A bench on the ST3025 and STS3250 (heating at the shuffle and turn duty, torque-speed line, firmware loop, STS3250 rotor inertia) would re-centre the r4 model. No bench has run.
5. **Plant from the selected body.** Rebuild the training model from the selected printed body and its motor map before training resumes, and fold the long-arm body into the exported model.
6. **Uneven ground without losing precision.** Mixed plane and world training, or arm terms on uneven ground, before the world becomes the default. Hills remain the weakest tile.
7. **Push and recovery.** The 0.1-0.2 m/s kick is too small to measure recovery. Nothing on push, fall or getting up is measured.
8. **Seeds.** Every comparison is one seed; checkpoint variance is not separated from the effect of a change.

## Evidence index

- WBC code and steps: `control/wbc/README.md`, `control/wbc/PLAN.md`, `control/wbc/agility/` (`library.py`, `demand.py`, `allocation.py`, `tape.py`, `target_check.py`, `world_tiles.py`), `control/wbc/compute/` (`launch.py`, `stage.py`, `report.py`).
- WBC records (`data/results.sqlite`): `cove50-wbc-t1-tier3-longarm-20261108`, `-t1-tier3-validation-20261108`, `-t1-tier3-world3-eval-20261109`, `-t2-varied-motions-20261109`, `-t2-varied-world3-widths-20261109`, `-slope-refs-smoke-20261112`, `-world-matched-20261112`; earlier steps `cove50-wbc-signal-mixed-20261106`, `-long1-mixed-20261106`, `-long2-mixed-20261106`, `-physics-step-20261106`, `-servo-compare-20261106`, `-reward-v2-20261106`, `-agility-timing-20261106`, `-loop3-ape-v5-20261106`, `-agility-demand-v2-20261105`. Idle root `T-xjjwrs`, servo and bridge task `T-qyajrv`, world tasks `T-9nn8az`, `T-vj9x22`.
- Plant and servos: `robot/model/current-model.json`, `robot/model/dynamics/README.md`, `robot/model/motor-map/`, `robot/actuators/tier-pick/README.md`.
- Earlier bodies: `control/skills/profiles.json`, `control/skills/registry.json`; Direct Frame artifacts under `artifacts/ape-cove50-*` and `artifacts/ape-low-level-training-2026101*/`; Idle `T-9yy63g`, `T-h3mkva`, `T-a2h676`, `T-2gke3v`, `T-gzqw3x`. The full per-skill narrative of 2026-09-26 to 10-16 is in git history of this file.
