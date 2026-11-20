# Landscape and references

Which robots and stacks exist, what each showed when we ran it, and the working rules those detours taught. Sizes and
prices are dated source claims (checked 2026-09-16 to 2026-10-16), not delivered costs. Hip-covering references (NAO,
QRIO, G1, Disney characters and others) are in `hip-shell-covering.md`; Cove's own control results in
`low-level-control.md`.

## 1. Current reading (2026-11-20)

- No open robot matches a 50 cm armed, playful ape. Microduck is the best small-biped learning stack, ToddlerBot the
  best size match with arms, Berkeley Humanoid Lite the best released ONNX / Isaac Lab humanoid stack. The rest are
  morphology, price or interaction references.
- Training stack in use: MJLab (MuJoCo-Warp, Isaac-style managers) with RSL-RL PPO on GPU pods, the stack Pollen uses
  for Microduck.
- The G1 / GR00T / VLA line was stopped by the owner on 2026-09-25 (G1 rejected as too large on 2026-09-22). It is a
  reference stack, not a body.
- Every reference that reports sim-to-real models actuator lag, friction, voltage and backlash explicitly. Unknown
  actuator parameters stay uncertainty bands until measured.

## 2. Existing robots

| Robot | Size / mass | Joints and compute | Use for us |
|---|---|---|---|
| Microduck (Pollen) | 25 cm, ~800 g (model 0.737 kg) | 14 policy axes (5 per leg, 4 neck/head), camera, LiDAR, two IMUs, 50 Hz policy | Closest compact learning reference; no arms. $399 preorder claim |
| Open Duck Mini v2 | 42 cm legs extended | 14 STS3215, Pi Zero 2 W, BNO055, foot contacts | Accessible build and sim-to-real guide. BOM EUR 398 |
| ToddlerBot | 56 cm, 3.4 kg | 30 DOF (6 per leg, 7 per arm, 2-axis waist and neck), Jetson Orin NX, stereo | Best size match with arms. About $6,000 BOM, 90% motors and computer. 19 min walking before overheating affected stability; up to seven falls before a break; repair 21 min print + 14 min assembly |
| Berkeley Humanoid Lite | 0.8 m, 16 kg | 22 joints, printed cycloidal gearboxes | Open URDF / MJCF / USD, Isaac Lab task, ONNX walker. Under $5,000. Authors note printed gearbox wear and heat |
| ROBOTIS OP3 | 0.51 m, 3.5 kg, 20 joints | brick servos like ours, MuJoCo model, learned soccer | Concurrent-hip layout with brick servos (`robot/body/hip-shell/DECISIONS.md` A-18) |
| Zeroth-01 | ~40 cm, 2.63 kg CAD | 16 STS3250 / STS3215 | Built reference; see section 4.3 |
| TonyPi / AiNex (Hiwonder) | 37-42 cm, 1.8-2.45 kg | 18-24 DOF bus servos, Pi | Retail price benchmark ($600-1,000); no RL stack shown |
| Reachy Mini | ~30 x 20 x 15.5 cm | 9 axes, XL330 + XC330, CM4 | Expressive interaction reference; stationary |
| SO-101 / LeKiwi / XLeRobot | arms / mobile arms | STS3215 followers | Manipulation and data-collection references |
| Upkie, DIABLO | wheeled legs | qdd100 / direct drive | Wheeled-leg control references, far above hobby scale |

How the cheap designs save money (reading of the sources): small scale and short links cut torque and fall energy;
smart bus servos bundle motor, gearbox, encoder and controller but hide the current loop and give no calibrated torque;
joints go only where the task needs them. ToddlerBot's durability reporting (minutes to overheat, falls per break,
repair minutes) is the model to copy, not an "unbreakable" claim.

## 3. Embodiment choices recorded early (2026-09-25/26)

- Walking feet are the main embodiment. Two wheels give no fore-aft support polygon, so a tall wheeled body is an
  inverted pendulum. Single-support gait, foot placement and contact transitions are core only to walking feet.
- Keep hip yaw / roll / pitch and shoulder yaw / roll / pitch as independent axes; compare reductions only as measured
  ablations. A kickstand or training support is a training mode, not balance evidence.
- Multi-axis packaging is separate from axis count. The best sourced pattern is a dual-axis module (ROBOTIS 2XC430 /
  2XL430) plus a separate third axis.

## 4. Reference robots we ran

### 4.1 Microduck (2026-09-28)

The released stack ran: Onshape-exported MuJoCo model, BAM voltage-driven actuator model, ONNX walkers. Policy: 61
observations in, 14 position offsets out at 50 Hz; BAM converts to torque at 200 Hz with back-EMF and load-dependent
friction. Training randomises battery 6.5-8.2 V, sag gain 0-0.2, motor lag 15-30 ms.

| Released walker, 4 s at 0.3 m/s (1.2 m commanded) | Walk | Single-foot support |
|---|---:|---:|
| 7.4 V, no extra lag | 0.391 m | 86.5% |
| 6.0 V, 20 ms lag, 1.75 A cap | 0.449 m | 82.5% |
| 5.0 V, 20 ms lag, 1.47 A cap | 0.378 m | 83.0% |

- It walks under low supply and delay, but tracks poorly (about 37% of commanded travel). At 0.20 m/s it barely moved
  (0.008 m): slow playful navigation is an open target for us too.
- Native retraining on an L4 pod: 4,096 environments at about 42,000 steps/s in about 5 GB; 500 updates walked 0.50 m
  forward and 0.12 m sideways. Iteration count and reward do not replace displacement, support and video.
- Pollen feeds a 2S pack straight to XL330s rated 6.0 V maximum and clears the shutdown bit. That is their choice, not
  a safe target.

### 4.2 ToddlerBot (2026-09-22)

The released `toddlerbot_2xc_walk` policy ran in native MuJoCo: backward -0.1 moved -0.96 m upright; forward 0.1 fell
at 9.2 s; the upstream demo input moved +1.11 m. A bounded working walker in its documented configuration, not
full-range tracking. Exit code is not evidence (the falling run exits 0). Open code (MIT), CAD under CC BY-NC-SA 4.0,
motor system identification, MJX / RSL-RL training, ONNX deployment; manipulation is task-specific diffusion, no
language.

### 4.3 Zeroth-01 (2026-09-28)

A builder model of an assembled Zeroth-01 trained to stable standing under a rating-bounded motor model, not walking
(mean travel 0.070 m against 0.427 m commanded, legs at the rated clamp 99% of samples).
- The builder's fitted torque caps (8.72 N·m STS3250, 5.47 N·m STS3215) exceed the manufacturers' 12 V stall figures
  (4.90, 2.94 N·m) with no raw identification behind them.
- Force limits were never enabled in the model (`actuator_forcelimited`), and a step function ignored the per-joint
  speed ceiling. Lesson: check that a "sysid" actuator file is backed by measurements and that limits are actually
  applied. Rated torque, not stall, is the honest cap for a servo-driven leg.

### 4.4 Berkeley Humanoid Lite (2026-09-25/26)

The released 22-joint ONNX walker ran in MuJoCo and Isaac Lab; low-command tracking and lateral drift were weak
(stationary command drifted 0.15 m). On top of the frozen walker, PPO skills in a small playroom beat constant-forward
baselines (ball to goal 8/16 vs 5/16; foot pad 15/16 vs 3/16), and a pretrained Cosmos Reason2-2B selector picked the
right next skill in 5 of 6 recorded cases but repeated finished skills. Lessons that transfer:
- Initialise PPO near an already-working command; a proximity-heavy reward let hovering dominate.
- Success predicates must require the physical event (active-stage foot contact above 1 N, an outside-to-inside
  transition); sensors updated once per 40 ms control step missed brief impacts.
- Sim contact peaks of 30-164 N are not hardware force limits.
- GR00T distillation from 20 PPO rollouts reached 3 of 10 full successes at best; imitation loss alone did not predict
  closed-loop contact.

## 5. G1 / GR00T / VLA line (2026-09-17 to 09-25)

A working but narrow learned loop on Unitree G1 in Isaac Lab Arena: released policies ran natively; a fine-tuned GR00T
N1.6 cart-push policy passed 4 of 4 fresh trials from clean behind-cart starts; a continuous tray-to-station run was
never achieved. Lessons that still apply:
- A restored simulator state is not a continuous handoff. Test the actual state at skill handoff; a byte-identical
  handoff gave different action chunks and outcomes.
- Check physical predicates natively (an HDF5 `success=True` meant the operator saved the file). Failed or partial
  trajectories are not positive demonstrations.
- Keep the demonstration controller and the deployment controller the same.
- Trainers can silently drop data (only the first dataset path used; zero-filled columns for missing fields).
- Camera-rendered Isaac evaluation needs an RTX GPU with a compatible driver; A100 / H100 have no RT cores. GR00T N1.6
  fine-tuning peaked at about 36 GB.
- No released language-to-whole-body stack on a compact body was found (2026-09-22). OpenHLM (π0.5 + SONIC on G1, 34-D
  action) was released but not reproduced; do not mix its action layout with GR00T SONIC or LeRobot G1.

Language-to-skill structure (reading): a language level selects a skill, a VLA emits short action chunks, a fast
whole-body controller handles balance and contact. Learned selectors repeat completed skills; guard with a valid-choice
mask and re-read progress after each skill.

## 6. Upstream simulators and stacks (source audits, 2026-09-17 / 09-22)

- MJLab: Isaac-style manager API over MuJoCo-Warp; the working Cove stack.
- Isaac Lab 3.0 early access: Linux / NVIDIA only, quaternions XYZW, a task needs a declared `newton_mjwarp` preset; pin
  the tag, not the branch.
- MuJoCo Playground, LeRobot (0.6.1: Python >= 3.12), LeIsaac, Isaac Lab Arena, ManiSkill, Gymnasium: audited. ManiSkill's
  default passive-force compensation must not silently assist a free base.
- WBC-AGILE (NVIDIA) was chosen over retraining SONIC as a motor RL recipe; joint maps and controllable base directions
  are ours to supply.
- Rerun as viewer and recorder: log time is not acquisition time; give episode and device times explicitly.
- Pretrained controllers (plan and source audit, 2026-10-16; no transfer result on Cove yet). Principle: copy a model's
  whole execution contract (topology, frames, action semantics, observation order, normalisation, timing, resets),
  run it with zero updates first, then change only what the body requires. LoRA comes after a correct interface; it
  cannot repair a wrong joint map or missing motor authority.

  | Lane | Released reference | Cove must supply |
  |---|---|---|
  | Locomotion | URMA ("One Policy to Run Them All"), GenBot-1K distilled weights | body description, joint masks, feet, action scale, motor response inside the checkpoint's morphology range |
  | Posture, reach, whole body | GR00T whole-body control (learned lower body + IK upper body), SONIC, OpenLoong | Cove kinematics and feasible targets; a G1 SONIC checkpoint has 29 outputs, renaming joints does not make it a Cove controller |
  | Visual reach and contact | FLUX action models, GR00T custom embodiment (fine-tune only) | camera calibration, timing, hand representation, IK layer, aligned examples |

  XHugWBC reports zero-shot cross-humanoid control through morphology-aware inputs; Any2Any aligns a pretrained tracker
  to a new robot, then adapts the dynamics-sensitive modules.
- World models (Cosmos, GR00T-Dreams, DreamZero) are data-generation or VLA research tracks. Generated video is not
  physical rollout evidence.

## 7. RoboCasa as a scene (2026-10-10)

Upstream RoboCasa (MuJoCo, 44 fixtures) works as a realistic kitchen for ape-height perception. At head height 0.46 m
and chest 0.30 m, a can on a lowered shelf was visible (100+ pixels) in 25 of 48 frames; an identical can behind the
counter was never visible. Consequence: a 50 cm robot cannot identify an item it never sees; the planner keeps "last
known or unknown" apart from "seen now". Third-party assets stay under ignored `.cache/`. Follow-on results are in the
perception and high-level behaviour files.

## 8. Agentic CAD / PCB / firmware workbench (2026-09-16 reading)

| Domain | Start with | Feedback the agent gets |
|---|---|---|
| Mechanical | build123d (OpenCascade); CadQuery or FreeCAD when needed | validity, dimensions, interference, mass, STEP, renders (today: `robot/body/checks/run.py`) |
| Robot model | one manifest generating MJCF and URDF | frames, limits, mass / inertia consistency |
| Electronics | KiCad CLI; atopile for new modules | ERC / DRC JSON, netlist, BOM, STEP |
| Firmware | PlatformIO or native SDK | build output, protocol tests, serial logs, timing |

- KiCad 9/10 IPC needs a running GUI; headless IPC is documented for KiCad 11.
- atopile's AI-Pin example disables its own PCB DRC; copying it is not evidence of a passing board. One authoritative
  source per fact: do not run two generators and hand edits side by side.
- A model grading its own geometry from its description is not a check. A result record carries source commit, tool
  versions, parameters, seed, artifact hashes, exit status and the measured value with units.

## 9. Evaluation rules from the 2026-10-07 source audit

An audit of earlier study notes against primary sources left these corrections:
1. Model existence, a paper result and working onboard integration are three different claims. Keep small PPO / BC,
   geometry and estimation baselines until a matched ape experiment supports a change.
2. A transform name needs an equation (`p_W = T_W_C p_C` maps camera into world). MuJoCo image projection is not the
   ROS optical convention.
3. Best-effort QoS does not expire stale actions; capture time and action age are still needed.
4. Median and p95 do not give the out-of-tolerance fraction; repeated frames are not independent trials.
5. Zero failures in five independent trials leaves about a 45% one-sided 95% upper failure bound. Do not add trials
   until a wanted answer appears.
6. Simulator realism is conditional on solver, integrator, actuator model and observability; rendering realism does
   not identify contact mechanics.
7. Official sources disagree; use version-pinned code and configuration.

## 10. Cost and GPU context (dated)

- GPU rental seen 2026-09-28: L4 $0.49/h (Microduck and Zeroth runs); A40 and RTX 4090 did the later Cove work. Listed
  "low" stock is not a reservation; a stopped pod is tied to its machine.
- Retail benchmark: twelve Hiwonder HX-35HM cost $431.88. Actuator prices are in the actuators file.

## 11. Open questions

- Whether a released locomotion controller runs on the Cove body with zero updates (section 6).
- Slow, playful navigation: no reference walker tracks well below 0.3 m/s.
- Five versus six leg axes, a waist hinge or wheel feet at 0.4-0.5 m with grippers: no published comparison.
- Safe hand-contact force limits on hardware: unknown.
- Real sim-to-real transfer for any of these small bodies is unverified here.
