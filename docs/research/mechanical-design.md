# Mechanical design: frame, joints, packaging and printable structure

Results for the 50 cm ape (APE0). Every number is a CAD, FEM or simulation measurement. No printed part has been
built, weighed or loaded. The check tool and its per-check results are in `design-checks.md`; the hip shell look and
its references are in `hip-shell-covering.md`; motors, power and electronics are in
`actuators-power-and-electronics.md`.

## 1. Current body (2026-11-20)

The selected robot is the complete printed FDM body refitted to the locked three-tier servos: `robot/design.json`
(`body: robot/body/motor-refit`, selected by the owner 2026-11-12, Idle M-zh6fxd). Design checks run on its manifest
`artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json`.

How it was built:

| Step | What | Date |
|---|---|---|
| Four FDM part studies | lower body, 6 stages (`robot/body/leg`); arm, 4 stages (`arm`); neck / head, 4 stages (`neck-head`); torso, pelvis and 9-cell battery carrier, 4 stages (`torso`) | selected 2026-11-02 to 11-06 |
| r1 assembly | the four studies joined (`robot/body/assembly`) | 2026-11-09 |
| Motor refit | r1 refitted to ST3025 / STS3250 / C047 (`robot/body/motor-refit`) | selected 2026-11-12 |
| Roll-flip | hip-roll STS3250 case moved into the thigh, output bolted to the pitch-to-roll connector; roll axis stays at z 190 (owner D-12 / D-13, `robot/body/hip-shell/DECISIONS.md`) | 2026-11-16 onward |
| Ape proportions | upper arm 55.6 → 87.2 mm (elbow and hand 32 mm lower), foot monocoque 60 x 86 → 78 x 112 mm (Idle T-gqwdrn) | 2026-11-19 |

Motor refit changes (2026-11-12, T-jrnxfn; history: the refit report `artifacts/robot/body/motor-refit/report.json` was
compiled by `robot/body/motor-refit/report.py` from stored outputs of the retired per-study sweep and FEM scripts and cannot
be recreated, so both were removed 2026-11-25; the script and these numbers are in git history before that date, e.g. `eb1fccc92`):
- Hip-pitch output faces +4 mm per side (spacing 118 → 126 mm). Knee and ankle output span 37.2 → 51.0 mm. Lower-pelvis
  cells lie horizontally on the midline.
- Printed structure 476.3 → 492.0 g. Leg and arm servos 1,100 → 1,685 g. Model mass 3.830 → 4.495 kg
  (`robot/model/motor-map/README.md`).
- The neck keeps its XM335 pair (owner, M-revtdr). The neck collar lost its front (6,033 → 2,558 mm³) because the head
  skin sweeps through it over neck pitch.

The refit's FEM table (pre-roll-flip connectors, retired evaluator, old material keys) is in git history with the report.
The FEM of the current parts is the shared check, `robot/body/checks/run.py --checks fem,fem_setup`, with each part's
setup in its body definition and the declared datasheet materials (`robot/body/checks/data/materials.json`).

Joint ranges: the refit kept the per-joint ranges. Corner contacts between limbs, or a limb and the body, are
handed to the WBC as self-collision limits (owner, 2026-11-12). The refit's final sweeps listed 47 such contact pairs over
the hip pitch range, 27 over hip roll, 28 over the elbow, 22 over the knee (its `self-collision-limits.json`, written by the
same removed report script; git history). On the current body the `range_sweep`, `collision_sweep`, `clearance` and `fit`
checks count and list these pairs (`other_limb_pairs`).

Open on this body:
- The WBC training plant still uses R3 kinematics; the 4 mm hip shift (and now roll-flip and ape proportions) needs a
  model re-export before the WBC uses the body.
- 18 r1 sleeves were dropped with the old C047 screw positions; the new M2 / M3 holes carry no pressed sleeves yet.
  Every sleeve must be pressable: flat counter face and a path for the heated press (owner D-14, 2026-11-17).
- Front shell lower attachment: two M2 posts at z 257.5 with screw heads on the lower belly (visible).
- Not measured: press-fit retention, bolt preload relaxation, interlaminar fatigue under landing, creep near warm
  servos, harness loops, enclosed-shell battery thermal behaviour.

Baseline for the printed connectors (r1, PAHT-CF dry, min FEM safety factor over load cases), against the sheet-metal
parts they replaced:

| Subsystem | Printed | Sheet metal | Change | Min safety factor |
|---|---:|---:|---:|---:|
| Lower body (12 parts) | 187.6 g | 332.8 g | -44% | 2.86 |
| Arm structure (8 parts, plus 8 hand parts) | 74.7 g | 136.1 g | -45% | 1.65 |
| Neck and head (4 parts) | 28.6 g | 52.7 g | -46% | 2.34 |
| Torso, pelvis, battery carrier (4 parts) | 178.2 g | 251.8 g | -29% | 1.55 |

## 2. Owner decisions that bound the body

- 2026-10-10: the exposed 21-axis "Direct Frame" (selected 2026-10-08) rejected (Idle D-sznrsa). The redesign kept the
  sculpted 50 cm Cove hull as exterior datum and rebuilt the body section by section. The October section selections
  (direct serial hip, double-moving-case neck, foot-fixed-roll ankle, bent-aluminium upper body, symmetric geared
  hand) live in Git tags `design/<study>`; the FDM studies replaced them.
- 2026-11-08: three-tier motor set locked (M-44khpr). 2026-11-12: neck stays XM335 (M-revtdr).
- 2026-11-16: sleek ape silhouette, no outward bulbous joint covers; keep today's hip actuators; two hip shells, each
  covering two axes, with edges as tight arcs about the joint axis; roll motor flipped into the thigh (D-09, D-11,
  D-12, D-13).
- 2026-11-17: finish the whole body design before any training or controller work (M-evsb2n). Every steel sleeve
  pressable (D-14).

## 3. Proportions and joint count

Current: the owner's Cove ape sketch (`robot/shell/sketch/reference/sketch.png`) sets the proportions: tall rounded
torso, outward shoulder line, short legs, brief-style pelvis, broad boots. 22 actuated joints: hip yaw, roll, pitch,
knee, ankle pitch and ankle roll per leg; shoulder pitch, roll, yaw and elbow per arm; neck yaw and pitch; a geared
gripper per hand. The 2026-11-19 ape proportions lengthen the arms and feet (section 1).

What the early screens showed (35-45 cm bodies, 2026-09-26 to 09-30, authored or short-PPO controllers; weak evidence
for ranking, kept for the direction):
- Low shoulders with arms long enough to reach the floor helped floor work most. Feet fixed, deep crouch, target 13 cm
  forward / 10 cm left / 5 cm high: a 45 cm low-shoulder body (shoulder 30 cm, arm 22 cm) reached it; a balanced body
  missed by 3 cm, ToddlerBot (56 cm) by 8 cm, long legs with short arms by 10 cm.
- A waist pitch joint restored far reach but peaked at 38.5 N combined sole force (22.8 N without) and saturated the
  waist actuator on 6.75% of steps.
- The elbow was added on 2026-09-29: a straight 25 cm arm at the 35 cm body's shoulder height would sit about 46 mm
  below the floor. Extra shoulder axes on a straight arm change hand orientation, not reach.
- A packaged multi-axis joint reduces parts, not controlled DOF. Passive or sequential motion is not an actuated axis.

Open: no proportion ranking with a controller trained per body exists on the 50 cm ape.

## 4. Mass and materials

Current: model mass 4.495 kg with the three-tier servos (section 1). Printed structure 492 g. Battery cells alone
639 g (nine supplier-maximum P50B cells). Nothing weighed.

- Density references: PAHT-CF 1.06 g/cm³ (Bambu data sheet; the checks' material table holds the data-sheet values),
  PETG-CF 1.25, PETG 1.24-1.30, PA12 SLS 0.93, 6061 aluminium 2.71, steel 7.7-8.1, brass inserts 8.3-8.8, TPU 1.12.
  A printed part's real mass needs a weighed sample.
- Datasheet servo mass (e.g. 55 ± 1 g) excludes horns, screws and cable.
- Lesson (35 cm and Direct Frame bodies): a simulator mass that no buildable part owns acts as a free training weight.
  Derive link mass and inertia from the part ledger and re-adapt the policy when it changes. The 35 cm ape compiled
  at 2.27 kg with 1.0 kg of unallocated "structure"; the first 50 cm body had 1.41 kg of its 3.20 kg with no
  buildable owner.

## 5. Hip

Current (right side; left mirrors x): hip yaw axis vertical through (x 59, y 2); hip pitch along x through (y 2, z 219);
they meet at C = (59, 2, 219). Hip roll runs along y through (x 56, z 190), 29 mm below C. The chain is yaw → pitch →
roll (stages hip yaw-to-pitch, hip pitch-to-roll); roll-flip puts the STS3250 roll case in the thigh. The hip shell over it is the open look task
(`hip-shell-covering.md`, Idle T-czv2d8).

- Hip roll is the load driver. In the older single-support sims it reached its rated torque first; in the trained
  tier-3 check it runs at 0.63 rated RMS over the target tape, 0.98 in the worst segment
  (`robot/model/motor-map/README.md`). A stronger roll servo moved the limiter to the stance ankle; the cause is stance
  load and foot placement as much as the motor.
- Yaw swings the pitch servo (45 mm inboard of C), the pitch bracket and the leg under the pelvis: 5 of the 6 worst
  realistic-pose contacts in the hip-shell study are yaw-driven (toes-in climb step: 9,200 mm³ against the battery
  cradle).

Lessons from the October exposed-frame hips (kept short; designs in `design/hip-direct-serial-low-roll`):
- A belt-driven yaw/roll/pitch stack had its vertical yaw spindle intersecting the roll spindle (144 mm³) and the
  pitch-fork bridge (113 mm³) at neutral. A controller that walked 16/16 on that body walked through solid metal.
  Check adjacent stages for interference before trusting a gait.
- Fixed-pose clearance gives no joint limits. A 59-pose grid kept 352-374 hard-part intersection records on every hip
  variant while the saved walk poses were clear. Task-like corridors (right turn ±12 deg yaw / ±4 deg roll) are not
  rectangular ranges.
- Supplier STEP spline datum sat 12.5 mm from the case bounding-box centre, not the 10.11 mm from the drawing: a
  2.39 mm axis error caught in review. Take axes from native frames and supplier solids, not drawings or part names.

## 6. Drive layout

Current (since 2026-10-08): direct drive. The servo case sits on the parent link close to the pivot, the output faces
the pivot, and a short hub bolts to the child. One-to-one belts add no torque. The switch cut hip motor-centre offsets
from 75/86/51 mm to 22/38/39 mm (yaw / roll / pitch) and removed 16 belts.

Not shown: radial and moment load on the servo output bearing, printed seats under cyclic load.

## 7. Knee, ankle and foot

Current: the six-stage lower body (shin, ankle coupler, foot monocoque, thigh, hip pitch-to-roll, hip yaw-to-pitch)
with ST3025 at knee and ankle pitch, C047 at ankle roll. The foot monocoque grew to 78 x 112 mm (section 1).

Lessons that still apply:
- Foot shape steers the gait. Changing only the sole in simulation moved a learned walk's yaw from +0.055 to
  +0.258 rad and worst lateral drift from 0.98 to 1.22 m over 8 s (same policy, 16 worlds). Soles must match CAD
  before a gait is judged.
- Static sole clearance does not measure contact release. Lifting the swing ankle +0.10 / +0.20 rad gave +3.3 / +7.0 mm
  static clearance, but natively the lowest corner stayed 0.6-0.9 mm below the floor and stance slip reached 42-48 mm.
  Sole-centre height misleads the same way: the foot pivots on its lowest corner.
- A stronger ankle moves the limiter instead of removing it (ankle ST-3215 → ST-3250 lowered the any-leg-at-rated
  share from 91.8% to 87.6% forward, while hip roll rose from 26.3% to 39.6%).
- Knee case versus stance ankle case: 0.88 mm penetration in single-support release on the old body; static legal
  swing shapes cleared it by 11.6 mm, but the native state did not reach the commanded shape.

## 8. Arm and hand

Current: four-stage FDM arm (shoulder pitch-to-roll, roll-to-yaw, upper-arm yaw-to-elbow, forearm elbow-to-wrist),
STS3250 at all four arm joints, symmetric geared gripper (8 hand parts). Upper arm lengthened by 32 mm (section 1).
FEM loads: 1.5 kg reach up to 4 kg at 1.5 g recovery, 6 kg at 1 g support.

Lessons from the older XL330-jaw hand (2026-10-09/10, 20 mm, 6.25 g floor cube), carried into the geared hand:
- Aperture alone does not make a pinch. The old jaw closed to 20.135 mm on a 20 mm cube (0.135 mm clearance, no
  compression margin); the forearm frame intruded into the jaw opening; 0 of 916 two-jaw contacts were clean.
- Pad contact height and opposing flat side faces set the grasp. Score a hand on side-face location and entry clearance
  before training a closure.
- Arm mass is a small COM lever for this body: moving both arms toward the swing side shifted the COM by tens of mm,
  not enough to unload the stance hip alone.
- A shoulder-yaw coupling once had screws at 30 mm spacing, not the 46 mm the slope arithmetic assumed. Check axes
  from native frames.

Open: the geared hand has not been tested on a floor cube; servo duty at the gripper.

## 9. Neck and head

Current: four-stage FDM neck and head (neck root cradle, yaw-to-pitch L-link, pitch-to-head cradle, head sensor /
compute carrier), two XM335, 28.6 g printed, min safety factor 2.34 (r1). Front camera window, torso neck recess,
underside service opening and tool bores are owner-accepted look and service choices.

Service-access lessons (older head, 2026-09-18/20):
- A 5.5 mm open-end wrench has a 13.4-15.1 mm head; M3 nuts 10 mm apart need side entry from outside.
- A CAD screw head without a socket proves no driver access. Measure the swept path of the real tool; the shared
  `driver_access` check now does this per screw.

Open: servo output bearing capacity under head loads; cable loop through neck motion.

## 10. Torso, pelvis and battery carrier

Current (2026-11-06, refit 2026-11-12): 7-cell honeycomb torso spine with hip / shoulder / neck tower; pelvis column
with the battery retention lid and 6-bolt waist U-clevis; front power supervisor and precharge frame; dorsal body-IMU
and foot-ADC carrier. Nine 21700 cells (3S3P) with nine 6063-T5 fin strips, two-piece centre shell plus chest
indicator. 178.2 g printed (r1), min safety factor 1.55 (1 g upright, 3 g gait, 6 g fall, multi-limb stall). The refit
lays the lower-pelvis cells horizontally on the midline. A lumped 12-node thermal model exists
(`robot/body/torso/thermal_evaluate.py`); enclosed-shell thermal cycling under gait is not measured.

## 11. Packaging facts for the selected servos

Prices, ratings and duty are in the actuator topic; these are the facts that shape the frame.
- ST3025 (fast leg tier): 40 x 20 x 40 mm, 124 g with wheels. STS3250 and C047 (strong and light tiers): 45.2 x 24.7 x
  35 mm, 74.5 g and 55 g.
- Shaft datum: the Feetech output is normal to the broad 45.2 x 24.7 mm face, about 10 mm from the case centre by the
  drawing, 12.5 mm by the supplier STEP. Several early cases were rotated as if the shaft left the long side.
- Brick servos are off-centre about their own axis (shaft 10 mm from one end: 9-14 mm off-centre, r 31-37 mm swept),
  and U-brackets reach r 48 (pitch) and r 67 (roll). Round joint covers must be sized for that sweep.
- Shell references (Cove redraw at 500 mm): 319.9 mm wide, 137.1 mm deep, 2 mm skin; open interior torso
  210 x 101 x 165 mm, thigh 70 x 72 x 134 mm, shin 55 x 52 x 97 mm.
- Direction kept from the 2026-10 shell-fit study: keep the exact motors, rotate and move cases inside their parent
  link, redesign the cover topology, and add transmissions last.

## 12. Clearance: CAD versus physics

The learning body, the CAD body and the hardware agree only where they are explicitly joined.
- Old collision proxies let an elbowed 35 cm ape stand on one foot with arm-thigh contact in 297 of 300 frames
  (5.36 mm penetration). A corrected proxy plus a self-collision cost removed it.
- Joint spacing alone broke an adapted walker: forward 0.3 m/s for 4 s went from +1.07 m to -0.15 m, and left-step
  foot clearance from 35.5 to 7.2 mm. A policy must be re-adapted for every body change.
- A catalogue torque-speed envelope cut an ideal-servo walk from 3.45 to 1.48 m in 8 s; 50 more PPO updates under the
  envelope restored 2.56 m. Train with the motor envelope in the loop.
- All CAD checks are sampled poses. Continuous swept-volume clearance and coupled corner cases remain open.

## 13. Printable structure

Current: home FDM (Bambu X2D, PAHT-CF or PETG-CF) with pressed-in 304 stainless compression sleeves and metal screws.
Print setup, material limits and printability results are in `design-checks.md`.

- Layer direction and moisture dominate strength: PA612-CF15 dry tensile 91.9 MPa in XY vs 48.3 MPa in Z (35.6 MPa wet
  Z). Do not infer fatigue or creep from a carbon-fibre label.
- An insert gives threads, not creep resistance in the clamped plastic. Measure insert pull-out and bolt retention on
  the real small bosses.
- Generator method (2026-10-29 shoulder pilot): a conditions-first generator (neural field or density topology
  optimisation, physics in the loop) responded to the load case locally; the direct density baseline had lower
  compliance than the neural family. Literature: topology-derived FDM brackets about 20% lighter at similar
  displacement; CF/PLA FEM overpredicted stiffness by 14-52%. Measure printed assemblies and fixture compliance
  separately. The topology and truss families stay in `robot/body/leg/build_topopt.py` and `build_truss_box.py`.

## 14. Open questions, ordered by what changes decisions

1. Printed structure under load: sleeve retention, preload relaxation, interlaminar fatigue at landing, creep near
   warm servos, tolerance stack. No printed part has been loaded.
2. Hip shell over the roll-flip hip (Idle T-czv2d8): a sleek closed shell over 80-85 deg flexion.
3. Simulation model of the current body: hip shift, roll-flip and ape proportions are not in the WBC plant.
4. Hip-roll, knee and ankle duty on a real servo: catalogue ratings only.
5. Continuous swept clearance; thermal and harness routing; the geared hand on a real cube.
