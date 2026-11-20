# r5-tier3: the locked three-tier motor set (2026-11-08)

The owner locked this set on 2026-11-08. It comes from the tier pick in
`robot/actuators/tier-pick/README.md`: brick-shaped servos on one Feetech bus, in the C047 case or
the 40 × 20 × 40 mm case.

## Which motor goes where (22 joints)

| Tier | Servo | Joints | Count |
|---|---|---|---|
| Fast leg | Waveshare ST3025 (Feetech ST-3025-C002): 40 × 20 × 40 mm, 124 g, 3.92 N m stall, 8.9 rad/s, brushless, rotor 0.0046 kg m² | R/L hip pitch, R/L knee, R/L ankle pitch | 6 |
| Strong | Feetech STS3250: C047 case, 74.5 g, 4.9 N m stall, 7.9 rad/s, rotor 0.024 kg m² (assumed) | R/L hip roll, R/L shoulder pitch, roll and yaw, R/L elbow | 10 |
| Light | Feetech ST-3215-C047 / Waveshare ST3215 12 V: 55 g, 2.94 N m stall, 4.7 rad/s | R/L hip yaw, R/L ankle roll, neck yaw, neck pitch | 6 |

The parts cost about $1,030–1,313 for all 22, depending on the STS3250 source. The machine-readable map is
`motor-map.json`.

## Model files

`build.py` writes `artifacts/robot/model/motor-map/` (`cove50-rigid.xml`, `cove50.urdf`,
`report.json`):
```sh
.venv-control/bin/python -B robot/model/motor-map/build.py
```
Each joint gets its tier's rotor inertia, friction, effort and velocity limits (12 V catalogue). The servo mass
change against the R3 CAD servo is added at the joint, by the same rule as the training plant. The mass rises from
3.830 to 4.495 kg: 6 × 69 g, 10 × 19.5 g and 2 × 28 g at the neck. MJCF and URDF agree.

`robot/model/current-model.json` points to this revision under `motor_assignment`.

**Training plant:** `walk_resume dynamics.ALLOCATIONS['tier3-st3025-sts3250-c047']`, selected with
`--actuator tier3-st3025-sts3250-c047`.
- It covers the 20 body joints.
- The neck keeps the XM335 law in training, because the shared boundary hard-codes it. The head is light, so this
  is a small, known difference.
- The long-arm ape body (upper arm ×1.6, forearm ×1.27, 30 mm fist, feet ×1.3) is applied at training time. It is
  not in these files.

## Trained check on the target (2026-11-08)

The WBC continued from the previous mixed-set run on this allocation. It was evaluated on 18 playful-ape-t1 segments,
nominal and pushed, 32 worlds each. Run `cove50-wbc-t1-tier3-longarm-20261108`, record of the same name, about 0.95
GPU-hours; measured by `control/wbc/agility/trained_measure.py`.

| Group | Servo | Tape RMS / rated | Worst segment |
|---|---|---|---|
| Hip roll | STS3250 | 0.63 (was 0.70 on the ST3025) | 0.98 in the turn (was 1.18) |
| Hip yaw | C047 | 0.41 | 0.78 in the walk |
| Ankle roll | C047 | 0.53 | 0.78 in the march |
| Hip pitch / knee / ankle pitch | ST3025 | 0.54 / 0.38 / 0.47 | 0.81 / 0.59 / 0.82 |
| Shoulders and elbow | STS3250 | 0.21–0.38 | shoulder yaw 1.06 in the 2.5 Hz chest beat |

Other results, across the 18 evaluated motions (all on the training tape):
- **Walk:** final error 8.4 → 3.9 cm.
- **Leg tracking:** 1–16% worse per family; walk and turn is the largest change.
- **Falls:** 0 of 1,152 episodes reach the fall condition.
- **Arm tracking:** the fastest arm motions track worst on both servo sets, so the servo tier does not explain it.

The score is agility across all motions (`AGENTS.md`). The motions are made-up stand-ins, and single-motion rows
above, such as shoulder yaw's peak in one motion, are evidence, not requirements.

## What changes in the CAD

| Servo | Same as the C047? | Pieces affected |
|---|---|---|
| ST3215-C047 (light) | It is the C047 | None at hip yaw and ankle roll |
| STS3250 (strong) | Same 45.2 × 24.7 × 35 mm case, 19.5 g heavier | No pocket change at hip roll, shoulders and elbow. Confirm the horn and screw pattern on a part |
| ST3025 (fast leg) | Different: 40 × 20 × 40 mm (5 mm shorter, 5 mm thinner, 5 mm taller), 124 g, its own mounting and horn | New pockets at hip pitch, knee and ankle pitch |
| Neck on the C047 | Replaces the XM335 (35 × 19 × 22 mm, 27 g) | Larger neck pockets |

The ST3025 pockets touch:
- the accepted hip (`hip-direct-serial-low-roll` r3, the hip-pitch case);
- the ankle/foot (`cove50-ankle-foot-fixed-roll` r4, the ankle-pitch case);
- the lower-body FDM connector stages around those joints: Hip Yaw-to-Pitch, Hip Pitch-to-Roll, Thigh, Shin and
  Ankle Coupler. Which stage holds a case and which a horn is to be confirmed in the CAD.

The neck change touches the neck/head design (`neck-double-moving-case-C` r3), the neck/head FDM connectors (Stage 1
Neck Root Cradle, Stage 2 Yaw-to-Pitch L-Link, Stage 3 Pitch-to-Head U-Cradle) and the torso's fixed neck-yaw mount.
In exchange, the Dynamixel bus goes away.

Not caused by the motors, but part of the body being trained:
- the long-arm and long-foot geometry: the arm connectors' Upper-Arm Yaw-to-Elbow and Forearm Elbow-to-Wrist stages,
  the fist, and the Foot Monocoque and ankle/foot sole at ×1.3.

**Electronics:** stall currents rise to 4.2 A (STS3250) and 4.4 A (ST3025), from 2.7 A on the C047, which affects
power distribution and bus wiring. All 22 servos are on one Feetech TTL bus protocol.
