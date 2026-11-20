# r4 best-estimate actuator dynamics (2026-11-05)

R4 keeps the selected R3 body exactly (links, masses, inertias, geometry,
collisions, sensors, timing) and adds best estimates for the actuator values
nothing has measured yet. `actuator-dynamics.json` holds every value with its
source and a low/high bound for sensitivity (bounds are not randomization
ranges). The values are for the C047, two candidates (STS3250 and Waveshare
ST3025) and the XM335 neck.

| Value | C047 | STS3250 | ST3025 | Source |
|---|---|---|---|---|
| Rotor inertia at the output (kg m2) | 0.024 | 0.024 | 0.0046 | BAM identification (STS3215, ST3025); STS3250 assumed |
| Coulomb friction (N m) | 0.052 | 0.052 | 0.05 | BAM |
| Firmware duty per rad of error | 5.31 | 5.31 | 6.43 | BAM firmware model (kp 32 x 0.166 x fitted ratio) |
| Firmware target rate limit (rad/s) | 5.2 | 8.66 | none | BAM STS3215 fit (1.1x no-load), scaled |
| Backlash (rad) | 0.0044 | 0.0044 | 0.0044 | half the 0.5 deg catalog bound (assumed) |
| Stall torque / no-load speed | 2.94 N m / 4.72 rad/s | 4.9 / 7.87 | 3.92 / 8.90 | catalogs |
| Mass (g) | 55 | 74.5 | 124 | catalogs |
| Supply rail (V) | 11.1 (10.5-12.6) | same | same | 3S3P pack nominal; servos assumed on the pack rail (catalog curves are at 12 V and scale with it) |

The servo law (`control/common/plant/boundary.py`, when
given `duty`) follows BAM's Feetech model:
1. The firmware moves its internal target toward the command at most at the
   rate limit.
2. Duty is gain x error, less half the backlash, clamped.
3. Torque is stall x (duty - speed/no-load speed), within the stall torque.

At full duty this is the catalog line; inside it, back-EMF supplies the
damping. Compared with R3's assumed PD (8 N m/rad, 0.16 N m s/rad, no rotor),
the C047 is about twice as stiff (15.6 N m/rad), four times as damped, and has
0.024 kg m2 of rotor to drive. That is 8x the knee's link inertia and 200x the
shoulder yaw's.

`build.py` writes `artifacts/robot/model/dynamics/`:
- `cove50-rigid.xml` with joint armature and frictionloss;
- `cove50.urdf` with `<dynamics friction>`, the stall-torque effort limit and
  the rotor inertia as a joint comment (URDF has no armature field);
- `report.json` with hashes.

```sh
.venv-control/bin/python -B robot/model/dynamics/build.py
```

`robot/model/current-model.json` names r4 under `actuator_dynamics`.
Its R3 fields are unchanged, so frozen runs and the boundary's R3 hash check
still hold. New work opts in with `robot_cfg(dynamics=dynamics.resolve(servo),
peak=True)`, or `tracking.env_cfg(dynamics='r4')` in the WBC track.

Limits: no value here is measured on a Cove-50 part:
- BAM measured a 7.4 V STS3215 and an ST3025, not a 12 V C047 or an STS3250.
- Load-dependent friction, thermal behaviour, overload protection and voltage
  sag are not modelled.
- The catalog line is kept even where BAM's motor fit gives more torque.

The bench should replace rotor inertia first. It moves the servo-demand result
most for the two 1:345 servos.

## Revision 2: hip ranges (2026-11-05)

The selected hip (`hip-direct-serial-low-roll` r3) specifies no range of motion.
R3 exports the six hip axes unlimited, and the runtime caps their commands at a
±0.35 rad diagnostic envelope that is "not a joint limit". Every WBC run before
tape2 inherited that ±20° cap. The agility motion set needs hip pitch down to
−82° (flexion; R3 hip pitch − flexes). The walk reference already exceeds the
cap on 35% of frames, and the deep squat, hop, crouch and knee bounce exceed it
on 36–82%.

`joint_ranges_deg` gives best-estimate ranges per side:

| Axis | Low | Nominal | High |
|---|---|---|---|
| Pitch | −90..20° | −110..30° | −125..40° |
| Roll | ±25° | ±35° | ±45° |
| Yaw | ±30° | ±45° | ±60° |

The nominal values follow primate/human hip range of motion, narrowed for a
compact servo hip. Low is the narrowest range that still covers the motion
set. `dynamics.resolve()` returns them as `ranges_deg`, with level key
`hip_range`. The boundary uses them in place of the cap, the spec makes those
joints limited, and both exports carry them. A CAD sweep of the real hip stops,
or a check on a printed hip, should replace the assumption.

## Bench plan: what is worth measuring

Each measurement replaces a named sidecar value. The order follows how much
each value moves the simulated evidence and the servo decision:
- the CPU demand check `cove50-wbc-agility-demand-v2-20261105`;
- the r4 walk runs `cove50-wbc-r4-*-20261105`;
- the tape runs `cove50-wbc-tape-*-20261105`.

Test one C047, one STS3250 and one ST3025 side by side on the same rig and
power supply, ideally with a spare of each.

| # | Measurement | Replaces | Why it matters | Method |
|---|---|---|---|---|
| 1 | Rotor inertia and friction | `armature_kg_m2`, `frictionloss_Nm` | The largest single swing in the demand check: up to 42 points of the frames-beyond-envelope share for the two 1:345 servos. The STS3250 value is assumed. | BAM-style pendulum (Rhoban). A printed arm with two or three known masses. Free swings plus logged position/current at ≥100 Hz; fit armature, Coulomb and viscous friction. |
| 2 | Thermal and overload at 1.0, 1.2 and 1.3× rated current | none yet (not modelled) | On r4, walking alone runs the hottest servo at 128% (C047), 116% (STS3250) and 104% (ST3025) of rated RMS current. Whether a C047 survives minutes of that decides whether it is usable at all. | A lever with a hanging mass at fixed angles, 2–5 min per level. Log temperature, current and position. Note when overload protection trips and how far torque then drops. |
| 3 | Torque–speed line at the pack rail | `stall_torque_Nm`, `no_load_speed_rad_s`, `supply.rail_V` scaling | The catalog line at 12 V is scaled to 11.1 V; the frames beyond the envelope depend on it directly. | Stall torque on a lever and load cell at 10.5, 11.1 and 12.6 V. No-load speed from logged position. Three or four loaded points with a pulley and masses. |
| 4 | Firmware position loop | `duty_gain_per_rad`, `max_duty`, `target_rate_limit_rad_s` | The simulated step responses differ most here: the ST3025 settles fastest, the STS3250 overshoots 10–17%. The WBC's tracking depends on it. | Step responses of 0.05, 0.2 and 0.5 rad, unloaded and with the pendulum, using the servo's own position feedback; also the P/D/I and protection registers as shipped. |
| 5 | Bus timing for 20 servos | `timing.json` (20 ms command, 10 ms feedback delay) | The policy's delays come from this assumption. | Sync-write plus sync-read round trip at 1 Mbps on one bus, measured with a logic analyser or host timestamps. |
| 6 | Backlash | `backlash_rad` | Assumed at half the catalog bound; small effect so far. | A dial indicator on a 100 mm lever with the servo holding, ± small torque. |
| 7 | Hip stops | `joint_ranges_deg` | Decides which squats, crouches and hops the real hip allows. | A CAD sweep first, then the printed hip with the servo horns fitted. |

A one-leg rig can follow once items 1–4 exist: hip pitch, knee and ankle pitch
on a vertical slide carrying half the body mass, running the knee bounce and
quick crouch. It tests the combined leg against the simulated demand. Items 1–4
need about a day per servo with printed fixtures, a bench supply, a USB bus
adapter and a cheap load cell.

Mixed sizing (`agility/allocation.py`, record
`cove50-wbc-servo-allocation-20261105`): the load sits in the hip pitch, hip
roll, knee and ankle pitch. On the 1:345 servos most of their walking torque is
the rotor-inertia term, so item 1 decides most of the C047 and STS3250 numbers.
An ST3025 on those 8 joints with STS3250 elsewhere comes within half a point of
an all-ST3025 robot (3.7% vs 3.2% mean frames beyond) at 4.62 kg instead of
5.21 kg. For the bench that means two ST3025 (leg candidates), two STS3250
(arm candidates) and one C047 reference. The one-leg rig should carry the
ST3025.

## Bench priorities after the step-5 servo comparison (2026-11-06)

Record `cove50-wbc-servo-compare-20261106`. Each servo set trained for one
hour on its own plant, from the same controller, on the 41-motion ape tape.
- **Mixed vs all-ST3025:** the mixed set tracks like all-ST3025 at 0.59 kg
  less. Both walk the full 1.54 m and stay below rated current except hip roll
  in the turn (1.06× and 1.17×).
- **C047:** runs 1.1–1.4× rated on the four loaded leg joints while walking
  (up to 1.5× on the faster walk).
- **STS3250:** reaches 1.1× rated at the knee and hip pitch in the hop.

For the knee-speed question (step 6), a separate inverse-dynamics demand
check adds a 24 V CubeMars knee (sidecar revision 3). It also screens two
12 V servos and an ST3025 knee through a 1:1.5 speed-up stage (revisions
3.1–3.2). These run reports hold the evidence:
`cove50-agility-allocation-knee-20261106`, `-knee12v-20261106` and
`-knee-geared-20261106`.

The order of the table above changes:
1. **ST3025 thermal and overload at hip roll loads (item 2).** On the two sets
   proposed for the legs, the ST3025 and the mixed set, hip roll is the joint
   closest to its rating: 1.06–1.17× in the turn, and 0.95–1.08× in the mixed
   set's shuffles. Run the 1.0, 1.2 and 1.3× levels on an ST3025 first. The C047
   level matters less now: in simulation its loaded leg joints run 1.1–1.4×
   rated on the walk alone; whether it survives that is for the bench.
2. **ST3025 torque–speed line at 11.1 V and its firmware loop (items 3 and 4).**
   The demand check of the reference motions (inverse dynamics, not a
   controller test) puts knee speed as the ceiling for faster moves:
   - marches below 0.5 s per step and hops need 15–17 rad/s at the knee,
     against the ST3025's catalog 8.9 rad/s;
   - an assumed 1:1.5 speed-up stage on the ST3025 knee brings the 0.4 s march
     from 15.5% to 3.5% of leg frames beyond stall on the same 12 V rail.
   Whether the real servo keeps its line under load decides that option.
3. **STS3250 rotor inertia (item 1).** On the arms, the assumed 0.024 kg m² is
   what limits the fast arm moves. The 0.75 Hz side rock needs 21 N m of
   shoulder pitch with it, against 6 N m on the ST3025's 0.0046.
4. Items 5–7 unchanged.

Buy list for the bench:
- four ST3025: one leg's hip pitch, hip roll, knee and ankle pitch, with one
  as a spare for the thermal test;
- two STS3250 (arm candidates);
- no C047 for the legs; one only as a cheap reference if the rig needs it;
- a printed 1:1.5 speed-up stage for one ST3025 knee, if step 6 shows it is
  worth it.

The 24 V AK45-10 knee needs its own 6S supply and CAN bus, so it is a
fallback only if one rail is given up.
