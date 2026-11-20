# Actuators, power and electronics

Results for the 50 cm ape (Cove-50). Every number is a catalogue value, a public listing price or a simulation unless
it says "measured". **Nothing in this topic has been measured on a physical servo, battery, rail or board.** Prices
are single-unit public listings before tax and shipping, not quotes. Purchases are the owner's decision; none has been
made. Sensors and the camera stack are in `perception-and-sensing.md`; the frame and servo pockets in
`mechanical-design.md`.

## 1. Current state (2026-11-20)

**Servos: the locked three-tier brick set** (owner, 2026-11-08, Idle M-44khpr; `robot/actuators/tier-pick/README.md`,
`robot/model/motor-map/motor-map.json`). All on one Feetech TTL bus, in the two pockets the body already has.

| Tier | Part | Joints | Case, mass | Stall / rated at 12 V | No-load | Price seen |
|---|---|---|---|---|---|---|
| Fast leg | Waveshare ST3025 (Feetech ST-3025-C002), brushless, rotor 0.0046 kg m² (identified) | hip pitch, knee, ankle pitch (6) | 40 x 20 x 40 mm, 124 g | 3.92 / 1.32 N·m | 8.9 rad/s | $74.95 at 4+ (Waveshare, 2026-11-08) |
| Strong | Feetech STS3250 (ST-3250-C001), 1:345 | hip roll, shoulder pitch / roll / yaw, elbow (10) | 45.2 x 24.7 x 35 mm, 74.5 g | 4.90 / 1.63 N·m | 7.9 rad/s | $45.67 (1 per buyer), $64.18 qty-20, $73.99 OpenELAB |
| Light | Feetech ST-3215-C047 / Waveshare ST3215 12 V, 1:345 | hip yaw, ankle roll (4) | 45.2 x 24.7 x 35 mm, 55 g | 2.94 / 0.98 N·m | 4.7 rad/s | $20.63 at 12+ (Waveshare) |
| Neck | ROBOTIS XM335 pair (kept, owner M-revtdr, 2026-11-12) | neck yaw, pitch (2) | - | - | - | - |

- Cost for 22 joints (with the neck on the light tier, as the tier pick priced it): $1,031 at the checked STS3250
  price, $1,216 at qty-20, $1,314 at OpenELAB. The neck XM335 pair is a second bus protocol and is not in that sum.
- Grades from the tier pick (inverse dynamics of the 59 target motions, each grade 2.5 at its reference: motor-line
  share 2%, heat 0.7x rated RMS, overload 1.6x rated RMS in the hardest motion): weakest group 3.1, mean 3.5.
  Hip roll moved to the STS3250 because it needs torque at low speed (2.3-2.7 N·m p95-p99 below 2 rad/s): 3.2 there
  against 2.5 on the ST3025.
- Trained check (WBC on the tier-3 plant, 18 target segments, 32 worlds, about 0.95 GPU-hours; table in
  `robot/model/motor-map/README.md`): tape RMS over rated 0.21-0.63 by group, hip roll highest (0.63, 0.98 in the worst
  segment); 0 falls in 1,152 episodes. Single-segment peaks are evidence of where motion degrades, not requirements.
- Where the tiers run past their motor line (share of a motion's frames): ST3025 hip pitch 4.4% (0.45 s knuckle drop),
  knee 3.9% (0.5 s march); STS3250 elbow 5.4% (0.8 s knuckle walk); C047 hip yaw 3.9% (forward walk x1.1).

**Power: 3S3P of Molicel INR-21700-P50B** (owner, 2026-10-26, Idle M-vb8vf4; `robot/electronics/power/battery.json`).
15 Ah, 162 Wh nominal, nine cells up to 639 g (supplier maximum). Runtime intent about one hour per charge (owner: half
an hour is too short). Electronics output reserve 28 W sustained / 45 W peak. Distribution, converters, fuses,
service and charge ports and the connected pack wiring are not placed yet (Idle T-42fga9).

**Electronics: integrated composition r1** (owner, 2026-10-26, Idle M-s77afy; `robot/electronics/`): sensing V24,
compute head R23 (ROCK 4D with a local 12 V regulator in the head), torso power R33. The body-to-head 12 V endpoint,
branch protection, moving neck loop, full data and power harness, supervisor and motor interfaces, and every loaded,
thermal and EMI observation are open (`robot/electronics/head.json`).

Open, in order of value:
1. Loaded torque-speed and thermal duty of one STS3250 and one ST3025 on a restrained bench at 12 V and 10 V.
2. Rail voltage: a 3S pack is 12.6 V full. An older note gives 6-12 V for the ST-3250-C001; check each tier's maximum
   supply against 12.6 V before the first power-up.
3. Rail sag and branch current: 10 STS3250 at 1.4 A rated already draw 14 A; stall is three times that. No workload
   trace on a rail exists.
4. Neck: XM335 (selected) against the light tier (motor map and tier pick). The motor map file still lists the neck on
   the light tier.
5. HLS line price: Feetech HL-3960 (C047 case, 6.5 N·m, current control) grades 3.7-3.9 on hip pitch, knee and hip roll
   at a search price of about $77; if real, it could remove the 40 x 20 x 40 pocket. Not verified.

## 2. Catalogue facts and traps

- **Torque labels mean different things.** Rated, stall, peak, locked rotor, maximum and static are not interchangeable.
  Waveshare "locked rotor" values are stall-type. ROBOTIS gives rated as 20% of stall as a selection estimate.
  Lynxmotion publishes dynamic torque at about 25% of static. Plan with rated (or stall / 3 where none is published);
  stall is a transient.
- **A family name is not a part.** "ST3215" covers Waveshare, Feetech C018, C001 (7.4 V, 0.49 N·m rated), C046 (1:147),
  C047 and clones. Check voltage, ratio, connector, firmware and supplier revision per unit. The early CAD proxy was a
  Waveshare ST3215 STEP (root `SCS215_v5`), not the Feetech part; the case datum matched, the horn and screw pattern did
  not. `robot/body/fastening/interfaces.py` now registers each servo case to its official definition.
- **Telemetry is not torque.** The bus "load" value is not a calibrated torque, and current through a lossy gearbox is
  only an estimate. A hanging mass on a lever is a gravity proxy that still includes friction.
- **No-load speed is not speed at rated torque.** No Feetech sheet gives impedance, latency, a loaded torque-speed law
  or thermal duty.
- Feetech (12 V TTL, own protocol) and Dynamixel (Protocol 2.0) are not bus-compatible: separate packet handling and,
  for XL / XM parts, their own rail.
- Feetech STS3215 turns torque off above 70 °C internal; STS3250 is rated -20..60 °C ambient (heat check input in
  `design-checks.md`).

## 3. Actuator landscape (search map, not a BOM)

Register of 267 model, family and build-route leads (2026-10-08): `robot/actuators/parts.csv`; price ledger
`robot/actuators/source-geometry/catalog-price-refresh/observations.csv`; brick survey for the tier pick (204 variants,
282 offers) `robot/actuators/brick-survey/`.

| Category | Leads | What it means for us |
|---|---:|---|
| Serial smart servos (Feetech, Waveshare, Hiwonder, Lynxmotion, Kondo) | 57 | The selected class: encoder and bus feedback in one brick |
| PWM / RC servos | 19 | No joint-state bus; expression or low-load only |
| DYNAMIXEL | 26 | Good docs and current modes, dearer (XM430 $269.90) |
| Integrated brushless joints (CubeMars, RobStride, SteadyWin, MyActuator, CyberGear) | 86 | Higher rated torque per gram, but 190-350 g, $110-186, 24-48 V and CAN |
| Bare motors, DIY QDD, linear, other | 79 | Need a drive built around them; builder BOMs $70-255 |

Reference points (2026-10-08 single-unit listings):
- RobStride 05: 46 x 46 x 44 mm, 191 g, 1.6 N·m rated at 48 V, $110. RobStride 00: 57 x 57 x 51 mm, 310 g, 5 N·m
  rated, $125. CubeMars AK40-10: D53 x 40.2 mm, 190 g, 1.3 N·m rated, $135.90. A QDD hip pair adds about 471 g over two
  STS3250 and needs a 48 V rail.
- Round actuators of our torque class are drums about the size of one brick (AK40-10, AK45-36 D55 x 57, GIM4310-36
  D55 x 47; $136-186 plus CAN), so pancake actuators do not by themselves make a slimmer hip.
- Slim alternatives: Waveshare ST3020 40.2 x 20.2 x 40 mm, 2.45 N·m locked rotor, $19.99 (cheaper light tier);
  Feetech FT-3332 34 x 15 x 36.5 mm, PWM; SteadyWin GIM3505-8 D43 x 23.6 mm, 0.65 N·m nominal, 24 V CAN.
- Hiwonder bus servos ($14-50) are cheaper on the survey frontier but bring a second protocol, potentiometer feedback on
  some and larger cases; they stay outside the pick. Their torque fields disagree between page sections.
- Clone servos (private-label "ST3215", LX-16A, DS3218 clones): identity unresolved; inspect on arrival.

## 4. How actuator limits shaped simulated motion

Earlier bodies (35 cm ape and the 21-axis Direct Frame), catalogue curves, rated torque as a hard sustained bound.
These set the training rule; the numbers belong to retired bodies.
- A policy tuned on ideal servos lost about a fifth (35 cm, forward 1.02-1.08 → 0.81-0.85 m in 4 s) to more than half
  (50 cm, 3.45 → 1.48 m in 8 s, yaw drift +0.19 → +0.94 rad) of its forward progress under the catalogue envelope. 50
  PPO updates under the envelope restored 2.56 m and -0.11 rad. Train with the motor envelope in the loop.
- Mass moves alone did not cause the loss; the drive law did (mass-only change: 1.03-1.11 m).
- Quasi-static gait screen (2026-10-08, 3.2 kg in single support): about 1.88 N·m at a 60 mm knee arm and 3.77 N·m at
  a 120 mm hip arm, against 1.57 N·m rated for the STS3250. A 2:1 or 3:1 reduction at an assumed 70% efficiency gives
  2.20 or 3.29 N·m at the rated point; 4:1 gives 4.39 N·m but about 1.97 rad/s no-load. Arithmetic, not duty: the tier
  pick's inverse-dynamics traces of the target motions replace it, and the body is now 4.5 kg.
- Leg joints sat at rated torque on 85-99% of walking frames. A stronger ankle moved the limiter to hip roll (any leg
  at rated 91.8% → 87.6% forward; hip roll 26.3% → 39.6%). Stance load and foot placement drive it, not one motor.
- Sim actuator coupling worth having (2026-09-21 proposal, now in the training plant in part): command delay
  (0-40 ms, 100 ms as a stress case), slew and overspeed guard, voltage-aware four-quadrant torque cap
  (tau_cap = tau_stall x (1 - |dq| / dq_no_load) x clamp(V / 12, 0, 1), a proxy, not a datasheet law), and a pack RC
  voltage update. Friction, backlash, winding resistance, thermal constants and bus delay are unmeasured; keep them as
  named sensitivities until a bench gives data.

## 5. First bench (prepared, not run)

Purpose: whether the rated torques used in training hold on a real joint, and real delay, response and
voltage-dependent speed for the plant.
- Kit: one servo per tier with horn and lead; Feetech FE-URT-2 USB-to-TTL adapter; a current-limited 12 V supply with
  voltage and current readout and an all-source disconnect; the servo restrained on a 100 mm lever with known masses
  and an independent angle reference.
- Steps: ID discovery at 1 Mbps, then read voltage, temperature, position and error status, no motion first. Small
  reversible steps at 12 V then 10 V, then gravity loads of about 0.3 / 0.6 / 0.9 N·m. Log command time, external
  angle, reported state, terminal voltage and current and case temperature on one clock. Torque =
  9.80665 x (m_payload x r + m_lever x r_com) x cos(angle).
- Interface cautions: connect only the adapter's TTL 3-pin port, never RS485. Pin order from the C018 spec is GND, Vcc,
  signal; meter the connector end to end before power. The adapter's USB power path is unresolved, and a USB cable is
  not a disconnect. The pack must never feed a servo directly during bench work.
- Sole friction on the same rig: TPU sole coupons at about 5, 11 and 22 N normal load, three pulls each, as breakaway
  and sliding ranges.

## 6. Servo protocol

- Manufacturer FTServo_Python is the protocol reference and first bench SDK; LeRobot's motor layer for datasets later.
  Do not write another decoder. An offline cross-check (2026-09, synthetic packets, no hardware) found the
  manufacturer helpers, LeRobot and our own C framing agreeing on every read, write and sync-write case and on all
  65,536 raw words for sign conversion.
- Register map used: position / speed / load at 56 / 58 / 60, voltage / temperature 62 / 63, status / moving 65 / 66,
  current 69. Position and speed sign bit 15, load sign bit 10. The manufacturer SDK caps packets at 250 bytes; our two
  buses stay under 30 rows.
- Unverified per part: LeRobot `sts3215` expects model number 777 and 4096 counts; the C018 spec writes 360° / 4095
  with midpoint 2047; velocity units are unverified. LeRobot calibration routines write registers: do not run them for a
  read-only identity check. Record identity, firmware, mode and original settings before any configuration change.

## 7. Power and energy

- Sizing model (2026-09-27, still the method): battery_kg = t x (P_fixed + p x dry_kg) / (e - t x p), with p about
  15 W/kg walking (ToddlerBot: 50 W at 3.4 kg, not a law), e = 160 usable Wh/kg (80% of nominal). [inferred] For the
  4.5 kg body at 15 W/kg plus the 28 W electronics reserve, about 95 W walking: 162 Wh nominal (about 130 Wh usable)
  gives roughly 1.3-1.4 h. A real duty trace replaces this.
- One energy hour is not one walking hour: ToddlerBot's longest uninterrupted walk was 19 min before motor heating
  changed behaviour.
- A pack must also deliver peak current and tolerate sag and heat; Wh and a C-rate label are not evidence that the
  rail or wiring is adequate.
- Regulator and connector lessons from the 2026-09 logic-board study (that board is superseded by the R33 power
  design): a buck converter's current limit can exceed the inductor's Irms rating, so the inductor rating is not a
  fault guarantee; ceramic capacitance falls with DC bias (a 10 µF 50 V X7R 1210 read about 7.2 µF at 12.6 V);
  1% feedback resistors can push a 5 V rail past 5.25 V at worst case; JST PH (2 A at AWG24) does not carry a 2.6 A
  camera-inclusive 5 V peak, JST XH (3 A, AWG22) does; low-insertion-force crimp contacts (JST SPH-002T-P0.5L) are less
  vibration-resistant and do not belong in a robot harness.

## 8. Rules that held up

1. Plan with rated, not stall.
2. Match voltage first: full-charge pack voltage must sit inside every servo's range; 7.4 V and 12 V variants are
   different parts.
3. Retail prices are comparison points. Compute cost per item and show the owner when one or two items drive the total.
   The early production-volume framing (a $1,000 robot at 1,000 units, 2026-09) is retired by the hobby-build intent.
4. Train with the actuator envelope in the loop.
5. Check identity, spline, connector and CAD per unit; never trust a family name.
6. Read before energising: meter the connector, check the adapter's power path, limit the current, start unloaded and
   restrained.
