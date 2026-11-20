# Three brick-actuator tiers for the 22 joints (2026-11-08)

This is the engineering pick the owner asked for: mixed, brick-shaped actuators (the C047 case family), so the current
CAD only needs small changes. About three tiers fill the 22 joints at the right performance and a good price. It uses
no new WBC training.

## Inputs
- **Candidates:** `../brick-survey/` (204 part variants, 282 offers). 114 of them have a 12 V torque and
  speed figure.
- **Requirements:** the inverse-dynamics traces of the 59 target motions (`playful-ape-t1`, long-arm body), run
  `cove50-agility-requirements-t1-mixed-20261108-81015e58019b`. Per-group summary:
  `../joint-requirements-playful-ape-t1.json`.
- **Grading tool:** `tiers.py` (run `cove50-actuator-tiers-playful-ape-t1-r3-15c6f13d49e2`):
  ```sh
  .venv-control/bin/python -B robot/actuators/tier-pick/tiers.py --out REPORT.json
  ```

## Method
Each candidate is checked on every joint group's real target traces:
- **Torque:** rigid torque, plus the candidate's own reflected rotor inertia times acceleration, plus friction.
- **Motor line:** the torque–speed line at the 3S rail (12 V catalogue values × 0.925).
- **Rotor inertia:** identified for the STS3215/C047, STS3250 and ST3025. Elsewhere it is estimated per motor type
  times the gear ratio squared, as in `categories.py`.

Three continuous grades, each 2.5 at its reference scale:

| Grade | Measure | 2.5 at |
|---|---|---|
| Motor line | share of frames past the line over the whole target | 2% |
| Heat | RMS torque over rated across the whole target tape, a 402 s play session | 0.7× rated |
| Overload | the hardest single motion's RMS over rated | 1.6× rated |

A joint group's grade is the lowest of the three. Motions where a part is past its line on more than 2% of their
frames are listed beside it, so no single motion sets a grade or a price.

Every set of up to three parts is ranked by its cost for all 22 joints against its weakest group grade. Each group
takes the cheapest member that matches the set's weakest grade; the neck takes the cheapest member.

**The view used for the pick:**
- Parts on the Feetech/Waveshare serial protocol the electronics already use (one bus).
- Parts in the two pockets the design already has: the C047 case (45 × 25 × 35 mm) and the 40 × 20 × 40 mm leg case.

The full survey frontier is cheaper still with Hiwonder parts. They bring a second protocol, potentiometer feedback on
some, a 115 kbaud bus on some, and larger cases, so they stay outside the pick.

## Proposed tiers

| Tier | Part | Joints (22) | Unit price | Grades on its joints |
|---|---|---|---|---|
| Fast leg | Waveshare ST3025 (40 × 20 × 40 mm, brushless, low rotor inertia) | hip pitch, knee, ankle pitch (6) | $74.95 at 4+, Waveshare, checked in the browser | 3.1–3.4 |
| Strong | Feetech STS3250 (C047 case) | hip roll, shoulder pitch, roll and yaw, elbow (10) | $45.67 checked on AliExpress (1 per buyer), qty-20 $64.18, OpenELAB $73.99 | 3.2–3.6 |
| Light | Waveshare ST3215 12 V 30 kg.cm (C047 case) | hip yaw, ankle roll, neck pitch and yaw (6) | $20.63 at 12+, Waveshare | 3.4–3.7 |

**Cost for 22:**
- $1,031 at the checked STS3250 price;
- $1,216 at its qty-20 price;
- $1,314 at OpenELAB.

Today's plan (8 ST3025, 12 STS3250 and two XM335 in the neck) costs $1,504 plus the neck. The weakest joint group in
this set grades 3.1 and the mean is 3.5.

**What moves against today's plan:**
- **Hip roll moves to the STS3250.** It needs torque at low speed (2.3–2.7 N m p95–p99 below 2 rad/s), not speed.
  It grades 3.2 there, against 2.5 on the ST3025, and it keeps today's C047 pocket. So the ST3025 pocket change
  shrinks to hip pitch, knee and ankle pitch.
- **Hip yaw, ankle roll and the neck go to the cheapest tier.**
- **The neck** holds two Dynamixel XM335 in the accepted neck design. Moving it to the light tier removes the only
  second bus protocol, but changes the neck pockets. The neck is not in the force-check traces (the head is light),
  so its grade is not computed.

**Where the tiers run past their motor line** (share of that motion's frames). These are the motions to watch, and
they are what a stronger part would buy:
- **ST3025:**
  - hip pitch: 0.45 s knuckle drop 4.4%, 0.27 s quick crouch 4.0%;
  - knee: 0.5 s march 3.9%, 0.4 s shuffle 3.8%, 60°/s turn 2.4%.
- **STS3250:** elbow: 0.8 s knuckle walk 5.4%. Hip roll and the shoulders have no motion above 2%.
- **ST3215 12 V:** hip yaw: forward walk ×1.1 at 3.9%. Ankle roll has none.
  - If it took the arms instead, the lively arm play would suffer: 1 Hz arm display 18%, 2.5 Hz wave 15%,
    2.5 Hz chest beat 14%. That is why the arms sit in the strong tier.

## Alternatives worth a price check
- **Feetech HLS line**, same protocol and pockets, with current control:
  - HL-3960 (C047 case, 6.5 N m): grades 3.7–3.9 on hip pitch, knee and hip roll;
  - HL-3955 (40 × 20 × 40 mm, 5.4 N m): grades 3.0–3.3.

  Only AliExpress search prices exist: HL-3960 $77, HL-3955 $11.83 (implausible, probably an option or accessory),
  HL-3950 $62.51. A reCAPTCHA stopped the browser check, which is left to the owner. Rotor inertia for the HLS is
  estimated, not identified. If an HL-3960 really costs about $60–80, it would grade higher than the ST3025 on the
  leg joints and the STS3250 on hip roll, in the C047 pocket. That could remove the 40 × 20 × 40 pocket change.
- **Waveshare ST3020** ($19.31, 40 × 20 × 40 mm): a cheaper light tier.

## Limits
- **Lower bound:** a perfect controller following the reference exactly. Trained controllers used more leg torque
  than this earlier.
- **Data quality:** rotor inertia is estimated for most parts, and ratings are stall/3 where none is published.
- **Masses:** the traces use the mixed set's masses (servo masses here range from 55 to 124 g).
- **Prices:** some are unverified AliExpress search prices, shipping and import VAT excluded.
- **Not yet folded in:** the trained per-joint current from run `t1-mixed-longarm-20261108`.
- **Purchases** are the owner's decision.
