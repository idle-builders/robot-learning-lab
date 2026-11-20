# Hip shell: brief and process (Idle T-czv2d8)

## Goal

One smooth shell over the hip, made of continuous curves that flow into explicit globes encapsulating the key joints
(owner, 2026-11-16). Some sections may be static and some may move. Hip flexion target: -80 to -85 deg (relaxed
from -90 by the owner, 2026-11-15). Actuators may be moved where the current layout limits the look, copying the
arrangements of the latest humanoids.

All decisions and assumptions are in [`DECISIONS.md`](DECISIONS.md), each with a visual explainer, for owner review.

## Process (owner, 2026-11-16)

| Stage | What | Output |
|---|---|---|
| S1 analysis | Cuts through the bare skeleton for each hip motor: turning point, swept region, internal contact, space for static and moving shell pieces. Cuts lie in the plane each joint keeps, so the 2D drawing is exact. | `sections/joint_cuts.py` → `sections/cuts-hip-{pitch,roll,yaw}.png`, `cuts-contacts.json` |
| S2 inspiration | Boards for the look (continuous curves into joint globes) and for the hip actuator arrangements of the latest humanoids. | `.cache/work/hip-refs/globe-joints-board.png`, `hip-actuator-architectures-board.png`, `hip-actuator-architectures.md` (local, third-party images) |
| S3 2D sketch | Shell profiles (and, where needed, actuator layouts) drawn as curves on the S1 cuts. | `sketch2d/` |
| S4 2D validation | Each sketch curve is checked against the S1 spaces in every cut and pose, and scored in mm of intrusion and gap. | `sketch2d/` |
| S5 CAD | Only for a sketch the owner has accepted in 2D. | later |

Every stage ends with drawings for the owner. A decision taken during a stage goes into `DECISIONS.md` before the
stage closes.

**Where we are (2026-11-16, later): D-11.** We keep today's hip actuators and the current FDM design with the 90-degree shell, and close its gaps in the style of the sketch. The owner is drawing on the cut sheets in `sections/drawing-sheets/` (`sections/drawing_sheets.py`). The concurrent-hip globe round below was rejected as bulbous (D-09, D-10).

**Earlier (2026-11-16): S3 on the concurrent hip (P-04, D-08).** All three hip axes meet at C, so every globe
layer is a surface about C (a sphere, or a slim surface of revolution about the pitch axis) owned by one link. That
makes the layers exact 2D curves in any cut through C. No CAD yet. Today's hip stays as the fallback.
- `sketch2d/block_sweep.py`: every concurrent servo arrangement and the globe it needs (F-01).
- `sketch2d/globe_concepts.py`: concepts K1 ball, K2 slim egg and K3 saddle as nested layers shaped from the measured free space (A-21), scored over the poses and drawn on the whole robot → `globe-concepts.png`, `globe-concepts.json`.
- `sketch2d/globe_cuts.py`: the same concepts cut through C, with servos, U-bracket and fork → `globe-cuts.png`.

## Facts (right side; the left mirrors x)

- **Hip axes:**
  - The yaw axis is vertical through (59, 2).
  - The pitch axis runs along x through (y 2, z 219).
  - Yaw and pitch meet at C = (59, 2, 219).
  - The roll axis runs along y through (x 56, z 190), 29 mm below C.
- **Chain:** torso → R_yaw_link → R_pitch_link → R_roll_link (thigh) → shin.
- **Hip-pitch servo:** ST3025 on the yaw link, case x 14..57 (shaft along x, 45 mm inboard of C).
- **Yaw-to-pitch connector:** rises from C to the torso-mounted yaw servo (x 49..72, z 224..275).
- **Pitch link:** carries the roll servo and the pitch-to-roll connector (z 177..229).
- **Battery cradle:** a keel between the hips (x ±18 below z 255) under a plate at z 258..276 (x ±55).
- **Fixed by the owner:** body proportions (arm 267 mm, hips 12 mm in, feet 4 mm in, feet x1.3). Joint positions and motors may now change only through an entry in `DECISIONS.md`.

## Tools

- `kit.py`: base body, part meshes (disk-cached), link transforms, revolve, render, and the 3D interference measure (3D is used in S5 only).
- `sections/joint_cuts.py`: the S1 cut sheets (single joint, superseded by `sections/leg_motion.py`, whole leg).
- `sketch2d/hip_layouts.py`: actuator layouts as block sketches, checked over the whole-leg poses.

Run every script under `python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B ...`, one heavy job at a time. Never run simulation or training locally.

## Records

The rejected rounds are kept as records only:
- `concentric/`, `converging/` and `moving-compliant/`: the 2026-11-15 diverge round, nine concepts;
- `concept-board.png`;
- `sections/hip-sections.png` and `sections/waist-line.png`: the first cut drawings.
