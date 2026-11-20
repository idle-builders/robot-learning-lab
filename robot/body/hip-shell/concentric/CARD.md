# Hip shell · lens A (concentric surfaces) · diverge (2026-11-15)

Three block-model concepts, all built and measured with `kit.py`. Scripts, renders and interference JSONs are in this
folder. Shared helpers are in `geom.py`. Design inputs and diagnostics are `probe*.py`, `blockers.py` and
`blockers.txt`.

Grades run from 0 (viable as is) to 5 (far from viable). Each grade has a confidence (low / med / high) and a trend
against the rejected swept-cut briefs (M-ck48bt): better (+), same (=), worse (-). Looks grades are my guess; the
owner decides.

## What lens A runs into (applies to all three cards)

- **Front coverage moves into the pelvis.** Whatever covers the front of the hip at standing has to turn 90 deg about
  the pitch axis at -90 flexion, which puts it directly above the pitch axis. That space is already full:
  - hip-pitch servo, r <= 31 mm about the pitch axis (yaw link);
  - lower-pelvis cradle, r 34..55 mm, at x 20..60;
  - yaw-to-pitch connector arm, r <= 57 mm, at x 49..72;
  - hip-yaw horn and servo, r_C >= 56 and 57 mm from C;
  - upper battery cell R2, r_C about 50 mm;
  - spine chassis.

  The pitch link's own lower part (bracket and roll servo, r <= 48) swings forward in front of the axis during
  flexion, so nothing fixed can sit there either.
- **Blocker map** (`blockers.txt`, real geometry, 2.5 mm clearance, full target grid). How much of each shell stays
  free, by blocker:

  | Shell | Torso internals block | Yaw-link servo and connector block | Arms and other leg block | Rolling thigh blocks | Free in total |
  |---|---|---|---|---|---|
  | Sphere about C, R 52 | 58 % | 27 % | 26 % | 18 % | 9 % |
  | Thigh dome about the roll axis, R 44 | 41 % | 34 % | not given | not given | 13 % |

  For the thigh dome, the pitch and yaw sweep carries the yaw-link parts and the pelvis through its top. So the
  "thigh top nested about the roll axis" survives only at its front and back ends.
- **Sphere size window.** A sphere about C has a narrow window:
  - it must clear the hip-pitch servo (r_C up to 51.8 at x 14, and 43 for x >= 30);
  - it must pass under the yaw horn and servo (r_C 56 and 57).

  The window is R 53 with a 2.5 mm wall if the ball starts at x 30.

## Method notes

- **Convex hulls in kit.interference.** `kit.interference` uses convex hulls for open meshes. The hull of the
  lower-pelvis cradle fills the whole region around the hip-pitch servo, and the hulls of the bracket and thigh
  structure are much larger than the parts. So every JSON also carries `real_geometry_check`. It uses the same grid
  and the same "growth over standing" rule. It counts real-surface points (2 mm spacing) that fall inside the posed
  concept solids (1.5 mm voxels). The unit is points, not mm3.
- **Neutral overlaps are not counted as contacts.** Both measures count only growth over standing. An invariant shell
  that already overlaps something at standing therefore shows no contact (A3), so I report `neutral_overlap_mm3`
  separately.
- **`kit.revolve` fails.** It fails with the installed manifold3d, because `Manifold.revolve` needs a `CrossSection`,
  not a list. `geom.rev` is used instead.
- **Render memory.** One `kit.render` call with all four LOOK_POSES peaks just above 6 GB here. `geom.render_phase`
  therefore calls it twice (two poses each) and stacks the two images.
- **Upper torso.** The upper torso (original shells above z 255) is drawn in the renders only. It is not in the
  interference measures.

---

```
hip shell · A concentric · A1 ball socket, range-first · diverge   (2026-11-15)
```

**Idea.** A doll hip with the briefs cut so that the full target range is contact-free:

- **Hip ball (pitch link).** Sphere about C = (59, 2, 219), R 52, 2.5 mm wall, x 14..102. It is kept only at the
  angles that no part (real geometry) reaches over pitch -90..+40, yaw ±40 and roll -45..+25, with 2.5 mm clearance.
- **Thigh dome (roll link).** Spheroid of revolution about the roll axis: centre (56, 6.5, 190), R 44 in the frontal
  plane, 34.5 front-back, upper part only. It is trimmed the same way.
- **Thigh covers.** The existing thigh covers, kept below z 172.
- **Briefs (torso).** An ellipsoid bowl below the waist z 255 (half-axes 80 x 45 x 78), cut by the voxel sweep (3 mm
  gap) of every right-leg part over the full combined grid, mirrored to the left.

**Evidence.**

- Render: `a1-ball-socket.png`. JSON: `a1-ball-socket-interference.json`.
- kit:
  - pitch -90..+40, yaw ±40;
  - roll -15..+25: the first contact is at -30, the thigh cover against the hanging forearm, 2.6 mm3;
  - 121 of 420 combined poses have contact. The worst total is 3851 mm3 at -90/+25/-40 (thigh cover against the
    cradle hull).
- Real geometry:
  - pitch -90..+40, roll -30..+25, yaw ±40;
  - 99 of 420 combined poses have contact. The worst is the right thigh cover against the right elbow servo at
    -90/-30/+40 (490 points). Next is the thigh cover crossing the midline into the left hip-pitch servo.
- Every contact is the thigh column against the body. None comes from the hip shells.
- The ball keeps only 5.9 cm3 per side, about 9 % of its band. The dome keeps 2.8 cm3. The briefs end at z 208: no
  crotch and no lower belly.

**Grades.**

| Criterion | Grade | Confidence | Trend |
|---|---|---|---|
| Range | 1 | med | + |
| Looks, standing | 5 | high | - (front of the pelvis, servos and battery fully open) |
| Looks, moving | 5 | high | - |
| Fit | 1 | med (no neutral overlap) | = |
| Buildability | 3 | low (the ball and dome are slivers) | |
| Robustness | 4 | med (open hip collects dirt) | |
| Mass / complexity | 1 | med | |

**Requirement challenges.**

- **Combined-pose corners.** In the target grid, -90 flexion combined with yaw ±40 and roll swings the thigh across the
  whole front of the pelvis and into the other leg. A fixed briefs that clears that has no front. Are the combined
  corners targets, or only the per-axis ranges?
- **Thigh covers against the forearm.** The thigh covers already touch the hanging forearm at roll -30 to -45. That
  is an arm-rest pose question, not a hip-shell one.

**Overall.** Distance about 4.5. This is the bound of the lens on the current pelvis: a concentric cap cannot fill
what a range-first briefs leaves open, because the cap is blocked in the same places. It is not a candidate. Keep it
as the reference.

---

```
hip shell · A concentric · A2 coaxial drum, look-first · diverge   (2026-11-15)
```

**Idea.** The classic action-figure hip, drawn for the sketch look and then measured:

- **Drum (pitch link).** A surface of revolution about the hip-pitch axis:
  - superellipse profile with p 4, R 50 at x 57, x 13..101, 2.5 mm wall;
  - the inboard end is an annulus with r 33 for the pitch servo;
  - it covers the pitch-axis angles from 160 to 380 deg: from the front at 20 deg above horizontal, under the hip,
    to the back at 20 deg above horizontal;
  - it is trimmed only where the rolling thigh passes, and outside the thigh dome.
- **Thigh.** Same dome and covers as A1.
- **Briefs (torso).** Ellipsoid bowl plus a coaxial drum socket (R 53..55.5, 3 mm gap), so pitch never sweeps the
  drum against the briefs. The V leg-opening plane runs through the crotch (12, 2, 182), the front (59, -60, 226) and
  the back (59, 64, 226).

**Evidence.**

- Render: `a2-coaxial-drum.png`. JSON: `a2-coaxial-drum-interference.json`.
- Neutral overlaps: drum against the cradle hull 431 mm3, against the pitch servo 20 mm3.
- kit:
  - pitch -15..0. The first contact is at -20 (drum against the cradle hull, 512 mm3) and at +5 (cradle hull,
    502 mm3);
  - roll -15..0. At +15 the thigh covers touch the briefs V, 28 mm3;
  - yaw 0. At ±20 the drum hits the cradle hull (1060 to 1460 mm3), the briefs socket (666) and the lower pelvis
    cell (657);
  - 415 of 420 combined poses have contact. The worst total is 23 746 mm3 at -90/+15/-40 (drum against the cradle
    hull, 6436).
- Real geometry:
  - pitch -35..0. The first contact is at -40, drum against the pitch servo;
  - roll -30..0, yaw 0;
  - 408 of 420 combined poses have contact.

**Grades.**

| Criterion | Grade | Confidence | Trend |
|---|---|---|---|
| Range | 5 | high | - |
| Looks, standing | 2.5 | low | + |
| Looks, moving | 4 | low | = |
| Fit | 4.5 | med | |
| Buildability | 2 | med | |
| Robustness | 3 | low | |
| Mass / complexity | 2 | med | |

- **Looks, standing:** this is the closest read of the sketch here: a briefs edge with a rounded hip under it. But the
  drums are bulbous, and the briefs has a ledge at the waist where the socket is wider than the torso (block-model
  artefact).
- **Looks, moving:** the drum's front turns into the pelvis. At -90 its open underside faces forward.
- **Fit:** the drum overlaps the cradle at standing.
- **Buildability:** two printed parts per side, with a tight 3 mm coaxial socket.
- **Robustness:** the 3 mm coaxial gap pinches and collects dirt.
- **Mass / complexity:** drum 31.8 cm3 per side, briefs 50.9 cm3.

**Requirement challenges.**

- **Yaw.** A drum about the pitch axis has no yaw. Any yaw swings its ends (46 and 42 mm from the yaw axis) into a
  tight socket. Doll hips get away with this because they have no hip yaw. Here yaw ±40 is a target.
- **Front coverage.** Covering the front needs the same pelvis pocket as A3.

**Overall.** Distance about 4.5. The dominating gaps are:

- yaw;
- the pelvis internals above the pitch axis, which stop flexion at -15 to -35.

It is superseded by A3, where the sphere handles yaw. Its use is to show what the sketch look costs on the current
pelvis.

---

```
hip shell · A concentric · A3 ball between the servos (pelvis-pocket challenge) · diverge   (2026-11-15)
```

**Idea.** A closed doll ball that covers the front of the hip at standing and turns up under the briefs in flexion.

- **The ball (pitch link).**
  - It is a sphere about C, R 53, 2.5 mm wall (inner 50.5), x 30..96.
  - The inboard end is an annulus with r 35 about the pitch axis: the hip-pitch servo passes through it.
  - The outboard end is closed.
  - It fits between the pitch servo (r_C <= 43 for x >= 30) and the yaw horn and yaw servo (r_C >= 56 and 57).
  - It is trimmed against the rolling thigh, the arms, motors, horns and cells.
  - A slot at x 47..74 spans pitch-axis angles -2.5..142.5. It is for a yaw-to-pitch connector rerouted to cross the
    ball at 45 deg (back-up), so the slot is under the briefs at standing.
  - It is not trimmed against the pelvis structure (cradle, spine chassis, front power carrier), the upper battery
    cell R2 or the current connector. That is the requirement challenge, and the contacts in the card measure it.
  - About 45 % of the ball's band survives, 25.6 cm3 per side.
- **Briefs (torso).**
  - It is the convex hull of the waist ellipsoid and sphere sockets about C (R 56..58.5, 2.5 mm wall), open at the
    waist. A sphere about C is invariant under pitch and yaw, so the ball slides under the briefs edge with a constant
    3 mm gap in both.
  - The leg opening is a plane through the crotch (12, 2, 176), the front 30 deg above horizontal and the back 5 deg
    above horizontal.
  - The leg opening also includes the voxel sweep of the thigh in pure pitch -90..+40.
- **Thigh.** The existing covers below z 180. The thigh top sits inside the ball's bottom opening.

**Evidence.**

- Render: `a3-ball-between-servos.png`.
- JSON, current pelvis: `a3-ball-between-servos-interference.json`.
- JSON, pocketed: `a3-ball-between-servos-pocketed-interference.json`. Same parts, with the cradle, chassis, front
  carrier, cell R2 (with its fin and insulator) and the connector removed.

As-is pelvis:

- Neutral overlaps: ball against the cradle hull 2090 mm3, the chassis hull 505 mm3, and the thigh-structure hull
  1260 mm3 (a hull artefact: no real overlap). Real-geometry neutral overlap: 49 cradle points and 51 chassis points.
- kit:
  - pitch -35..+40. The first contact is at -40, ball against the current connector, 58 mm3;
  - roll -15..0. At +15 the ball hits the thigh-structure hull (artefact);
  - yaw 0. At ±20 the ball hits the cradle (2389 to 2657 mm3) and the chassis (543 mm3);
  - 403 of 420 combined poses have contact. The worst total is 11 034 mm3.
- Real geometry:
  - pitch -5..0 (chassis), roll -30..+15, yaw ±20;
  - 399 of 420 combined poses have contact.

Pocketed:

- kit:
  - pitch -90..+40;
  - roll -15..0 (hull artefact; real geometry is clean to +15);
  - yaw ±20. At ±40 the ball's inboard end reaches the lower-pelvis cell at x ±11, 88 mm3;
  - 334 of 420 combined poses have contact. The worst total is 3855 mm3 (ball against the thigh hull, artefact).
- Real geometry:
  - pitch -90..+40, roll -30..+15, yaw ±20;
  - 271 of 420 combined poses have contact. The worst is the thigh cover crossing into the other leg's pitch servo
    at -90/+25/-40, and the briefs against the knee servo at -80/+25/-40.

**Grades.**

| Criterion | Grade | Confidence | Trend |
|---|---|---|---|
| Range | 2 pocketed, 5 as-is | low | + |
| Looks, standing | 3 | low | + |
| Looks, moving | 2.5 | low | + |
| Fit | 2 pocketed, 4 as-is | med | |
| Buildability | 3 | low | |
| Robustness | 3 | low | |
| Mass / complexity | 3 | low | |

- **Looks, standing:** the front of the hip is closed by the ball under a V briefs edge. The ball's front and side
  views read differently:
  - From the front it reads as a tall rounded capsule (x 30..83, z 166..271), not a ball. The arm's yaw sweep trims
    it outboard of x 83, and it starts at x 30.
  - A strip at x 11..30, between the ball and the narrow crotch strip, shows the pitch bracket and roll servo.
  - From the side it reads as a round hip under the edge.
- **Looks, moving:** one smooth ball turns under the edge, with no sweep against the briefs in pitch or yaw. The
  connector slot shows at the back in deep flexion, and the ball's bottom opening shows the cover tops.
- **Buildability:** the sphere prints in two halves. The cost is the pelvis rework, the connector reroute and the
  slot.
- **Robustness:** a 3 mm socket gap, plus dirt entering through the slot and the bottom opening.
- **Mass / complexity:** about 32 g ball per side plus about 30 g briefs in PETG, plus the rework of other parts.

**Requirement challenges.**

1. **Pelvis pocket.** A sphere of r about 56 around each C must be free of fixed structure. That means pocketing the
   lower-pelvis cradle, the spine chassis and the front power carrier: about 2.6 cm3 of their hulls inside the ball
   at standing. Battery cell R2 grazes the sphere (3 mm3 at R 54).
2. **Connector reroute.** The yaw-to-pitch connector must cross the ball at about 45 deg (back-up) through a slot,
   instead of dropping straight down onto C.
3. **Yaw ±40.** The ball's inboard end (x 30) still swings into lower-pelvis cell 08 at yaw ±40: 88 mm3 in kit,
   28 points in the real-geometry check. Options:
   - trim the ball inboard further, which widens the exposed inboard strip;
   - move the cell;
   - accept yaw ±30.
4. **Width and the inboard strip.** The hanging arm limits the ball to x <= 83, so there is no hip flare. The inboard
   strip x 11..30 needs either a wider crotch gusset on the briefs or a ball that reaches further inboard, which
   collides with the pitch servo and the cells.
5. **Combined corners.** The worst remaining combined poses are thigh-against-body (other leg, knee servo against the
   briefs), as in A1. They are the same question about whether combined corners are targets.

**Overall.** Distance about 2.5 to 3, conditional on the pocket. The dominating gaps are:

- whether the pelvis packaging (cells, cradle walls, electronics carrier) can give up a sphere of r about 56 around
  each hip;
- the connector reroute.

**Next most valuable step.** Overlay the r 56 spheres on the pelvis packaging and the battery layout to see what moves
and what it costs. Then model the rerouted connector and re-run this card with it in place of the slot assumption.

---

## Recommendation within lens A

**A3** is the only concentric pattern that reaches -90..+40 pitch with the front of the hip covered and one shell
reading. In the pocketed measure it reaches pitch -90..+40 and yaw ±20, and roll -30..+15 in the real-geometry check.
The cost lives in other parts: the pelvis pocket and the connector reroute.

A1 and A2 bracket it, both on the current pelvis:

- A1 keeps the full range and loses the front.
- A2 keeps the front and loses the range (-15 to -35 pitch, no yaw).
