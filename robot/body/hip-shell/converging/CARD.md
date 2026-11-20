# Hip shell, lens B (converging geometry): diverge cards

Designer B, 2026-11-15. Block models only. Every number comes from the files in this folder (scripts run under
memguard, ~4.5 min and 5.7 GB peak each). Right side; the left mirrors.

## Shared geometry (all three concepts)

- **Fixed part (torso).** The torso shells are trimmed at z 238 (waist line) and continue down into rounded briefs:
  the hull of the waist outline and a bowl ellipsoid (crotch at z ~180), 2.5 mm wall. The leg opening is a **wedge
  about the hip pitch axis**. Its front and back walls are ruled surfaces through the axis line (y 2, z 219), so
  they slope inward toward the turning point. Each wall angle psi(x) is measured from straight down. B1 and B3 set
  it from the leg's own angular envelope: pitch-link parts plus the egg thigh over roll -15..15, points at
  r >= 36-40 mm, plus the design flexion and 3 deg. B2 sets it from the sketch line. Thin **dark lips** lie in the
  walls from the skin to 36-40 mm from the axis, standing in for the sketch's dark band. The new fixed parts below
  the waist keep out of the yaw sweep (±40) of the hip-pitch servo and the yaw-to-pitch connector. The upper torso
  shell is left as it was.
- **Egg thigh (roll link).** It replaces both thigh hard covers. It is the hull of three ellipsoids, with the top
  converging toward C (x 11..89, y -26..39, z 130..205). The top is open: the roll sweep (-45..25) of the pitch-link
  roll servo and the pitch-to-roll connector is cut out, so roll against the pitch link is clear by construction.
- **Trade-off curve** (`edge_vs_flexion.json`, B1-style wedge, lips on; front edge height = where the front wall
  leaves the briefs skin, x 14..80):

  | design flexion | front edge z, min / median / max (mm) | standing front, visible mechanism (mm2) | standing side (mm2) |
  |---|---|---|---|
  | -30 | 196 / 207 / 212 | 612 | 1044 |
  | -45 | 205 / 216 / 216 | 1018 | 1176 |
  | -60 | 215 / 223 / 225 | 1488 | 1292 |
  | -75 | 224 / 232 / 235 | 2024 | 1436 |
  | -90 | 229 / 245 / 249 | 2554 | 1590 |

  The current body (uncut shells, original thigh covers) measures 1524 mm2 front and 1588 mm2 side standing
  (`visible_gaps.json`, baseline). Lips change the area by less than 5%.
- **What converging geometry can and cannot do (the main finding).** Take a fixed wall through the pitch axis and a
  rigid thigh. The angle the wall must leave open is the same angle that stays open when standing: flexion plus the
  leg's angular half-width plus clearance, about 125-130 deg at mid-hip for -90. Sloping the walls toward the axis
  does not make that angle smaller. It only moves the seam inward, where its width r * dpsi is smaller. On the skin
  (r 40-55 mm) each 10 deg of flexion raises the front edge by about 6-7 mm. Full -90 and the sketch's V
  (z ~205 at mid-hip) are about 40 mm apart in edge height.
- **Gaps that no fixed brief can close.**
  - Between x 14 and 72, nothing can converge onto the axis: the hip-pitch servo sits on the axis (x 14..57) and the
    yaw-to-pitch connector rises above C (x 49..72). Only the lateral strip x 72..95 is free all round the axis.
  - The egg's open top, where the pitch-link mechanism enters, leaves the largest standing window (about 40-70 mm
    tall at x ~22) in every concept.
- **Contacts that are not the hip shell.** Every concept shows the same three, and they cap the reported roll and
  yaw ranges:
  - Yaw ±40 grid contact: the kept upper torso shell against the yaw-to-pitch connector, 1.0 / 10.9 mm3. The
    original shells measure the same; the new briefs and lips are clear (checked separately).
  - Roll -45: the egg against the hanging right forearm covers, 856 mm3. That is an arm pose question.
  - Roll +25: the egg against the crotch strip, 63 mm3.
- **Method notes.** The kit grid steps are pitch 5 deg, roll -45/-30/-15/0/15/25 and yaw ±20/±40. So "contact-free
  ±20" means the next grid point, ±40, touches. Visible gap is from ray casts (`visible_gaps.py`). The largest-run
  number is inflated at silhouette columns and has a floor near 33 mm in many poses, so the mm2 area is the steadier
  measure.

---

```
hip shell · B converging · B1 axis wedge · diverge   (2026-11-15)
Idea: Range first. The leg opening is a wedge about the hip pitch axis. Its walls are ruled surfaces through the axis,
  placed from the leg's own envelope for -90..+40: psi_front 107 deg at x 6, 130 deg at mid-hip (x 46..62), 116 deg
  at x 86; psi_back -65 to -93 deg. On the skin the front edge runs from z 229 (x 78) to 249 (x 38) and the back edge
  from z 200 (x 6) to 221 (x 54). The wedge runs through the side, so the side seam is a "<" whose point is the
  pitch-axis exit, the place that does not move. A 12 mm crotch strip stays. Dark lips reach from the skin to 40 mm
  from the axis. Egg thigh as shared.
Evidence: robot/body/hip-shell/converging/b1_axis_wedge-look.png, b1_axis_wedge-interference.json,
  visible_gaps.json, edge_vs_flexion.json (script b1_axis_wedge.py).
  - Contact-free (grid): pitch -80..+30, roll -30..+15, yaw -20..+20.
  - First contacts: pitch -85 egg x briefs 5.7 mm3 and egg x yaw-to-pitch connector 3.8 mm3 (the known
    mechanism stop); +35 egg x briefs 12.1 mm3.
  - Combined grid: 297 of 420 poses grow overlap. Worst total 11 607 mm3 at (pitch -90, roll +25, yaw -40), egg x
    pelvis battery cradle 5 469 mm3.
  - Visible mechanism (front / side / back, mm2): standing 2556 / 1610 / 1176 (largest run 70 mm at x 22, z 239);
    -90 flexion 1398 / 2114 / 1360.
Grades (0-5 distance to viable, confidence, trend):
  - range 2 (medium, new): -80/+30 measured; the mechanism stop at -85 is shared with today's body.
  - looks-standing 4 (low, lead's guess): high-cut front with a 40-45 mm window onto the hip servo and connector.
    It reads as a cut belly, not as briefs, and has 1.7x the current body's visible area.
  - looks-moving 3 (low): tidy at -90 and when abducted; the window shows through every mid pose and at +40.
  - fit 2 (medium): clear of the mechanism by construction except at combined extremes, where the egg meets the
    battery cradle.
  - buildability 2 (low): two printed shells, no moving parts. The briefs and lips print as one torso piece, but the
    lips are thin cantilevers.
  - robustness 2 (low): wide open gaps barely pinch, but they collect dirt and fingers.
  - mass/complexity 1 (medium): adds no parts.
Requirement challenges: Full -90 with fixed briefs costs the sketch look about 40 mm of edge height. At -75 the
  median edge drops 13 mm and the visible area by 21%. The owner may want to trade the last 15 deg of flexion for
  the look, or accept a moving piece (lens A or C) in the window.
Overall: distance 3. Dominating gaps: the standing window and the egg's open top. Next step: none on its own. Keep
  its radial-wall seam geometry as a piece.
```

```
hip shell · B converging · B2 sketch crease · diverge   (2026-11-15)
Idea: Look first. The seam runs where the sketch draws it. On the actual briefs skin the front edge is
  z = 186 + 43 * x / 88 (measured z 189 at x 6 to 224 at x 78) and the back edge is z = 196 + 30 * x / 88
  (198 to 223). The walls behind the seam are still ruled surfaces through the pitch axis: psi_front 40 deg at x 6
  to 107 deg at x 78, psi_back -62 deg at x 6 to -100 deg at x 78. The visible seam is therefore the sketch's V line, and the
  wall slopes inward from it toward the turning point. Dark lips from the skin to 36 mm stand in for the dark band.
  Egg thigh as shared; flexion is whatever the V leaves.
Evidence: robot/body/hip-shell/converging/b2_sketch_crease-look.png, b2_sketch_crease-interference.json,
  visible_gaps.json (script b2_sketch_crease.py).
  - Contact-free (grid): pitch -30..+25, roll -30..+15, yaw -20..+20.
  - First contacts: pitch -35 lips x pitch-to-roll connector 6.1 mm3 and egg x briefs 3.5 mm3; +30 egg x briefs
    38.7 mm3.
  - Combined grid: 363 of 420 poses grow overlap. Worst total 18 627 mm3 at (-90, +25, -40).
  - Visible mechanism (front / side / back, mm2): standing 710 / 1332 / 832 (largest run 33 / 32 / 34 mm). That is
    the lowest front area of the three and 47% of today's body. At -90 the leg passes through the briefs, so the
    render there shows overlap, not a reachable pose.
Grades (0-5 distance to viable, confidence, trend):
  - range 4 (medium, new): -30 of the -90 target.
  - looks-standing 2 (low, lead's guess): reads as the sketch's V briefs with the thighs below. The orange
    pitch-to-roll connector still shows at the lateral top of each leg opening, and the dark band is thin.
  - looks-moving 4 (low): any flexion past -30 collides; nothing hides that.
  - fit 2 (medium): same as B1.
  - buildability 2 (low): same as B1.
  - robustness 2 (low): narrower gaps than B1. The lips are the first thing the pitch-to-roll connector touches
    at -35, so they are a pinch and wear point at the range ends.
  - mass/complexity 1 (medium): adds no parts.
Requirement challenges: This is the measured cost of the sketch line with fixed briefs: 60 deg of flexion. It
  shows that the look the owner drew and -90 need either a moving piece or a different seam.
Overall: distance 3.5. Dominating gap: range. Next step: none on its own. It is the look reference point that the
  others are judged against.
```

```
hip shell · B converging · B3 side skirt (file b3_side_walls) · diverge   (2026-11-15)
Idea: Seams in pitch planes, balanced. Hip pitch moves every leg point inside its own plane x = const, so a fixed
  panel lying lateral of the leg's whole x range is never swept by pitch, whatever the angle. B3 adds such a skirt to
  the briefs over each hip joint: x 91.5..94.5 (the thigh ends at x 89), y -12..42 (behind the flexed thigh), z 207..240
  (above the egg top, so yaw cannot reach it), rounded. It covers the side view of the hip servo and connector. Front
  and back are opened by the axis wedge set for -75..+40 (front edge on skin z 221..235, back 200..221). Lips at
  40 mm; egg thigh as shared.
Evidence: robot/body/hip-shell/converging/b3_side_walls-look.png, b3_side_walls-interference.json,
  visible_gaps.json (script b3_side_walls.py).
  - Contact-free (grid): pitch -65..+30, roll -30..+15, yaw -20..+20. The skirt itself shows no contact anywhere on
    the grid.
  - First contacts: pitch -70 egg x briefs 27.4 mm3, short of the -75 design because the egg near the axis
    (r < 40) is not in the wall envelope; +35 egg x briefs 13.7 mm3.
  - Combined grid: 310 of 420 poses grow overlap. Worst total 12 145 mm3 at (-90, +25, -40), egg x cradle.
  - Visible mechanism (front / side / back, mm2): standing 2030 / 878 / 1176. The side is the lowest of the three,
    55% of today's body. At -90 (past the contact-free range) 1216 / 1456 / 1358.
Grades (0-5 distance to viable, confidence, trend):
  - range 3 (medium, new): -65/+30.
  - looks-standing 3 (low, lead's guess): the side reads closed and the front window is 15 mm lower than B1's. In
    the block model the skirt floats about 5 mm off the bowl, so the bowl must widen to x 94 above z 207 to join it.
  - looks-moving 3 (low): the side stays covered in every pose; the front window shows through mid poses.
  - fit 2 (medium): the skirt clears roll and yaw on the grid.
  - buildability 2 (low): one more panel, printed with the briefs.
  - robustness 2 (low): the skirt is an exposed thin flange.
  - mass/complexity 1.5 (medium): about 2 x 4 cm2 of extra shell.
Requirement challenges: Same as B1, plus whether the owner's -90 is a hard target or a stress test. The curve puts
  -75 at 232 mm median edge and 2024 mm2 visible, against 245 mm / 2554 mm2 at -90.
Overall: distance 3. Dominating gaps: the front window and the egg's open top; neither can be closed by fixed
  geometry. Next step: develop it.
```

## Which to develop next

B3. Its side skirt is the one gain lens B gets for free: a fixed wall in a pitch plane costs no pitch, and it showed
no roll or yaw contact on the grid. Its radial front and back walls are the right seam geometry for whatever flexion
the owner settles on. The front window and the egg's open top need a part that turns with the leg, so the next step
is a hybrid:

1. B3's skirt and its radial walls.
2. Lens A's pitch-link hip cap, filling the egg's open top and the window at r < 40 mm.
3. The flexion target taken as an owner decision against the edge-height curve above.
