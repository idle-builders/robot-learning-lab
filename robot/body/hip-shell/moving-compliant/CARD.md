# Lens C: moving or compliant covers, diverge cards (2026-11-15)

All three concepts share these block-model parts, built in `mc_common.py`:

- **Torso cut.** The belly and back shells are cut back to a waistband rim at z 250 between x 8 and x 66. Further out, only the part in front of y -12 and behind y 14 is cut, so a side strip around the hip-yaw connector keeps the original rim at about z 231. The 16 mm centre strip (|x| < 8) runs down under the crotch as a briefs gusset.
- **Thigh covers.** Original shape, with the front-top corner chamfered on the plane through (y 4, z 207) and (y -25, z 190). Without the chamfer the cover hits the yaw-to-pitch connector before -90.
- **Side hip panel.** An oval plate on `R_yaw_link` at x 93..95.5 (y -24..28, z 195..229), outside the thigh's outer wall. It yaws with the leg and never pitches. It stays below the torso side rim, because above z 229 it would sweep into the torso side when the hip yaws.

Concept parts that move by a fraction of the hip motion sit on virtual links: `mc_common.VBody`, `follower_link`, `frac_link`. Because of that, `kit.render` and `kit.interference` measure them unchanged. Each concept's JSON also has a `pitch_sweep_growth_mm3` table: every contact pair along the hip pitch alone, which is the full picture behind the kit's first-contact walk.

**Reference: today's shells, same grid** (`reference-body-interference.json`, from `reference_body.py`):

| Contact-free pitch | Roll | Yaw | Combined poses with contact | Worst combined |
|---|---|---|---|---|
| -45..+35 | -15..+15 | -20..+20 | 342 / 420 | 8995 mm³ at (-90, +25, -40) |

The worst combined pose is the thigh cover against the battery cradle. That pose and the second (thigh against the hanging arm at -90/-45/+40) stay worst in every concept below. They belong to the leg and torso layout, not to the hip cover.

**Screen, not proposed: a half-angle flap about the pitch axis** (`half_angle_check.py`, `half-angle-check.json`). A cylindrical front flap at k = 0.33 / 0.5 / 0.67 of the hip pitch has nowhere to go at -90:

- At -90 the thigh's front face (27 mm ahead of the axis) becomes its top, at z ~246. It fills the arc in front of the axis up to 30-35 deg above horizontal.
- Behind the belly wall the torso is full: battery cradle, yaw-to-pitch connector and front electronics.
- At k = 0.5, a flap at r 46 spanning -8..+35 deg hits the cradle and the connector: 1595 mm³ at -90, 156 mm³ already at -30.
- The only arrangement that nearly works is a narrow 25-deg band (z ~198-224), standing 15 mm proud of the belly at r 60. It hits the rim and the thigh with 195 mm³ at -90.

So the front cover has to leave by moving outward or upward, not by turning about the pitch axis. That is why the concepts below slide, hinge high, or are fabric.

---

## hip shell · C moving/compliant · C1 sliding briefs (thigh-pushed telescoping panels) · diverge (2026-11-15)

**Idea.** Below the waistband rim each hip has two rigid curved panels, 1.6 mm thick, that follow the belly or seat outline 0.5-1.4 mm inside the shell's inner wall.

- **Front panel** (x 8..~64). Its lower edge is the briefs leg opening: z 192 at the crotch, rising to ~208 at the panel end, so it overlaps the thigh front by 5-15 mm and the thigh top sits tucked under the edge. In flexion the thigh top pushes a low-friction lip on the panel's lower edge, and the panel slides straight up inside the belly. Light springs and gravity return it.
  - Lift: 0 down to -15, 6 mm at -30, 21 mm at -60, 45 mm at -90.
- **Back panel.** Does the same at extension: 8 mm lift at +40.
- **Side hip panel** on the yaw link closes the side.

No drive parts cross the hip. A positively driven version (pin on the pitch link, or a gear sector on the pitch axis) has to route through the space the thigh cover sweeps in roll, so it was not modelled further.

**Evidence.**
- Render: `robot/body/hip-shell/moving-compliant/sliding-briefs.png`
- Interference: `robot/body/hip-shell/moving-compliant/sliding-briefs-interference.json`, with the pitch sweep and the panel lift per pitch.

Key numbers:
- **Contact-free walk:** pitch -15..+20, roll -15..+15, yaw -20..+20. Combined: 385/420 poses with contact; worst 6341 mm³ at (-90, +25, -40), the thigh cover against the cradle, as in the reference.
- **Why the pitch walk stops early.** It stops on the panel's channel, not on the leg:
  - An 11 mm³ sliver against the battery cradle's convex-hull proxy at the panel's crotch end (x 7-11) appears as soon as the panel lifts.
  - At -80/-90 the top of the lifted panel hits the front electronics carrier (94 / 279 mm³) and the precharge board (86 mm³ at -90).
  - Panel against the leg and mechanism: clear from -90 to +40.
  - Extension: back panel against the pitch-to-roll connector, 2-4 mm³ at +30..+40.
- **Sensitivity to the leg line.** An earlier run with the leg line at z 200 instead of 192 needed 38 mm of lift instead of 45, and the -90 contacts dropped to the carrier (94 mm³) plus the same 11 mm³ cradle sliver. The look (thigh tucked under) trades directly against internal clearance.

**Grades** (0-5 distance to viable, confidence, trend):

| Criterion | Grade | Confidence | Trend | Reason |
|---|---|---|---|---|
| Range | 2.5 | medium | new | The leg is clear to -90/+40. Reaching that needs a 2.5 mm × ~50 mm tall free channel behind the belly wall, which the cradle front and the front electronics occupy today. Roll and yaw are unchanged from the reference. |
| Looks-standing | 2 | low; the owner decides | new | Briefs with a crisp waistband seam and a leg opening, thigh tops tucked under. Seen from the front-side, the pitch-to-roll connectors still show in the gap between the panel end (x 64) and the side panel. |
| Looks-moving | 2 | low | new | The panel vanishes into the belly as the thigh rises, so at -90 the belly sits directly on the thigh with no hole. The open-topped thigh cover still shows the roll servo at -90, which is common to all lenses. |
| Fit | 3.5 | medium | new | Needs ~1-2 mm relief on the cradle front face, the carrier's lower front corner moved or notched, and the precharge board kept above z ~300 at x < 41. Alternatively, a lower lift. |
| Buildability | 3 | low | new | Four thin curved panels plus two side panels, guide rails or a channel in the shell wall, four light springs, and a PTFE or roller lip. Thin printed PETG panels in a closed channel need tolerance work. |
| Robustness | 3 | low | new | The lip rubs on the thigh-cover top at every step past -20 (wear, noise). Yawing the hip loads the panel sideways, which can jam it. The channel collects dust. Little pinch risk: the panel moves away from the crease. |
| Mass/complexity | 2 | medium | new | ~4 g per panel, ~4.5 g per side panel, so ~25 g a side plus springs. Four sliding parts per side. |

**Requirement challenges.**
- Is a 45 mm internal channel behind the belly wall acceptable, or can the leg line rise to ~200? At 200 the lift is 38 mm and the thighs tuck less.
- Roll -45/+25 and yaw ±40 stay limited at ±15/±20 by the existing thigh covers (against the pitch-to-roll connector) and by the torso side strip (against the yaw connector). This concept does not address those limits.

**Overall.** Distance to viable ~2.5.
- Dominating gaps: internal packaging behind the belly, and lip wear and jamming under yaw.
- Next most valuable step: check the real (watertight) cradle and carrier geometry against the panel path. The sliver is measured against a convex-hull proxy and may not exist. Then test a 200 mm leg line and a split panel (x 8-40 / 40-64) so the inner half needs less lift.

---

## hip shell · C moving/compliant · C2 hinged tassets (spring-closed flaps the thigh pushes open) · diverge (2026-11-15)

**Idea.** Same torso cut, but 6 mm higher (rim z 256), so an opened flap clears the thigh top, which sits at z ~246 at -90. Below the rim the briefs are flaps hinged on the outside of the rim, like armour tassets or a cat flap. Per hip there are four 2 mm flaps, 0.8 mm outside the belly or seat outline:

- **Front flap** (x 8..~32): hinges along x.
- **Front-side flap** (x ~32..~64): hinges along the rim tangent at its middle. A single straight hinge for the whole wrap-around flap swings its side part down into the thigh. With one, the flap could not open past ~68 deg at -70/-80 (screened).
- **Back and back-side flaps:** the same at the back.
- **Side hip panel** as in C1.

Torsion springs hold the flaps closed. Thigh, roll servo and pitch-to-roll connector push them open:

| Flap | -30 | -60 | -90 | +40 |
|---|---|---|---|---|
| Front | 2 deg | 33 deg | 75 deg | — |
| Front-side | 12 deg | 44 deg | 84 deg | — |
| Back-side | — | — | — | 26 deg |

At -90 the front flaps lie almost horizontal on top of the thigh, like a lifted apron. Nothing slides inside the torso.

**Evidence.**
- Render: `robot/body/hip-shell/moving-compliant/hinged-tassets.png`
- Interference: `robot/body/hip-shell/moving-compliant/hinged-tassets-interference.json`, with the pitch sweep and the flap angle per pitch.

Key numbers:
- **Contact-free walk:** pitch -40..+15, roll -15..+15, yaw -20..+20. Combined: 356/420; worst 5453 mm³ at (-90, +25, -40), the same thigh-cradle pair.
- **Contacts along pitch.** Only the flap hinge ends rub the cut torso shell:
  - Front-side flap against the side strip: 1.4 mm³ at -50 rising to 30 mm³ at -90. A hinge-end relief removes it.
  - Back-side flap against the back shell: 0.5-2.2 mm³ at +20..+40.
- No contact with the mechanism from -90 to +40.

**Grades** (0-5 distance to viable, confidence, trend):

| Criterion | Grade | Confidence | Trend | Reason |
|---|---|---|---|---|
| Range | 2 | medium | new | Clear from -90 to +40 apart from small hinge-end rubs. Roll and yaw unchanged from the reference. |
| Looks-standing | 2.5 | low; the owner decides | new | Same briefs outline as C1, but the flaps stand 1-3 mm proud of the belly, and the flap seams show as extra lines. |
| Looks-moving | 3.5 | low | new | The flaps visibly swing out: 10-30 deg in a walking range, and a 55 mm shelf sticking forward at -90. Gaps open under their edges while they are lifted. Reads as armour or tassets, not one smooth shell. |
| Fit | 1.5 | medium | new | Needs no space inside the torso; hinges sit on the rim. |
| Buildability | 2 | medium | new | Eight printed flaps, eight small torsion springs and hinge pins; prints flat-ish. |
| Robustness | 3.5 | low | new | Open flaps can snag on the floor or obstacles in crawling and kneeling and in falls, which matters for an ape that crawls. Flap edges ride on the thigh cover (wear). Spring fatigue at every step. Finger pinch where the flap closes onto the thigh crease. Hinge pins collect dirt but are easy to clean. |
| Mass/complexity | 2.5 | medium | new | ~5 g per flap, ~40 g plus springs per robot. Eight moving parts per side. |

**Requirement challenges.**
- "Reads as one shell when moving" versus a cover that must leave outward: the flaps answer coverage, not that requirement.
- If crawling with the hips at -60..-90 is common, the open-flap snag risk counts more than the range.

**Overall.** Distance to viable ~2.5-3.
- Dominating gaps: the look in motion and snagging. The packaging is the easiest of the three.
- Next most valuable step: show the owner the -45 and -90 frames. If the apron look is acceptable, check flap snag clearance against the ground in the crawl poses.

---

## hip shell · C moving/compliant · C3 segmented core + stretch fabric (owner's direction) · diverge (2026-11-15)

**Idea.** The core shell is no longer one belly part and one back part.

Rigid islands:
- Upper torso front and back (torso), cut to the rim at z 250.
- Centre briefs gusset (torso), running under the crotch.
- One side hip panel per hip (yaw link). It overlaps the thigh's outer top, so the outer side needs no fabric.
- Thigh covers (roll link), chamfered.

Fabric per hip: 4-way nylon/elastane stretch knit, mounted with 10 % pre-stretch at standing, clamped or sewn into edge rails. It is rendered dark grey.

| Zone | Upper edge (torso) | Lower edge (thigh) | Rulings at standing |
|---|---|---|---|
| Front (groin crease) | waistband rim | front cover at z ~186, lower toward the crotch | 9, 68-90 mm |
| Back (seat) | back rim | back cover at z 168 | 9, 85-90 mm |
| Back, long variant | back shell at z 265 (15 mm above the rim, fabric over the shell) | z 158 | 109-114 mm |
| Inner | gusset edge | thigh inner wall at z 165 | 11, 24-46 mm |
| Inner, high variant | gusset edge | z 182 | — |

A longer seat panel sees less strain, because the length change is set by the hip motion, not by the panel. Per pose each ruling's taut length is the shortest free path in its own plane; it wraps over any part that pokes across. Strain is measured against the natural length (standing length / 1.10). Working strain used: 40 % for a 4-way knit (inside the owner's 30-60 % range); for comparison, ~25 % for 2 mm neoprene and ~15 % for a printed TPU 95A skin.

**Evidence.**
- Render: `robot/body/hip-shell/moving-compliant/fabric-core.png`
- Interference (rigid islands only): `robot/body/hip-shell/moving-compliant/fabric-core-interference.json`
- Strain: `robot/body/hip-shell/moving-compliant/fabric-core-strain.json`, per ruling and pose, with shear, excess fabric and crease gap.

Rigid islands:
- **Contact-free walk:** pitch -90..+40 (no contact at all along pitch), roll -15..+15, yaw -20..+20. Combined: 302/420 (fewest); worst 5443 mm³ at (-90, +25, -40), the same thigh-cradle pair.

Fabric strain (min..max over the rulings; + stretch, - slack that must fold; 10 % pre-stretch included):

| Pose | Front (groin) | Back (seat, z 168) | Back long (z 265 to 158) | Inner (gusset) |
|---|---|---|---|---|
| -90 flexion | -80..-50 % (61 mm excess folds into a 6 mm crease) | +25..+71 % | +15..+62 % (most ~+50 %) | +16..+253 % |
| +40 extension | +22..+40 % | -31..-17 % | -24..-14 % | -29..+106 % |
| -45 abduction | -21..+37 % | -12..+35 % | -12..+28 % | +66..+164 % |
| +25 adduction | -12..+27 % | -11..+21 % | -7..+18 % | -85..-31 % (crease gap 2 mm) |
| yaw ±40 | 0..+36 % (shear ≤15 deg) | +3..+30 % | +5..+25 % | +33..+123 % |
| -90 & yaw +40 | -35..-16 % | +34..+87 % | +26..+66 % | +88..+376 % |

The "inner high" variant (lower edge at z 182) still reaches +149 % at -90 and +204 % at +40, and some of its rulings already wrap over the roll servo at standing.

**Grades** (0-5 distance to viable, confidence, trend):

| Criterion | Grade | Confidence | Trend | Reason |
|---|---|---|---|---|
| Range | 2.5 | medium | new | The rigid islands never limit pitch. The fabric does: the seat needs ~50-65 % at -90 (above the 40 % working figure, inside 30-60 % only at its top), and the inner zone needs 150-380 %, which no knit takes with a fixed gusset. |
| Looks-standing | 2.5 | low; the owner decides | new | Reads as soft briefs between hard shells. Coverage is continuous with no gaps, but the material is visibly different from the sketch's hard shell. A shell-coloured knit would read closer. |
| Looks-moving | 2 | low | new | The best continuity of the three: no openings, no flaps. Bunching at the groin at -90 (61 mm of excess per ruling) shows as wrinkles; the seat goes taut and shiny. |
| Fit | 2 | medium | new | Needs only clamp rails on the shell and cover edges, and no space inside the torso. |
| Buildability | 3 | low | new | A patterned 3D knit piece per hip (or one "underpants" piece), clamp strips and a hook or zip seam for service. Easy to replace, but patterning by trial. |
| Robustness | 3.5 | low | new | Folds trapped in a 5-6 mm crease between the thigh top and the belly rim at -90 (pinch and wear). On the inner side, fabric rulings wrap over the roll servo and pitch-to-roll connector, so they can be drawn into the gap between the thigh-cover top and the pitch-link parts. Abrasion at the cover edges, dust and hair pickup, needs washing; knit fatigue at ~50 % cyclic strain. |
| Mass/complexity | 1.5 | medium | new | ~2-3 g of fabric per hip plus rails. No moving rigid parts beyond the two side panels. |

**Requirement challenges.**
- A fixed briefs gusset cannot be joined to the inner thigh by fabric at -90 flexion or -45 abduction. Either the gusset leaves, with one continuous crotch piece from front rim to back rim that each thigh's inner edge is sewn to, or the inner thigh gets a rigid tuck-under like the side panel.
- The groin needs a folding allowance, a pleat or gather, or a rigid front (C1) instead of fabric.
- Roll and yaw limits are unchanged from the reference (thigh covers and torso side strip), as in C1 and C2.

**Overall.** Distance to viable ~2.5.
- Dominating gaps: the inner/crotch zone (strain 4-9× the working figure) and the seat at -90 (~1.3-1.6×). Second: pinch and bunching at the groin crease.
- Next most valuable step: re-run `fabric_core.py` with three changes: no rigid gusset (one continuous crotch zone), the long seat variant, and C1's sliding front panel in place of the groin fabric. Then judge whether every zone stays below ~40-50 % at -90/+40/-45 abduction.

---

**Ranking for the next round (lens C):** develop **C3**, borrowing C1's front panel:

- **C3** is the only one whose rigid parts reach the full pitch target with no contact, the fewest combined contact poses (302 vs. 342 reference), the lightest and simplest build, and the most continuous look.
- **C3's fabric gaps** (crotch, seat) are geometry and attachment questions that a cheap strain re-run can test.
- **C1's sliding front panel** solves the one zone where fabric does worst (groin compression to -80 % with a 6 mm crease), and hides completely at -90.
- **C2** has the easiest packaging but the weakest look in motion and a snag risk.

**Reproduce.** Each run takes about 2-3 minutes at ≤ 4.1 GB peak:

```
python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/moving-compliant/<script>.py
```

where `<script>` is `sliding_briefs`, `hinged_tassets`, `fabric_core`, `reference_body` or `half_angle_check`.
