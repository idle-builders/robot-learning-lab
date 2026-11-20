# Hip shell: covering a hip that flexes 80-90 deg with one smooth shell

Reference research for the ape body's hip shell (Idle T-czv2d8), surveyed 2026-11-15: how humanoid and legged robots
cover the hip, and which cover mechanisms from other fields keep a seam closed over a large angle. The design process,
the decisions and the cut drawings live with the work in `robot/body/hip-shell/` (`BRIEF.md`, `DECISIONS.md`,
`sections/`). Evidence tags: **[src]** read in a source, **[obs]** seen in a linked photo, **[inferred]** reasoning or
arithmetic.

## 1. Current state (2026-11-20)

Owner decisions that filter this survey (`robot/body/hip-shell/DECISIONS.md`):
- **D-04 / D-09 (2026-11-16):** one smooth shell of continuous curves; joint globes stay within the limb and pelvis
  silhouette, like the round hip discs and rounded thigh tops of G1, GR-2, Walker S2 and Digit. No outward bulbous
  shapes. Every look is compared against the Cove sketch (`robot/shell/sketch/reference/sketch.png`: brief-style pelvis,
  cylindrical thighs tucked under a diagonal leg-hole line).
- **D-05:** no loose plates, pods, bolted-on covers, and no dark-gap layering copied from current humanoids.
- **D-03, D-10:** the 3D concept rounds and the concurrent-hip globe concepts (ball, slim egg, saddle) are rejected as
  bulbous. Already tried and rejected before the survey: a fixed-pelvis leg window, nested caps or a hard ball about
  the hip centre, spring-closed flaps, a panel sliding up into the belly, stretch-fabric patches (150-250% strain).
- **D-11 / D-12 / D-13:** keep today's hip actuators. Static core down to z 271. One green shell per leg turns with hip
  yaw and holds the hip-pitch servo; its top slides inside the core's lower edge. The yellow pitch-to-roll connector is
  the only visible part and moves in slits in the green shell (about 125 deg of pitch arc plus its width). The roll
  servo is flipped into the thigh; the blue thigh shell holds roll and knee servos. **Each shell edge is a tight arc
  about its joint axis.** Order of work: roll flip and new Stage 4 / 5 connectors, then the leg shell, then the hip
  section.
- Flexion target -80 to -85 deg (owner, 2026-11-15).

How the survey maps onto that direction:
- The owner's edges-as-arcs rule is the surface-of-revolution rule below (F1): a seam stays closed under a rotation
  exactly when both sides lie on a surface of revolution about that rotation's axis. Each of our shells covers two
  axes, so each needs its moving edge on a surface invariant under both: about C for the yaw + pitch green shell, about
  the roll axis for the thigh's top.
- The robots that look like our target (QRIO, NAO, aibo, RoboHoN) hide the hip with a concentric boss or a band, never
  with a hard one-piece pelvis across 90 deg. The dark-band variants are excluded by D-05; the concentric-edge
  variants match D-12.
- Patterns that stand out at flexion (skirt plates, tassets, flaps) or bulge (external ball, mantlet) are excluded by
  D-09.

## 2. Our hip in section (right side, measured from the CAD, 2026-11-15)

Script: `robot/body/hip-shell/sections/hip_sections.py` (drawing recreated locally).
- Axes: yaw vertical through (x 59, y 2); pitch along x through (y 2, z 219); they meet at C = (59, 2, 219). Roll along
  y through (x 56, z 190), 29 mm below C. A globe centred on C slides up to 22 mm at roll -45 and 13 mm at roll +25.
- The pelvis is shallow at hip height: belly skin about 25-30 mm in front of C, back skin about 30 mm behind. The thigh
  is as deep as the body.
- At -90 flexion the thigh's front face becomes its top, at z about 245 (25 mm above C). Standing, the belly-thigh line
  is near z 200. Anything static on the front of the pelvis between those heights is in the leg's path.
- Static under hip pitch: the yaw servo and yaw-to-pitch connector above C, the pitch-servo horn at C, the pitch servo
  (x 14..57, 45 mm inboard of C) and the battery cradle above it. The only room a moving front cover can retreat into is
  in front of and above C, inside the belly skin.
- Hip roll sweeps the outer side of the hip below about z 200.
- Swept arc for 90 deg of rotation: 71 mm at r 45 mm, 94 mm at r 60 mm [inferred].

## 3. How robots cover the hip

| Robot | Size | Covering pattern | Published hip ranges | Deep flexion | Source |
|---|---|---|---|---|---|
| Sony QRIO (SDR-4X) | 58 cm, 6.5-7.3 kg | dark recessed belt over silver spherical thigh-top domes; box thigh shell below | not found | crouches with the domes still under the belt [obs] | Commons photos; US6917175 |
| Aldebaran NAO v6 | 57 cm | brief-shaped pelvis, exposed grey hip band (about 15-20 mm), thigh shell | pitch -88..+27.7, roll -21.7..+45.3 (L), yaw-pitch -65.6..+42.4 about a 45 deg axis, one motor for both legs [src] | sit and crouch with the mechanism visible [obs] | [joints](http://doc.aldebaran.com/2-8/family/nao_technical/joints_naov6.html) |
| Sharp RoBoHoN | 19.5 cm, 13 servos | pill-shaped side hip boss, black on black, no pelvis shell to cut | not found | thigh horizontal with no shell conflict [obs] | [robohon.com](https://robohon.com/) |
| Robosen Buzz Lightyear | 40.7 cm, 1.5 kg | briefs shell with leg-hole cutouts, thigh cups tucked under | not found | not documented | Robosen product page |
| Disney Kid Cosmo | 1.45 m, 25 kg | fixed "pants" shell on rubber washers, openings enlarged beyond the VFX model | pitch -130..+55; skewed 45/135 deg yaw-roll pair -40..+20 / -45..+40 [src] | femur sometimes hits the shell; shell compliance (5.77 N/mm) absorbs it [src] | [arXiv 2508.11884](https://arxiv.org/abs/2508.11884) |
| Disney Olaf | 88.7 cm, 14.9 kg | PU-foam skirt plus 4-way stretch costume; legs hidden; one leg inverted to avoid roll / knee collisions | not published | the skirt deflects for recovery steps; costume forces handled by domain randomisation [src] | [arXiv 2512.16705](https://arxiv.org/abs/2512.16705) |
| Disney BD-X | 0.66 m | exposed styled actuators with rounded caps | not published | - | [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1) |
| Sony aibo ERS-1000 | dog, 22 DoF | large haunch shell overlapping the body side | not found | in the sit pose the seam stays a curve; haunch edge roughly concentric with hip pitch [obs / inferred] | Commons photos |
| Boston Dynamics Spot | quadruped | hip actuators in dark corner pockets, thigh shell outboard of the body | HX ±45, HY ±91 with a 50 deg bias [src] | the leg swings beside the body | [BD docs](https://dev.bostondynamics.com/docs/concepts/about_spot) |
| Honda ASIMO 2011 | 130 cm | torso hem over a separate hip module, thigh cups with grey inserts (three nested layers) | not found | - | Honda technical review 2007 |
| AIST HRP-4C | 158 cm | silver briefs shell, dark thigh tops under it; also wore dresses | not found | - | AIST 2009 |
| 1X NEO | 1.65 m class | seamless whole-garment knit over a soft lattice | not published | squats and sits in chairs [src] | [1x.tech](https://www.1x.tech/neo) |
| Figure 03 | full size | textile over foam on limbs and torso, hard black pelvis block | not published | - | [figure.ai](https://www.figure.ai/news/introducing-figure-03) |
| Unitree G1 | 1.32 m | open gap: small pelvis block, hip motor cans visible | P ±154, R -30..+170, Y ±158 [src] | large range because nothing is covered [inferred] | [unitree.com/g1](https://www.unitree.com/g1) |
| Unitree R1 | ~1.2 m | shell panels over an exposed-joint body | Y ±157, P -168..+146, R -60..+100 [src] | - | R1 user manual |
| Booster T1 | 1.2 m | mostly exposed actuators | P ±118, R -21..+88, Y ±58 (retailers) | - | retailer pages |
| Fourier GR-3 | 1.65 m | upholstered soft shell over foam | L: P ±150, R -10..+90, Y -40..+90 [src] | - | Fourier support docs |
| iCub 2.5 | 104 cm | fabric jacket per segment | pitch software -30..85, roll 0..85, yaw ±70 [src] | crawls; roll lower limit set by a link collision [src] | iCub docs |

Packaging patterns [src]: Sony put the hip pitch and roll actuators in the thigh and offset the pitch actuator through
a pulley to keep the waist small and avoid bulges (US6917175). Kid Cosmo and Olaf use skewed or inverted hip axes to
avoid actuator collisions in a tight pelvis. NAO shares one 45 deg yaw-pitch motor between both legs. Disney sized the
Groot hip and knee actuators to fit inside the costume.

**What nobody in the sample does:** a hard, glossy, one-piece pelvis surface across the hip at 80-90 deg flexion. Every
example with that range shows a deliberate band or gap (NAO, QRIO, RoBoHoN), uses soft goods (NEO, Olaf), or accepts
occasional contact with a compliant shell (Kid Cosmo).

Recurring robot patterns:

| Pattern | Examples | What it costs |
|---|---|---|
| R1 briefs shell + tucked thigh cup | Buzz, Kid Cosmo, HRP-4C, ASIMO, Figure 03 | openings must be enlarged for flexion; at deep flexion the cup rises out and a rear gap opens. The leg-window variant was already rejected |
| R2 dark recessed joint band | NAO, QRIO, Optimus, Figure 03, Spot, Atlas 2025 | reads as deliberate with a clean constant edge and colour contrast, allows the full range; excluded by D-05 |
| R3 spherical thigh-top dome under a band | QRIO | works only if the dome is centred on a point that does not translate; on our hip that is C for yaw and pitch, so roll needs its own seam [inferred] |
| R4 concentric boss / side-mounted hip | RoBoHoN, aibo haunch, Spot, Atlas 2024 drum caps | constant gap for one axis; yaw and abduction break it unless the boss rides on the right link |
| R5 continuous soft suit | 1X NEO, Figure 03, GR-3, XPeng IRON, Olaf, iCub | the only true one-piece look; needs shape-defining cushioning and extra fabric at the crotch; changes the dynamics |
| R6 skirt or hem | Olaf, HRP-4C in dresses, ASIMO hem | hides the legs partly; the skirt must deform |
| R7 exposed styled actuators | BD-X, G1, Optimus gen 1, Atlas 2024 | mechanical look; the largest published ranges |

## 4. Cover mechanisms from other fields

| Mechanism | Example | Angle | Room it needs behind the surface | Seam look | Source |
|---|---|---|---|---|---|
| Nested spherical covers at two intersecting axes | Honda humanoid ankle | full pitch + roll | one shell plus a small gap at radius R; nothing may poke out inside R | one fine constant line, "no step or gap" in roll | [US7240747](https://patents.google.com/patent/US7240747B2/en) |
| Surface of revolution about the joint axis | LEGO minifigure hip; BJD thigh tops; ASIMO 2007 "spherical" joints | LEGO about 135 deg total | a thin layer at R over the swept arc R·θ | constant line | [Honda 2007](https://global.honda/en/technical-review/PDF/019-1/19-1_6e.pdf) |
| Body-side cover with relocated pivots | Sen-Ti-Nel figure hip | "large" | curved shafts inside the pelvis | small groin gap | [US12465867](https://patents.google.com/patent/US12465867B2/en) |
| Floating intermediate thigh piece | Iplehouse / Enchanted Doll BJDs | knee to chest | none beyond the silhouette | two moving cut lines | Enchanted Doll blog |
| Hinged skirt plates pushed by the thigh | Gunpla skirts | to near horizontal | none inside; plates stand proud | plate tips out | Gunpla101 |
| Overlapping lames (fauld, culet) | 15th-16th c. plate armour | waist bend, riding | stack thickness, 1-2 mm per lame | shingle lines read as one garment | armour references |
| Concentric lames on one pivot | lobster tail, couter, Almain gauntlets | 90 deg+ | N x (t + clearance) radially, one lame arc tangentially | concentric lines | myArmoury |
| Rotating truncated-sphere rings | NASA / Vykukal hard-suit hip, AX-5 | 3 x 23 deg = 138 deg cone | none: the shell is the structure | beaded bearing rings | [US4594734](https://patents.google.com/patent/US4594734A/en) |
| Rigid hip brief + canted bearings | NASA Mark III | about 84 deg flexion (model) | structure is the shell | bearing rings | Cullinane 2017 |
| Rolling convolute, bellows, gaiters | suit soft joints, CV boots | 40-90 deg+ | fold depth, several mm | ribbed | [US4091464](https://patents.google.com/patent/US4091464) |
| Hard plates + folding membrane | arthropod sclerites | large | membrane folds under the overlap | plate edges only | Springer 2020 |
| Telescoping bands / shingled scales | armadillo, pangolin | roll-up | overlap depth | banded or scaly | Teeple 2025; Meyers group |
| Child-mounted or rod-linked door | landing-gear doors | 90 deg swing | the bay holds the door path | flush in one pose | [US8167241](https://patents.google.com/patent/US8167241) |
| Split fairing with deflector blades | Embraer flap-track fairing | tens of deg | thin; blades stow inside | opened seam, cavity masked | [US9567063](https://patents.google.com/patent/US9567063B2/en) |
| Compliant morphing skin | FlexSys FlexFoil | -9..+40 deg | internal compliant ribs | seamless | New Atlas |
| Hidden hinge spine (barrel) | foldable phones | 180 deg | the barrel between the housings | flat one end, barrel the other | [US12461568](https://patentsgazette.uspto.gov/week44/OG/html/1540-1/US12461568-20251104.html) |
| Gear-synchronised dual-axis hinge | 360 deg laptops | 360 deg | small gear block | two parallel lines | [US11907027](https://patents.justia.com/patent/11907027) |
| Shaped foam + stocking | hip-disarticulation prosthesis covers | sitting | foam 10-30 mm | smooth standing, folds seated | OPedge |
| Vacuum-tensioned sliding skin | Disney | elbow / knee full range | thin | smooth | [US9452538](https://patents.google.com/patent/US9452538B2/en) |
| Casing with soft local inserts | Sony quadruped casing; ASIMO elbows | full leg range | casing clearance | hard casing, soft patches | [US12060124](https://patents.google.com/patent/US12060124B2/en) |

Details that carry numbers or a rule:
- **Honda nested covers [src]:** two part-spheres about the intersection of the pitch and roll axes. The outer cover
  rides the pitch shaft; the inner one rides the roll member and is rotationally symmetric about the roll axis; the
  parent ends in a concave sphere with a small uniform gap. Pitch turns both covers together; roll turns only the inner
  one, which is symmetric about its own axis, so nothing steps. [inferred] Our yaw and pitch meet at C, so a surface
  about C on the yaw / pitch link is invariant; roll is 29 mm off C, so the thigh top needs its own surface of
  revolution about the roll axis, nested under it. A single ball about C went bulbous because R is set by the farthest
  servo corner.
- **Surface-of-revolution rule [inferred]:** for flexion θ the concealed arc is R·θ (71 mm at R 45, 94 mm at R 60 for
  90 deg) plus about 5 mm overlap, in a layer of wall plus gap (about 3-4 mm) under the parent lip. The hardware only has
  to stay inside R - 4 mm over that arc. Pinch risk below about 1 mm gap.
- **Concentric lames [inferred arithmetic]:** N = 3 lames of 38 deg (30 deg + 8 deg overlap), 2 mm thick, 0.6 mm
  clearance: about 8 mm radial stack, and only one lame arc (about 33 mm at R 50) hidden behind the fixed lip instead of
  the full 90 deg a single sliding dome needs. Pivots must sit on the axis at both ends; the crotch side has little room
  for a boss.
- **Hard-suit rings [src]:** maximum deflection is the sum of the wedge angles, so 90 deg needs at least 3 x 30 deg.
  As a passive cover it is over-determined (one free spin) and needs a coupling.
- **Half-angle plate [inferred]:** a plate pivoting on the pitch axis at half the thigh angle (2:1 gear pair or a
  half-angle four-bar, about 10-15 mm envelope) needs about R·π/4 (35-47 mm) of hidden arc; a bell crank can add dwell
  (tucked in the first 45 deg, then still). It must ride on the yaw output so yaw does not disturb it.
- **Bellows [inferred]:** a 2-3 fold rolling convolute (fold pitch 6-8 mm) keeps material strain under about 20%,
  against the 150-250% measured with plain stretch fabric. It reads ribbed unless hidden under a hard lip.
- **Olaf [src]:** the foam skirt holds its shape while deflecting for larger leg motion and absorbs fall impacts;
  costume forces were not modelled, only randomised in training.

## 5. Cover families for 90 deg in a 45-60 mm envelope

Where gaps open under flexion about the pitch axis: the front of the thigh top rolls up and back into the full space
above and in front of C; the back rolls down and forward, opening a gap at the buttock; yaw (±40 deg) and abduction
(45 deg about the roll axis) open the side seams.

| Family | Principle | What 90 deg needs here [inferred] | Fit with the owner decisions |
|---|---|---|---|
| F1 concentric seams | every seam on a sphere or cylinder about the axis (or axis intersection) that moves across it | a continuous 4 mm layer at R over 71-94 mm of arc; use only the visible bands (front-side flexion band, side abduction band) as exact surfaces | the basis of D-12's arc edges |
| F2 overlapping lames | N plates each take θ/N | about 8 mm radial stack, about 33 mm hidden arc; 2-3 extra seam lines; coaxial pivots or arc guides | possible inside the green shell; adds parts and lines |
| F3 linked fractional plate | an intermediate plate driven at a fixed fraction | 35-47 mm hidden arc plus a 10-15 mm drive; replaces the rejected spring flap | possible; adds a mechanism |
| F4 recessed dark band | a real opening, recessed, with a dark soft floor | 6-10 mm band depth plus membrane fold space | excluded (D-05) |
| F5 flexible skin | the cover deforms: folds, sliding skin, shape-holding foam | stretch fails; folds or sliding required; foam 8-15 mm | stretch already rejected; foam changes dynamics |
| F6 hidden barrel | a barrel about the pitch axis, covered standing, shown as a rounded crease at flexion | thin shell of radius R; servos inside R - 4 mm in that sector; pitch only | compatible with arc edges at the groin |
| F7 outward-swinging plates | skirts, tassets, flaps | always stand off at flexion | excluded (D-09) |

## 6. Open

- Whether the green shell's arc edges about C clear the pitch-servo corners within a 4 mm layer at the radius the
  sleek silhouette allows (D-09), over -80..-85 deg flexion and the yaw range. Measured on the cut sheets in
  `robot/body/hip-shell/sections/`, not yet in CAD.
- The thigh-top seam about the roll axis, 29 mm below C, nested under the green shell.
- Hip flexion limits behind the published robot ranges were not found for QRIO, RoBoHoN, aibo and the toy robots.

Main sources not linked inline: Sony QRIO and aibo Commons photos (upload.wikimedia.org); Sony patent
[US6917175](https://patents.google.com/patent/US6917175); iCub joint limits (icub-tech-iit.github.io) and HeiCub
([arXiv 1607.08525](https://arxiv.org/pdf/1607.08525)); ranges overview humanoid.guide.
