# Design kit: shells and connectors

The look and build rules for APE0's printed parts. They come from what the owner approved and rejected (quotes and
decision IDs below). Read this before you design or redesign a shell or a connector. Start from a reference builder.
Do not ask the owner to explain the aesthetics again. If a part departs from the kit, say so next to its renders and
give the measured reason. The kit grows: when the owner approves or rejects a new pattern, add it here, edited in
place (owner, 2026-11-24).

Like everything else here, the numbers are reference values, not gates (`evaluation-principles.md`). A design is
shown with its measurements and renders, and the owner judges the look.

## 1. Shared principles

- **Few parameters, simple analytic functions.** Owner 2026-11-19: "simple smooth parametric courses, the less
  parameters the better". 2026-11-22: "Make use of gently and simple parametric functions to define the shape".
  Shape parameters are named code constants with a comment, set by a rule (joint centre, axis, neighbour) or by a
  small fit. They are never free-form edits.
- **Design about the joint axes.** Every surface that sits next to a moving neighbour is a surface of revolution
  about that joint's axis: an arc bar, a seam, a cut-in or a dish. Then the gap stays constant through the motion and
  nothing has to follow a swept volume.
- **Sleek ape silhouette.** Parts stay inside the limb and torso outline. Owner 2026-11-16 (D-09): "anything that
  creates an outward bulbous design you can directly disregard ... a sleek body".
- **Not sheet metal, not organic.** No bent-plate look. No visible topology-optimisation webs on the selected body.
  Topology optimisation may show where the load goes, but the part that ships is a plain U / H / I / Y (section 3).
- **Render before judging.** Show the owner camera (front + outside, 49 deg above), front, side and top, at standing
  and at the range ends, together with the measurements. Look trade-offs go to the owner as renders.

## 2. Shells

**References:** the arm shell, `robot/body/arm/arm_shell.py` (owner-approved 2026-11-24, M-eb6cxz, candidate
arm-shell-3), which is the template. Also the thigh shell (`robot/body/leg/thigh_shell.py`) and the shoulder shells
(`robot/body/arm/shoulder_shells.py`).

### The look

Owner 2026-11-23: "an aesthetic design of simple continuous and calm curves ... think of the entire arm as one
continuous design".

- **Calm and continuous.** No kinks, no steps, no local bumps. Curvature changes slowly along the limb.
- **Looks closed.** From the normal viewpoints the mechanism is not visible, apart from the few parts the design
  shows on purpose (the hip's yellow pitch-to-roll connector, D-12). Openings are small, simple, and only where a
  moving part has to pass.
- **One body, not a kit of covers.** No loose plates, pods, bolted-on covers or dark-gap layering copied from current
  humanoids (D-05). Neighbouring shells continue each other's outline: the thigh's lower section runs into the shin
  covers' top outline, and the forearm end is as wide as the hand's top.
- **Round, soft sections.** Superellipse sections lean toward round (exponent about 2.4). Hoods and caps are domes.
- **Do not follow the mechanism closely.** Owner 2026-11-22: the shell "follows the sweep of the motors to closely".
  The shell holds the parts with a margin, and the margin varies smoothly.

### How to build one (the arm-shell recipe)

1. **One master surface per limb, or per region,** made as a sweep of planar sections along a spine:
   - **Stations:** a few planar stations (the arm has 9). Each has a centre, a tilt, four half-widths (out, in,
     front, back) and one exponent n.
   - **Centres and tilts:** set by rule, from the joint centres and axes.
   - **Splines:** every station column is one natural cubic spline over the chord length of the centres (C2).
   - **Section:** a superellipse, |u/a|^n + |v/b|^n = 1.
   - **Cap:** g(q) = (1 - h^q)^(1/q) shrinks the section to a dome.
   - **Wall:** constant, offset inward along the surface normal (arm 0.9 mm, thigh 2.1 mm).
2. **Requirements as point sets with a radial need.** These are the only inputs the fit sees:

   | Kind | Need |
   |---|---|
   | static parts of the shell's own link | GAP + WALL |
   | parts turning inside it, swept over the declared range | RUN_CLEAR (1.5-2 mm, owner) + WALL |
   | screw heads in their wells | 0.3 mm |
   | parts that must stay outside (EXCLUDE sets, e.g. the torso beside the arm) | RUN_CLEAR |

   Look targets may be requirements too, such as the forearm end held as wide as the hand.
3. **Fit the half-widths and exponents.** Minimise section area + smoothness (integrated squared second derivative
   of each half-width) + a pull of n toward 2.4, with every requirement point inside with its need. Arm settings:
   `FIT` (n 2.2-3.4 off the hood). Centres, tilts and seam circles stay fixed by rule.
4. **Cut the master into parts at analytic seams,** after it is made:
   - **Rotating joint, axis along the limb (shoulder yaw):** a plane normal to the axis. Around it the master is a
     circle about the axis (two stations, n = 2). The lower part carries a thin lip up inside the upper one.
   - **Hinge (elbow):** sectors bounded by planes through the axis, plus a cylinder about it, so each part turns
     inside the other. The sector angles = range + margin.
   - **Static split (forearm front / back):** a plane with a 0.3 mm gap.
5. **Free edges** that a moving neighbour passes (the thigh top) are low, calm periodic curves: 3 Fourier harmonics,
   peaks and dips capped about the mean (4.5 mm), slope 0.35 or less, below the swept neighbour. Fabric covers what
   lies above them.
6. **Openings and cut-ins** are one simple closed curve in the joint's own coordinates. Example: the arm hood's rim
   is a superellipse in (angle about the roll axis, position along it) over the sector the blue U sweeps. Nothing
   else is open.
7. **Mounts:**
   - Screws go through wells in the shell into pads or posts on a connector. Never into a floating boss.
   - Button or flat heads sit flush in the wells.
   - Every clamped printed hole carries a pressed 304 sleeve (4.5 x 3.2 under M3; owner 2026-11-18).
   - Captive nuts sit in open slots.

### What to report with a shell candidate

- **Spine smoothness** (`spine_smoothness`): minimum bend radius (arm 32.8 mm), maximum curvature change, maximum
  half-width slope (arm 0.45) and its curvature, exponent range (arm 1.98-4.33).
- **Requirement margins** per point set, in mm, worst first.
- **Growth** against the previous skin (p05 / p50 / p95 and the maximum outward, in mm).
- **Visible open area** (mm²) from the owner camera, front and side: at standing and at the range ends.
- **Checks:** `run.py --checks candidate` on the shell parts and their neighbours, `clearance` over the library
  poses, and `fit` once on the candidate you show.

## 3. Connectors

**References:** the shoulder U connectors in `robot/body/arm/shoulder_clevis.py` ("Much better the two U connectors
look clean now", owner 2026-11-20, M-rya3ty; arc bar approved 2026-11-24, M-eb6cxz). Also the thigh H-cage
(`robot/body/leg/build_upper_hip_connectors.py`, D-15) and the forearm clevis
(`robot/body/ape-proportions/forearm_clevis.py`).

### Vocabulary: I, U, H, Y

| Shape | What it is | Example |
|---|---|---|
| **U (clevis)** | Two flat arms grip one actuator from both sides, horn arm and idler arm, joined by one bar | blue and green shoulder U |
| **H** | Two U's back to back on one web, each gripping its own actuator; may turn 90 deg in the middle | thigh H-cage (roll case + knee case; owner 2026-11-19: "an H connector with a 90 deg rotation in the middle") |
| **I** | A straight column or tube between two two-sided ends, going the direct way | forearm column (owner 2026-11-13: "the connector should go straight down, not along the side") |
| **Y** | One hub (a horn or a round boss) that forks into a U | hip pitch-to-roll: a plate on the pitch wheel, two arms down to the roll horn and idler |

Build a connector from these shapes. If a part fits none of them, rethink the layout before adding features.

### Rules

- **Grip every actuator from two sides.** Use the horn on one side and an idler on the other, with a symmetric
  screw pattern on each face (2 + 2). If the servo has no idler, add a rear idler (the shoulder roll got one) before
  you accept a one-sided mount.
  - The single-sided hip pitch-to-roll connector, on the pitch side, was the weakest leg link: strength ratio 1.70 at
    stall in PAHT-CF, 1.28 in PETG-CF.
  - Mirror both faces where the actuator allows it (owner 2026-11-20: "why not make it symmetric the wings of the H?").
- **Flat, plain members.** Use flat arms, a plain plate or bar, and flat outer cheeks. Owner 2026-11-22 on a sloped
  cheek: "the outside has a weird sloped part, what is the reason for that?". The cheek was then made flat.
  - Green U: "No ribs / gussets / fillers".
  - Add a gusset (45 deg) or thicken an arm only where FEM shows the need. Then name the case that needs it.
- **Hug the axis.** Where a bar passes the joint it turns about, make it an arc concentric with that axis (blue U:
  ARC_R 15.9 mm, ARC_T 2.9 mm; owner 2026-11-23: "hug tight"). It then sweeps a thin shell of revolution, and a
  closed cover can follow it.
- **Rounded everywhere:**
  - Plan corners: radius 1-3 mm (arm corners and plate 3.0, arc ends 1.0).
  - Root fillets: 1.2-1.5 mm, larger where the load enters.
  - 3D edges: `bevel_3d_edges` in `{arm,leg,neck-head}/common.py`, with r_outer about 0.4 (touch) and r_inner about
    0.75 (stress).
  - Bevels: 45 deg chamfers on flat cheeks.
  - Keep sleeve faces flat.
- **Hubs and bosses:**
  - Hub radius = bolt circle + sleeve radius + about 2.75 mm wall (shoulder: r 7 + 2.25 + 2.75 = 12).
  - Boss wall around a sleeve: 1.5-2 mm. Head pockets keep at least 2 mm of wall.
- **Fasteners:**
  - A pressed 304 sleeve in every clamped printed hole: 4.5 x 3.2 under M3, 3.6 x 2.2 under M2.
  - Every sleeve must be pressable, with a flat counter face and a path for the heated press (owner 2026-11-17, D-14).
  - Inserts only where no press path exists. Horizontal holes are teardrops.
- **Gaps that survive printing and hand assembly:**
  - Seated faces: arm on horn or idler face, 0.1 mm per side.
  - Separate bodies that move against each other: 1.5-2 mm.
  - Never a slit under 1 mm between members. Owner 2026-11-19: the 0.15 mm plate-to-arm gaps "will not hold for
    printed parts and hand assembly".
- **Print:**
  - Choose one print direction per part, so that bridges are under 40 mm, overhangs are within the material's limit,
    and the layers are not peeled by the main load.
  - Walls: at least 2 line widths.
  - No knife edges, and no rims thinner than one print line.
- **Mount pads for the shell belong on the connector,** with the sleeve and nut slot designed in. No floating
  bosses.

### Patterns to avoid (each one happened)

- **Thin bridges:** a 21 mm² neck bridge on arm r12, stage 1.
- **Floating mount bosses:** shell bosses with no bracket behind them.
- **Slits under 1 mm:** a 0.15 mm slit beside the H-cage bridge gave a 49 MPa peak.
- **Knife edges:** a 57 deg edge with 0.03-0.75 mm walls on hip stage 5.
- **Thin walls left by a hole:** a 0.55 mm wall where a hole was carried through the bar.
- **One-sided cantilevered servo mounts.**
- **Sleeves the press cannot reach:** 280 of 281 on the body before T-gckc6p.

### What to report with a connector candidate

- **Mass** per part.
- **FEM:** strength ratio and layer ratio for each load case and material (PAHT-CF and PETG-CF), and displacement.
- **Press check** for each sleeve.
- **Clearance and fit** over the library poses.
- **Print checks:** overhang, walls, bed, holes.
- **Which kit shape the part is,** and how it grips each actuator.

## 4. Code

The kit is also code, in `robot/body/design_kit/` (`from design_kit import shell, connector`, with `robot/body` on
`sys.path`). Like the checks, it knows no part, link or joint: positions, axes and faces come in as arguments.

- **`shell.Sweep`** is the master surface of section 2. It takes the stations, the tilt axis, the up direction, the
  U reference direction, an optional dome cap and the wall. Its methods:
  - `frames`, `grid` and `master` build the surface: the section frames, the skin grid and the outer / inner solids.
  - `local` gives the local coordinates of points against the surface.
  - `margin` and `report_margins` measure the requirement point sets.
  - `smoothness` gives the bend radius, curvature change, slopes and exponent range.
  - `fit` finds the half-widths and exponents. Stations can be fixed or tied, and the bounds come from the caller.
  - `stations_text` prints the fitted `STATIONS` literal, ready to paste.
- **The rest of `shell`:**
  - `growth` measures the new skin against the previous covers.
  - Seam cutters: `seam_sector` (hinge), `seam_cylinder` and `tube` (the lip under a rotating seam).
  - Mount pieces: `boss` and `sleeve_tube`.
  - The trimesh / manifold3d converters.
- **`connector`:**
  - The sleeve table `SLEEVES` and `hub_radius`.
  - OCC primitives: `box`, `cyl`, `boolean`, `largest`, `transformed`, `mirror_x`.
  - `sleeve` and `bore`.
  - 2D outlines: `rounded_rect`, `wedge`, `arc_band` (the arc bar about an axis), `hub_fan` (an arm: a hub plus a
    fan to the bar).
  - `root_fillet`, `fillet_2d`, `extrude_y` / `extrude_z` and `bevel_3d`.

`robot/body/arm/arm_shell.py` and `robot/body/arm/shoulder_clevis.py` are built on these modules. Read them as
worked examples. A new shell or connector builder imports the kit and does not copy these functions.

## 5. Using the kit

1. Copy the closest reference builder's structure (the section, the parameters as constants, `fit`, `margins`,
   `views`), on top of `design_kit`. Do not start from scratch.
2. Before you write CAD, write the part's kit pattern down in its builder docstring: which shell sections, seams and
   openings, or which connector shape, and which actuators it grips on which faces.
3. Run the checks (`docs/operations/working-rules.md`, CAD design loop). Render the owner views.
4. Show the candidate with its measurements, its renders, and the list of places where it departs from the kit,
   with the reason for each.
