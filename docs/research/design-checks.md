# Design verification before printing: which checks a hobby FDM robot should have

Research for the shared design-check tool, 2026-11-18. It lists the checks that
commercial CAD and print-prep tools, engineering release processes and hobby/research robot builds use, says which of
them we already run, and ranks the missing ones for this robot. Value order (owner): **P1** wasted prints, **P2** parts
that do not assemble, **P3** owner time spent on obvious errors, **P4** failures in use. Every check here reports
measurements in native units and counts of bad events; none accepts or rejects a part (AGENTS.md). Evidence tags:
**[src]** read in a source (numbered list at the end), **[repo]** seen in this repository, **[inferred]** reasoning.
Laptop cost: **s** = seconds per part, **min** = minutes for a region, **heavy** = one heavy job under memguard.

## Current state (2026-11-20)

- One shared tool, `robot/body/checks/run.py`, with 31 registered checks (`--list-checks`; command, scopes and the
  per-check table in `docs/operations/working-rules.md`). It runs on the selected manifest named in `robot/design.json`.
  The per-section check scripts it replaced are gone; `robot/body/leg/fast_check.py` and `roll_flip_checks.py` call the
  same checks.
- Built from the catalogue below: geometry hygiene, plate fit, print setup and coverage, overhang and bridges, walls
  and gaps, small features, bed contact and warp, elephant foot, holes and pilots, slicer pass, sharp edges, standing
  gaps and audit, clearance and fit over poses, range and collision sweeps, fit classes, press, screws, driver access,
  fastener motion, connectivity, assembly order, BOM, mass, heat screen, FEM and FEM setup, render, shell offset.
- Not built: auto-orientation reference score, servo horn and spline alignment, tolerance stack (by hand for the
  tightest chains), bearing fits, change diff between runs, nut-slot insertion, cable path across joints, threadlocker
  notes, serviceability (reverse walk of the assembly order).
- Data that still limits the numbers: `printers.json` `hole_offset_mm` is null until a test coupon is printed; the
  fit-class bands are reference values, not measured on this printer; print directions are seeded (unconfirmed) for 34
  of 61 printed parts; 20 of the 25 body shells have no native STEP; the assembly order is a seeded draft for the right
  leg only.

## Printer and process facts

**Bambu Lab X2D exists.** Bambu announced it on 2026-04-28 as a dual-nozzle printer for support removal and
multi-material printing [S1][S2][S3]. I could not read Bambu's own spec page (bambulab.com/en/x2d/specs returns 403 to
automated fetches). The numbers below come from Bambu's search snippets, trade press and resellers, and they agree with
each other.

| Item | Value | Source |
|---|---|---|
| Build volume, main (left, direct-drive) nozzle | 256 x 256 x 260 mm | [S1][S2][S3] |
| Build volume, auxiliary (right, Bowden) nozzle and both nozzles together | 235.5 x 256 x 256 mm | [S2][S3] |
| Max nozzle temperature | 300 °C | [S1][S2] |
| Active chamber heating | up to 65 °C | [S1][S2] |
| Max bed temperature | 120 °C (resellers and reviews only) | [S4] |
| Hotends | hardened steel; 0.4 mm supplied, 0.2 / 0.4 / 0.6 / 0.8 mm available, standard and high flow | [S4][S5] |
| Other | closed-loop servo extruder, automatic flow calibration, jam detection; H12 HEPA + carbon filter | [S2] |

Material limits that checks need (Bambu technical data sheets V2.0, read in full) [S6][S7]:

| | PAHT-CF (PA12-based) | PETG-CF |
|---|---|---|
| Nozzle / bed / chamber | 260-290 / 80-100 / 45-60 °C | 240-270 / 65-75 / 35-50 °C |
| Nozzle size | 0.4, **0.6 recommended**, 0.8 mm | 0.4 / 0.6 / 0.8 mm, hardened |
| Drying | 80 °C, 8-12 h; print and store below 20 % RH | 65 °C, 8 h; below 20 % RH |
| Max overhang / max bridge (Bambu's figures) | about 70° / about 40 mm | 70° / about 30 mm |
| Support material | Bambu Support for PA/PET (the X2D's second nozzle can print it) | supports on |
| Density | **1.06 g/cm3** | 1.25 g/cm3 |
| Tensile strength X-Y / Z (dry) | 88 / 64 MPa | 59 / 38 MPa |
| Young's modulus X-Y / Z | 3860 / 2180 MPa | 2460 / 1340 MPa |
| Bending strength X-Y / Z | 140 / 68 MPa | 83 / 62 MPa |
| HDT at 0.45 / 1.8 MPa | 194 / 170 °C | 74 / 68 °C |
| Glass transition | 70 °C | 68 °C |

What this means for the checks [inferred]:

- Plate limit: every printed part must fit 256 x 256 x 260 mm in its print orientation, or 235.5 mm in X when the second
  nozzle prints support. Our largest printed parts (centre front / back shells, about 180 x 55 x 206 mm by their
  bounding boxes) fit; the check is still worth having because orientation changes the fit.
- Nozzle: Bambu recommends 0.6 mm for PAHT-CF. Our build notes say 0.4 mm [repo]. A 0.6 mm nozzle makes walls and
  features under about 1.2 mm print poorly, so minimum-wall checks must take the nozzle as a parameter.
- Layer direction matters a lot: across layers PAHT-CF keeps about 73 % of its tensile strength and 49 % of its bending
  strength; PETG-CF keeps 64 % / 75 %. Print orientation is a first-class input for FEM and for any strength screen.
- Heat: PETG-CF softens at about 68-74 °C, and the Feetech STS3215 turns its torque off above 70 °C internal temperature
  (STS3250 rated -20..60 °C ambient) [S8]. A PETG-CF part clamped to a hot servo case is at risk; PAHT-CF is not.
- **Density mismatch:** the old `fast_check.py` and the manifest use 1.22 g/cm3 for PAHT-CF; Bambu's sheet says
  1.06 g/cm3. The shared `mass` check uses the data-sheet value and still reports the legacy one; PAHT-CF masses from
  1.22 read about 15 % high (open question below).
- Printer data in the manifest is prose today: in the selected `motor-refit` manifest 36 of 63 printed parts carry a `print_orientation` string and 12 a
  `manufacturing` string, and materials are free text such as "PAHT-CF ... or PETG-CF" [repo]. Most printability checks
  need a machine-readable print direction and material key per part.

## Starting point (2026-11-18) [repo]

Before the shared tool, each section had its own scripts: per-section CAD-rule checks (valid solids, slivers,
micro-edges, wedges, tangent continuity, L/R mirror, sleeve counts), per-section fit checks over pose lists with
screw and captive-nut insertion rays (arm), `robot/body/leg/fast_check.py` (poses, gaps, mass, press, render),
per-study FEM scripts, and motor-refit range / collision sweeps and static audit. Their logic now lives in
`robot/body/checks/`. Still separate: `robot/body/fastening/mount_check.py` (per screw, called by `screws`; servo
tapped holes from `fastening/interfaces.py`) and `robot/body/torso/thermal_evaluate.py` (coupled battery thermal model).
None of the old scripts looked at printability: plate fit, overhangs, wall thickness against the nozzle, bed contact,
warping, holes that print undersize, or slicer output.

## Catalogue

Columns: what it catches; inputs (STEP = native solid, man = assembly manifest, poses, hw = fastener / hardware
data, ori = print orientation, mat = material, prn = printer and nozzle); cost; what we have; value (P1-P4); sources.

### A. Geometry hygiene (does the file describe a printable solid?)

| Check | Catches | Inputs | Cost | Have | Value | Src |
|---|---|---|---|---|---|---|
| B-rep validity, solid count, closed shells | broken booleans, extra bodies, open shells the slicer "repairs" wrongly | STEP | s | `geometry` | P1 | S9 S10 |
| Sliver faces, micro-edges, acute wedges | knife edges and zigzags that print as stringy or missing walls | STEP | s | `geometry` (per-part thresholds) | P1/P3 | S9 |
| Mesh watertight, orientation, shells, holes | export mesh differs from the solid; Netfabb / Magics standard analysis | mesh | s | partial (`geometry` STEP vs mesh volume, `solid_volume_off_over_5pct`) | P1 | S11 S12 |
| STEP vs manifest hash and units | stale mesh or STEP, mm vs m mix-ups | man, STEP | s | caches keyed by sha256; no report | P3 | [repo] |
| L/R mirror identity | left part differs from right by accident | STEP pairs | s | `geometry` | P3 | [repo] |
| Tangent continuity, unrounded corners | stress raisers, owner-visible facets | STEP | s | `geometry`, `edges` | P4/P3 | [repo] |

### B. Printability per part (P1: wasted prints)

| Check | Catches | Inputs | Cost | Have | Value | Src |
|---|---|---|---|---|---|---|
| Fits the plate in its orientation | part larger than 256 x 256 x 260 (235.5 in X with the second nozzle) | STEP, ori, prn | s | `geometry` (print frame, both nozzles, best yaw) | P1 | S1 S2 |
| Print orientation and material present, machine-readable | checks below cannot run; part printed in a guessed orientation | man | s | `print_setup` (34 of 61 printed parts seeded, 2026-11-19) | P1/P3 | [repo] |
| Overhang area by angle | downward faces steeper than the material limit (Bambu: about 70° for both CF materials, generic guides 45°), support scars on mating faces | STEP/mesh, ori, mat | s | `overhang` | P1 | S6 S7 S13 S14 |
| Unsupported bridges | flat ceilings longer than about 30-40 mm | mesh, ori, mat | s | `overhang` | P1 | S6 S7 |
| Minimum wall thickness | walls thinner than about 2 line widths (0.8 mm at 0.4 mm, 1.2 mm at 0.6 mm) that drop out or print as one weak line | mesh (ray or medial-axis thickness), prn | s-min | `walls` | P1 | S14 S15 S16 S17 |
| Minimum gaps and slots | gaps narrower than a line width that fuse shut | mesh, prn | s-min | `walls` | P1/P2 | S14 S16 |
| Minimum feature / pin size | pins and ribs under about 1-3 mm that snap or vanish | mesh, ori, prn | s | `small_features` (free-standing islands; attached thin fins are `walls`) | P1 | S16 S18 |
| Z steps smaller than the layer height | features that disappear between layers ("empty layers") | STEP, ori, layer height | s | `small_features` | P1 | S19 |
| Bed contact and first-layer area | small footprint that tips or detaches; floating regions | mesh, ori | s | `bed` | P1 | S19 S20 |
| Warp risk of the footprint | long flat bases and sharp plan-view corners in PA-CF; Bambu calls for an enclosure, glue and drying | mesh, ori, mat | s | `bed` | P1 | S6 S21 |
| Elephant foot on mating faces | bottom-layer flare on faces that must seat or slide; a 0.2-0.5 mm chamfer avoids it | STEP, ori, man (mating faces) | s | `elephant_foot` (covered 2026-11-19) | P2 | S22 S23 |
| Horizontal holes and bores | round holes across the layers sag at the top; teardrop or reaming needed | STEP, ori | s | `holes` | P2 | S24 |
| Hole size compensation | holes print about 0.1-0.3 mm undersize; screw clearance and sleeve press holes need a printer-specific offset | STEP, hw, prn | s | `holes` (offset key in printers.json, coupon not printed yet) | P2 | S24 S25 |
| Trapped support / unreachable cavities | support printed where it cannot be removed | mesh, ori | s-min | `small_features` (voids, support standing on the part) | P1 | S13 |
| Slicer pass (headless Bambu Studio / OrcaSlicer) | the slicer's own warnings (floating regions, empty layers, out of volume), support volume, print time, filament mass | mesh, ori, printer / process / filament profiles | min | `slicer` (Bambu Studio 2.8.2) | P1 | S19 S26 S27 S28 |
| Auto-orientation score as a reference | how far our chosen orientation is from the lowest overhang / support / contact score | mesh | s | none (the slicer runs with `--orient 0`) | P1/P3 | S29 S30 |
| Layer direction vs main load | a joint loaded across layers (Z strength about 50-75 % of X-Y) | STEP, ori, FEM loads | s (screen), heavy (FEM) | `fem` along `process.up` (layer tension, layer strength ratio) | P4 | S6 S7 S21 |

### C. Fit and assembly (P2: parts that don't assemble)

| Check | Catches | Inputs | Cost | Have | Value | Src |
|---|---|---|---|---|---|---|
| Static interference at standing | parts that overlap in the assembled pose | man, meshes | s-min | `static_audit`, `gaps` | P2 | S31 |
| Standing gaps to neighbours | zero-gap faces that need sanding, rattle gaps | man | s-min | `gaps` | P2 | [repo] |
| Screw axis vs hole centre | screw off its hole, hole missing in one of the stacked parts | man, hw | s-min | `screws` (mount_check) | P2 | [repo] |
| Thread engagement and screw length | PA2 screw too short to bite (rule of thumb 2-3 x diameter in plastic), too long so it bottoms or pokes into a servo, stack wrong | man, hw | s-min | `screws` (mount_check; servo tapped-hole data from `fastening/interfaces.py`) | P2 | S32 S33 S34 |
| Pilot hole size for self-tapping | hole too large (strips) or too small (cracks the boss); about 80-90 % of the major diameter | STEP, hw | s | `holes` | P2/P4 | S32 S33 |
| Boss wall around sleeves and screws | wall less than about 1.5-2 mm or under about 2 x the bore, cracks on pressing | STEP, hw | s | `holes` | P2/P4 | S35 S36 |
| Sleeve press-tool path and anvil face | sleeve that cannot be pressed in (owner rule M-evmrv6) | STEP, sleeves | s (screen), min (exact) | `press` (spacer-block counter side, `--exact-press`) | P2 | [repo] |
| Driver / tool access | screw head that a hex key or driver cannot reach in the assembled state | man, hw, tool sizes | s-min | `driver_access` | P2 | S37 |
| Nut and captive-nut insertion | nut slot blocked or too tight | man, hw | s | none (the old arm insertion rays are not ported) | P2 | [repo] |
| Assembly sequence | a part that cannot go in after the parts already fitted (no straight insertion path), screws closed in by later parts | man, assembly order | min | `assembly` | P2 | S38 S39 |
| Servo horn and spline alignment | horn holes at the wrong angle for the 25-tooth spline, zero pose not reachable; ToddlerBot-style zero-point jigs | man, motor interface data | s | none | P2 | S40 S41 |
| Clearance per fit class | sliding fits under about 0.2-0.3 mm, press fits outside a tested range | STEP pairs, man | s | `fit_class` (covered 2026-11-19) | P2 | S24 S25 |
| Tolerance stack along a chain | worst-case sum across 3-5 printed interfaces (servo case + connector + sleeve) exceeding the slot | man, per-feature tolerance | s | none | P2 | S42 |
| BOM vs manifest | screws or sleeves in the CAD missing from the BOM, wrong lengths, mixed sizes | man (`bom`, parts) | s | `bom` | P2/P3 | S43 S44 |
| Fastener kinds and counts | many screw lengths and heads; DFA says fewer kinds and fewer separate fasteners | man, hw | s | `bom` | P3 | S38 S39 |
| Bearing fits | bearing bore and shaft against the bearing's spec | man, hw | s | none | P2 | [inferred] |

### D. Obvious errors the owner would find in renders (P3)

| Check | Catches | Inputs | Cost | Have | Value | Src |
|---|---|---|---|---|---|---|
| Six-side render of the scope | anything visibly wrong; agent shows the candidate with its findings | man | s-min | `render` | P3 | [repo] |
| Unattached parts | a part on a link with no screw, sleeve or contact to anything | man, hw | s | `connectivity` | P3/P2 | [inferred] |
| Duplicate or coincident parts | the same part placed twice, a hidden copy inside another | man | s | partial (`static_audit` overlap) | P3 | [inferred] |
| Mass and COM vs manifest | wrong density, part not updated in the manifest | STEP, mat | s | `mass` | P3 | [repo] |
| Change diff | which parts, holes and sleeves changed since the last commit, with their new numbers | git, man | s | none | P3 | [inferred] |
| Coverage report | printed parts without orientation, material, FEM setup or check results | man | s | partial (`print_setup`, `fem_setup` coverage counts) | P3 | parent task |

### E. Failures in use (P4)

| Check | Catches | Inputs | Cost | Have | Value | Src |
|---|---|---|---|---|---|---|
| Static FEM per part | stress and deflection under the part's load cases, dry / wet / PETG-CF | STEP, FEM setup, mat, ori | heavy | `fem` | P4 | S45 |
| FEM setup still matches the part | supports and loads on screw holes that moved; parts with no setup | STEP, FEM setup | s | `fem_setup` | P4/P3 | parent task |
| FEM mesh quality and convergence | slivers in the tet mesh, stress that changes with mesh size | FEM mesh | min | convergence via `--fem-mesh-h` (`checks/fem_convergence.py`); no mesh-quality statistics | P4 | S45 S46 |
| Collision over motion | contact over the pose library and joint ranges | man, poses | min | `clearance`, `fit`, `collision_sweep`, `fastener_motion` | P4 | [repo] |
| Joint range before self-contact | range loss after a change | man | min | `range_sweep` | P4 | [repo] |
| Heat at servo and battery mounts | PETG-CF (HDT 68-74 °C) touching servo cases that reach about 70 °C; battery heat | man, mat, servo data | s (contact screen), heavy (thermal) | `heat` (screen); torso `thermal_evaluate.py` (thermal) | P4 | S7 S8 |
| Cable path across joints | wire stretched or pinched over the joint range, bend radius, no strain relief or service loop | man, poses, harness route | min | none (no harness model) | P4/P2 | S47 S48 |
| Creep and preload | bolt preload on PA-CF without a sleeve; self-tapping joints opened repeatedly | man, hw | s | `screws` (sleeve under the head) | P4 | S33 |
| Threadlocker compatibility | anaerobic threadlocker on PA / PETG causes crazing; use plastic-safe products or none | BOM notes | s | none | P4 | S49 |
| Serviceability | parts to remove before a servo can be swapped | man, assembly order | min | none | P4 | S38 |
| Process risk review (DFMEA-style) | a ranked list of how each joint can fail, with severity first | judgement over the catalogue | owner/agent time | none | P4 | S50 |

## Ranked build list (2026-11-18) and its status

Ranked by the owner's value order, then by cost. Built on 2026-11-19 unless noted: 1 print setup in the manifest
(`print_setup`); 2 plate fit in print orientation (`geometry`); 3 overhang and bridge map (`overhang`); 4 wall and gap
thickness against the nozzle (`walls`); 5 headless slicer pass (`slicer`); 6 bed contact and warp (`bed`); 7 servo
tapped-hole data for `mount_check` (`screws`); 8 holes across layers and hole compensation (`holes`; the printer
offset is still null); 9 boss walls and pilots (`holes`); 10 BOM vs manifest and fastener kinds (`bom`); 11 driver
access for every screw (`driver_access`); 12 elephant foot on mating faces (`elephant_foot`); 13 FEM setup check and
coverage (`fem_setup`); 14 heat contact screen (`heat`). Added after the list: assembly sequence (`assembly`, seeded
order), connectivity, sharp edges, fastener motion, fit classes.

Still later: cable path across joints (needs a harness route) and serviceability. Tolerance stack-up is worth doing by
hand for the two or three tightest chains rather than as a check.

## Where each check sits in the framework

The framework is `robot/body/checks/` (module per check, registered by name; scope, poses, per-part settings and the
common result shape are shared). Module kinds:
- **G** per-part geometry (one scoped part, its STEP / tessellation in the print frame): `geometry`, `mass`, `overhang`,
  `walls`, `small_features`, `bed`, `elephant_foot` (with mating faces from neighbour contact), `holes`, `edges`.
- **N** neighbours over poses (scoped part against parts that come close, standing or over a pose set): `gaps`,
  `static_audit`, `clearance`, `fit`, `range_sweep`, `collision_sweep`, `fit_class`, `press` (G + N), `heat`,
  `fastener_motion`, `shell_offset`.
- **F** FEM setup, meshing and solve: `fem`, `fem_setup`.
- **A** assembly-level (manifest-wide, or hardware over several parts): `print_setup` (coverage), `screws` (calls
  `mount_check.run_screws`), `driver_access`, `assembly`, `bom`, `connectivity`, `render`.
- **X** external tool (a subprocess under the same memguard limit, output parsed into the common shape): `slicer`.

Inputs: STEP, manifest, poses (`data/pose_sets.json`), per-part settings (`data/part_settings.json`: `process`,
`expected_hardware`, `geometry_thresholds`, `fit_classes`, `fem`, ...), `materials.json`, `printers.json`,
`hardware.json`, `tools.json`, `fit_classes.json`, `heat_sources.json`, `assembly_order.json`, and the standard-part
definitions in `robot/actuators/interfaces/standard-parts.json`. The design review of 2026-11-18 found that no
catalogue check needed a structural change: new per-part keys are free-form entries every check reads through
`ctx.settings(name)`, new global data is a small loader in `Ctx`, and the slicer is the only external binary. The
checks not built (see Current state) need new data rather than new structure: a harness route (cable path), per-feature
tolerances (stack-up), bearing specs (fits), the previous report (change diff), an assembly order walked in reverse
(serviceability).

## Print setup as data (2026-11-19)

`checks/process.py` resolves each printed part's print setup: part setting (`process.up`, `material`, optional `nozzle_mm`,
`layer_height_mm`, `line_width_mm`, `plate_yaw_deg`), else the material's recommended nozzle (`materials.json`), else the
printer's values for that nozzle (`printers.json` `by_nozzle`). `run.py --process KEY=VALUE` replaces any of them for one
run, so a nozzle or material can be cross-evaluated without editing the settings; `mass` follows the material override
(stage4 carrier 10.38 g in PAHT-CF, 12.24 g in PETG-CF from the same 9794 mm3). `materials.json` entries carry the data
sheet's print limits (`print`: temperatures, max overhang, max bridge, HDT, Tg) for the printability checks.

Seeding (34 of 61 printed parts, flagged `seeded`): the axis comes from the manifest's free-text orientation (or the FEM
layer-normal axes for the roll-flip parts), the sign from the larger flat bed contact, then the smaller downward area
steeper than 45 deg; L parts take the mirror (x -> -x) of their R part. The free text follows no single sign convention
(palm_upper's "-Z" puts its stated seam face on the bed, palm_lower's "+Z" puts its stated wrist face off it), so the text's
sign is kept only where the estimate is ambiguous (14 parts, flagged). The estimate overrode the text for stage4 (written
+Y: 682 mm2 on the bed; -Y: 2612 mm2, the whole plate) and stage1 (+Z 99 mm2 vs -Z 141 mm2). Renders of every seeded part
in its print frame confirmed a flat face on the bed for most parts; stage1, stage3, the head sensor carrier, hip
yaw-to-pitch and the roll-flip thigh have no good flat face along the written axis (large downward areas either way),
which is for the overhang check to measure.

Coverage on the selected body: 4 parts with role `body_shell` are bought M2 / M3 nuts (`process.printed: false`); 25
shells and both palm-lower parts have no print direction; 20 of the 25 shells have no native STEP, so no printability
check can run on them; both palm-lower STEPs hold 326 faces and no solid (`geometry` now counts `steps_without_solid`).

## Overhang and bridges (2026-11-19)

`overhang` measures in the print frame. The data sheets state "max overhang angle ~70°, max bridging ~40 mm" (PAHT-CF;
PETG-CF 70° / 30 mm) without an angle convention; the check measures from the vertical (0 wall, 90 ceiling) and also
reports the area beyond 45° so either reading is visible. A flat ceiling's bridge span is twice the distance to its
nearest supported edge (an edge whose neighbouring face goes down), tested on synthetic shapes: a 20 mm bridge between
pillars reads 20.02 mm, a 6 mm ledge 11.63 mm (12 expected), a 40 x 20 plate on a centre post 38.87 mm (39.4 expected).
A ceiling with no supported edge floats and is flagged. First readings: stage4 carrier 0 mm2 beyond the limit (plate on the
bed); head sensor carrier 3648 mm2 beyond the limit, its plate segments bridge about 24 mm between ribs and the ribs hang
14.7 mm above the bed with nothing below (floating, 97 mm long); stage3 carrier 5596 mm2 beyond the limit (a 4010 mm2 frame
7.4 mm above the bed on four feet); roll-flip thigh 1194 mm2, its base bar floats 2 mm above the bed (41 mm span), 35 mm2 of
beyond-limit surface in the sleeve bores and 19 mm2 on the roll servo case seat.

## Walls and gaps (2026-11-19)

`walls` casts a ray from each surface sample (about 4 per mm2) along the inner normal (wall) and the outer normal (gap,
up to 3 mm), keeping hits whose face looks back within 60 deg. A synthetic part with a 0.5 mm fin and a 0.4 mm slot reads
0.5 mm over 402 mm2 (400 expected) and 0.4 mm over 120 mm2 (120 expected). First readings at the 0.6 mm nozzle (0.62 mm line):
R_left_finger has a 0.007-0.02 mm slit about 22 mm long inside its single solid between the C-web and the body (a CAD
defect: the slicer fuses it or leaves a weak seam) and 0.71 mm gear-tooth walls; the stage4 carrier has two 0.81 mm end
walls 6.5 mm tall (under two lines: one perimeter per side, no infill). With `--process nozzle_mm=0.4` the area under two
lines drops from 105 to 15 mm2 (finger) and 260 to 67 mm2 (stage4). R_shin_front_hard_cover's STEP tessellates to nothing
(another STEP defect) and the shell has no material, so no line width.

## Bed contact and warp (2026-11-19)

`bed` measures the first layer in the print frame. Synthetic tests: a 40 x 20 plate reads 800 mm2, aspect 2, four 90 deg
corners, centre of mass 10 mm inside; a 30 deg triangle finds its 30 / 60 / 90 deg corners; a T on an off-centre 4 x 4
post reads its centre of mass 10 mm outside the contact; a 2 mm strip counts as a thin strip (73.6 of about 83 mm2, the
0.2 mm raster loses about one row per edge). On the body: contact areas agree with `overhang` (foot 7325, stage4 2609, thigh
280 mm2). Three seeded directions stand the part on a patch its centre of mass is outside of: the roll-flip thigh (10.75 mm
outside its 280 mm2 flange tab, checked independently from the contact vertex hull), the shoulder roll-to-yaw connector
(21 mm2 ring, 3.1 mm outside) and the fingers (41 mm2 gear ring, 6 mm outside); stage1 stands 77 mm tall (centre of mass)
on two 70 mm2 feet. Review follow-ups in the same commit: the best plate yaw is searched over 0-180 deg (the both-nozzle
volume is not square) and bridge-span edges are sampled at half the raster step (a 20 mm tunnel read 26.9 mm at 200 mm
length with fixed sampling, now 20.00 mm).

## Small printability features (2026-11-19)

`small_features`: synthetic part with a 0.8 mm and a 0.5 mm pin, a 0.15 mm pad, a closed 2 mm cube void and a ledge over the
base reads 0.8 / 0.5 mm pins (raster 0.05 mm), a 0.15 mm step at z 3.0 / 3.15, an 8.0 mm3 void at its centre and 124 mm2 of
support standing on the part (ledge 120 mm2 plus its ends). Free-standing islands only: a thin fin attached to a wider
section is measured by `walls`. On the body: R_left_finger has a 0.05 mm Z step at the same place as its internal slit
(C-web and body flat faces 0.05 mm apart), the roll-flip thigh 367 mm2 of support standing on the part. Review fix: islands thinner
than two raster steps (slivers) read the strip formula 2 x area / perimeter at a point inside them (0.02 / 0.044 mm strips
read 0.0199 / 0.0435 mm; a disc that small reads half its diameter), and one pin or rib over several layers is one feature.
The stage1 chassis has 30 sliver features under one line, the thinnest 0.011 mm over three layers at z 26.9-27.5 mm in its
print frame, consistent with 6374 mm2 of wall under one line in `walls`: CAD slivers to clean up.

## Holes, sleeves and pilots (2026-11-19)

`holes` reads concave B-rep cylinders (faces on one axis line merge only where their axial ranges touch: the shin's two
cheeks have 4.5 mm bores on the same line) and matches sleeves (the coaxial sleeve overlapping the hole most) and screws
(the coaxial screw whose shank passes through it). Thread sizes and the 80-90 % pilot band are in `data/hardware.json`; sleeve
sizes come from each part's sleeve geometry, because the body uses four (4.5 / 3.2, 3.9 / 2.6, 3.6 / 2.2, 3.2 / 2.1 mm).
`printers.json` `hole_offset_mm` is null until a test coupon is printed. First readings: the shin's four 4.5 mm sleeve bores
around its 8 mm centre hole have 0.75 mm walls (7 mm centre distance - 4 - 2.25, checked by hand); all shin and thigh
sleeves sit in line-to-line bores (0.000-0.001 mm interference in CAD), the stage4 carrier's 3.9 mm sleeves in 3.98 mm
bores (0.08 mm clearance in CAD: they press only if the printed holes come out that much small); PA2 pilots are 1.7 mm (85 %);
with the seeded up +x, 12 of the roll-flip thigh's hardware holes run across the layers. Review fixes: a sleeve bore's boss wall is measured over the sleeve's length only (a chassis bore that opens into a slot past its sleeve read 0.021 mm, now 1.33 mm where the sleeve sits), and bores wrapping 180-300 deg with hardware in them are kept as partial bores that break out of the part (6 on the chassis). Screws whose CAD position does not
reach the hole (the shin's PA2x6 proposals end 0.6 mm short of their sleeve bores) are left to `mount_check`.

## BOM against the CAD (2026-11-19, rebuilt 2026-11-20)

Until 2026-11-19 the manifest carried a hand-written subsystem rollup (9 lines: printed parts and sleeves per subsystem,
battery fins) and `bom` mapped it to links through `data/bom_groups.json`: printed parts matched except the arm (CAD 12 with
the palms, BOM 8), sleeves CAD 262 vs BOM 316, and no screw, nut or washer had a line. On 2026-11-20 (owner, Idle T-4jjry4)
the rollup and `bom_groups.json` were retired and the BOM is generated: `robot/body/bom/build.py` joins the selected
manifest with the hand-maintained `lines.json`; every part lands on a line. On the selected body (691 parts, manifest
4554f506): 180 lines (78 make, 101 buy, 1 included with the servos), 0 lines whose quantity differs from the CAD, 101 buy
lines still without supplier or part number; 40 hardware kinds (27 screw kinds by thread x length x head standard, 3 nut,
6 washer, 4 spacer), 258 sleeves in 32 sizes (the head carrier's sleeve STEP not counted). `bom` runs in about 5 s.

`robot/body/bom/hardware.py` measures each of the 470 screws, nuts, washers and spacers on its old mesh (axis, head,
outer / inner diameter, thickness; nut and spacer axis by a through-hole ray), snaps it to the standard sizes and writes a
STEP / STL at the same placement. Most fit the old mesh to a volume ratio of about 1.0 and bounds within 0.004 mm; 88
differ more (old M2 washers 0.5 mm thick vs ISO 7089 0.3 mm, low-head M2x4 plate screws modelled as button heads, the
round old standoffs vs hex), and 8 wrist screws have no mesh and get a spec only.

Per-part audit (`robot/body/bom/inventory.py`): 411 parts are written by a script in the tree; about 338 start in old
study folders whose code is only in the private history or nowhere; 47 file references resolve nowhere, 20 parts have no
geometry; 618 parts have no assembly step (the order covers the right leg); 37 printed parts have no print settings. 38
printed parts lack a scripted native STEP (`robot/body/bom/defects.json`, Idle T-jsnes2 for the ones with no script).
Connectivity (manifest of 2026-11-19, 27.6 s): 3 isolated parts (chest indicator relief, head neck collar, B0410 camera),
5 groups not reaching the core (head skin with B0196; ROCK4D compute stack; fan stack; pelvis cells 8 and 9), 175 parts
reach the core by contact only, the largest contact-only break points the pelvis battery cradle (55 parts beyond it) and
the lower yaw servo cases (47 each). On the 2026-11-20 manifest the same check did not finish in 900 s and in 2400 s;
the cause and the reclassified findings are under Connectivity below (unscoped run now 42-49 s).

The chain that writes the selected manifest is data (`robot/body/bom/chain.json`, 10 steps) and `rebuild_chain.py` reruns
it in an isolated copy: steps 1-2 reproduce main exactly (747 / 707 parts, meshes byte-identical, STEPs identical up to
the header, bounds 0.0 mm); step 3 (`ape-proportions/build.py`, untracked) did not finish at 6 GB nor at 12 GB / 3600 s
(Idle T-be4dt7).

## Slicer pass (2026-11-19)

`slicer` runs Bambu Studio 2.8.2.61 headless (`--slice 0 --orient 0`, Textured PEI plate: the default Cool Plate refuses
PAHT-CF) on the print-frame mesh with the flattened X2D system profiles named in `printers.json` / `materials.json`
(0.6 mm: "0.30mm Standard @BBL X2D 0.6 nozzle" and "Bambu PAHT-CF @BBL X2D"; the profiles confirm the 0.62 / 0.42 mm line
widths and 0.30 / 0.20 mm layers used as printer defaults). Support grams are the net extrusion inside the G-code's support
sections (all sections together give 18.49 g against the header's 18.45 g on the head carrier; summing only positive moves
counted every un-retraction and read 21.8 g). Readings agree with `overhang`: stage3 15.6 g support of 31.3 g (5596 mm2 beyond
the limit), head carrier 11.1 of 18.5 g (3648 mm2), roll-flip thigh 2.7 of 14.3 g (1194 mm2), stage4 and foot none (8.0 g /
21.8 min, 21.0 g / 48.8 min). `--process nozzle_mm=0.4 --process material="Bambu PETG-CF"` switches to the 0.20 mm / PETG-CF
0.4 profiles (stage4 32 layers, 8.18 g).

## Sharp edges and corners (2026-11-19)

`edges` (owner: a finished part has no sharp edges and corners) classifies every B-rep edge by the angle between its two
face normals (smooth < 5 deg, soft kink < 20 deg, sharp above), except edges of chamfer strips (planar, at most 2 mm wide)
and fillet faces; convexity comes from the mesh crease. Synthetic tests: a box reads 12 sharp convex edges; the same box
filleted 1 mm reads none and a 1.0 mm smallest fillet; one 1 mm chamfer leaves 11 sharp edges and reads a 1.414 mm chamfer
width; an L reads one concave edge. Sharp edges within 0.3 mm of a neighbour at standing are reported as on a mating face.
First readings: stage4 carrier 52 free sharp edges (851 mm, no fillet at all: every outline, rail and hole rim edge),
R_foot_monocoque 289 (1290 mm, smallest fillet 0.4 mm), R_shoulder_pitch_to_roll 144 (535 mm). Review fixes: `edges.sharp_ok_near_parts` and
`edges.sharp_ok_regions` [{centre_mm, radius_mm}] mark edges meant to be sharp (counted as intended; a 30 mm region on stage4
moves 38 edges there), and every sharp edge and corner with its location goes to edges-<part>.json next to the picture
(the chassis has 972 free sharp edges and 468 corners, too many for the report's top ten).

## Connectivity (2026-11-20, T-c4sy9d)

`connectivity` (owner: no part unconnected, every part linked to the core) builds the manifest's graph. Nodes: every
non-hardware part. Edges: a screw joins the parts that encircle its shank (>= 5 of 8 points inside on one ring; rings
every 0.5 mm at r_shank + 0.3 / 0.8 / 1.5 / 2.5 mm, and around the head) or bear its head (>= 3 of 8 points 0.3 / 0.8 mm
under it); a threaded spacer entered by two screws joins both screws' parts; a sleeve joins its part; joints join case
and horn parts of their two links; contact is within 0.2 mm at standing. Every part is screened with the coarse
neighbour solid whatever the scope, and probed on a watertight mesh. The core is the stage1 chassis.

Why it stalled (2026-11-20). Not the manifest: between the 19 Nov and 20 Nov manifests the node set grew by 4 parts (the
thigh shell sleeves, 218 -> 222), and with the 19 Nov scope the graph builds in 53 s. The 27.6 s runs on 19 Nov were
scoped to one part (`--parts combined__stage1_center_torso_spine_chassis`, `R_shin_combined`); the stalled runs were
unscoped (`--checks connectivity`, BOM cleanup). Unscoped, every non-hardware part (190) is in scope, and `ctx.solid()`
gives a scoped part the fine 0.03 mm native STEP tessellation: 101 scoped parts have a STEP, 54 of them with no 0.03 mm
cache entry, among them supplier STEPs of 78 / 44 / 26 / 22 MB (power supervisor board, ROCK4D, B0196, B0410) whose
thousands of component solids are each tessellated and unioned. A stack dump at 60 s and 120 s sat in
`build123d tessellate` under `core.step_mesh` for `Power_supervisor_source_board`, the first solid the screw loop asked
for; it was still there at 170 s. Ruled out: no part was voxel-filled that was not before (6 voxel fills both days), the
pair loops are bounds-filtered (contact 5-6 s), no gmsh call, no lock (memguard waits were 15-51 s and reported). Fix
`1df655de4`: the graph is manifest-wide, so it always uses `core.screen_solid(..., changed=set())` with its own cache.

Screw model (`a6c243bfe`, `a3fd2ce6b`, `5ecdd6cef`). The old single ring at r + 0.8 mm fell into the gap of a pressed
steel sleeve, 8 stations stepped over 1.6 mm boards, decimated closed meshes left open made trimesh's ray-parity
`contains()` answer differently on repeated calls (microphone, foot ADC), the 8 wrist screws have bounds but no mesh
(`bodylib.Body` drops 20 mesh-less rows: wrist screws, both hand servos and horns, 8 bushes), and the fan screws are named
without a length (`fan_M3_11_430`), which `mount_check.kind` calls other hardware. Against `mount_check`'s clamp stacks
on the 52 screws of the torso and head carriers: 44 identical (old ring 9), no part outside the stack; the 8 that differ
miss a sleeve whose parent is joined (4), a countersunk head (2) or the back shell 0.2 mm under a socket head on a
decimated mesh (2). Screw edges 79 -> 164.

Timing (selected manifest, unscoped, memguard 6 GB): roll-flip body 222 parts 40 s wall (graph 32 s, 3.6 GiB); shoulder U
body 224 parts 42-49 s wall (graph 36-44 s, 3.7 GiB): solids 26-35 s, screws 29-37 s including solids, contact 6 s.

Findings on the shoulder U body (roll-flip body the same except the shoulder lines). Capture = new overlap (mm3) when the
part moves 2 mm along +-x / +-y / +-z at standing against every part within 3 mm; 0 means it slides free that way.

| Finding (19 Nov) | Today | Measurement | Class |
|---|---|---|---|
| ROCK4D compute stack not reaching the core | reaches the core by screw | compute M2.5 screws: ring at r + 0.8 mm 0.04 mm outside the carrier's sleeve, carrier 0.55 mm | artefact: sleeve gap in the screw model (fixed) |
| neck cradle -> head carrier contact-only | screwed | neck_M3x6_countersunk_1 clamps both (mount_check stack) | artefact: same (fixed) |
| microphone (isolated 20 Nov) | screwed | mic_side_M2x4_lowhead_0/1 head bears on it; the open decimated mesh flipped contains() | artefact (fixed) |
| pelvis battery cradle, 55 beyond | 1 beyond (itself) | captured: min new overlap 309 mm3 (+y, chassis); 9 of 9 cradle sleeves and 29 of 41 chassis sleeves have no screw within 0.5 mm of their axis (nearest 8-146 mm), coaxial pairs at x +-17 / +-11.2 / +-33.6 | real CAD gap: holes and sleeves exist, the chassis-cradle screws are missing; geometry captures it meanwhile |
| stage3 power carrier, 3 beyond | same | free along +-z (0 mm3); 4 of 10 sleeves empty, coaxial with chassis sleeves at x +-17, z 288 / 338 | real CAD gap: 4 screws missing |
| stage4 IMU / ADC carrier, 4 beyond | same | free along +-z; 6 of 14 sleeves empty, coaxial with chassis sleeves at x +-11.2 / +-33.6, z 304 / 335 | real CAD gap: 6 screws missing |
| lower yaw servo cases, 47 / 49 beyond each | same | gap 0.04 mm, 1235 mm2 within 1 mm of the chassis pocket; free along -x and -y (L; +x / -y R): only its own idler, 35 mm3; no hardware within 2 mm, none in the waist-pelvis source design either; none of the case's 8 defined mount holes has a screw | real design gap: the whole leg hangs on a pocket seat |
| upper dorsal / core cells 01-07 with insulators and fins | contact-only | captured in all six directions, min about 100 mm3 (+z, chassis) | artefact: held by the honeycomb pocket, which the graph does not model |
| pelvis cells 08 / 09 with insulator and fin (groups) | same | 0.83 mm from the cradle; free along +-y (0 mm3); cradle mesh note: floor under the lower cell removed, "hold that cell at its ends" until the CAD pass | real design gap (known, pending the cradle CAD pass) |
| head skin + B0196 camera (group) | same | skin: no screw (nearest compute screw 1.92 mm), free along -y and +z; B0196: camera screws enter only the camera, plate screws only the carrier and end 1.1-2.3 mm short; the 4 M2 F-F 5 mm standoffs between them exist in the sensor layout parts but not in the manifest | real gaps: head skin has no mount; B0196 standoffs missing from the CAD |
| B0410 camera (isolated) | same | 1.23 mm from the carrier; plate screws end 2.1-2.7 mm short; its 4 F-F standoffs are missing the same way | real CAD gap: standoffs missing |
| fan stack (group) | same | 4 M3 x 15.7 screws clamp only the fan (8 / 8 ring points) and the lead guide; nearest other part 1.05 mm (Radian heatsink, beside the axis) | real design gap: the fan bolts to nothing |
| head_neck_collar (isolated) | same | 0.82 mm from the head skin, 0.98 mm from the neck link; no screw; free in four directions; its `native_step` names the head skin STEP | real design gap (and a data defect) |
| chest indicator relief (isolated) | same | 0.229 mm from the front shell (0.2 mm contact tolerance); blocked +-x 29 mm3, free +y | real design gap: no retention (bond, press or lip not specified) |
| palms (new 20 Nov), palm upper and fingers | palm lower screwed (wrist screws from bounds); palm upper and fingers contact-only | the hand servo case and horn have no mesh, so no joint edge can form | artefact: mesh-less hand servo |
| forearm front covers | contact-only | L_fore_front_M3x8_187 inside the forearm (12 points), 1.07 mm from the cover | unresolved: screw placed before the shell morph, or missing hole |
| Radian heatsink + TIM | contact-only | 0.11 mm from ROCK4D, TIM 0.0, guard 0.5 mm, no screw | unresolved: intended hold (adhesive TIM, guard clip) not in the CAD |
| Wuerth 61304021821 header | contact-only | 0.18 mm from the neck pitch case, not on a board (ROCK4D beyond 0.34 mm) | unresolved: placement looks stale |
| center back shell | contact-only | T02_back_mount1/3 clamp it per mount_check; probe 0.2 mm short on the decimated mesh | artefact: residual of the probe |
| upper arm covers (shoulder U body) | contact-only | cover posts removed with the shoulder U connectors | expected until the shell redesign (T-tph7xb) |

Counts (shoulder U body): 224 parts, 3 isolated, 4 groups not reaching the core (14 parts), 150 parts reach the core by
contact only, 37 contact-only break points; edges: screw 164, contact 378, joint 34, sleeve 32; 8 screws probed from bounds.

## Elephant foot and clearance per fit class (2026-11-19)

`elephant_foot` slices each printed part in its print frame at 0.5 mm above the bed (mesh profile, not B-rep: it reads a
planar chamfer, a fillet or any other relief alike) and compares that wall with the first-layer outline every 0.5 mm. Signed inset =
how far the first-layer edge is pulled in from the wall (+ chamfer, 0 square foot, - the wall leans out). A wall within 0.3 mm
of another part at standing is a mating wall; no chamfer = inset under 0.2 mm. It reports the mating outline length, the
length and area (length x 0.5 mm) without chamfer, runs with locations and the mating part. `fit_class` takes every pair
within 1 mm (or with a class) from the `gaps` logic: clearance = minimum distance at standing, negative = deepest
interpenetration (sampled); the class comes from the part settings (`<part>.fit_classes = {mate: class}`), its band from
`data/fit_classes.json` (press -0.15..0, slide 0.15..0.35, loose 0.35..0.8 mm: reference values, not measured on this printer;
replace with coupon results), and the signed margin to the nearer band edge is reported. Pairs without a class are counted, not judged.
Synthetic tests (`robot/body/checks/test_fits.py`): a 20 mm block with a mating wall reads 80 mm of outline, 80 mm without
chamfer, 40 mm2, inset 0.0; with a 0.4 mm chamfer 76.8 mm (the smaller bed outline), 0 mm without chamfer, inset 0.40; with no neighbour 0 mm
mating. Two blocks 0.15 mm apart read 0.150 mm; overlapped by 0.3 mm they read a depth of 0.300 mm; clearance 0.15 / 0.5 / 0.05 mm
against slide gives margins 0 / -0.15 / -0.10, 0.2 mm against press reads too loose by 0.2.
First readings: R_hip_pitch_to_roll_roll_flip (-x) has the four 4.5 mm steel-sleeve holes (and no others) on the bed as
square mating edges, 55.5 mm outline, 27.8 mm2; confirmed with independent cross-sections of the solid at 0.02, 0.25, 0.5 and 0.8 mm
from the bed: the hole diameter stays 4.50 mm and circumference 14.13 mm, so no chamfer. The hip-pitch / thigh pair reads
0.150 mm clearance (slide margin 0.00) and the sleeves read -0.003 mm (press margin +0.003: CAD interference is nil; the
printer's hole offset in `holes` is what makes it tight). R_thigh_roll_flip (+x) has no mating bed edge. Slide and press classes of
the first pair are set in `part_settings.json` as a trial; the remaining 14 pairs of the two parts have no class.

## FEM along the decided print direction (2026-11-19)

The print direction of a part (`process.up`, print setup) is now the single source of the FEM layer normal: `fem` runs the axis of `up` (sign irrelevant for the orthotropic axis) and reports p99 layer tension and the across-layer strength ratio (`layer_strength_ratio`) next to the in-plane p99 von Mises and ratio in every worst case. The old free list `process.layer_normal_axes` stays in the data as provenance and is the input of the comparison variant `--fem-variant studied_axes`. A part without a print direction (or one not along an axis) falls back to its stored axes and the report says so (`layer_direction.source`). `fem_setup` reports per part the direction and its source and counts parts whose direction is missing, seeded (unconfirmed) or not among the stored axes.

Verification: fem_setup on R_shin_combined, stage1 chassis, R_thigh_roll_flip, neck_root_cradle agrees with part_settings by hand (4 seeded; 2 disagree). fem on R_ankle_coupler_combined (print +x, stored x): default and studied_axes give identical numbers (heavy_6kg_1p5g PAHT-CF: p99 von Mises 31.97 MPa, p99 layer tension 6.03 MPa, in-plane ratio 2.75, layer ratio 10.61).

Of the 18 parts with a setup: 2 have no print direction (R_thigh_combined, R_hip_pitch_to_roll_combined: stored axes used), 16 are seeded or set from text, and 2 disagree with the stored axes: R_shin_combined (print +x, stored z) and the stage1 chassis (print -z, stored y). Their FEM now runs along the print axis; confirm the direction in the print setup, `studied_axes` gives the old numbers.

## Screening solids: exact, and what the reference is (2026-11-19)

Pairwise checks screen each part as a closed solid: native STEP tessellation (fused), else the part's own closed mesh, only an open mesh is voxel-filled. The voxel fill is the one that over-states: it counts the surface layer of a thin shell, so a 1 mm fill of the four shells below reads +67 % to +137 % (0.5 mm: +34 % to +68 %), and the ST3025 cases read +33 % (12988 against 9753 mm3), which made a case x thigh connector standing overlap 97.5 mm3 where the exact value is 0.

The "-15 % ... -29 %" that the report printed for shells was the reference, not the solid. `volume_mm3` in the manifest of a mesh-only shell is the volume of the STEP before the ape-proportions morph / stretch (R_upper_arm_hard_cover: the unmorphed STEP is 21383.1 mm3, the morphed mesh 15336). The report now compares with a reference it can defend: the STEP volume (all solids fused: a servo's six overlapping solids sum to 41257 mm3 but fuse to 32983), else the signed volume of the part's own closed mesh, else the recorded volume labelled as possibly stale. Every solid in the report carries `reference_volume_mm3` and `reference`.

Independent measurement (signed mesh volume / ray-parity Monte Carlo 400k points / voxel fill 1.0 mm / 0.5 mm, mm3): center_front_shell 62560 / 62309 / 104439 / 84005; L_fore_front_hard_cover 6375 / 6381 / 12156 / 9099; L_fore_back_hard_cover 8755 / 8822 / 17064 / 12550; R_upper_arm_hard_cover 15336 / 15585 / 36344 / 25836. The screening solids are the signed volumes (decimation within 0.02 %); the voxel fills are 1.7 to 2.4 times too large at 1 mm. Solids of those four shells are exact, no further fix; they simply disagree with a stale manifest number.

Bug fixed: `core.step_mesh` returned an EMPTY mesh when a solid's tessellation was not a manifold (the stage-1 chassis at 0.1 mm: 9123 faces, NotManifold): the fallback concatenated `kit.tm` of empty manifolds, and the empty result was cached. It now treats a non-manifold tessellation as a failed try and re-meshes on a fresh import (0.03, then 0.1, then 0.5 mm); the chassis comes out at 0.03 mm, 78567 mm3 against 78433 in the STEP (+0.17 %). Empty cache entries are rebuilt.

Verification (`--parts L_knee_ST3025_case,R_upper_arm_hard_cover,L_fore_front_hard_cover,L_thigh_roll_flip --checks gaps,static_audit`): all four solids 0.0 % off their reference; L_knee_ST3025_case x L_thigh_roll_flip standing overlap 0 mm3 (touching, distance 0), and an exact OCC common of the two STEPs gives 2.6e-11 mm3. What still differs: a part whose mesh is open is still voxel-filled or hulled (the report names them), and the STEP tessellation of a mirrored ST3025 case is 0.1 mm (chordal error, not a volume error: +0.13 %).

## Heat contact screen (2026-11-19)

`heat` migrates the HDT margin of the torso `thermal_evaluate.py` (HDT minus the carrier wall temperature) into the shared tool as a screen; the coupled cell / air simulation stays in that script. Per scoped printed part at standing it measures, against every heat source (manifest roles `motor`, `actuator`, `battery_cell`, `source_power_board`; `data/heat_sources.json`), the contact area (surface within 0.3 mm), the closest distance, the area within 3 mm, the source's worst-case temperature, the material's HDT at 0.45 / 1.8 MPa and Tg (`materials.json` `print.*`, resolved per part by `process.effective`, so `--process material=...` changes every margin) and the margin in C (limit minus source temperature). Source temperatures: servo case 70 C (Feetech internal torque-off limit [S8], taken as the surface worst case: an assumption that over-states the surface), battery cell 60 C (P50B continuous limit from the old script's `P50B_SPEC`), power board 85 C (assumption, no data). The 0.3 mm contact distance also catches the 0.2 mm bore clearance of a cell in its honeycomb pocket; distances read up to ~0.3 mm high (sample spacing). Margins to Tg are shown next to HDT: the data sheets give PAHT-CF Tg 70 C, so a 70 C source reads 0 C to Tg but 100+ C to HDT.

Verification: synthetic two blocks (`test_heat.py`): 10 x 10 mm face on a 20 mm block reads 107 mm2 contact (100 true; the rim's side faces within 0.3 mm add 7 %) at distance 0, and 2.0 mm at a 2 mm gap. Real: `combined__stage2_lower_pelvis_battery_cradle` to cell 05 closest 0.27 mm against an exact manifold `min_gap` of 0.275 mm; to `R_hip_pitch_ST3025_case` 2.49 mm against exact 2.25 mm (sampling reads 0.24 high).
Through `run.py` on `L_thigh_roll_flip`: 352 mm2 on `L_knee_ST3025_case` at distance 0 (866 mm2 within 3 mm; the gaps check read touching, exact overlap 0); margin to HDT 0.45 / 1.8 MPa +124 / +100 C in PAHT-CF and +4 / -2 C in PETG-CF (`--process material=Bambu PETG-CF`). The cradle touches the nine cells (about 30-47 mm2 each for the seven upper ones) and sits 2.5 mm from the two hip-pitch cases: against 60 C cells +134 / +110 C (PAHT-CF), +14 / +8 C (PETG-CF). The cradle's `run.py` call is slow only because run.py's own reference-volume tessellation of that part takes minutes; the check itself takes 15 s.

## Accuracy of the existing checks: poses, thresholds, load paths, summary (2026-11-19)

- **Pose sets per region.** Fit and clearance on parts of the arm, neck and torso links ran standing only in effect: the key / library sets move leg joints. `pose_sets.json` now holds `region_poses` built from the manifest joint ranges (each joint at both ends, named pairs such as shoulder pitch + elbow, fingers open; torso: every joint of the chains hanging on it, both sides together). Checked: clearance of the neck yaw-to-pitch link at `key` now finds its closest approach to the pitch servo case at neck pitch +35 deg, and the arm upper-arm connector at shoulder yaw -60 deg (0.14 mm to the yaw case); before, the same parts only saw leg poses. No old arm or neck pose list survived in Git history beyond the joint ranges (the per-section scripts used each joint's own sweep), so the lists are new data; edit them when joint limits change.
- **Per-part geometry thresholds.** Sliver area, micro-edge length, acute angle and G1 kink are `geometry_thresholds` in `part_settings.json` (defaults = the old leg values). Example: the arm roll-to-yaw connector counts 3 sliver faces and 11 micro-edges at the defaults, 0 and 0 with 0.05 mm2 and 0.10 mm.
- **Load-path slices for torso and neck.** Axis and range of the neck root cradle, yaw-to-pitch link, pitch-to-head cradle, head carrier and torso chassis are the dominant-axis span between the centroids of the FEM support holes and load holes (labelled `set_by`; a first reading, not designer-confirmed). Thinnest slabs: chassis 252 mm2, root cradle 16 mm2 (1 severed branch), yaw-to-pitch link 47 mm2, pitch-to-head cradle 91 mm2, head carrier 2.5 mm2.
- **Summary noise.** The bed margin of a part that fits and the STEP-vs-mesh difference are `context` cases: listed in the report, counted in the summary; a negative bed margin stays a finding. Setup timers are no longer added to the check seconds; collision-sweep `overlapping_pairs` counts distinct pairs, not pair x joint; same-link pairs print no pose.

## FEM completeness: origins, shells, left arm, published evidence (2026-11-19)

- Wrench origins are derived: `origin_mm` takes `{"joint": J | "joints": [A, B], "offset_mm": [...]}` (resolved in `fem_core.origin_of` from the manifest joints; two joints give their midpoint). The three setups still typed (roll-flip thigh and hip pitch-to-roll, pelvis cradle) are converted; derived equals the typed value exactly (thigh [32, 4, 141.65] from R_knee, hip pitch-to-roll [37, 7.2, 190] from R_hip_roll, cradle [0, 6, 231.35] from the R/L hip pitch midpoint). The `legacy_coordinates` variants keep their typed old-body points, which lie 12 mm further out in x than the derived ones (the old legs were 12 mm narrower), except the cradle (0 mm).
- `fem_setup` coverage counts printed body shells (role `body_shell`) next to connectors and carriers.
- The four left arm connectors carry the mirror (x -> -x: Fx, My, Mz flipped) of the right setups. `fem_setup`: every patch of all 8 arm parts lands on holes of the part (L vs R patch areas within 1.5 mm2 (0.3 %), node counts within 0.5 %, no mating part without contact, 0 shared support / load nodes). `fem` on the shoulder pitch-to-roll pair at h 1.2: PAHT-CF p99 recovery 197.6 (L) vs 196.2 MPa (R), heavy 222.1 vs 220.9 MPa; the whole arm L vs R in the evidence (each part along its print direction, all load cases): PAHT-CF p99 -1.30 % to +0.78 %, up to 2.88 % across the three materials; peak displacement L lower than R by 0.2-3.14 %.
- Published FEM evidence (the publishers `publish_{arm,neck_head,torso}_studies.py` build the FEM section of their record from `fem-evidence.json` only; `--fem-section` prints it without publishing): `fem_evidence.py --study cove50-{arm,neck-head,torso}-fdm-combined` prints the `run.py --checks fem,fem_setup` command and turns its report into `fem-evidence.json` (per part and per load case / material the p99 and peak stress, layer tension, ratios, displacement; per chain the RSS displacement). Arm 8 parts, neck 4 parts, torso stage 1 only: stage 2 is mesh-only in the manifest (setup not checked) and stages 3 and 4 have no setup. The old multi-family tables (baseline / truss / topopt) of the study publishers are not recreated: those families are not in the tree.

## Screw connections and driver access (2026-11-19)

`screws` and `driver_access` take the screws that touch the scope (>= 2 points of a ring 3 mm around the shank, along it and
under the head, inside a scoped part or its sleeves at standing; 3 mm is the window in which mount_check still reads a hole as
the screw's). `screws` runs `mount_check.run_screws` on them, so its per-screw numbers are the full report's (17 screws on
`R_shin_combined` and `R_hip_yaw_to_pitch_combined`: engagement, hole offset, head-seat and layer gaps all identical to the
263-screw report), and reports per screw the engagement against the standard part's reference, hole offset, clamp gaps, the
sleeve under the head, and the reverse view restricted to the scope (sleeves without a screw, mount holes of scoped standard
parts). `driver_access` puts a cylinder of the driver's blade diameter along the axis from the head outward (`data/tools.json`:
drive per head type and thread; review fix: only the M2 / M2.5 socket-cap hex sizes have a source (ASME B18.3.1M), ISO 7380-1
starts at M3, PA2 is set to PH0 by analogy with ISO 7045 (M2: 0, M2.5 / M3: 1), every other drive size and all blade diameters are
assumptions until the screws' data or the owner's driver set replaces them) to where the axis leaves the body's bounding box (40 mm at least, 250 at most), and
measures at standing its overlap with every other part, sleeve and shell: volume over the whole path and over the first
40 mm, share of the path, first obstructing part and distance from the head. A ray integral over the same path agrees with
the boolean on the yaw M3x14 screws (first obstruction 8.20 vs 8.25 mm, 136 vs 138 mm3 of the connector itself; 55 vs 55 mm3
of the pitch servo case). Other screws' heads are not in the path. Not covered: a nut slot or the insertion of the tip side.

## Fastener heads and tips over motion (2026-11-19)

`clearance`, `fit` and `collision_sweep` leave fasteners out, which hid a real collision on roll-flip Stage 4: the upper STS3250 case-ear PA2 heads (`R_A13_roll_face1_PA2x3_2/_3`, x 45.75 / 66.25, z 181.7) overlap `R_hip_pitch_to_roll_roll_flip` at every roll angle. `fastener_motion` measures each screw (scoped links plus screws touching a scoped part) as a solid against non-fastener parts on other links over the pose set, optionally the range of the joints named with `--joints`: overlap (manifold boolean, split into head, shank and the last 3 mm tip), minimum distance, worst pose and location, and the overlap at that pose against the part's full manifest mesh (the screening solid reads about 9 % low on the hip part: 2.04 against 2.245 mm3). A second measure is at standing: screw head volume inside a same-link part or its steel sleeves (above 1 mm3), which flags screw models placed inside their own plate (stale heads of `R_A13_roll_face0/1_PA2x3_*` inside the thigh connector sleeves, 5.5 - 7.6 of 12.5 mm3). Steel sleeves are not cross-link neighbours (their boolean costs minutes). Checked on `R_thigh_roll_flip`: the face1 heads at z 181.7 overlap the hip part 2.2 mm3 at standing, 6.2 mm3 at hip roll -45; an exact OCC boolean of the two STEP solids at standing gives 2.254 mm3. Scores, not verdicts.

## Assembly sequence (2026-11-19)

`assembly` walks `data/assembly_order.json` (assemblies -> base parts + ordered steps; a step is a part / group or a set of
screws, a direction, `seeded` and `derived`). Part step: the part (with its sleeves) is translated back along its direction in
1 mm samples to 60 mm; per sample the exact overlap volume against the parts assembled earlier only; reported: first contact
distance (last sampled overlap, bisected to 0.01 mm), max overlap and where, overlap at the end of the range (path never
clear), seat overlap at t = 0. Screw step: the `driver_access` path against what is assembled at that step, plus the later
parts that would close it in, and the parts the screw touches that are not placed yet. Coverage: parts and screws on the
assembly's links that no step names. The right leg (hip-yaw servo case to foot, 43 steps, all seeded) is drafted by
`checks/assembly_seed.py` from the connectivity graph (fastened distance from the servo case; direction from the screws that
join a part to what is placed, else toward its placed neighbours snapped to an axis). The drafted directions are guesses, so
its blocked steps first show where the draft is wrong; the numbers become findings once the designer confirms the order. Thin
obstacles under the 1 mm sample can be stepped over; the screw's own shank path is not swept (driver path only). Not built:
serviceability (parts to remove before a servo swap) would be a walk of the order in reverse.

## Open questions

- Bambu's official X2D spec page could not be fetched; the bed maximum (120 °C) and nozzle list rest on resellers.
- Which X2D nozzle prints the CF parts (direct-drive left is likely) and whether the Bowden right nozzle should print
  only support: affects the usable plate width (256 vs 235.5 mm).
- Density: 1.06 g/cm3 in Bambu's PAHT-CF sheet (used by `mass`) vs 1.22 g/cm3 in the legacy code and manifest masses.
  A weighed printed sample decides.
- Nozzle: the checks default to 0.6 mm (Bambu's recommendation for PAHT-CF); older build notes say 0.4 mm. The
  minimum-wall numbers depend on it (`--process nozzle_mm=0.4` gives the other reading).
- Hole offset and fit-class bands: no coupon printed yet, so press and slide margins are CAD margins only.
- Which printed parts are PETG-CF rather than PAHT-CF: the manifest says "or" for most of them, and the heat check
  depends on it.

## Sources

- S1 Bambu Lab, X2D product and spec pages: https://bambulab.com/en-us/x2d , https://bambulab.com/en/x2d/specs
- S2 3D Printing Industry, X2D launch, 2026-04-28: https://3dprintingindustry.com/news/bambu-lab-launches-x2d-dual-nozzle-3d-printer-targeting-reduced-post-processing-and-material-waste-251005/
- S3 DEVELOP3D, Bambu Lab X2D: https://develop3d.com/3d-printing/bambu-lab-x2d/
- S4 Reseller spec listings: https://www.3djake.com/bambu-lab/x2d , https://www.matterhackers.com/store/l/bambu-lab-x2d-3d-printer/sk/MM4MFV83 , https://printer-hub.com/printers/bambu-lab-x2d
- S5 Bambu hardened steel hotends for H2 / X2 / P2: https://us.store.bambulab.com/products/bambu-high-flow-hotend-h2-p2s-x2d
- S6 Bambu PAHT-CF technical data sheet V2.0: https://cdn.shopify.com/s/files/1/0380/5149/files/Bambu_PAHT-CF_Technical_Data_Sheet.pdf
- S7 Bambu PETG-CF technical data sheet V2.0: https://cdn.shopify.com/s/files/1/0380/5149/files/Bambu_PETG-CF_Technical_Data_Sheet.pdf
- S8 Feetech STS3215 datasheet (translated) and STS3250 listing: https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf , https://openelab.io/products/feetech-sts3250-c001-ttl-servo , SO-ARM100 thermal test https://github.com/TheRobotStudio/SO-ARM100/issues/134
- S9 OCCT BRepCheck_Analyzer and shape healing: https://dev.opencascade.org/doc/refman/html/class_b_rep_check___analyzer.html , https://dev.opencascade.org/doc/overview/html/occt_user_guides__shape_healing.html
- S10 Siemens NX Check-Mate (part / assembly validation, SASIG-PDQ): https://blogs.sw.siemens.com/nx-design/ensure-cad-quality-with-nx-validation-tools-whats-new/
- S11 Autodesk Netfabb repair and standard analysis: https://help.autodesk.com/cloudhelp/2026/ENU/NETF/files/GUID-91B0DBFF-A4A8-4FFB-84BA-20904FFE83F9.htm
- S12 Materialise Magics fixing tools: https://www.materialise.com/en/inspiration/articles/5-basic-3d-model-repair-functions
- S13 Fusion 3D printing analyses (minimum wall, upskin / downskin) and generative-design overhang limits: https://up1.autodesk.com/prd/2027/NETFE/47BEF3B6-8946-5271-AD2B-43C733B67A09/3D_Printing_Essentials_Usage_Instructions.html , https://help.autodesk.com/cloudhelp/ENU/Fusion-GenerativeDesign/files/GD-ADDITIVE-REF.htm
- S14 Protolabs, mistakes to avoid in 3D printed parts: https://protolabs.com/resources/design-tips/7-mistakes-to-avoid-when-designing-3d-printed-parts
- S15 PrusaSlicer layers and perimeters (thin walls): https://help.prusa3d.com/en/article/layers-and-perimeters_1748
- S16 University of Zurich FDM design guideline: https://www.amf.uzh.ch/dam/jcr:fd1ee024-1a98-488b-89ec-ffd85befa786/Design-Guideline-FDM.pdf
- S17 Stratasys Direct FDM design guide: https://info.stratasysdirect.com/rs/626-SBR-192/images/FDM_Design_Guide_Stratasys_Direct.pdf
- S18 HLH Rapid FDM design guide: https://hlhrapid.com/knowledge/design-guide-fdm-3d-printing/
- S19 Floating regions / empty layers in Bambu Studio: https://help.simplyprint.io/en/article/fixing-floating-regions-and-empty-layer-slicer-errors-1qyt2r6/ , https://forum.bambulab.com/t/why-do-i-get-empty-layers-and-floating-regions-after-slicing/73629
- S20 OrcaSlicer precision settings (elephant foot, XY hole compensation): https://www.orcaslicer.com/wiki/print_settings/quality/quality_settings_precision
- S21 Warping fixes (brim, corner radius, enclosure) and PA-CF forum reports: https://printpal.io/wiki/warping-and-curling , https://forum.bambulab.com/t/warping-issues-w-large-thin-paht-cf-print-on-x1c-help-needed/58816
- S22 Elephant foot causes and chamfer fix: https://www.wevolver.com/article/3d-printing-elephant-foot
- S23 Rob Miles, the power of the chamfer: https://www.robmiles.com/journal/2020/11/1/the-power-of-the-chamfer
- S24 Undersized and horizontal holes: https://www.goodprints3d.com/blogs/3d/why-do-3d-printed-holes-come-out-too-small-and-how-do-you-fix-them
- S25 ISO 286 fits on desktop FDM (MDPI Processes 2023): https://mdpi.com/2227-9717/11/10/2810
- S26 Bambu Studio command line usage: https://github.com/bambulab/BambuStudio/wiki/Command-Line-Usage
- S27 Printago slicer CLI reference (third party): https://printago.io/blog/bambu-studio-cli-reference
- S28 PrusaSlicer man page (`--info`): https://manpages.ubuntu.com/manpages/jammy/man1/prusa-slicer.1.html
- S29 OrcaSlicer auto orientation: https://www.orcaslicer.com/wiki/print_prepare/prepare_auto_orient
- S30 Tweaker-3 auto-orientation (Salzburg Research): https://www.salzburgresearch.at/en/?p=17279
- S31 Onshape analysis tools and interference detection: https://cad.onshape.com/help/Content/analysis_tools.htm
- S32 Screws into printed parts (pilot hole, engagement): https://kingroon.com/blogs/3d-print-101/how-to-screw-into-3d-printed-parts , https://makerworld.com/en/community/post/1182854
- S33 Fictiv, fasteners for 3D printed parts: https://www.fictiv.com/articles/how-to-choose-the-best-fasteners-for-3d-printed-parts
- S34 Threaded fastener site design (engagement 2-3 x diameter): https://www.taylorfrancis.com/chapters/mono/10.1201/9780203747889-6/fastening-site-design-considerations-threaded-fasteners-brayton-lincoln-kenneth-gomes-james-braden
- S35 CNC Kitchen, heat-set insert tips: https://www.cnckitchen.com/blog/tipps-amp-tricks-fr-gewindeeinstze-im-3d-druck-3awey
- S36 SPIROL, hole and boss design for inserts: https://www.spirol.com/resources/white-papers/how-to-design-the-proper-hole-for-heat-ultrasonic-inserts/
- S37 SolidWorks DFMXpress (inaccessible features, hole rules; machining processes only): https://help.solidworks.com/2021/english/SolidWorks/dfmxpress/c_rules.htm
- S38 Boothroyd Dewhurst DFA, minimum part criteria: https://dfma.com/design-for-assembly.asp , https://www.dfma.com/resources/part-consolidation.asp
- S39 Boothroyd, Design for Assembly, the key to DFM: https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/BF02601481.pdf
- S40 ToddlerBot zero-point calibration and assembly manual: https://hshi74.github.io/toddlerbot/tools/01_zero_point_calibration.html , https://hshi74.github.io/toddlerbot/_static/assembly_manual.pdf
- S41 Berkeley Humanoid Lite hardware docs (insert jig, printing): https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware
- S42 Fictiv, tolerance analysis for 3D printed parts: https://www.fictiv.com/articles/how-to-conduct-a-tolerance-analysis-for-3d-printed-parts
- S43 University of Florida design lab, common design review mistakes and detailed design checklist: https://web.mae.ufl.edu/designlab/TA/Common%20Design%20Review%20Mistakes.pdf , https://web.mae.ufl.edu/designlab/Lab%20Assignments/EML2322L%20Detailed%20Design%20Checklist.pdf
- S44 SolidWorks Design Checker (standards, feature and assembly checks): https://help.solidworks.com/2025/English/swconntutorial/Design_Checker_Overview.htm
- S45 SimScale, checking and improving mesh quality: https://www.simscale.com/knowledge-base/how-to-check-and-improve-mesh-quality/
- S46 Digital Engineering, accuracy and checking in FEA: https://www.digitalengineering247.com/article/accuracy-and-checking-in-fea-part-1
- S47 igus, robotic cable management: https://toolbox.igus.com/wp-content/uploads/2023/08/robotic-cable-management-made-easy-.pdf
- S48 Why robot cables fail (bend, torsion): https://blog.igus.ca/?p=6000
- S49 Anaerobic threadlockers on plastics: https://www.adhesivesmag.com/articles/96391-can-we-use-anaerobic-threadlockers-for-screws-used-in-plastics
- S50 AIAG-VDA FMEA seven steps and action priority (secondary): https://www.qualitymag.com/articles/98478-failure-modes-and-effects-analysis-fmea

## Press check: spacer-block counter side (2026-11-20)

The fixed coaxial dia 20 x 40 mm anvil post always met the opposite plate of a 32 mm servo-case pocket, and the coverage
score read annulus points inside an adjoining integral wall as "no surface". The counter side is now a spacer block on the
exit plane. Per sleeve (annulus sleeve OD/2 + 0.15 .. 5 mm, equal-area samples): `flat_pct` (air above, surface within 0.2 mm),
`wall_pct` (material above the plane, the spacer cannot enter but the part is supported), `recess_pct` (surface more than
0.2 mm below, within 10 mm), `open_pct`; `support_pct` = flat + wall; `free_depth_mm` = how deep a spacer on the flat points
reaches before meeting the part (cap 40), `axis_free_depth_mm` the same on the axis. Old post volumes and coverage stay.
Synthetic checks (`checks/test_press_spacer.py`): flat plate 100 % flat / 40 mm; wall on x > 3 mm 18.3 % wall (analytic
18.4 %); plate 32 mm opposite 31.8 mm free; counterbore 100 % recess; through hole 100 % open.
R_thigh_roll_flip: free depth 31.5-31.8 mm for all 8 sleeves; wall share 23-24 % on the four inboard sleeves (1, 3, 5, 7), which the
old coverage read as no surface; sleeve 1 wall share 24.2 % against 25.0 % from an OCC slab boolean, axis free depth 31.8 mm
against 32.0 mm from OCC pin booleans (0.5 mm steps). Limits: spacer width is not a variable (a narrow block is assumed);
recess / open points are not supported.
