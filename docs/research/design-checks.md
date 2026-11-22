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

2026-11-22 (T-ghvncz), shoulder-shells body (673 parts): 196 lines (79 make, 116 buy, 1 included), 0 quantity
differences; buy lines: supplier filled from a repo source 15, part number 47 (standard designations for ISO hardware,
the actuator catalog, the electronics selection; source in each line's notes), both `unknown` 68, empty 0; 44 hardware
kinds (29 screw), 273 sleeves in 44 sizes; parts with an assembly step 673 of 673. Dangling file references: 31 native
STEPs (47 on the 20 Nov body), all with the exact bytes recorded in the bucket while the local blob index
(`data/blob-store.sqlite`) is empty: flagged `file_in_bucket_not_indexed`, explained per group in `defects.json`
`dangling` (T-ysavyb); 0 unexplained. `bom` check 4.4 s, 0.6 GiB.

The chain that writes the selected manifest is data (`robot/body/bom/chain.json`, 15 steps, ending at the shoulder-U body
of `robot/design.json`) and `rebuild_chain.py` reruns it in an isolated copy (Idle T-be4dt7, 2026-11-20). Run times (s /
peak GB): assembly 323 / 8.2 (data complete, render phase over its then 8 GB cap, now 10), motor-refit 583 / 3.5,
ape-proportions 264 / 4.4, shell-pass 334 / 3.6, foot-screws 46 / 1.5, hip-yaw-screws 11 / 0.7, torso-screws 5 / 0.6,
roll-flip-cad 220 / 1.1, roll-flip bootstrap 13, ankle-coupler-screws 17, roll-flip manifest 13, thigh-shell 46 / 2.6,
shin-cover 32 / 3.0, shoulder-clevis cad 9, manifest 2: about 32 minutes. The rebuilt shoulder-U manifest against the
selected one: 651 / 651 parts, none added or removed, no link or role change, bounds within 0.001 mm; 2 STEPs (ankle
couplers) and 13 meshes differ: surface distance <= 0.033 mm on the ankle couplers and thigh shells (tessellation), up to
0.68 mm (p99 0.26) on the boot and torso shells (the voxel sweep sampled surfaces unseeded; seeded now), 0.58 mm / +0.47 %
volume on the head neck collar (not traced), triangulation only on the pelvis cradle and head skin.

What had blocked it, with evidence:
- Step 3 never finished (past 3600 s at 6.5 GB). Stack dumps every 30 s all sat in `hips.py` at `res.to_mesh()` of the
  pelvis-cradle cut: manifold3d records booleans lazily, so the "batched" subtraction of 3,538 swept hip-part copies was
  one CSG tree evaluated in a single C++ call. The gmsh sleeve mesh logged just before takes 0.5-0.9 s; the gmsh
  faulthandler limit was already cancelled (and was checked to stop gmsh inside `mesh.generate`). Evaluating after every
  batch (`hips.cut_cradle`, `num_tri()`): 185 s, 4.9 GB alone, cradle volume 22348.7 mm3 as selected.
- `ape-proportions/` was untracked; the edits after the selected step-3 manifest were reviewer fixes (elbow motor sign,
  hip pitch-to-roll volume), which the selected downstream manifest already carries, and a cradle tool size change made
  for the lazy tree's memory (reverted to the selected 2500 faces / 150 per batch).
- `roll-flip-cad` stopped at 997 s with no output: the Mac slept 18:06-18:21 (pmset log) and memguard's timeout is wall
  time. Steps now run under `caffeinate -i -s`; awake it takes 220 s.
- The roll-flip manifest's hooks read and wrote section folders through links into the main checkout (neck screws
  rewritten in main, byte-identical; torso records read from main; foot and hip-yaw builds missing, so the unedited foot
  stayed). Their builds are chain steps now and their folders belong to the copy.
- `compare` reported "raw_equal" for a file compared with itself through links, and compared against a setup snapshot
  the selection had moved past. It now says `same_file`, counts reference files that no longer match their recorded
  sha256, and compares against the live manifests. The older intermediates in main (assembly ... shell-pass) carry 12-32
  such stale references (neck connectors rebuilt today, leg section meshes), so they differ by up to 1.99 mm there.

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

Whole body (2026-11-22, Idle T-ghvncz). The order covers every subassembly of the selected body
(`artifacts/robot/body/arm/shoulder-clevis/shells/assembly-parts-shoulder-u-shells.json`, 673 parts), each built in place
on the ones in its `after`: torso (base: stage-1 chassis and its sleeves), neck_head, right_arm, left_arm, right_leg,
left_leg, final_body (torso shells). Seeded by `checks/assembly_seed.py seed` from one connection graph (66 s, 4.0 GB),
the right leg's 16 designer hip steps kept, the left leg mirrored from the right, then `--search` (six axis directions per
blocked seeded part step, 89 + 12 directions changed), `reorder` (a screw's obstacle moved after it, a part moved before
the part it touches; covers stay last; 14 + 2 x ~10 moves) and `confirm` (a seeded step becomes `confirmed` with the run
and measurement when its path touches nothing over 60 mm, seat interference let go within the range, or its driver path
is free). Changes to the check: coarse screening solids (`screwlib.screen`, the T-c4sy9d pattern), a box test per sample,
screws as the BOM classifies them (the 24 shoulder-U `*_screw_<i>` proxies included, their thread from the BOM spec),
`with` lists, seat interference (`seat_parts_released_at_mm`) apart from path contact (`path_first_contact_mm`), and
coverage over every manifest part. Unscoped: 76.7-93.9 s, 3.2-4.0 GiB (was 76 s for the 42 right-leg steps).

| assembly | steps | confirmed | designer | seeded | parts | screws | carried (`with`) |
|---|---|---|---|---|---|---|---|
| torso | 62 | 44 | 0 | 18 | 44 | 37 | 48 |
| neck_head | 41 | 28 | 0 | 13 | 30 | 41 | 35 |
| right_arm | 41 | 26 | 0 | 15 | 26 | 40 | 40 |
| left_arm | 41 | 26 | 0 | 15 | 26 | 40 | 40 |
| right_leg | 59 | 38 | 16 | 5 | 48 | 52 | 6 |
| left_leg | 59 | 54 | 0 | 5 | 48 | 52 | 6 |
| final_body | 8 | 6 | 0 | 2 | 3 | 5 | 4 |

Coverage: 0 parts, 0 screws and 0 other hardware items in no step (was 14 parts and 8 screws on the right leg's links
alone, 587 parts without a step body-wide). Remaining seeded steps, best direction found: 41 part steps touch a part
that is not in their seat (55 have seat interference), 24 screw steps have an obstructed driver path. Registered as
proposed Idle tasks with the measurements: torso cells captured by the stage-1 honeycomb (22.5 mm, 1882-2802 mm3 per
cell; T-f7p9ke), head compute stack and the one-piece head skin (skin still 1235 mm3 into ROCK4D at 60 mm; T-dj4tpn),
arms (T-f5gcm8), legs (T-xp29jk), torso screw driver paths (T-8zj89w). The BOM shows the order per line and writes the
per-subassembly sequence (`ASSEMBLY.md`, `assembly.json`), and the chain's last three steps are the assembly check, the
BOM and the bom check.

Body definitions (2026-11-22, owner-approved, T-ghvncz). The BOM, the build sequence and the body rebuild no longer come
from hand-kept files in `robot/body/bom/` (chain.json, lines.json, defects.json, assembly_order.json and inventory.py are
deleted): each builder has an annotated `<builder>.bom.yaml` next to its script (schema `robot/body/bom/schema.json`) with
its generators, the parts it owns (line, kind, spec, supplier or `unknown`), its assembly steps (ordered by `after`) and its
defects; the BOM tool only resolves them and reports findings that name the file to fix. Migrated: 23 definitions, 693 / 693
parts claimed, 26 generators (8 section builders `rebuild: false`), 311 steps (+6 for parts added the same day, written into
the owners' files from `assembly_seed.py propose`). The derived rebuild order reproduces the chain's folders (9 produced,
8 read-only, 2 writable copies) and exposed 5 records `roll_flip_checks.py manifest` reads that chain.json never declared.
Replay with definition edits only, `robot/body/bom/` unchanged: shoulder-shells definition absent -> 16 parts unclaimed,
18 without a step, manifest without generator; the file alone restores the baseline; two parts renamed -> 2 unclaimed, 2 stale
globs, 2 stale steps; editing torso_screws.bom.yaml alone restores it. Phase 2: the checks overlay each part's settings from
its definition (part_settings.json as fallback, source per value in every report, the definition named on every worst case);
neck/head, torso and arm entries moved (39 parts, 63 blocks, effective settings identical before / after), heat sources
declared per part (34). Measured on the shells body: bom 3.8 s / 0.6 GiB (196 lines, 0 quantity differences, 46 findings:
22 printed parts without a print setup, 17 connectors without an FEM setup, 5 hardware lines without a spec, 2 settings for
absent parts); assembly 86.8 s / 3.9 GiB over 317 steps, 0 parts / screws / hardware without a step, 248 steps supported by
the measurement, 69 not (Idle T-f7p9ke, T-dj4tpn, T-f5gcm8, T-xp29jk, T-8zj89w).

## Worktrees and the generator cache (2026-11-22)

Why (from the code, git and 18 session transcripts of 2026-11-17..22): built parts did not come back with a merge
(`artifacts/` is Git-ignored; the lead copied outputs into main by hand after each builder merge); the selected body was
one file rewritten in place by whoever ran a chain step (thigh shells dropped by a rebuild that skipped two steps, a
selection overwritten by unreviewed shell cuts, a rebuild during a review); 7 different selected-manifest sha256 over 11
checkouts; main's working tree held other sessions' edits (nearly every commit staged single hunks, one fast-forward failed
silently); the roll-flip manifest script imported each section's builder by name; new worktrees cloned `artifacts/` file by
file (1083 s measured under load) and did not link `.artifact-store`, so 138,758 of the 229,054 links in `artifacts/`
dangled there.

What (owner decisions 2026-11-22): `robot/body/bom/make.py` + `gencache.py` reuse a generator's outputs from any checkout
when every file its run read has the same content (reads recorded while it runs, including STEPs read inside OpenCascade;
STEP / npz / STL by content); `compose` entries in the definitions replace the section imports; `infra/tools/worktree.py`
add / list / land / remove / prune; gc keeps what a live checkout holds plus 3 days; nothing derived is uploaded.

Measured (real runs on the selected body, 2026-11-22/23, laptop under other sessions' load):
| | before | now |
|---|---|---|
| new worktree | 1083 s (`cp -c -R`, 111,566 files), `.artifact-store` missing | 14 s (one clonefile), linked |
| selected body in a fresh worktree from main's artifacts | chain by hand (about 47 min) or copying | 3 s, 18 / 18 reused, manifest byte-identical to the checkout that built it |
| one builder changed (comment in torso_screws.py) | full or hand-picked partial chain | 75 s: torso-screws and the two roll-flip manifest passes that import it ran, 15 reused |
| make on an up-to-date checkout | | 0.8 s |
| main moved 29 commits (hip connectors, held parts, foot, neck collar) | | 15 ran, 3 reused, 1194 s, settled in one pass |
| populating the cache from nothing | | 18 generators, 2572 s, peaks 0.15-6.8 GB |
| composition by `compose` entries vs the old imports | | manifest identical (707 parts) apart from the new `composed` record; log identical |

What kept reuse from working, found by these runs and fixed: OpenCascade writes the time into every STEP header and the
manifests record raw sha256, so each rerun with unchanged geometry changed 27 manifest rows (make fixes the header time);
entries listed only outputs whose content changed (a restore would have left 290 of motor-refit's 291 files stale; cache
version 2 lists every file written); `roll-flip/report.json` timings kept four steps rerunning (merged reports are declared
`report_files`, outputs but not inputs); feedback reads (a builder reading its own previous output or a manifest a later step
writes: assembly's GLB, roll-flip-cad and neck-collar) made a fresh checkout miss (not inputs when no earlier generator writes
them); shin-cover peaked at 6.41 / 6.82 GB against the 6 GB default (declared 8 GB on main meanwhile). Still open, as
findings make prints: generators that read files a later step writes (roll-flip-cad, foot-screws, the bootstrap pass and
neck-collar read the roll-flip / shoulder-U manifests), and outputs outside the declared ones (assembly's GLB, manifest.json
and nesting audit, roll-flip report, shoulder-clevis screw meshes); the chain still needs a second pass whenever one of those
loops changes.

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
| pelvis cells 08 / 09 with insulator and fin (groups) | same | 0.83 mm from the cradle; free along +-y (0 mm3); cradle mesh note: floor under the lower cell removed, "hold that cell at its ends" until the CAD pass | real design gap (known, pending the cradle CAD pass). 22 Nov: the lower cell has no cradle wall over its whole lower half (ring at r + 0.85 + 0.8 mm: wall only at 15-60 and 120-165 deg) and the sleeve (y -27..32) is shorter than the cells (y -29..41); retention goes into the cradle B-rep pass (T-3npyjq follow-up) |
| head skin + B0196 camera (group) | same | skin: no screw (nearest compute screw 1.92 mm), free along -y and +z; B0196: camera screws enter only the camera, plate screws only the carrier and end 1.1-2.3 mm short; the 4 M2 F-F 5 mm standoffs between them exist in the sensor layout parts but not in the manifest | real gap: head skin has no mount. B0196 standoffs: artefact (T-bv7bwe): the 4 `B0196_M2_FF5_spacer_*_v23` are in the manifest (role `spacer`), but `mount_check.HARDWARE_ROLES` lacked `spacer`, so the graph dropped them; fixed, camera and plate screw join through each spacer (8 / 8 ring points) |
| B0410 camera (isolated) | same | 1.23 mm from the carrier; plate screws end 2.1-2.7 mm short; its 4 F-F standoffs are missing the same way | artefact (T-bv7bwe): same `spacer` role fix; B0410 now reaches the core by screw |
| fan stack (group) | same | 4 M3 x 15.7 screws clamp only the fan (8 / 8 ring points) and the lead guide; nearest other part 1.05 mm (Radian heatsink, beside the axis) | real design gap: the fan bolts to nothing. Fixed 22 Nov (T-3npyjq, `robot/body/assembly/held_parts.py`): the compute M2.5x30 nuts floated 2.0 mm off the bridge standoffs, where the 2 mm fan bridge plate was never modelled; plate added (4 flush sleeves), fan screwed to it by its two upper holes (M3x16); fan stack reaches the core by screw |
| head_neck_collar (isolated) | same | 0.82 mm from the head skin, 0.98 mm from the neck link; no screw; free in four directions; its `native_step` names the head skin STEP | fixed (T-a2mt33): screwed to the yaw-to-pitch link plate by 2 x M2x8 through 3.6 / 2.2 sleeves, chain to the core by contact+screw; `native_step` dropped |
| chest indicator relief (isolated) | same | 0.229 mm from the front shell (0.2 mm contact tolerance); blocked +-x 29 mm3, free +y | real design gap: no retention (bond, press or lip not specified). Fixed 22 Nov (T-3npyjq): printed with the front shell on the X2D second nozzle; rebuilt to fill its pocket (0.00 mm, 0.04 mm3 to the shell), reaches the core by contact (bonded) |
| palms (new 20 Nov), palm upper and fingers | palm lower screwed (wrist screws from bounds); palm upper and fingers contact-only | the hand servo case and horn have no mesh, so no joint edge can form | artefact: mesh-less hand servo |
| forearm front covers | contact-only | L_fore_front_M3x8_187 inside the forearm (12 points), 1.07 mm from the cover | unresolved: screw placed before the shell morph, or missing hole |
| Radian heatsink + TIM | contact-only | 0.11 mm from ROCK4D, TIM 0.0, guard 0.5 mm, no screw | unresolved: intended hold (adhesive TIM, guard clip) not in the CAD |
| Wuerth 61304021821 header | contact-only | 0.18 mm from the neck pitch case, not on a board (ROCK4D beyond 0.34 mm) | unresolved: placement looks stale |
| center back shell | contact-only | T02_back_mount1/3 clamp it per mount_check; probe 0.2 mm short on the decimated mesh | artefact: residual of the probe |
| upper arm covers (shoulder U body) | contact-only (screwed via the cover posts on the roll-flip body) | cover posts removed with the shoulder U connectors | expected until the shell redesign (T-tph7xb) |
| center front shell | contact-only on both bodies | no screw on either body; on the shoulder U body both U connectors (moving, on the shoulder pitch link) come within 0.2 mm of it at standing | real gap: shell has no mount; the U connector contact is a clearance item for the shoulder work |

After the 18:42 manifest (639d0069ed6c: B0196 / B0410 standoffs restored as spacers, 5e3e81fa0): 2 isolated (chest
indicator relief, head neck collar), 3 groups (pelvis cells 08 and 09, fan stack), 153 contact-only, 38 break points,
screw edges 172; connectivity 38 s wall. Before it: Counts (shoulder U body): 224 parts, 3 isolated, 4 groups not reaching the core (14 parts), 150 parts reach the core by
contact only, 37 contact-only break points; edges: screw 164, contact 378, joint 34, sleeve 32; 8 screws probed from bounds.
After the `spacer` role fix (T-bv7bwe, same body): 2 isolated (chest indicator relief, head neck collar), 3 groups not reaching
the core (12 parts: pelvis cells 08 / 09, fan stack), 151 by contact only (the head skin now hangs on B0196 by contact), 35
contact-only break points; edges screw 172 (+8 through the 8 F-F standoffs), contact 372.

Held parts (T-3npyjq, 22 Nov, `robot/body/assembly/held_parts.py build`, swapped in by `roll_flip_checks.py manifest`):
fan bridge plate (1307 mm3 PAHT-CF, 2 mm, sleeves 4.5 / 3.2 x2 and 3.9 / 2.7 x2): press support 83-89 % (flat), boss wall min 1.75 mm,
one 0.05 mm wall tip (2.8 mm2) where the lead-guide cut-out meets the air opening, bed contact 655 mm2; the fan's lower right screw stack
is gone (the Radian fins stand 0.6 mm in front of the plate there). The fan model's corner counterbores (3.7 mm) leave the plate sleeves
facing a pocket: mount_check reads +3.7 mm sleeve-to-fan on both plate screws. center_front_low_L/R_M2x10 (r1 front-shell screws whose
shelves motor_refit removed) dropped. R_roll_servo_case_reference: vendor solid 2 rebuilt from its outer shell (7 void shells, +17 mm3),
invalid solids 1 -> 0. Connectivity on the patched selected body: isolated head_neck_collar only; groups not reaching the core: pelvis cells
08 and 09 (2 x 3 parts); 156 contact-only, 38 break points; edges screw 185, contact 387, sleeve 33, joint 34.

Neck collar mount (T-a2mt33, 22 Nov, `robot/body/assembly/neck_collar.py build`, swapped in by `roll_flip_checks.py manifest` after `neck_parts`):
head_neck_collar is screwed to the left plate of the neck yaw-to-pitch link (x -24.5..-21, the collar's inner wall stood 1.9-3.8 mm off it at y 5-14). An inward
tab (y 5-14.2, z 394.4-405.6, +182 mm3 PETG-CF) grows to the plate face (layer gap 0.000 mm at the tab, 3.35 mm where the wall curves away), two 3.6 / 2.2 sleeves
(4.5 mm) at y 10.5, z 398.3 / 402.8, M2x8 ISO 7380-1 heads in dia 4.4 recesses whose flat floor is the sleeve end (the outer wall slopes about 15 deg; nothing
stands out of the outer surface), dia 1.6 pilots through the plate (-14 mm3 on the link). mount_check: engagement 3.25 of the 4.0 reference (shortfall 0.75: the plate
is 3.5 mm), hole offset 0.000, head seat gap 0.000, head bearing ring 0.655 mm, tip margin 0.5 mm.

Neck collar as a B-rep (T-vjw5z8, 22 Nov; owner: "fix the collar so that it is fully BOM compliant"): the collar now has a native STEP made by neck_collar.py, so the
mesh-only exception is gone. The pure ring (ruled elliptic lofts, 7845 mm3) lost about 5 285 mm3 to the head-envelope mesh and the neck-pitch notch (the notch alone about 55 mm3,
back of the ring, theta 70-117 deg, down to z 395.0 at the inner wall). Both cuts are parametric: the envelope as a ruled spline surface z = z28(theta) + slope(theta) (rho - 28)
(2 deg knots, front arc below the base clipped at z 390), the notch as a ruled surface z = z27.5(theta) + 1.2 (rho - 27.5), prisms up; knots in `neck_collar_envelope.json`,
measured by vertical rays on the old mesh (the old head envelope exists only as a mesh). Against the old mesh outside the tab zone: mean surface distance 0.017 / 0.024 mm
(new to old / old to new), p99 0.18 / 0.28 mm, max 0.58 mm (new to old) at the notch end; old to new reaches 2.4 mm at thin fins the 2.5 deg pose grid left in the old notch (0 volume);
volume 2734 vs 2740 mm3 (tab and bores included; the outer wall is the same ellipse cone, so the visible surface is unchanged). GProp's default integration is 8 % off on the spline faces
(2777 mm3), the record carries the volume at eps 1e-5 (matches the closed mesh to 0.01 %). Library poses: 15.7 mm3 with the neck pitch case at pitch +35 (y 25.5; baseline 15), cradle
0.3 mm3 at +35 (baseline 0.1 mm3 on its sleeves at the corner pose; the sleeves now keep 0.92 mm), head skin gap 0.883 mm (baseline 0.822). Screws, driver access (0 mm3) and press as before: counter support 73 / 62 % flat from the
head side, entry tool path 45 / 18 mm3 (the 8 mm tip is wider than the 4.4 recess), boss wall 0.79 mm, sleeve interference 0.0. Print direction measured on the STEP for the six axes (overhang beyond 70 deg / bed
contact / support standing on the part): +z 15 / 712 mm2 / 15 mm2 (only the sleeve-bore tops), -z 159 / 0 / 13, +x 243 / 0 / 113, -x 274 / 0 / 158, +y 132 / 0 / 12, -y 83 / 0 / 24; +z confirmed
in part_settings (the flange is the only flat face). walls: a 0.004 mm sliver where the flange tip meets the head-envelope cut (128 mm2 under one line). Connectivity: isolated parts 0, groups not reaching the core 2 (pelvis cells 08 / 09).
BOM row: STEP disk, producer neck_collar.py, +z (no seeded), no flags; defects.json no longer lists it.

Neck collar redesign for the checks (T-6rr3sa, 22 Nov; owner: the collar passes every design verification check). Baseline (17831ca46, library poses, all checks): engagement 3.25 of 4.0 mm,
wall 0.004 mm (127.8 mm2 under one line, the flange tip where the head cut meets the base) and a 0.055 mm gap at the tab, press entry tool 45 / 18 mm3 (8 mm tip, 4.4 recess) and counter
support 73 / 62 %, boss wall 0.79 mm, clearance 15.7 mm3 (neck pitch case at +35), 0.3 mm3 (cradle), 9.3 mm3 (head skin at pitch -30; collision_sweep), 0 sliver faces by tolerance but a 2.5 mm2 / 10 mm2 scatter of micro-edges at the tab, heat margin -2.0 C (PETG-CF 68 C against the 70 C case assumption), no FEM setup.
Changes (all B-rep, `neck_collar.py build` / `sweep`): (1) link: 1.1 mm boss on the plate's outer face (y 5-13.3, z 394.4-405.4) plus a plug in the plate's lightening window beside the pilots,
the tab grows to the boss face (x -25.6), sleeves 3.6 / 2.2 x 3.5, M2x8: thread 4.6 deep, engagement 4.25 of 4.0 (0.0 shortfall); (2) screws moved to y 9.0, z 397.4 / 401.9 (boss wall, window edge, counter disc);
(3) a vertical flat channel in the collar's outer left wall (8.1 mm wide, floor x -29.1, 0-1.8 mm deep, replaces the two dia 4.4 recesses): heads bear flat, the dia 8 press tip has a free path (entry tool 0 / 0 mm3);
(4) `neck_collar.py sweep`: what the moving head parts reach over neck_pitch -30..35 deg (2.5 deg steps, + 0.5 mm) is measured against the collar and written to neck_collar_envelope.json:
the notch is now a flat floor per theta (62-136 deg; the old notch kept a 1.2 slope that made acute tips) and the head cut drops 0.3-0.7 mm at theta 358-6 and 90-97 deg where the skin tilts in;
(5) a trim removes the footprint where the head cut leaves under 1.4 mm over the flange (sliver 0.004 mm gone), the notch floor (z 394.8 at the deepest, 0.6 mm over the bed) is the only link between the ring's halves;
(6) 0.4 mm chamfer on the tab's bed edge (elephant foot). Look change: the collar's left side shows a flat channel 8 mm wide, 1.8 mm deep at most, the flange is cut back where the head cut came down to the base, and the back notch is wider and flat;
nothing stands out. Renders: .cache/work/collar-renders/collar-before-after.png (before = HEAD build, after).
Collar after (full run, library poses, all checks incl. exact fit: no collar overlap, 0.59 mm to the cradle, 5.8 mm to the case at standing): clearance min distance 0.32 mm (case at +35), 0.58 mm (cradle), collision_sweep skin 0.16 mm3 at yaw -60 pitch -30 (was 9.3), geometry 4 sliver faces
(0.0039 mm2) and 2 micro-edges (0.08 mm) left at the tab corner and channel flank, walls 2.0 mm2 under one line (was 127.8), boss wall 0.9 mm, press support 71 / 75 % (flat tab face; limit: 2 sleeves 4.5 mm apart
need a dia 10 disc each, the collar is 11.8 mm high), overhang 10.1 mm2 (sleeve-bore tops, support 0 g in the slicer), heat margin +100 C at 1.8 MPa (material now PAHT-CF, nozzle 0.4: the collar is screwed to a PAHT-CF link; the 0.6 mm default made the thin walls worse), driver access 0 mm3, assembly steps clear, connectivity by screw, BOM row without flags.
FEM (new setup on the collar: sleeve bores at the link clamped, load = rim within 2 mm of the head skin, 5 N / 20 N down and 10 N sideways, nominal assumptions): p99 strength ratio PETG-CF 1.79 / 0.45 / 2.69, PAHT-CF 2.68 / 0.67 / 4.01; the 20 N hard press bends the C-shaped ring (displacement 50-77 mm at the free ends).
Link: p99 von Mises (adverse neck stall, PAHT-CF) 34.7 -> 30.0 MPa, strength ratio 2.54 -> 2.94, fall 3.13 -> 3.42; volume 4809 -> 4964 mm3; walls under one line 0 -> 0.

Neck collar round 2 (T-6rr3sa, 22 Nov; replaces the screw count, press, sliver, wall, overhang and collision_sweep numbers of the round above; `neck_collar.py build` / `sweep`). Remaining findings of round 1 and what
the redesign does: ONE M2x8 at the collar's mid height (y 9.0, z 399.8) instead of two stacked in z (4.5 mm apart on an 11.8 mm high collar: each dia 10 press counter disc ran off the collar top / bottom, support 71 / 75 %).
Screen of the options (counter support of the sleeve's dia 10 disc, flat + wall / link pilot or bore wall / FEM p99 strength ratio / look): (a) two screws side by side in y at mid height: at y 5.6 / 10.6 support 92.5 / 92.5 % (each disc
loses the neighbour's bore) but the boss wall round the second bore is 0.56 mm (the outer wall slopes in: the bore nearly breaks out above the channel floor) and a 0.045 mm wall at the channel end; at y 2.2 / 9.2 (7 mm apart, no neighbour loss)
the head cut is low at y 2 (collar top about 402): the bore roof breaks through the top, support 64 / 98 %, boss wall 0.20 mm with 25 % of the rays open, so (a) needs a collar taller than it is; (b) one screw + a locating key: support 98.3 %,
boss wall 2.16 mm, no look change (the 8 mm flat channel is unchanged), 1 sleeve / 1 screw / 1 pilot fewer; FEM p99 strength ratio 0.36 / 0.54 (PETG-CF / PAHT-CF, 20 N hard press; round 1 two screws 0.45 / 0.67), 1.45 / 2.16 at 5 N (1.8 / 2.7),
side push 2.19 / 3.26 (2.69 / 4.01): the single bore support is about 20 % weaker, the peak stays in the ring arch at the back (z 394.8), not at the screw, and the ridge support does not change it. A key that stands out of the tab
face into the plate's lightening slot (tried first) blocks the collar's way on (the collar slides in along y over the neck: 24 mm3 overlap along the insertion path), so the key is a ridge along y (trapezoid, 45 deg flanks, 0.8 high, z 395.6-397.6)
in a groove of the link boss (0.25 mm clearance, 3 deg of play): assembly path overlap 0 mm3, the screw driver path 0 mm3. Also: the tab runs into the ring wall (no wedge at y 14.3) and through the head cut (no flat top
grazing it); the tab is cut from a box far outside the ring so it is bounded by the ring's own outer surface (no seam edges on the flange cone); the trim of the thin flange applies to the ring and the tab alike; the flange top radius is the
cone's at z 397.4 (the 0.08 mm ledge that made five 0.075 mm micro-edges is gone); the channel flank's top edge, where the head cut rises towards the flank in a 38 deg knife edge, has a 0.7 mm fillet; the tab and ridge bed edges 0.5 mm fillets
(a fillet of the ring's whole bed outline took its 4.1 mm wide foot under the 4 mm strip width: thin strips 51 -> 250-370 mm2, and OCC cannot round the 1.4 mm high vertical foot corners); the bore has a 45 deg teardrop roof;
`sweep` takes the head poses collision_sweep uses (neck_pitch step 5 x neck_yaw neighbour grid, reduced to the distinct head-in-neck_yaw-link transforms: yaw turns collar and head together, so only the pitch matters) plus the 2.5 deg pitch grid,
and re-measures against the collar made with the stored cutters until no overlap is left (3 rounds: the head cut drop is now 0.78 mm at most, 48 knots instead of 18). The link's plug in the lightening window reaches z 402.5 (the 0.03 mm2 sliver and
0.023 mm micro-edge where it met the window's rounded top are gone), the groove edges have 0.3 mm fillets.
Before (round 1, f2573de17) -> after, head_neck_collar, library poses, PAHT-CF 0.4 nozzle: press counter support (flat + wall) 71.4 / 74.7 -> 98.3 % (flat 80.8, ridge wall 17.5, teardrop notch 1.7; the old fixed-post coverage 71 / 75 -> 80.8 %), entry tool path 0 -> 0 mm3,
sleeve interference 0.001 mm; geometry sliver faces 4 -> 0 (smallest face 0.0039 -> 0.96 mm2), micro-edges 2 -> 0 (0.08 -> 0.4 mm), acute wedges 0, unrounded corners 0; walls under one line 2.0 -> 0.0 mm2, thinnest wall 0.004 -> 0.477 mm, under two lines 12.8 -> 7.0 mm2;
overhang beyond 70 deg 10.1 -> 0.0 mm2 (floating ceilings 2 -> 0; the holes check now counts the teardrop bore as 1 partial sleeve bore, 20.8 % open rays, boss wall 2.16 mm), support in the slicer 0 g, print 506 s / 2.02 g (519 s / 2.03 g);
free sharp edges 75 -> 84 (length 695 -> 747 mm; the 0.5 mm fillets end in short creases and the ridge adds edges; OCC rounded none of the long spline rims), sharp bed corners 6 -> 5, thin features under one line 5 -> 7 (knife tips of the head-cut rim at the channel flank / floor top and the two ring ends);
collision_sweep head skin 0.16 mm3 at yaw -60 pitch -30 -> 0 mm3 (0 poses, clearance 0.013 mm; clearance check 0.32 mm case, 0.59 mm cradle at +35), engagement 4.25 of 4.0 mm (screw hole offset 0.004), thread 4.6 deep, assembly step overlap 0 mm3,
driver 0 mm3, connectivity by screw, fem_setup patches (bore 108 mm2, ridge 34 mm2, load 11 mm2), BOM findings for neck_collar.bom.yaml 0 (1 sleeve, 1 screw), connected to the core, heat no source in contact. Not cleared: fit pair collar x neck pitch case over its 120 s exact budget after 1 of 5 poses
(472 s of exact work, no overlap seen; distance 0.32 mm in clearance); link own findings unchanged by this work (press support 69-76 % on its 8 other sleeves, 171 mm2 overhang beyond 70 deg, skin overlap 2.05 mm3 at yaw -60 pitch -30, two 0.29 mm edges at z 378).
Renders: .cache/work/collar-renders/collar-single-screw-before-after.png, collar-ridge-groove-section-z396.6.png.

Torso stage interface sleeves (T-bv7bwe; cradle, stage 3, stage 4 rows above): not an obvious screw fix. The stage-1
`aux_bores` (torso/common.py) pair end to end with the carrier sleeves along y, both plain clearance sleeves (bore 2.1 /
2.6 mm, no thread). Past the stage-1 end there is no room for a nut: at the front pairs (cradle x +-17, z 271.5; stage 3
x +-17, z 288 / 338) the stage-1 sleeve end lies 0.77 / 0.48 mm inside battery cells 06 / 07 (vertical cylinders, dia 21.5);
at the rear pairs (cradle x +-11.2 / +-33.6; stage 4 x -33.6 / +-11.2, z 304 / 335) the bore runs into the valley between
cells 02-05: axis free about 3 mm at r <= 1.2 mm, cell or pocket wall at r 2.0 mm from the exit (M2 nut corners r 2.3).
Fastening them needs a design choice (threaded insert instead of the stage-1 steel sleeve, bores moved off the cells,
or another joint); the cradle's two sleeves at x +-16, z 252 have no partner on their axis.

Threaded stage-1 sleeves (owner 2026-11-22, T-bv7bwe; `robot/body/torso/torso_screws.py build`, chain step `torso-screws`).
The stage-1 sleeve of each pair becomes an internally threaded 304 sleeve of the same outer diameter and length (press fit
unchanged); screw from the carrier side, no nut; the shortest ISO 4762 length that fully engages the threaded sleeve with
its tip 0.3 mm clear of the cells, fins and boards; head seated on the outermost surface under it. Placed 8 of 16:
| pairs | screw | threaded sleeve | engagement (plan / mount_check) | tip past sleeve | tip margin to cells |
|---|---|---|---|---|---|
| cradle rear x +-11.2, +-33.6 (4) | M2 x 8 | M2, 3.3 OD x 2.8 | 2.8 / 2.75 mm | 1.85 mm | 0.95 mm |
| stage 4 x -33.6, +11.2, z 304 / 335 (4) | M2.5 x 10 | M2.5, 3.9 OD x 3.0 | 3.0 / 2.75 mm | 0.2 mm | 1.9 mm |
Skipped (recorded in `records.json` `interface_joints`): the 6 front pairs (cradle x +-17, stage 3 x +-17): the existing
stage-1 sleeves overlap cells 06 / 07 by 10.2 mm3 each side (exact mesh boolean; the chassis itself 0), only 1.9 / 2.3 of
3.8 mm of the sleeve is 0.3 mm clear; and stage 4 x -11.2 z 304 / 335: a 4.5 x 2.5 mm head on the carrier face meets the
IMU / ADC board 0.45 mm above it. The printed carrier faces stand 0.10 mm (cradle) and 0.35 mm (stage 4) proud of the
clearance sleeve ends, so the heads bear on PAHT-CF, not on the sleeve. The stage-1 sleeve part is now the shell-pass
STEP minus the 8 threaded positions (41 -> 33; its shoulder pitch sleeves stay at x +-42.4, not common.py's +-30.4).
The cradle sleeves at x +-16, z 252 are the cradle half of the front-shell lower mount (boss present, about 50 % ring
cover); the front shell's posts and its M2x10 screws sit at z 224, where the cradle has neither boss nor sleeve: a
centre-front-shell design item, not a leftover.

Checks, scoped `--parts 'combined__stage[1-4]_*'`, before -> after: connectivity contact-only break points 38 -> 35,
parts reaching the core by contact only 153 -> 147, screw edges 172 -> 177 (cradle and stage 4 now screwed to stage 1),
isolated 2 -> 2, groups 3 -> 3; screws 26 -> 34, engagement shortfall 8 -> 8 (none of the new ones), hole offset > 0.1 mm
0 -> 0 (new: <= 0.002 mm), head seat gap of the new screws 0.0 mm, sleeves without screw 48 -> 32, screws with a clamp gap
> 0.05 mm 20 -> 28 (the 8 new: stage-4 carrier to chassis 0.05-4.0 mm, cradle to chassis 0.05 mm, existing carrier
geometry); bom lines 191 -> 195 (M2 x 8 ISO 4762 x4, M2.5 x 10 ISO 4762 x4, two threaded sleeve lines x4 each, supplier
unknown), quantity differences 0 -> 0, sleeves in CAD 276 -> 266 (STEP count 274 -> 266); press: sleeves with support
< 90 % 51 -> 43 and tool obstruction 41 -> 33 (the 8 removed clearance sleeves; the threaded sleeves are not in `press`,
which reads only `*_steel_sleeves` parts); fit_class and gaps unchanged (67 pairs, 2 standing overlaps). `mount_check`
lists the chassis as a clamped hole without sleeve for the new screws: it links sleeves to parts by the `_steel_sleeves`
name only.

Stage 4 IMU / ADC carrier on the threaded sleeves (Idle T-htfaaz; `torso_screws.py build` rebuilds carrier, rail sleeves, Stage 1
and the 8 board screws; rail sites `common.stage4_rail_sites`, board sites `common.stage4_board_sites` from the boards' STEP).
The carrier's outer face is the boards' back face (y 42.0, layer gaps 0.10 -> 0.00 mm). The two x -11.2 rail pairs moved from
z 304 / 335 (under the IMU z 295.1-312.9 and ADC z 330.1-347.9 boards) to z 316.5 / 326.5 between the boards, so main's two
skipped pairs are now placed: Stage 1's old bores there (dia 3.98) are filled outside the battery pockets, new bores of the same
size drilled at the new sites, the old clearance sleeve solids dropped and threaded sleeves added (Stage 1 body edited from the
shell-pass STEP here; no chain script yet). All 6 rail pairs: M2.5 x 10 ISO 4762, threaded sleeve M2.5 3.9 OD x 3.0 (span
y 32.1-35.1), head on the carrier face (flush sleeve end, y 42.0, head seat gap 0.0), engagement 3.0 mm in the plan (2.75
mount_check), tip 0.1 mm past the sleeve exit, rail screws 1.10 mm from the cells (reviewer measurement; not 2.0). Carrier rail holes carry full-thickness flush clearance
sleeves (y 35.4-42.0, bore 2.6). The 8 board screws: M2 PA2 x 6 into printed pilots (dia 1.6) in bosses on the plate's rear
face, engagement 3.93 (IMU) / 4.17-4.25 (ADC) of 4.0, pilot depth left past the tip 0.62 / 0.38 mm (that is the pilot, not a clearance; the ADC tips are 0.108 mm from fins 02/03, the IMU tips 0.39 / 0.44 mm). Measured on the shoulder-u manifest, main
(dbe8201ca) -> this branch: screws 28 -> 30, gaps > 0.05 mm 28 -> 18, sleeves without screw 23 -> 19, engagement shortfall
8 -> 8 (shoulder), hole offset > 0.1 mm 0 -> 0; carrier press: sleeves 14 -> 6, tool obstruction 6 -> 0, anvil 14 -> 0,
counter support < 90 % 4 -> 6 (flat 60 % at the 5.6 mm rail end), free depth 0 -> 40 mm; carrier walls min 0.81 -> 0.81 mm;
driver path blocked by center_back_shell on all screws (5.8-14.5 mm3), as before; Stage 1 walls min 0.004 -> 0.004 mm (under
1 line 6374 -> 6318 mm2), wall around the rail bores 0.46-0.48 mm at the sleeve inner end at the new sites as at the old;
fit overlaps 3 -> 3 pairs, identical (own sleeves 0.498 mm3, both pitch cases 5430 mm3); FEM p99 PAHT combined stall
15.126 -> 15.157 MPa (mesh h 2.2, 36468 nodes; the mesh differs, ratio and peak not re-read), strength ratio 5.82 -> 5.82; BOM lines 196 -> 193 (M2.5 x 10 and threaded
sleeve M2.5 4 -> 6, rail clearance sleeves 3.9 x 2.6 x 5.8 / x 3.0 and the 8 board sleeves 3.2 x 2.1 x 2.2 dropped,
3.9 x 2.6 x 6.6 x 6 added). The moved rail bores
(x -11.2, z 316.5 / 326.5) are stepped: sleeve bore r 1.99 to y 31.9, tip clearance bore r 1.5 on to y 31.4, so the M2.5 x 10
tip (y 32.0) stays 0.6 mm off the floor (ISO 4762 length tolerance +-0.29) and the floor keeps 0.72 mm to the battery pocket
at r 1.5 (0.38 mm at the r 1.99 rim, as before); a full-radius bore to y 31.4 broke into the pockets (0.0057 mm3 per site).

## Whole-body run of the other checks (2026-11-22, T-zfejzf)

Every check outside screws / sleeves / fits, connectivity, fem and slicer, run unscoped on the selected body
(`assembly-parts-shoulder-u-shells.json`; scope = 193 non-hardware parts, 62 of them printed, 42 with a print direction) under
memguard 6 GB (clearance 8 GB) and `caffeinate -i`, 4 batches in parallel with other sessions' jobs. Wall times are of the batch
child, without memguard's slot wait; peak = the child's resident set. Counts only: the findings are not triaged here (owner
2026-11-22: reliability and data first; the design triage is in the Idle children of T-zfejzf).

Why the first unscoped runs looked stalled (13:40-14:12). Not the code alone: the laptop went to sleep twice during the runs
(pmset: 13:41 and 13:58:23-14:11:49, 806 s), and the part each job was on when it slept carried the time (heat on
`center_front_shell` 1001 s, walls on the stage1 chassis 804 s, overhang on the stage3 carrier 995 s, all 0.1-17 s awake). Long
runs go under `caffeinate -i` (as the body rebuild). Real causes, found from `CHECKS_STACK_DUMP_S` dumps and a per-part timing run,
fixed in `1599e4209`:
- `geometry` B-rep audited every part with a STEP, unscoped also the 52 bought parts (supplier STEPs up to 80 MB: power supervisor
  board, ROCK4D 45, ST3025 cases 15 each, thousands of component solids): no part finished in 1800 s, the stack sat in the edge loop
  (`_face_normal_at`). It now audits printed parts and made roles (cut stock, produced backing / holder, manufactured) and lists the
  bought ones with their STEP size (`bought_not_audited`). It also died on the first of 23 scoped `native_step` paths that resolve
  nowhere (FileNotFoundError): now counted (`dangling_native_step`) and skipped like a mesh-only part.
- `gaps` measured every pair twice when both parts are scoped (2322 evaluations, ~400 s warm): memoised.
- `clearance` re-measured every pair at every pose, also pairs whose two links did not move relative to each other (torso parts over
  leg poses). Such a pair now takes its standing result (overlap and distance are invariant under a common rigid motion; checked
  against the old code on the chassis + R shoulder connector: identical except one no-finding row at the 5 mm cap that the world-axis
  box prefilter had let through when the shoulder pitch turned both links), 32 -> 5 s there.
- `bed` crashed the whole check on the pelvis cradle (seeded +z: only 0.001 mm2 faces the bed, `min()` of no islands): now a row with
  0 mm2 contact.

| Check | Wall s | Peak GB | Counts (bad events) | Worst items |
|---|---|---|---|---|
| geometry | 56 (45 check; before the fix: none audited in 1800 s, then a crash at 32 s) | 1.6 | 51 audited, 52 bought not audited, 23 dangling STEP; 0 invalid solids, 270 sliver faces, 456 micro-edges, 66 acute wedges, 160 G1 steps, 244 unrounded corners, 0 over build volume, 10 parts with part/sleeve overlap > 0.05 mm3, 8 severed load-path branches, 2 STEPs without a solid | palm lower R / L: STEP holds 326 faces, no solid; forearm R / L 80.6 mm3 part-vs-own-sleeve overlap, L hip yaw-to-pitch 1.13 mm3, neck yaw-to-pitch link 0.77 mm3 |
| mass | 5.5 (24.8 when it builds the solids) | 2.6 | 37 printed parts compared, 0 without material | manifest masses are volume x 1.22 g/cm3 (legacy fast_check), PAHT-CF is 1.06: -63.1 g over the 37; chassis 95.69 vs 83.28 g, foot monocoque 36.77 vs 31.94 g each |
| print_setup | 0.0 | 2.9 | 62 printed: 20 without print direction, 18 -> 8 without material, 22 without STEP, 28 directions and 24 materials seeded | still without material: centre front and back shells, head skin, chest indicator relief, 4 foot boot / sole shells |
| fem_setup | 69 | 3.1 | 42 printed parts without a setup, 1 setup not checked, 2 supports sharing nodes with the load, 13 setups on a seeded print direction, 2 disagreeing with `layer_normal_axes`, 0 empty patches | smallest patches 308 / 310 mm2 (shoulder roll-to-yaw roll-case support) |
| gaps | 146 | 2.4 | 135 parts overlapping at standing | foot boot shell R / L 0.0 mm to the foot monocoque |
| static_audit | 2.4 | 2.4 | 60 overlapping pairs at standing | foot boot shell vs monocoque 10973 / 10570 mm3, foot sole shell vs sensor pocket pad 6526 mm3 each side |
| shell_offset | 2.3 | 2.4 | 2 shells with a `covers` setting | thigh shell back R 0.11 mm, front R 0.16 mm min inner-wall offset |
| heat | 45 | 2.3 | 62 parts, 20 in contact with a source, 41 within 3 mm, 41 contact / 89 near pairs | margin to HDT 1.8 MPa -2 C: thigh shell front R / L (2.5 mm from the knee ST3025) and, once seeded PETG-CF, the 4 upper-arm covers; every PAHT-CF part >= +85 C |
| bed | 28 | 2.7 | 42 parts: 1 with small islands, 258 sharp contact corners, 12 with the centre of mass outside the contact hull | pelvis cradle 0 mm2 on the bed (seeded +z), fingers 10.7 mm2, head compute carrier 15.7 mm2 in 2 islands |
| walls | 3.3 (7.8 after) | 2.7 | 62 parts: 18 -> 8 without line width, walls under 1 line 31 -> 35, under 2 lines 44 -> 49, gaps under 1 line 26 -> 29 (the 10 seeded covers now measured) | stage1 chassis 0.004 mm wall, 6374 mm2 under one 0.62 mm line; pelvis cradle 1995 mm2 |
| small_features | 19 | 3.2 | 42 parts: 315 thin features under 1 line, 550 under 2, 389 Z steps under a layer, 1 internal void, 35 parts with support standing on the part | stage1 chassis 0.011 mm feature, 980 mm2 of support on the part |
| overhang | 42 | 3.5 | 42 parts: 40 with area beyond the limit, 18 bridges over the limit, 192 floating ceilings, 35 with support on mating faces | stage3 carrier 5593 mm2 beyond 70 deg (seeded -y), chassis 4210 mm2, head compute carrier 3781 mm2 |
| edges | 43 | 3.9 | 40 parts: 11374 free sharp edges, 2823 on mating faces, 4074 sharp corners, 200 soft kinks | chassis 8865 mm of free sharp edge, head skin 6667 mm |
| elephant_foot | 3 | 3.9 | 42 parts: 25 with a mating bed edge, 20 unchamfered, 107 unchamfered runs | stage4 carrier 155 mm unchamfered on its own sleeves' wall, shin R / L 79 mm |
| clearance (`--poses key`) | 200 (184 check) | 3.9 | 82 poses (leg key + arm / neck / torso region), 23029 pairs: 290 pairs with new overlap, 393 pose-pair events over 20 mm3 | R palm lower vs R shin 2404 mm3 (key: leg out), L sensor pocket pad vs R foot roll case 1475 mm3; clearance keeps cross-limb pairs, the sweeps leave them out |
| render | 12 | 0.5 | 0 render errors | six-side drawing written |

Screening solids more than 5 % off their reference in every batch: upper L / R pitch C047 case +25.1 % (the native STEP is the same
shape 12.0 mm off in x, so it is rejected and the open mesh, 41252 vs 32983 mm3, is screened; needs `native_step_offset_mm` +-12 mm in
the manifest), stage1 chassis sleeves -7.0 % (STEP 31 solids 603.6 mm3 and mesh agree; the recorded `volume_mm3` 643.45 predates the
removed sleeves). Both sent to the BOM session. Data items sent on: the 31 dangling `native_step` paths of the manifest
(foot sensor stack 18, cut actuators 5, forearm cover nuts 8), the legacy-density masses and the palm STEPs to the BOM session
(T-ghvncz); part / sleeve overlaps to the screws and fittings work. Seeded in `part_settings.json` (`e5c670637`): Bambu PETG-CF on
the 10 arm and shin covers whose manifest text names PETG, marked seeded; print directions left to the designers. Reports:
`.cache/work/cad-speed/checks/triage-{A,B,C,D,E,F,G}/report.json` (recreated by the commands above).

Rerun after the stage-4 carrier, head carrier and ankle coupler merges (same manifest path, sha 314067c7e493, 22 Nov evening): every
check ran again without a stall or a failure (batches 325 / 254 / 824 s wall incl. 10 / 81 / 627 s slot wait, peaks 3.9 / 4.5 / 4.0
GB; geometry 68 s, gaps 138 s, clearance 186 s). Counts that moved: sliver faces 270 -> 296, bed sharp contact corners 258 -> 290
and centre of mass outside the contact hull 12 -> 14, thin features under 1 line 315 -> 320, overhang bridges over limit 18 -> 16 and
floating ceilings 192 -> 201, free sharp edges 11374 -> 12059, unchamfered mating bed runs 107 -> 93 (19 parts), mass difference
-63.1 -> -62.8 g; the rest unchanged. New screening solid off by more than 5 %: R / L ankle coupler sleeves +6.3 % (closed mesh vs
reference). Reports `triage-re{1,2,3}`.

## Reliability of the check tool itself (2026-11-22)

Why: sessions kept repairing `run.py` instead of using it. Transcripts of 2026-11-17..22 (136 sessions and subagents,
about 525 invocations): memguard stopped run.py 49 times (39 timeouts, 10 memory caps, about 17 h of wall time in the timed
out runs); one session lost 4.5 h to runs that stalled with no output and no cause found; about 15 commits repaired the tool.
Recurring modes: a killed run printed nothing and its report gave no hint where it stood; one input that resolves nowhere
(a STEP, an empty B-rep) blanked a whole check for every part in scope; reports replaced by a run under the same label (macOS
paths ignore case); worktrees without `.venv` / `artifacts` or with cold caches; laptop sleep stretching parts to 800-1000 s;
timeouts guessed without knowing what a run would cost; light checks waiting in memguard behind heavy ones.

How it was judged: `robot/body/checks/evaluate.py` runs the real run.py under memguard on the selected body
(`assembly-parts-shoulder-u-shells.json`) and on copies with one injected fault, against main `a88fc4d78` (baseline, a frozen
worktree) and the changed tool, on the same artifact snapshot, with other sessions' jobs running (load 10-20 on 10 cores).
Seconds are of run.py itself; memguard's slot wait is apart. Ground truth for the injected interference is computed outside
run.py: exact OpenCascade common volume and distance of the native STEP solids at standing.

| Scenario | Baseline | Now |
|---|---|---|
| light iteration, `R_thigh_roll_flip` (`light`) | 19.8 s | 14.5 s |
| light, arm connector `R_shoulder_pitch_to_roll_combined` | 13.5 s | 16.6 s |
| candidate (`light`, `--poses library`) | 23.8 s | 22.8 s |
| one part, no `--checks` (all 29 checks) | 665 s, 4.95 GB | 462 s (screws 185 -> 6 s: shared registrations) |
| Hip section, 26 parts, light per-part checks | 56.7 s | 55.6 s |
| whole body unscoped, 193 parts, 6 per-part checks | 272 s | 260 s |
| bought board with a 44 MB STEP in scope (ROCK4D) | 15.5 s | 14.5 s |
| new worktree, first light + screws run on one part | 337-343 s cold | 28-29 s (shared caches) |
| same run twice | identical measurements | identical |
| part moved 3 mm into its hip neighbour (exact truth 22.09 mm3) | gaps 20.82, static_audit 20.78, clearance 20.80 mm3 | the same values |
| STEP rebuilt in place 2 mm lower | new geometry measured (sleeve overlap 0.2 -> 88.8 mm3, knee case 0 -> 213 mm3) | the same |
| one of two scoped parts with a missing STEP | `holes`, `press` lost for the healthy part too | healthy part measured by all 7 checks; the bad part listed per check with cause and code line |
| scoped STEP truncated | `geometry`, `press` fail with a traceback | the same two fail, one line each naming cause and code line |
| memguard timeout during `fit` | killed at 40 s, nothing printed | stops at 20 s before the deadline, exit 124, finished checks kept, the stopped one with its stack (`import_step` <- `brep` <- `fit._shape`), summary printed |
| two runs on one label at once | both exit 0 and print the same report path; the file holds only the second run's part | the second stops at once naming the holder (pid, command); the first's report intact |
| labels differing in case, one after the other | the second replaces the first silently | lower-cased, the second says which report it replaces |
| `--changed-since last-run`, nothing changed | exit 1, "selects no part" | exit 0, "nothing changed ... not an error" |
| typos (part, check, section) | exit 1 in 4.2 s with the known names | the same |

What changed: per-part fault isolation by bisection in the check child (`failed_parts`), a child crash costs one check and
the group goes on in a new child, the stop before memguard's deadline (memguard exports `MEMGUARD_DEADLINE`) and on
SIGTERM / Ctrl-C with the Python stack of the stalled call (faulthandler), atomic report writes, the label lock, the idle-sleep
assertion (`caffeinate -i -w`), the plan line with an estimate from earlier reports and the checks that may not be reached,
check presets, content-addressed caches shared by all worktrees with atomic writes (`checks/cachefs.py`; the old writers
could leave half-written `.npz` and `sha256.json` for a concurrent reader), `infra/tools/worktree.py`, and memguard admitting
by resident memory (owner 2026-11-22; at the time 4 jobs used 9.3 GB of 24 GB caps). Measurements of the checks are
unchanged (truth rows). Not covered by a scenario: a child crash in native code (no fault found that triggers one on demand),
the memory-cap path (memguard's own), and the exact `fit` / `fem` numbers beyond the baseline equality.

Follow-up the same day (owner 2026-11-22): no job count in memguard, as many runs as fit in the 24 GB, and a result cache
(`checks/resultcache.py`). memguard counts a running job at its full `--limit-gb` for its first 60 s, then at the peak memory
of its process tree (without the warm-up, ten jobs started together would all fit at 0.1 GB each). The cache stores each
check's result with the inputs recorded while it ran (files read by the check child and its Python subprocesses, STEPs
handed to OpenCascade / gmsh, the set of body definition files, the run options, the code it loaded) and reuses it while every
recorded file keeps its sha256; up to 8 input versions per check and scope, so going back to an earlier candidate finds its
results. Measured on the selected body:

| Scenario | Result |
|---|---|
| light run on `R_thigh_roll_flip`, again with nothing changed | 17.6 s computed, 5.2 s cached, identical results |
| fem + screws + fit_class + driver_access on the same part, again | 133.7 s computed, 5.2 s cached, identical results |
| a neighbour's / the part's own STEP and mesh rewritten in place 1.5 mm closer, manifest byte-identical (hip pitch-to-roll connector, knee ST3025 case through its moved path, front thigh shell, the part itself) | 0 of 24 check results differ from a `--no-cache` run; the checks that read the changed files ran again, `clearance` stayed cached where the changed part is on the scoped part's own link (not paired) |
| the files restored | all 6 checks cached again (earlier input version) |
| six light runs started at once on six parts, other sessions' jobs running | all done in 65 s, slot waits 0-26 s, peak resident memory of every guarded job on the machine 7.6 GB |

The light checks share one child, so each records what that child read up to its end: a neighbour's edit also reruns
`geometry` and `mass` of the scoped part (seconds; never stale). Heavy checks and each FEM part run in a child of their own,
so their inputs are their own. 5.2 s is the floor: start-up, plan and the input hashes.
Found on the way, fixed: the owner six-side drawing (`render`) and leg_motion's closed solids cached their meshes under the
`mesh_sha256` recorded in the manifest, so a mesh rebuilt in place was drawn where it had been: a 15 mm shift of
`R_thigh_roll_flip` changed 0 of 2.5 M pixels; keyed by the file's own sha256 (`hip-shell/kit.py mesh_tag`) the same shift
changes 62-64 thousand, and a stored render is not served after the edit (`render-mesh-edit`). The scenarios that rewrite
artifact files in place (`cache-stale-hunt`, `render-mesh-edit`) refuse to run in the main checkout.

## Tool defects reported by the body sessions (2026-11-22, T-vftxyn)

Seven sessions (hip, torso, whole-body triage, screws, floating parts, shoulder, BOM) sent about 50 tool defects; owner:
tooling fixes go to one session. Each item was first reproduced on the selected body with run.py (or shown not to reproduce),
then fixed and rerun; the reproductions are scenarios in `robot/body/checks/scenarios/` (runner, pairs, part_checks,
fasteners, fem), run with `evaluate.py`. Before -> after, measured:

| Item (task) | Measured |
|---|---|
| selected body stale against its upstream, nothing warned (torso; T-aprdsc) | the plan names each recorded source that changed and each upstream manifest written later: selected body 5 findings, a consistent roll-flip manifest 0, the regenerated shoulder-U manifest 0 |
| stale recorded volume (triage) | injected 643.45 mm3 on the stage-1 sleeves: -7.0 % reported before, nothing after (fused STEP reference); the reported case no longer reproduced, the torso-screws merge had rewritten the volume |
| STEP rejected silently (triage) | pitch C047 cases: "frame or manifold mismatch" -> "moved by [+-12.0, 0, 0] mm (all six box faces alike)", +25.1 % still listed |
| a tool exception remembered as an unusable STEP | one part's STEP skipped since a `savez` bug: such entries no longer remembered; solid v3 re-evaluates the 35 earlier rejections once |
| checks/data ignored by git (BOM) | a bare `data` line in .git/info/exclude (written by worktree.py): anchored, `git add` works again |
| big shells pull the body in (shoulder; T-fdczae) | does not reproduce: the shoulder's scope (both torso shells, both upper-arm covers, 4 checks) 386 s complete; a surface-sample neighbour test trimmed neighbours 68 -> 50 / 94 -> 62 for no time gained (384 s, same results) and was dropped |
| manifest outside the repository (floating parts) | refused -> measured (absolute path in the report, result cache off) |
| voxel-filled overlap shown as real (floating parts) | fan bridge plate x fan 13.8 mm3: the line now says `[approximate: SameSky_4010_fan: voxel fill of an open mesh]` |
| memguard: big jobs could starve under memory-only admission | a job starts ahead of the longest waiter only if that one still fits after it |
| clearance kept cross-limb pairs (triage, T-xfkwv2; T-g72tnq) | whole body: 290 pairs with new overlap -> 37 (7764 cross-limb pairs left out, counted, listed); R palm x R shin 2403 mm3 -> left out |
| D-15 hip range corners as design overlaps (hip) | thigh shells collision_sweep: worst design case 818 -> 57 mm3; the corner contacts (818, 791 mm3 vs Honeycomb_Cell_08) listed in `controller_limit_contacts` |
| driver paths 200 mm away "closing in" (hip) | `closed_in_by` within the 40 mm reach, distances given; farther parts apart |
| collision_sweep other limbs, fit_class seeded cross-link, fit progress (hip) | do not reproduce since 863102d49; fit prints per pose, stops pairs over budget, a killed run reads unfinished with its stack |
| stale load-path slices (shoulder; T-r96hnj) | shoulder pitch-to-roll 3 severed / 0 mm2 -> 0 / 75.5 mm2 with 4 of 12 slabs counted as outside the part (the data is the shoulder session's) |
| teardrop / slot bores as boss wall 0 (hip, shoulder) | walls at 0 mm: 8 -> 0 (1.15-3.20 mm, open share reported) |
| mesh-only parts skipped silently (floating parts, hip) | geometry measures 4 shells from their mesh; press 3 parts (cradle sleeves 0-5 equal the STEP numbers); holes lists them as not measured |
| sleeve overlap from coincident tessellation (hip yaw) | R / L hip yaw-to-pitch 0.42 / 1.13 -> 0.0 / 0.0 mm3 (exact OCC, matching an independent truth) |
| bed 0 mm2 at -x, +-z crash, floating bridge at 0 mm, tapped bore as open support | do not reproduce on the current body (148.7 mm2 at -x, slicer prints it); bed now reports all six directions |
| ROCK4D hole offsets (screws; T-vza9m4) | CS10_2 / CS10_3 / CS30_0 / CS30_1 0.118 / 0.244 / 0.795 / 0.224 -> 0.028 / 0.0 / 0.006 / 0.008 mm |
| coplanar / board gaps, layer pairing, clamp stack (screws, torso stage 3, floating parts) | head-to-sleeve 0.44 -> 0.0 mm; Power_supervisor H2 1.595 -> 0.005 mm; fan plate 3.7 mm and H-cage shell posts 5.4 / 4.4 mm -> gone |
| screw identity (T-fdvj93) | shoulder-U proxy screws seen by screws / driver_access / fastener_motion 0 -> 12; ISO 4762 screws PH0 -> 1.5 mm hex; H-cage shell obstruction 14.8-16.0 mm3 -> reported apart |
| R ankle output registered as the horn (T-fbpzck) | rms 0.213 -> 0.048 mm, all 8 holes, all 16 ankle screws in defined holes |
| bonded / co-printed edge (T-5ptw44) | `joined` edge from `joined_to` on a definition entry; waits for the schema property and the chest indicator's entry (BOM session's files) |
| FEM released members, empty patch, progress, collars (T-r922sr) | two released idler members: 324 -> 654 nodes released (results = stored setup); singular matrix -> a named error in 0.2 s; 131 s silent -> a line at most every 13.5 s; collars from the bore (head carrier ratio 3.58 -> 3.57: its bores equal the sleeve OD) |

Integrated run on the merged code (all 44 truth / robust scenarios of every area in two shards in a fresh worktree, under
load 15-30 from other sessions' jobs): it found one real defect the per-area runs had not: `mass` took the solid centroid of any
part another check had loaded in the same process, so mass alone and mass after gaps / clearance differed and the result cache
served one for the other (cache-stale-hunt: 2 stale results); fixed (d37241152), then 0 stale and `check-context-independence`
gives identical results for each light check alone and grouped. Three checks stalled for 970-1010 s once (geometry, clearance,
screws on the fan plate) and took 4-16 s when rerun at once; the stack named plain computation each time; each check now
reports its CPU seconds next to its wall seconds (`cpu_s`), so the next stall says whether it computed or waited. The first
whole-body run after the solid v3 bump took 1547 s (scoped solids 804 s: the earlier rejections re-evaluated once); the next
365 s (CPU equal to wall; geometry 56 -> 79 s for the mesh-only parts and the exact sleeve overlaps).

Left with their owners: stale load-path slice ranges (shoulder), the `joined_to` schema property (BOM), shoulder_clevis._density
still reading part_settings.json (shoulder: the selected body cannot be regenerated at HEAD), seeded single-horn assembly steps
(BOM), the ankle-coupler hole stopgap (lower leg), the clamp width vs clamped part measurement (hip: proposed).

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

Unscoped run (2026-11-20, T-c4sy9d). `run.py --checks connectivity,screws,bom` unscoped on the shoulder U body hit
memguard's 1800 s timeout in `screws`. Stack dumps every 12-45 s over 200 s all sat in `screwlib.screws_of_scope` ->
`ctx.solid()` -> `core.step_mesh` (build123d tessellate): the selection asked for the fine 0.03 mm STEP tessellation of
every scoped part, the cause found for connectivity. Its coarse watertight solid is now `screwlib.screen()`, shared by
both. With that fixed the selection takes 30 s, and the next stage passed 6 GB in 48 s: `mount_check.run_screws` took
all 229 selected screws in one process, holding the full mesh of every part within driver reach of any screw (in practice
the whole body, the 587k-face power board and the shells among them) and two embree scenes over them. The check now runs
mount_check in chunks of 30 screws per process, as `mount_check.run` already does for the whole manifest (embree leaks per
scene). Not a per-screw reload and not an O(screws x parts) loop: candidates are bounds-filtered per screw. Timed
unscoped, shoulder U body (manifest 639d0069ed6c): 124 s wall, selection 30 s, mount_check 85 s in 8 chunks, tree peak
5.5 GiB under a 6 GB cap; 218 screws analysed, 47 with an engagement shortfall, 32 with a hole offset over 0.1 mm, 121
with a clamp gap over 0.05 mm, 125 sleeves without a screw, 87 mount-hole faces with unused holes.

Other checks unscoped (2026-11-22, T-c4sy9d). `assembly` did not stall (74 s for its one assembly, the right leg,
42 steps), but stack dumps showed the same two traps: `_screw_info` probing screws on `ctx.solid()` (0.03 mm STEP
tessellation) and `run.py`'s reference volume fusing every scoped multi-solid STEP. Every check that calls `ctx.solid()`
on its scope or on what lies near it inherits them in an unscoped run: on the scope `gaps`, `mass`, `clearance`,
`static_audit`, `fits` (printed parts only: `edges`, `holes`, `walls`, `heat`, `overhang`), on neighbours or paths
`driver_access`, `fastener_motion`, `range_sweep`, the `assembly` insertion booleans. One rule in
`core.screen_solid` and `run.py` covers all of them: a part outside the STEP roles (bought) with a STEP over 5 MB takes the
coarse path even when scoped (power supervisor 78 MB, ROCK4D 44, B0196 26, B0410 22, ST3025 outputs 12, precharge board
9.5, foot roll cases 7.8, head skin and collar 5.5, microphone 5.3). Printed parts and servo cases keep the fine
tessellation. `assembly` selects screws on `screwlib.screen()`. A `native_step` that resolves nowhere
(`R_toe_lateral_sensor`) crashed unscoped `mass` in `sha()`; it now counts as no STEP and the mesh is used. Timed
unscoped on the shoulder U body: `assembly` 29 s (was 104 s; one more screw not named, 9 vs 8, from the coarse selection),
`mass` 83 s (crashed before; solids for 192 parts), peaks 1.8 / 3.4 GiB. The first run after a servo STEP changes still
fuses its reference volume once (cached afterwards).

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
