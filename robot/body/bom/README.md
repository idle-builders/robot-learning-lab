# Robot BOM and assembly order: start here for hardware

Everything needed to build the selected APE0 body starts here: what to print and buy, from which scripts and files,
with which print setup, and in which order it goes together (owner 2026-11-20 / 2026-11-22, Idle T-4jjry4, T-ghvncz).

**The BOM tool knows no part.** What goes into the BOM is decided by the agents working on the body, in one annotated
definition file per builder, next to its script: `robot/**/<builder>.bom.yaml` (schema [schema.json](schema.json)). The
tool only finds those files, resolves them against the selected body (`body_manifest` in
[robot/design.json](../../design.json)), adds what it can read from files, writes the BOM, and reports what is missing,
naming the definition file to fix. Nothing is filled in for you.

## Regenerate

```bash
.venv/bin/python -B robot/body/bom/build.py            # seconds; --publish also updates the design dashboard BOM
python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/checks/run.py --checks assembly --label assembly
python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 900 -- .venv/bin/python -B robot/body/checks/run.py --checks bom --label bom
.venv/bin/python -B robot/body/bom/publish_body.py --publish   # main checkout: the selected body's dashboard record (below)
```

## Bring a checkout's body up to date (make.py)

```bash
.venv/bin/python -B robot/body/bom/make.py status     # per generator: reused or runs, and why; runs nothing
.venv/bin/python -B robot/body/bom/make.py            # restore or run every declared generator in order, here
.venv/bin/python -B robot/body/bom/make.py --report   # ... then build.py, the bom check and the findings that changed
```

Same in main and in every worktree (`infra/tools/worktree.py`). A generator's outputs are reused, from whatever checkout,
branch or commit made them, when every file its run read still has the same content here; else it runs under memguard and
its outputs go into the shared cache ([gencache.py](gencache.py): reads recorded while it runs by
[gensite/sitecustomize.py](gensite/sitecustomize.py), including STEPs read inside OpenCascade; STEP / npz / STL compared by
content, not header bytes). Folder listings count as inputs, except the definitions' search over `robot/`, which counts
as the set of `*.bom.yaml` paths (untracked leftover folders in one checkout no longer stop reuse in another).
`make.py --rerun ID[,ID]` runs those generators even when an entry could be reused (after a change to what make records),
the ones after them reused when the outputs match. `make.py gc` deletes entries no live checkout holds after 3 days, and
their blobs. Findings printed per run: a file read that a later generator writes, outputs outside the declared ones.

## Composed manifests (compose.py)

A builder that edits a manifest another generator composes declares it in its own definition; the composing generator
declares `composes: STAGE` and calls `compose.apply(STAGE, manifest)`:

```yaml
compose:
- id: torso-screws
  into: roll-flip                      # the stage (roll_flip_checks.py manifest composes roll-flip)
  call: robot/body/torso/torso_screws.py:swap_parts
  takes: parts                         # parts: f(parts) -> parts | manifest: f(manifest) -> log lines
  after: [ankle-coupler-screws]        # compose ids applied before it
  needs: [torso-screws]                # generators it reads: they run before the composing generator
```

## The dashboard record (publish_body.py)

The design dashboard's view of the selected body is generated from the same source, so it cannot drift: `publish_body.py`
(no part or study knowledge, like build.py) builds the record of a study from its `study` block (in any definition; the
selected body's is [robot/body/complete_body.bom.yaml](../complete_body.bom.yaml), the study selected for Whole robot in
robot/design.json), the manifest, every definition, the BOM and assembly order of `build.build()`, and the newest complete
check report per check and part on the same manifest sha256. The record's `body_view` section feeds the study page (the
subassemblies in assembly order with their steps; per part its BOM line, STEP / mesh and how they resolve, generator and
regenerate command, assembly step and direction, print setup, check worst cases, BOM findings and the definition file to
fix), the CAD panel's BOM table and the part card shown when a CAD part is clicked; `design.bom` is the same BOM for robot
BOM & costs. Without `--publish` it writes `.cache/work/bom/<study_id>-record.json` (the rebuild copy does only this; it
refuses `--publish` outside the main checkout). `--publish` packages the display GLB
(`artifacts/robot/body/selected/assembly.glb`, one named node per manifest part) under memguard when the manifest changed,
then runs `results_hub.py publish`; the record's sources (definitions, generator scripts, these tools, robot/design.json)
must be committed: their last commit is the version. `--study ID` publishes another declared study (e.g.
`shoulder-shells-parametric`, scoped to its own definition's parts); `--definitions F... --manifest M` builds from copies
(replay tests). Study text changes go into the study block, never into a publisher.

Outputs (derived, `.cache/work/bom/`): `BOM.md` (the builder's page, findings first), `ASSEMBLY.md` (build sequence per
subassembly, with the assembly check's measurement per step), `findings.json`, `bom.json`, `assembly.json`, `bom.csv`,
`dashboard-bom.json`. The body rebuild (`rebuild_chain.py setup && run && compare`, isolated under
`.cache/work/body-rebuild/`) runs the declared generators in the order their inputs and outputs give, then the same three
commands and the dashboard record (to a file).

Dashboard: `.venv/bin/python -B robot/body/bom/publish_body.py --publish` (main checkout) publishes the selected-body record
from a consistent snapshot: the definitions as committed at HEAD and a manifest whose files hash-match what it recorded.
Other sessions' uncommitted work is listed on the record as not included and never blocks it; the record's version is
that commit. A manifest file that no longer matches its recorded sha256 is a finding (`file_changed_since_manifest`)
naming the definition. `rebuild_chain.py setup --commit HEAD` builds the isolated copy from a commit the same way.

## The definition file

```yaml
# robot/body/leg/foot_screws.bom.yaml: comments are welcome; the owner of foot_screws.py maintains it
schema: cove50-body-definition/1
definition: foot-screws            # unique id
owner: leg screws session (T-rptay5)
generators:                        # what the builder writes from what; the rebuild orders and runs these
- id: foot-screws
  command: [.venv/bin/python, -B, robot/body/leg/foot_screws.py, build]
  inputs: [artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json, robot/body/fastening/connector_screws.json]
  outputs: [artifacts/robot/body/leg/foot-screws/records.json, artifacts/robot/body/leg/foot-screws/parts]
  after: [shell-pass]              # only where the files alone do not order it (several writers of one file)
  resources: {limit_gb: 6, timeout_s: 900}
  # rebuild: false                 # a section builder whose outputs the rebuild links read-only
parts:                             # which manifest parts this builder owns, and what each is
- match: [R_foot_monocoque_combined, L_foot_monocoque_combined]    # names or globs
  kind: printed                    # printed | purchased | hardware | included | made | other
  line: print:foot-monocoque       # BOM line id; the same id in several files is one line (quantities add)
  make_buy: make
  quantity: 2                      # what you expect the manifest to hold (checked)
  # process: {printer: bambu-x2d, material: bambu-paht-cf, up: [0, 0, 1]}   # phase 2: print setup lives here
- match: "*_foot_monocoque_combined_steel_sleeves"
  kind: hardware
  line: steel-sleeve-sets
  make_buy: context
  contains:                        # bought items inside each matched CAD part (checked against the STEP's solids)
  - {line: "sleeve:4.5x3.2x5", quantity: 4, make_buy: buy, spec: {kind: sleeve, outer_dia_mm: 4.5, inner_dia_mm: 3.2, length_mm: 5.0},
     supplier: unknown, part_number: unknown}
- match: "*_foot_M3x8_*"
  kind: hardware
  line: "hw:M3-x-8-ISO-7380-1"
  make_buy: buy
  spec: {kind: screw, thread: M3, family: M, length_mm: 8, standard: ISO 7380-1}
  supplier: unknown                # from a repo source (say which in notes), else unknown; never guessed
  part_number: M3 x 8 ISO 7380-1
assembly:
  subassemblies:                   # declared once, in the file that owns the subassembly
  - name: right_leg
    after: [torso]                 # built in place on these
    plan: {links: [R_yaw_link, R_pitch_link, R_roll_link, R_shin, R_ankle_pitch, R_foot]}   # for the proposer
  steps:
  - id: right_leg:p050             # unique over all files
    subassembly: right_leg
    part: [R_foot_monocoque_combined, R_foot_monocoque_combined_steel_sleeves]   # a printed part with its pressed sleeves
    direction: [0, 0, 1]           # manifest-frame vector it travels going in (+X the R_ side, -Y the face, +Z up)
    after: [right_leg:s049]        # the step(s) before it, in any file
    status: designer               # designer (your order) | seeded (a draft)
  - id: right_leg:s051
    subassembly: right_leg
    screw: "R_foot_M3x8_*"
    with: ["R_foot_M3_nut_*"]      # nuts, washers, spacers, inserts that go in with it
    direction: screw axis
    after: [right_leg:p050]
defects:                           # what is known to be wrong with these parts and what fixing it takes
- {id: foot-fsr-backings, parts: "*_backing", kind: no_scripted_step, fix: script the backing, task: T-54avce}
```

## Configured values: joint ranges and builder values (single source)

Owner 2026-11-23: every configured value (a joint range, a clearance, a count, a material choice) is set in the
definitions, and the builders, the design checks and the design dashboard read it from there. No second copy in a
script, a JSON side file or the dashboard, and no silent default: an undeclared value is an error or a finding.
A definition holds robot information only (owner 2026-11-24): dimensions, joints and their frames, materials, hardware,
the motion the body must clear (the pose library), print orientations and part colours. A tool's own settings stay in
its code: mesh and voxel sampling, solver steps and tolerances, plot colours that colour no part, role filters, task ids,
legacy part-name lists and export provenance (as the checks keep their measurement parameters).

```yaml
joints:                            # the joints this builder owns: each joint is declared in exactly one definition
- match: [R_knee, L_knee]          # exact joint (or control) names of the body manifest; both sides may share one entry
  range_deg: [0.0, 95.0]           # [lower, upper], manifest sign convention, lower < upper
  doc: Knee flexion; the A-11 pose library reaches 95.
  decision: owner 2026-11-23       # optional: who / which decision set it
config:                            # this builder's other configured values, one key each
  SHELL_GAP_MM: {value: 1.5, unit: mm, doc: running gap between the shell and the moving link}
```

Reading them: [bom_config.py](bom_config.py), `joint_ranges()`, `joint_range(name)` (raises naming where to declare
it), `value(definition, NAME)`, `values(definition)`, `apply_joint_ranges(manifest)` (writes `joints[]` / `controls[]`
`range_deg`; the assembler calls it, so every manifest after it carries the declared ranges). Under
make.py the reads are recorded as the values asked for (`values:joints`, `values:config:<definition>`), not as every
definition's bytes, so editing one definition reruns only the builders whose values changed. build.py exports them
(`bom.json` `joints`, `config`), and publish_body.py puts the ranges in the dashboard record (`cad.joint_ranges`), which
the CAD motion sliders use in place of the manifest copy.

Where the joint ranges live today: legs `robot/body/leg/package_complete_leg_hip.bom.yaml`, arms
`robot/body/arm/build_arm_connectors.bom.yaml`, neck `robot/body/neck-head/build_neck_head_connectors.bom.yaml`, hand
opening `robot/body/assembly/assemble_complete_fdm_body.bom.yaml`. Findings: `joint_declared_twice`, `joint_range_empty`,
`joint_range_undeclared` (an undriven manifest joint or control nobody declares), `joint_not_in_manifest`,
`joint_range_differs` (a manifest built before the declaration changed: rebuild it with make.py).

## What the checks report (each finding names the file to fix)

`bom` and `BOM.md`: manifest parts no definition claims; entries or globs that match nothing; parts claimed twice;
declared quantity against the manifest count; declared sleeve contents against the solids of each sleeve STEP; files that
resolve nowhere (or only in the bucket); hardware without a declared spec; printed parts without a print setup; lines whose
name, spec or supplier disagree between files; generator inputs no generator writes and not on disk; parts without an
assembly step; steps whose `part` / `screw` / `with` match nothing (stale) or whose `after` names no step; defects that
name no part. `assembly`: walks the declared order and measures each step (insertion path, driver path); ASSEMBLY.md shows
per step whether the measurement supports it.

## Keeping it current (every agent)

- Start hardware work from `BOM.md` / `ASSEMBLY.md`.
- When you add, rename or remove a part or screw, or a builder step: edit your builder's `.bom.yaml` only (parts, spec,
  supplier, generator, assembly steps), rerun `build.py` and the bom check, and work the findings that name your file down.
  Nothing under `robot/body/bom/` needs an edit for a design change.
- Drafting aids (they never edit a definition): `PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/assembly_seed.py
  propose [--names right_leg] [--search]` (under memguard, about 1 min / 4 GB) writes proposed steps per file into
  `.cache/work/bom/assembly-proposals/`; `robot/body/standard-hardware/propose_specs.py --only GLOB` proposes hardware specs from names
  and meshes.
- A new builder: a new `<script>.bom.yaml` next to it with its generator; the rebuild picks it up from its inputs / outputs.
- Design findings (a part that cannot go in, a screw no driver reaches) go to an Idle task for the part's owner with the
  measurement.

## Files here (tools only)

| File | What |
|---|---|
| `schema.json` | the definition schema |
| `definitions.py` | finds, validates and resolves the definitions; derives the generator and assembly order; findings |
| `build.py` | the BOM and build sequence from the definitions plus file facts |
| `files.py` | how a manifest file resolves (disk, regenerable, blob index, path moves, bucket records) |
| (moved) | standard hardware STEP / STL: `robot/body/standard-hardware/generate.py` (declared as a generator in its `generate.bom.yaml`); spec drafting aid `propose_specs.py` next to it |
| `rebuild_chain.py` | isolated rebuild of the selected body from the declared generators, then the BOM steps |
| `make.py`, `gencache.py`, `gensite/` | bring a checkout's body up to date from the declared generators, reusing unchanged outputs from the shared cache; gc |
| `compose.py` | apply the declared `compose` entries of a composed manifest |
| `bom_config.py` | read the declared joint ranges and configured values (`joints:`, `config:`) for builders, checks and the dashboard |
| `publish_body.py` | the design-dashboard record of a declared study (default the selected body) from the definitions, the manifest and the rebuild outputs |

## State

The dated summary is in [docs/research/design-checks.md](../../../docs/research/design-checks.md) (BOM against the CAD;
Assembly sequence).
