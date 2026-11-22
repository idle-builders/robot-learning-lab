# Robot BOM and assembly order: start here for hardware

Everything needed to build the selected Cove-50 body starts here: what to print and buy, from which scripts and files,
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
```

Outputs (derived, `.cache/work/bom/`): `BOM.md` (the builder's page, findings first), `ASSEMBLY.md` (build sequence per
subassembly, with the assembly check's measurement per step), `findings.json`, `bom.json`, `assembly.json`, `bom.csv`,
`dashboard-bom.json`. The body rebuild (`rebuild_chain.py setup && run && compare`, isolated under
`.cache/work/body-rebuild/`) runs the declared generators in the order their inputs and outputs give, then the same three
commands.

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

## State

The dated summary is in [docs/research/design-checks.md](../../../docs/research/design-checks.md) (BOM against the CAD;
Assembly sequence).
