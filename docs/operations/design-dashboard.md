# Robot design dashboard

[Robot design](http://127.0.0.1:8805/design/) shows the robot body, its parts and its bill of materials. [Robot learning](http://127.0.0.1:8805/) is for policies, perception, motion control and planning. Start both with `.venv/bin/python infra/dashboard/results_hub.py serve`.

## The robot is the sum of its parts

Every part of the robot is declared in a body definition, `robot/**/<builder>.bom.yaml` (schema `robot/body/bom/schema.json`): its generator, BOM line, make / buy, spec, supplier, print setup, assembly step and its robot `section`. The body manifest is the output of the declared generators (`make.py body`). Nothing else defines the robot or a section (owner 2026-11-25: "bom only").

The dashboard holds only records that `robot/body/bom/publish_body.py` generates from these definitions (owner 2026-11-22, T-zmmdkm):

- **The body record** (`cove50-complete-fdm-body`, the `study` block of `robot/body/complete_body.bom.yaml`): every part of the body manifest.
- **Scoped views** of the same body (a `study` block with `scope: {definitions | parts}` in a builder's definition, e.g. `shoulder-roll-u-connectors`, `shoulder-shells-parametric`): the same manifest, the part view narrowed to those parts, the rest shown as CAD context.

A record is built from the study block (title, summary, observation, measurements, limitations, owner statements, evidence and render links), the body manifest, every definition, the BOM and assembly order of `robot/body/bom/build.py` and the newest complete check report per check and part on the same manifest sha256. `results_hub.py publish` refuses a design record without that generated body view. Publish from the main checkout:

```sh
.venv/bin/python -B robot/body/bom/publish_body.py --publish                                  # the body record; packages the display GLB when the manifest changed
.venv/bin/python -B robot/body/bom/publish_body.py --study shoulder-shells-parametric --publish
```

After a redesign edit the definitions (parts, steps, study text), commit and publish again; never hand-edit a record or a publisher. A new study is a `study` block, not a new publisher.

## Sections

A record's sections are the distinct `section` values its parts' definitions declare (part view `body_view.parts[*].section`); `results_hub.py` derives them when it publishes and serves the record. The names are the definitions' own (`Hip`, `Knee`, `Ankle / foot`, `Shoulder`, `Elbow`, `Hand / wrist`, `Neck / head`, `Torso`, `Waist / pelvis`); there is no "Whole robot", "Electronics" or alias section. A part without a section belongs to no section view; fix its definition.

Choosing a section on the drawing or in the list opens the same records filtered by part section: the part view, its subassemblies and steps, the CAD BOM table and the cost lines list only the parts whose definition declares that section, and the CAD view isolates them (**Show assembly** restores the whole body). There is no selection: nothing is "selected for" a section, and no command or button writes one.

## What a record shows

Under **Body: subassemblies and parts** (or **<section>: …**): the subassemblies in assembly order with every step (parts / screws, carried items, direction, declared status, assembly-check measurement, definition file), and a part card per part: BOM line, STEP and mesh with how each resolves, generator and its regenerate command, assembly step, print setup with the source of each value, newest check worst cases, BOM findings and the definition file to fix. The CAD panel's **Bill of materials** table uses the same data; clicking a CAD part opens its card.

The CAD view (the display GLB, one named node per manifest part) has joint sliders over the ranges the definitions declare (inspection commands, not qualified travel), shell visibility, overlay layers (`overlay_id` on manifest parts, labels from the study block's `cad.overlays`), X/Y/Z section cuts and display-only explosion. **Hide menu** gives the CAD view the full width. **Snapshot & annotate** freezes the view for owner feedback: numbered marks with comments are resolved to the parts under them, and **Save for Claude / Codex** writes the annotated PNG, a Markdown summary and the full context to `.bookkeeping/design-feedback/` (`results_hub.py design-feedback --open` lists open items; see `docs/operations/working-rules.md`).

**Versions.** Git holds them: a record's version is the commit of the definitions it was built from (`snapshot.commit`); publishing at a new commit keeps the previous one as `study_id@commit` in the version menu, which serves that version's files by SHA-256.

## Robot BOM & costs

[Robot BOM & costs](http://127.0.0.1:8805/design/?view=bom) is the BOM of the body record: the lines `robot/body/bom/build.py` builds from the definitions (the record carries them as `design.cad.bom`; a `design_boms` row, e.g. from `build.py --publish`, takes precedence). It shows known cost subtotals by currency, known mass, the lines with unknown cost, mass or source, and exports CSV and JSON (`/api/robot-bom`, `format=csv`). Each record's page shows the same BOM; a section view lists the lines of that section's parts. There is no composition of study versions: to change a part, quantity, supplier or price, edit its definition and republish.

## Archive

`results_hub.py archive-study STUDY_ID --reason '…'` moves every version of a record out of the current list (`restore-study` brings it back). The design Archive lists only such archived records; old artifact folders are not shown.
