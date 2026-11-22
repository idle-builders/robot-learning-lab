# Robot design dashboard

[Robot design](http://127.0.0.1:8805/design/) is for CAD, mechanisms, electronics packaging and design simulations. [Robot learning](http://127.0.0.1:8805/) is for policies, perception, motion control and planning. Start both with `.venv/bin/python infra/dashboard/results_hub.py serve`.

## Explore the robot

Select a joint or body part on the aesthetic drawing, or use the section list beside it. Both controls select the same section. **Design studies** shows the current body: the selected complete printed FDM body, its part studies and the electronics design. Studies named in `robot/design.json` carry a **Selected** mark. Earlier bodies, finished earlier designs and alternatives are in **Archive**, each with the reason it left the current view.

A study is **exactly one design variant**, owning its CAD, measurements and simulations. **Git holds its versions.** Opening a study shows its current version; the version menu lists earlier versions by date and commit (`2026-11-09 · commit 8b54a8a`). An earlier version shows its own registered files: the dashboard serves them by SHA-256 from the blob index or from Git history, even after the working files changed or left the tree. Records published before 2026-11-12 keep their old label (`r3`, `v16`) as provenance, shown on hover.

Use a stable, descriptive **mechanism name** as the study title, for example **Direct serial neck**, and the same content name for `study_id` and the study folder. Keep designer letters, dates, rounds and version tags out of titles, ids and folders. Old links that named a round (`?result=…-r3`) still open, through `result_aliases`.

Continue an existing mechanism by editing its sources in place and committing; do not mint a new study because a designer, session or round changed. A genuinely different mechanism gets a separate study. Curate abandoned or superseded branches into Archive with a written reason. Never delete or silently reassign version evidence, preferences or reviews while tidying names.

Each robot section has exactly one selected study, kept in `selected` in `robot/design.json` (section → study_id). One study may be selected for several sections: the complete body covers every section, the arm covers Shoulder and Elbow, and a newer hip study can be selected for Hip while the selected complete body still contains an older hip. Agents record a selection with `infra/dashboard/results_hub.py select-design STUDY_ID --robot-section Hip [--robot-section Knee]`; the owner uses **Select for [section]** on a study. Both write the same file, and opening a study still defaults to its newest version. Selection does not qualify hardware. `section_selections` in the same file is provenance (statements, Idle decisions) and drives nothing.

The CAD view supports native joint sliders, play/pause, speed, reset, saved-configuration scrubbing, shell visibility, X/Y/Z section cuts and display-only explosion. Section cuts show open surfaces rather than solid caps. Explosion is a visualization, not a disassembly simulation. Kinematic motion does not simulate contact, loads, flexible cables or attained motor behavior. Unsupported continuous mechanisms offer their exact saved configurations; missing CAD stays explicitly source-only.

**Hide menu / Show menu** on the CAD view of a study collapses the side panel (sliders, toggles, summary, review) so the CAD view takes the full width; the choice is remembered in the browser. **Snapshot & annotate** freezes the current view for owner feedback to Claude, Codex or ChatGPT: numbered pen, arrow and box marks are resolved to the parts under them with the snapshot camera, and **Save for Claude / Codex** writes the annotated PNG, a Markdown summary and the full context to `.bookkeeping/design-feedback/` (`results_hub.py design-feedback --open` lists open items; see `docs/operations/working-rules.md`). The CAD model library has the same snapshot button. Each mark takes its own comment; **Copy image**, **Download PNG** and Save all use one picture: the numbered marks plus a legend strip below (comment and parts per mark, study, camera, joint angles). The remote view cannot save; paste the copied image into the Claude or ChatGPT chat. **Show / Hide sections & studies** in the header folds the left navigation away while a study is open (folded by default, remembered in the browser); the landing page always shows it.

Use `cad.motion_controls` as an optional list of source joint names to show only those sliders. The complete joint hierarchy still drives their descendants. Slider ranges are inspection commands, not qualified physical travel; label their basis in `cad.motion_note`.

Set `cad.motion_mode: "saved_configurations"` when a coupled mechanism has exact solved poses but
its continuous closure is unsupported by the viewer. Its complete source joint manifest remains
available; independent angular sliders stay disabled. Composed body references can use the existing
`has_body_reference`, `fit_reference_on_load` and `body_panel_label` fields. Mark each reference panel
with `inspection_reference: true` and `display_group: "ape_body_shell"`, give it explicit rigid link
ownership, and exclude it from installed mechanism counts, purchase cost and mass. A reference-only
CAD BOM line may map those named meshes without implying that they are shoulder purchases.

## The selected body: generated, never hand-built

The record of the selected complete body (`cove50-complete-fdm-body`, the study selected for Whole robot) is generated
by `robot/body/bom/publish_body.py` from the same source as the BOM (owner 2026-11-22, T-zmmdkm): the body definitions
(`robot/**/<builder>.bom.yaml`), the selected manifest (`body_manifest` in `robot/design.json`), the BOM and assembly
order of `robot/body/bom/build.py`, and the newest complete check report per check and part on the same manifest sha256.
The study text (title, sections, summary, observation, measurements, limitations, owner statements, evidence and render
links) is declared in a `study` block of a definition (`robot/body/complete_body.bom.yaml` for the selected body); the
publisher knows no part and no study. The body rebuild (`robot/body/bom/rebuild_chain.py`) ends with the record written
to a file; publish from the main checkout:

```sh
.venv/bin/python -B robot/body/bom/publish_body.py --publish          # selected body; packages the display GLB when the manifest changed
.venv/bin/python -B robot/body/bom/publish_body.py --study shoulder-shells-parametric --publish
```

What the study page shows from it: under **Selected body: subassemblies and parts** (above the costs BOM), the
subassemblies in assembly order with every step (parts / screws, carried items, direction, declared status, assembly-check
measurement, definition file), and a part card per part: BOM line (make / buy, quantity, spec, supplier, tool, mass),
STEP and mesh with how each resolves, generator with its regenerate command, assembly step and direction, print setup
with the source of each value, newest check worst cases, BOM findings, and the definition file to fix. The CAD panel's
**Bill of materials** table uses the same data; clicking a CAD part opens its card and names its BOM line and step in the
CAD label, and a part name in the table or the part view selects it in the CAD view. The remote export follows the same
record (the part view's file names are text, not uploaded). After a redesign edit the definitions (parts, steps, study
text) and rerun; never hand-edit a publisher. Sources must be committed (their last commit is the version).

## Register a study and publish a version

Generate the maintained authoring template, fill in real values, commit the study sources, then publish:

```sh
.venv/bin/python infra/dashboard/results_hub.py design-template --robot-section Knee > /tmp/knee-study.json
git add robot/body/knee/ && git commit -m "Knee: supported direct drive, deeper horn seat"
.venv/bin/python infra/dashboard/results_hub.py publish /tmp/knee-study.json
.venv/bin/python infra/dashboard/results_hub.py upload
```

The template is not a result. Its `null` measurements mean unknown. Do not invent values to fill it.

| Field | Meaning |
|---|---|
| `study_id` | Stable content name of one variant, e.g. `knee-supported-direct`; no round, version or date (the publisher refuses them) |
| `source` | The committed study sources (folder or file list); their last commit is the version |
| `dashboard` | `"design"` |
| `workspace` | Usually `"Body"` |
| `robot_sections` | Explicit body part(s), e.g. `["Knee"]` |
| `design` | One object containing this version's CAD, renders and exact-variant evidence |
| `assets` | Complete list of registered local sources for this version |
| `clips` | Source-matched simulation/inspection videos |
| `sections` | Additional version-specific evidence; use `design_evidence` for structured results |

How versions work:

- The publisher refuses `source` paths with uncommitted changes: commit first, the commit is the version.
- Publishing at a new commit makes that the current record (`id` = `study_id`) and keeps the previous one as `study_id@commit`, with its CAD, evidence, reviews and BOM.
- Publishing again at the same commit corrects text. Changed CAD needs a new commit; the publisher compares the registered CAD hashes.
- Older publish scripts that send `id: "…-r3"` and `version` still work: the record goes to `study_id`, the old id becomes a link, and without `source` the repository HEAD is the version. Name `source` when you touch such a script.
- Republication replaces the assets and sections lists, so include their complete inventories. Human comments and judgments stay with the version they were written on.

Do **not** put multiple variants in a study, use a designer's lane as a study when its architecture changes, or create a new study for each iteration of the same variant.

Use these section names: Hip, Knee, Ankle / foot, Shoulder, Elbow, Neck / head, Hand / wrist, Torso / shell, Waist / pelvis, Actuators, Electronics, Whole robot. The publisher maps older spellings (Neck, Head, Torso, Pelvis, Whole body) onto them.

## Curate current studies and Archive

Archive an unsuitable or misleading study by human judgement, with a written reason:

```sh
.venv/bin/python infra/dashboard/results_hub.py archive-study STUDY_ID --reason 'Why this work is retained only as a reference'
.venv/bin/python infra/dashboard/results_hub.py restore-study STUDY_ID
```

This moves all versions out of the current list without deleting CAD, assets, reviews, BOMs or
evidence. Archive shows one entry per study; direct study links retain version inspection and show
the archive reason. Restoration returns its versions to the current list. Republishing a record or
adding a version does not silently restore an archived study. Archival is curation, not an automatic
engineering verdict from a metric. Label useful but incomplete current references explicitly;
their presence does not make them a build recommendation. Schematics must not acquire a related
reference model as though it were CAD of that concept.

## Register CAD and motion

Inside the single `design` object, register:

```json
{
  "id": "knee-supported-direct",
  "title": "Supported direct knee",
  "summary": "One short finding or trade-off.",
  "visual_kind": "native_CAD_assembly",
  "maturity": "State engineering maturity and physical unknowns.",
  "image_path": "artifacts/knee-supported-direct/preview.png",
  "cad": {
    "model_units": "m",
    "views": [{"label": "Neutral assembly", "model_path": "artifacts/knee-supported-direct/assembly.glb"}],
    "parts_path": "artifacts/knee-supported-direct/parts.json",
    "step_path": "artifacts/knee-supported-direct/assembly.step",
    "report_path": "artifacts/knee-supported-direct/report.json"
  },
  "source_refs": [{"label": "Native report", "path": "artifacts/knee-supported-direct/report.json"}],
  "evidence": []
}
```

Register every referenced file in `assets`: `kind: "model"` for GLB, `"image"` for source renders and `"data"` for STEP, reports, manifests, configuration and full traces. The publisher hashes registered assets. Keep exact source/CAD/configuration hashes in native reports and provenance too. A GLB must contain a complete scene or clearly named part; separate meshes are not assembled by the viewer.

Put the **neutral assembly first** in `cad.views`. Additional views can be exact posed, exploded or service-state exports. `cad.model_units` is `"m"` by default; use `"mm"` for millimetre GLBs. Do not apply a source registration transform twice.

For continuous articulation, `cad.parts_path` supplies source-matched names and kinematics:

```json
{
  "parts": [{"name": "shin_carrier", "owning_link": "shin", "role": "frame"}],
  "joints": [{"name": "knee", "parent": "thigh", "child": "shin",
    "origin_world_mm": [60, 4, 123], "axis_world_neutral": [1, 0, 0], "range_deg": [0, 125]}]
}
```

GLB node names match manifest part names. Vertices, axes and pivots use the same neutral coordinate frame. The manifest's pivots are millimetres; `model_units` describes the GLB. Older `origin_mm` / `axis` keys also work. The viewer composes parent-to-child rigid transforms. It supports explicit `driven_by.linear`, two-column `driven_by.table_deg`, and `coupled_to` / `multiplier` relationships.

For live neutral models, optional `cad.inspection_anchor_links` names reference frames, for example `[{"link":"shin","label":"Calf fixed"},{"link":"foot","label":"Sole fixed"}]`. The first is the default. **Hold still** applies the inverse transform of that link to the complete assembly, preserving relative joint motion. This is a kinematic viewing frame, not simulated ground support, balance or load capacity. Saved posed configurations remain static inspection references.

Preserve native coupling. A closed linkage with multiple joints for one child needs a source-defined reduced kinematic model; otherwise use saved-configuration playback. Unknown ranges do not become guessed limits. For retained records with a physical-order contract and native pose report, the viewer may label slider bounds as the saved-pose envelope. Source-only/static versions remain clearly labeled.

An integrated print may have separate diagnostic skin and frame meshes for inspection. Partition the actual produced solid without duplicating material, map both named meshes to the same physical BOM item, and give them the same `physical_part_id` and `owning_link`. Mark only the skin as `display_group: "ape_body_shell"`; the frame then stays visible when panels are hidden. Explosion moves both regions together. Retain the complete fused STEP/STL as the manufacturing source and record partition coverage evidence; diagnostic meshes do not add manufactured parts.

A reduced mechanism may define angular command coordinates separately from its physical joints:
`"controls": [{"name": "knee", "range_deg": [0, 125]}]` in the part manifest.
Driven hinges reference that command through their source-defined linear/table mapping. This exposes
the total bend of a double hinge without making either half-angle independently adjustable. Retain
an audit and an explicit correction note if this inspection mapping differs from an older evaluator.
Fit view frames visible parts, so hidden shells do not shrink an exposed or isolated mechanism.

## Body panels and neighboring assemblies

**Ape body panels** controls the robot exterior, never a purchased motor housing. New part manifests declare `display_group: "ape_body_shell"` for those panels. Motor cases retain a mechanism role such as `motor` or `actuator_housing`. Legacy `shell`/`skin` panel roles remain supported; generic covers are not assumed to be ape panels.

Use `inspection_reference: true` on visual body datums and other contextual geometry, with `display_opacity` when transparency helps. Use `display_group: "neighbor_reference"` on an immutable neighboring joint and `cad.neighbor_reference_label` for its explicitly named checkbox. Those parts remain outside the designed section's BOM, mass, and engineering ownership. Context is excluded from display explosion. **Fit mechanism** frames the visible owned mechanism; `cad.has_body_reference: true` adds **Fit ape body** for the complete visible context. Register the exact body source, proper neutral placement transform, frame/units, and its limits; a reference hull is not an engineered panel assembly.

Set `cad.show_neighbor_reference: true` when actuator/context geometry should be visible initially. Optional condition layers use `cad.overlays: [{id, label, visible, description}]` and a matching `overlay_id` on each part. Mark those parts `inspection_reference: true`, give them the correct `owning_link`, and optionally set `display_opacity`. Each layer has an independent checkbox; layers are excluded from mechanism fitting, physical BOM/mass and explosion. A sampled movement reservation remains a sampled reservation while the instantaneous assembly articulates; it is not a continuous collision calculation. `cad.fit_reference_on_load` and `has_body_reference` can expose a full-assembly fit view alongside mechanism fit.

For hip-only studies, expose the hip axes. An unchanged knee at a declared context pose is reference geometry, not a new knee design or a hip-owned part. Retain the complete neighbor source/hash and hip-side mounting-interface parts; names beginning with `knee_` alone do not determine ownership.

## Purchased and produced bill of materials

Set `design.cad.bom_path` to a registered JSON asset and optionally
`bom_csv_path` to a registered CSV. The shared CAD panel displays an expandable,
filterable parts table. The JSON has `items`, with one row per BOM line:

```json
{"items": [{"id": "pitch-motor", "name": "Pitch actuator", "quantity": 1,
  "category": "purchased", "manufacturer": "Actual manufacturer",
  "part_number": "Actual SKU", "source_url": "https://example.com/actual-product",
  "part_names": ["pitch_motor_body"], "material": null, "process": null,
  "unit_mass_g": null, "mass_basis": "Unweighed", "notes": "Open identification or integration issues"}]}
```

Categories are `purchased`, `produced`, `modified_purchased`, `context` or `unresolved`. Purchased parts needing a secondary operation remain distinguishable from stock items. Map `part_names` to the
exact GLB/manifest identities; multiple solids may belong to one purchased item.
Use **Isolate** beside a BOM row to show its named solids and **Show assembly** to restore the view. Search by part, supplier or material; clicking a CAD part highlights its BOM row when that row is visible. These controls work alongside the joint, section and explosion controls.

Name each produced part's material/process and source drawing. Include unmodeled
fasteners, harness and covers explicitly; keep placeholders/reserves distinguishable
from specified purchased or produced items. Unknown mass remains null. State the
quantity scope in the source BOM. A catalog listing does not establish installed
load capacity, stock, delivery, or a complete ankle assembly.

## Attach measurements, simulations and results

Attach exact-version results in `design.evidence`. An item contains:

```json
{
  "id": "loaded-flexion", "title": "Loaded flexion sweep", "kind": "simulation",
  "observation": "Describe actual motion and where behavior degrades.",
  "measurements": [{"name": "Peak effort", "value": null, "unit": "N m", "condition": "Actual load and pose"}],
  "conditions": {"cad_commit": "the source commit", "solver": "Version and step", "load_case": "Payload, gravity and contact assumptions"},
  "limitations": "Actuator, flexibility, contact and physical transfer unknowns.",
  "source_refs": [
    {"label": "Native report", "path": "artifacts/knee-supported-direct/report.json"},
    {"label": "Full trace", "path": "artifacts/knee-supported-direct/trace.jsonl"},
    {"label": "Exact config", "path": "artifacts/knee-supported-direct/config.json"}
  ],
  "clip_ids": ["knee-supported-direct:loaded-flexion"]
}
```

Evidence items may also contain `image_groups`: a list of `{title, images: [{path, caption}]}`. These appear as expandable, lazy-loaded picture groups with full-size image links. Register every image path in the same version's `assets`; the publisher checks those links. For FEM comparisons, retain the exact case, source/mesh/field hashes, common colour limits, camera and deformation-display factor in the evidence sources, and label geometry-only alternatives explicitly.

Supported kinds are `measurement`, `simulation`, `analysis`, `physical_trial`. Register every local `source_refs[].path` in the version's complete `assets` list. Register each `clip_ids` entry in that version's `clips`, including exact report and trace paths; `upload` enables verified private playback. A `design_evidence` section uses the same item format in `payload.items` and still belongs to this one variant/version.

Retain exact input trajectories, CAD/body/configuration hashes, solver/time step, actuator/contact hypotheses, full traces, early terminations and severe trajectories. For cohorts, show native-unit distributions by condition and distinguish episodes, starts, worlds, transitions and optimizer updates. Distinguish requested kinematics, dynamic simulation, policy rollout and physical trial. Grade dimensions separately with qualitative judgment; never reduce engineering to a binary verdict.

An inspection revision may reference an existing recording by reusing its exact clip ID, source path and hash. The publisher retains the original recording owner and links the additional study version through `design_study_clips`; it does not move the recording away from the older version or require another upload. A different source recording still needs a new immutable clip ID.

Keep the first view human-readable: a short title, one finding, named measurements with units, and explicit open issues. Put detailed setup, reports and provenance in labeled evidence/source sections. Surface aesthetic changes as renders for human discussion.

## Existing material

`infra/dashboard/results_hub.py migrate-design-studies` converts the retained published catalogs into one-variant study/version records. It keeps original multi-variant comparisons and unattributed cross-checks in Archive. Existing option identities and review histories survive. Architectural replacements become different studies; actual source revisions remain versions. Source-identified fixture reports and recordings attach to their exact CAD versions using embedded native paths/hashes. Shared or ambiguous material stays labeled comparison evidence instead of being assigned by guesswork. No native CAD, report, trace or video is deleted.

## Bills of materials and robot costs

Each study version has a **Bill of materials & costs** panel with an editable parts inventory, quantity and cost extensions, mass, supplier/part numbers, material, price date, lead time, sources and exclusions. Blank numeric inputs remain unknown. The quantity, unit cost and unit mass all use the same quantity unit: for cable priced per metre, use `unit: "m"`, cost per metre and grams per metre. Quantities describe **one assembly of this exact version**; left/right repetition belongs in the robot composition's assembly count.

Publish a BOM as `design.bom` (also included in `design-template`), or add/edit it in the study panel:

```json
{
  "scope": "One knee assembly, including its actuator and installed hardware",
  "completeness": "partial",
  "notes": "State currency, tax, shipping, fabrication and batch-size assumptions.",
  "exclusions": ["Cable harness and assembly labor are not yet estimated"],
  "items": [{
    "id": "actuator", "name": "Source-identified actuator", "category": "Actuators",
    "quantity": 1, "unit": "ea", "part_number": "", "supplier": "", "material": "",
    "unit_cost": null, "currency": "", "cost_basis": "unknown", "price_date": "",
    "unit_mass_g": null, "lead_time_days": null, "source_refs": [], "notes": ""
  }]
}
```

This example is an authoring shape, not a purchasing claim. Part IDs must be unique within a BOM. Known costs need a three-letter currency code. `cost_basis` is `unknown`, `estimate`, `quote` or `paid`; `price_date` is empty or `YYYY-MM-DD`. Register every local `source_refs[].path` in that version's `assets`; HTTP(S) source URLs can identify supplier pages or quotes. The editor accepts one URL or registered file path per source line. No current supplier pricing is inferred from the existing reports.

Costs are shown as **known subtotals by currency**, with no implicit exchange rate. Mass uses grams. Missing quantity, cost, mass and source counts remain visible. `completeness: "complete"` is the author's declaration about the stated assembly scope, never an engineering assessment. Record shipping, tax, labor, print material or machining as explicit cost lines with their assumptions if they should be included. Pack pricing must be converted to the stated quantity unit before entry. No procurement batching, automatic shipping/tax, nesting or lead-time scheduling is inferred.

Open [Robot BOM & costs](http://127.0.0.1:8805/design/?view=bom) from the dashboard header. Add the **exact study version** for each assembly and its positive integer count, then **Save robot BOM & calculate**. One study can appear once, at one version; counts express repeated assemblies. The saved composition persists in `robot_bom` in `data/results.sqlite`. It does not populate from section preferences or change `robot/design.json`.

The robot view multiplies each part quantity by the assembly count and shows parts with their owning study/version. A newer design version does not replace a pinned version. Multiple included studies covering the same section are called out for ownership review; all their lines are still counted. A knee BOM that already includes its actuator must not also include that same actuator through another assembly BOM. Whole-robot studies are an alternative to their overlapping subassembly inventories. Missing/empty BOMs, partial inventories, exclusions and sections without an explicitly included study remain visible, so a subtotal cannot imply a complete robot estimate.

BOM estimates are revisioned independently of CAD. Dashboard saves use optimistic concurrency and retain prior values in `design_bom_history`. `design_boms` stores each version's current BOM. Republication with `design.bom` replaces that current estimate and creates a revision; omitting `design.bom` preserves dashboard edits and their registered sources. To clear an inventory, explicitly save/publish an empty `items` list. Geometry changes still require a new study version. The robot composition uses the latest BOM estimate for each pinned design version; exports identify every BOM revision used.

Both views export CSV part lines and JSON reports. **JSON includes scope, exclusions, missing assemblies, composition counts and revision metadata**; keep it with a CSV when sharing a robot estimate. CSV is a parts table, not a complete scope record. Read reports through `/api/design-bom?result=RESULT_ID` and `/api/robot-bom`; add `format=csv` for CSV. A previous estimate is available through `/api/design-bom?result=RESULT_ID&revision=N`. Local-origin POSTs to those endpoints require `expected_revision`, plus `result_id` and `bom` for a study, or `plan` for the robot. The plan contains `title`, `notes`, `exclusions` and `assemblies: [{"result_id": "…", "count": 2}]`. Stale saves return a conflict and preserve the editor's inputs until reloaded.

## Explicit study cleanup

`data/design-study-retirements.json` records user-directed removal of obsolete studies. Deleted result IDs cannot be republished by stale generators. Old links explain that the study was removed; the Archive excludes retired artifact entries, while shared legacy sources can lose a Hip label without removing their other sections. This curation record is separate from engineering measurements and never applies a numeric design veto.

For a bilateral inspection scene with separate kinematic roots, an inspection anchor may also declare `links_by_root`, mapping each root link to its own held link (for example `{"shin":"foot","left_shin":"left_foot"}`). This keeps both soles at their separate neutral world placements; a single global inverse transform would only change the common reference frame. Mirrored geometry remains an explicitly labeled reference, excluded from one-ankle BOM quantities. Optional `fit_reference_on_load: true` and `fit_reference_label` frame and name the visible comparison. None of these display transforms supplies hip/knee closure, ground support or dynamics.

A reference part may declare additional `display_groups` alongside its primary `display_group`. A mirrored body panel uses both `neighbor_reference` and `ape_body_shell`, so the reference checkbox and the body-panel checkbox each control its visibility. Existing single-group manifests retain their behavior.
