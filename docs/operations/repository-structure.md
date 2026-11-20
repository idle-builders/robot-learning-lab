# Repository structure

The tree holds the current robot, the current control system and the infrastructure around them, as the files they are built from. Git holds every version. Large outputs live in Git-ignored folders and the private bucket, and each one can be traced to the commit and command that made it. The past is explored through `git log` and tags, never through numbered folders.

```
robot/          the current robot, as code
  design.json   what is selected, part by part
  inputs.json   pinned outputs of earlier designs that the current build reads
  body/         printed FDM structure: leg, arm, neck-head, torso, assembly, motor-refit
  actuators/    catalog, joint bank, selection (tier pick), source geometry
  electronics/  integration build, power, sensing, compute (head)
  shell/        the ape look reference (sketch and the shell built from it)
  model/        body + actuators → URDF/MJCF (cove50_description), physics ledger
control/        the control system
  common/       MJLab tasks, motions and tapes, simulation models
  wbc/          whole-body controller: training setup, evaluation, compute jobs, registry.json
  skills/       per-skill policies, until the WBC supersedes them: training setups, registry.json
  perception/   perception system: identity, people, range, localization, sensor models, worlds
  behaviour/    learned option head and viewpoint routing
infra/          everything around the robot
  dashboard/    results hub server and UI, design BOM, session inspector
  storage/      blob store, storage lifecycle, catalog backup, move_paths
  compute/      Runpod, serverless workers
  tools/        bootstrap, name checker, memguard, public-source check
  lab           entry point: `infra/lab storage …`
  config/       spend limits, targets, public-source policy
  requirements/ pinned environments
docs/           operations, research by topic, design references
data/           local catalog, Git-ignored and backed up to the private bucket
.agents/skills/ agent instructions
artifacts/ .cache/ .bookkeeping/   Git-ignored: outputs, scratch, local process records
```

## What is checked in

Checked in: scripts, parameters, specs, configurations, small hand-written data, selection pointers and registries. These are the files a design, a controller or a policy is made from.

Not checked in: generated geometry, renders, FEM fields, traces, checkpoints and videos. They go to `artifacts/<area>/<part>/` (or `artifacts/runs/<run-id>/` for training) and to the private bucket. The bucket keeps every content a path ever had (`file_versions` in the blob index), and the dashboard records which commit produced which output.

## Versions

- **Design.** One folder per part, named by content, edited in place. A version is a commit. Publish with `results_hub.py publish` naming the folder in `source`; the dashboard keeps the previous version as `study_id@commit` and shows each version's own outputs.
- **Control.** The training setup is in Git. A run is the commit plus the configuration, and its outputs are kept under its run id. Promoting a policy is a commit that updates that part's `registry.json`.
- **Inputs from earlier designs.** `robot/inputs.json` pins each output of an earlier design that the current build reads: path, SHA-256, the tag of the design that made it and its build command. A fresh clone restores these from the bucket, or rebuilds them by checking out the tag.
- **History.** Designs that are no longer current exist only in Git history. Each kept dashboard version has an annotated tag `design/<study>` on the commit where its sources exist as built; `git checkout design/<study>` gives the full tree of that time. The `history/*` tags are listed in `git-and-bookkeeping.md`. These tags and the history before 2026-11-20 are on the owner's computer only, not in the public repository (`public-upstream.md`).

## Fresh clone

1. `python3 infra/tools/bootstrap.py …` rebuilds the environments from `infra/requirements/`.
2. `.venv/bin/python infra/storage/catalog_backup.py restore` restores the dashboard and blob catalogs and the local records under `data/`.
3. Build a part from its folder (each part's README names the command), or let the dashboard and tools restore outputs on demand by path.

## Rules

- No numbered or dated names: `.githooks/pre-commit` refuses them (see `git-and-bookkeeping.md`).
- Move files only with `infra/storage/move_paths.py`, which keeps the blob index, the dashboard catalog and references in step.
- Process records go to `.bookkeeping/`; research results go to the topic documents in `docs/research/`.
