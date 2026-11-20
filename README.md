# Robot Learning Lab

A private hobby project: a 50 cm ape robot that moves in an agile, playful way, like a little monkey. The owner's intent, cost rules and decision rules are in [AGENTS.md](AGENTS.md); procedures are in [working rules](docs/operations/working-rules.md). Work is planned in Idle tasks.

## Current state

- **Body:** the complete 3D-printed FDM body is selected ([robot/design.json](robot/design.json)); its parts are built in [robot/body/](robot/body/README.md). Nothing has been printed or loaded yet.
- **Control:** the whole-body controller ([control/wbc/](control/wbc/)), trained with MJLab on the GPU and judged by agility across all motions. The per-skill policies ([control/skills/](control/skills/README.md)) stay until it supersedes them; perception is in [control/perception/](control/perception/README.md).
- **Model:** [robot/model/current-model.json](robot/model/current-model.json) names the current `cove50_description` MJCF and URDF.
- **Results:** the dashboards (`.venv/bin/python infra/dashboard/results_hub.py serve`): [learning](http://127.0.0.1:8805/) and [design](http://127.0.0.1:8805/design/). Findings by topic are in [docs/research](docs/README.md).

## Layout

| Path | Contents |
|---|---|
| [`robot/`](robot/README.md) | The current robot as code: body, actuators, electronics, shell reference, model, pinned inputs from earlier designs |
| [`control/`](control/README.md) | The control system: whole-body controller, skill policies, perception, behaviour, shared simulation code; registries of promoted policies |
| [`infra/`](infra/README.md) | Dashboard, storage and archive, compute (Runpod), tools, config, pinned environments |
| `docs/` | Operations, research by topic, design references ([repository structure](docs/operations/repository-structure.md)) |
| `data/` | Local catalog (Git-ignored, backed up to the private bucket) |
| `.agents/skills/` | Agent instructions per track |
| `artifacts/`, `.cache/`, `.bookkeeping/` | Git-ignored: outputs (private bucket), scratch, local process records |

## Fresh clone on another computer

1. `python3 infra/tools/bootstrap.py --python-core /path/to/python3.12 --python-atopile /path/to/python3.14` rebuilds the local environments from `infra/requirements/`.
2. Set `ROBOTLAB_ARCHIVE_BUCKET` (see `.env.example`) and the Infomaniak S3 credentials, then `.venv/bin/python infra/storage/catalog_backup.py restore`. This brings back the dashboard and blob catalogs and the local records under `data/`.
3. Outputs restore on demand by path; each part's README names its build command. See [storage lifecycle](docs/operations/storage-lifecycle.md).

## Versions and history

Git holds every version; names never do ([git and bookkeeping](docs/operations/git-and-bookkeeping.md)). A design version is the commit of its part folder; a policy is promoted by a commit to its `registry.json`. This repository is public; private records stay Git-ignored ([public repository and private evidence](docs/operations/public-upstream.md)). Its history starts on 2026-11-20. The lab's earlier private history is kept only on the owner's computer: earlier designs as tags (`git tag -l 'design/*'`, then `git checkout design/<study>`), the old `main` as `archive/private-main-20261120`, and the tree before the 2026-11-12 restructure as commit `6fbb278e4`. [Finished designs](docs/design/finished-designs.md) lists the accepted earlier ones.
