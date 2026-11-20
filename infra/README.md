# Infrastructure

| Folder | Contents |
|---|---|
| `dashboard/` | Results hub server (`results_hub.py serve`, port 8805) and UI, design BOM, session inspector |
| `storage/` | Python package `infra.storage`: blob store and private archive, storage lifecycle, catalog backup, `move_paths.py`, restore tools |
| `compute/` | Runpod Terraform, serverless workers, `runpod_infra.py`, `runpod-flash` |
| `tools/` | `bootstrap.py` (environments), `check_repo_names.py` (pre-commit), `memguard.py`, `public_source.py` |
| `config/` | Spend limits, targets, public-source policy |
| `requirements/` | Pinned environments |
| `lab` | `infra/lab storage run|finish|status|sync|clean` |
