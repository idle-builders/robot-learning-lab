# Control

The robot's control system. Training and heavy evaluation run on the GPU (Runpod), never on the laptop CPU.

| Folder | Contents |
|---|---|
| `wbc/` | The whole-body controller: tracking task, parcour, motion tapes, agility measures, compute jobs, `registry.json` |
| `skills/` | Per-skill policies, kept until the whole-body controller supersedes them: one folder per training setup, `registry.json` (promoted policies by run id), `profiles.json` (how each policy is used) |
| `perception/` | The perception system (identity, people, range, localization, sensor models, worlds) |
| `behaviour/` | The learned option head and viewpoint routing on top of perception |
| `common/` | Shared by all: `plant/` (the batched R3 simulation boundary), `compute/` (remote launch, bundling, transport), `mjlab_ape/` (MJLab task package `ape_walk`), simulation models and motions |

A run is the commit plus its configuration (each skill's `setup/`, the WBC's launch arguments); outputs live in `artifacts/runs/<run-id>/` and the bucket. Promoting a policy is a commit to the part's `registry.json`.

Moved code is a new version: older runs pinned their inputs by path and hash, so an exact rerun of a past run uses its commit (`git checkout <commit>`).
