# Git versioning and local bookkeeping

Owner rule (2026-11-12). Git keeps the versions; file names never do. Process records stay local. Research is kept as results, by topic.

## One file per thing; git holds its versions

- Never create a copy to change something. Edit the file and commit. A new iteration is a new commit, not `build_v2.py`, `card-r3.md`, `integration-v16/`, `final.json`, `new_`, `old_`, `copy`, `backup` or a date-stamped twin of an existing file.
- Name files and folders by what they contain: `hip_yaw_bracket.py`, `obstacle-world/`, not `study-r4-20261030/`.
- A real alternative design that lives side by side with another gets a descriptive name for how it differs (`swarm_reference_knee_shell.py`, not `swarm_reference_knee_v2.py`). Behaviour variants of one script are settings or flags on that script, not copies.
- To refer to an earlier version, cite the commit: `git log --follow -- path`, `git show <commit>:path`. Put the commit hash in a record, a dashboard provenance field or an Idle note when a result depends on a specific version.
- To keep a version for a published result, commit it before changing it further, and record that commit in the result's provenance. Do not freeze a copy under a new name.
- Commit small, coherent changes with a message that says why. Stage explicit paths, never `git add -A` across the tree. Do not commit generated bulk; `artifacts/`, `.cache/` and `.bookkeeping/` are ignored.
- Run identifiers (a run id with a date and hash, a bucket key) are identifiers, not versions. They belong in `artifacts/runs/<run-id>/`, `.cache/work/<run-id>/` and `.bookkeeping/`, not in the names of source, configuration or documentation files.

`.githooks/pre-commit` refuses new paths that carry a version token (`_v3`, `-r2`, `rev4`, `final`, `copy`, …) or a date stamp outside `data/`, and process records outside `.bookkeeping/`. Enable it once per checkout with `git config core.hooksPath .githooks`. Existing names are grandfathered until they are next renamed.

## Bookkeeping goes to `.bookkeeping/`

`.bookkeeping/` is the Git-ignored folder for local process records, next to `.cache/` (disposable scratch) and `artifacts/` (native evidence, offloaded to the private bucket). It is local, not uploaded, and may be deleted when the work it records is finished.

Bookkeeping is any record of how work was coordinated rather than what it found or what it needs to run:

- reviews, challenges, independent-review packets, lead reviews;
- authorizations, activations, dispatch, provider status, endpoint lifecycle, preflight, teardown, transport and cleanup records;
- verifications, receipts, attestations, bindings, claims, locks, preimages, custody and retention records;
- publication checks and publication regeneration records, handoff packets between agents;
- input snapshots, frozen-input copies, executed-input copies and other duplicates of tracked sources;
- logs, stdout/stderr captures and lock files;
- one-off scripts that only produced the above (publish, verify, cleanup, close, restore helpers for one run).

Write them under `.bookkeeping/<task-id-or-run-id>/…`. Migrated older records keep their original relative path under `.bookkeeping/` (for example `.bookkeeping/data/basic-skills-retry-20261027/compute/…`).

What stays tracked:

- code, configuration and the small exact inputs that code reads (`robot/`, `control/`, `infra/`);
- the docs in `docs/`: operations, design references and topic research in `docs/research/`.

Everything else under `data/` is the local catalog (the dashboard and blob databases, storage run state, run and build records that code reads). All of `data/` is Git-ignored, because the repository is public (since 2026-11-20 also the catalog pointer, the spend ledger, the regenerable-artifact list and the design retirements). It travels to the bucket with `infra/storage/catalog_backup.py backup`; `restore` brings it back on a fresh clone (it fetches the pointer from the bucket when `data/catalog-backup.json` is missing).

## History

The tree holds only what is current. Earlier designs, runner chains and tools live in Git history; tags make them easy to reach:

- `design/<study>`: each kept design version, on the commit where its sources exist as built (the tag message names the folder);
- `history/perception-runners`: the perception and high-level runner chain behind the best whole-task numbers;
- `history/design-fitkit`: the sheet-metal design studio tools and joint inventory.
- `history/ankle-foot-fixed-roll`, `history/neck-double-moving-case`, `history/hip-rear-mount`, `history/sheet-metal-leg-v16`, `history/bent-aluminium-upper-body`, `history/symmetric-gripper`, `history/body-sensors-selection`: the October design branches that preserved finished designs' sources and reviews; never merged, kept as tags when the branches were deleted (2026-11-13).

`git tag -l 'design/*'` lists them; `git checkout <tag>` gives the whole tree of that time.

The private blob index (`data/blob-store.sqlite`) and the dashboard catalog (`data/results.sqlite`) look files up by path. Move or rename files only with `infra/storage/move_paths.py OLD NEW`, which moves them (like `git mv`, including evicted files) and rewrites the path in both catalogs and in tracked code and docs. Then commit and run `infra/storage/catalog_backup.py backup`. Files under `artifacts/` and run ids are evidence named by the run that made them; they keep their names.

## Dashboard versions come from Git

A design study's version is the commit of its sources: commit, then `results_hub.py publish` with `source` naming the study folder. The previous version stays in the dashboard as `study_id@commit`, and an earlier version shows its own files (served by SHA-256 from the blob index or from Git history). Learning results record the commit they were published from. See `docs/operations/design-dashboard.md`.

## Research is kept as results, by topic

`docs/research/` holds one document per topic (for example `low-level-control.md`, `perception-and-sensing.md`). Add a finding to the topic it belongs to, in place: current best result first, native-unit numbers with conditions, limits and open questions, the date and the evidence id. Replace a finding when a later result supersedes it; git history keeps the old text. Do not create a note per experiment, run or iteration. A routine training run is reported as a dashboard row, not a document.
