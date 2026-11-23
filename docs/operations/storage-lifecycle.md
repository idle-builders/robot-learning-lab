# Local storage lifecycle

Use one folder for disposable work and another for retained evidence. A training
checkpoint, full episode, original recording or expensive solve is native evidence
even when a script could rerun the experiment. A viewer export, temporary download,
render or other cheap derivation belongs in work. Keep the required inputs, exact
configuration and a compact regeneration command outside disposable work. Verify
that command before declaring an output disposable. Never put a sole original,
an active input, or the only copy of a configuration in scratch.

| Location | Contents | Lifetime |
| --- | --- | --- |
| `robot/`, `control/`, `infra/`, `docs/` | Authored code, configuration and documentation | Git |
| `data/` | Compact catalogs, regeneration recipes and recovery pointers | Local, with private catalog backups |
| `data/storage-runs/` | Exact run command, closed selection, sync status | Compact local records |
| `.cache/work/<run-id>/` | Declared cheap, reproducible output and transport scratch | Removed automatically after successful command completion |
| `artifacts/runs/<run-id>/` | Native evidence and compact report/config | Bulk verified remotely and evicted after closure |
| `.cache/asset-requests/` | Dashboard restores and regeneration inputs | Removed when the request ends |

Existing historical paths stay valid through the cold index. Move a path only with
`infra/storage/move_paths.py`, which keeps the index in step. New runs use the layout above.

Run a producer through the shared entry point:

```sh
infra/lab storage run shoulder-study -- .venv/bin/python robot/body/arm/my_study.py \
  --scratch '{work}' --output '{evidence}'
```

The wrapper substitutes those two arguments and also sets `LAB_WORK_DIR`,
`LAB_EVIDENCE_DIR` and `LAB_RUN_ID`. Producers should accept the two folders or
read these environment variables. It records the expanded command locally.
Write the original observations, full episodes, decisive failures, configuration,
hashes and native reports in evidence. Write cheap derived outputs in work after
verifying their regeneration. Scratch is removed after a successful command;
nonzero commands keep scratch until native evidence is verified remotely, then
remove it too. Interrupted commands keep their scratch for inspection. Native evidence
from completed nonzero commands is retained and offloaded in the same way as any
other completed invocation. Scientific assessment is independent of command exit.

At closure, regular native evidence is sealed by SHA-256, size and modification
time, including small episode/field files. Compact files below 64 KiB in `records/`
and `source/`, plus named top-level report/config/provenance files, remain local.
The wrapper preserves direct script/config arguments; producers must record their
transitive dependencies. Registered disposable
exports are excluded from native upload; put new ones in scratch. Existing shared
CAS input links are not silently evicted as owned output. No recursive upload or
deletion guesses which arbitrary historical folders are inactive.

Native packs and remote per-file manifests are fully read back, hashed, and
sampled through actual range restoration before originals are evicted. A failed
transfer remains queued with local originals intact. A changed selection stays
local with a visible error. Inspect it before declaring a fresh closed selection.
Retained bytes use the existing private Infomaniak bucket and credentials.

Existing scripts can close a completed run explicitly:

```sh
infra/lab storage finish artifacts/EXACT_COMPLETED_RUN
# Or publish and close the same completed native run together:
.venv/bin/python infra/dashboard/results_hub.py publish RECORD.json \
  --close-run artifacts/EXACT_COMPLETED_RUN
```

Do this only after all writers and consumers of its native inputs have finished.
Publication alone does not declare closure. An interrupted managed invocation is
left marked `interrupted`; inspect its evidence and use `finish` on its exact run
directory when all work has stopped. This also releases its scratch after verified
offload. Keep original failure observations in evidence, where they remain retained.

```sh
infra/lab storage status    # running, interrupted, queued and cold runs
infra/lab storage sync      # retry closed runs only
infra/lab storage clean     # delete declared completed scratch only
infra/lab storage install   # macOS: install hourly retry at login
```

The installed LaunchAgent is `ch.robot-learning-lab.storage`; it also runs at
login. Its log is `.cache/storage-sync.log`. It processes only the explicit
closed-run queue, with locks protecting active managed invocations. It does not
upload arbitrary cache contents or evict active experiment directories. To stop
the local retry worker, unload and remove its plist in `~/Library/LaunchAgents/`.
The script is still usable manually on other platforms.

Recover an exact native file or subtree with `infra.storage.blob_store restore`, or use
the dashboard, which creates transient request-scoped copies. To inspect a cheap
registered export without making it permanent:

```sh
.venv/bin/python -B -m infra.storage.regenerable_artifacts EXACT_PATH \
  --output .cache/inspection/assembly.glb
```

Remove that inspection copy when finished. See [storage and recovery](../design/artifact-storage.md)
for index recovery, private snapshots, optional tool/environment setup and the
existing regeneration registry. Moving bytes to private storage does not make a
costly scientific experiment cheaply reproducible.

## Fresh clone on another computer

Git holds code, configuration, selected design sources, recovery pointers and topic
research. Two Git-ignored catalogs connect a clone to everything else:
`data/results.sqlite` (dashboard records) and `data/blob-store.sqlite` (the path to
private-pack index that restores cold files). `data/catalog-backup.json` names their
latest verified copies in the private bucket.

1. Set `ROBOTLAB_ARCHIVE_BUCKET` and `ROBOTLAB_ARCHIVE_ENDPOINT` (see `.env.example` and `public-upstream.md`) and the S3 credentials.
2. `.venv/bin/python infra/storage/catalog_backup.py restore`
3. Restore inputs on demand: native tools and the dashboard call `infra.storage.blob_store.ensure`.

Run `.venv/bin/python infra/storage/catalog_backup.py backup` at the end of every task that
published results or offloaded evidence. It updates the Git-ignored `data/catalog-backup.json`
and the pointer copy in the bucket, which `restore` fetches when the local file is missing.
Without it, a new computer cannot reach anything offloaded after the previous backup.

## Delete proposal for old study folders (measured 2026-11-23, nothing deleted yet)

Owner decision (2026-11-23): delete all of tier 1 and prune all of tier 2 locally, remove the
156,815 dangling symlinks and the folders left empty; keep the hold tier until each folder is
confirmed as derived; keep the whole bucket (the five perception clips are current state).
Re-measure right before deleting and run `rebuild_chain.py --dry-run` afterwards.

Done 2026-11-23 on a fresh measurement (same numbers): `artifacts/` went from 111,617 files and
229,013 symlinks (7.9 GB on disk) to 25,323 files and 31,785 symlinks (4.1 GB). Afterwards
`rebuild_chain.py --dry-run` reports 0 declared inputs missing, and every file kept as build,
code, current dashboard or hold is present at its measured size. One slip to avoid next time:
a file directly in a study folder (`artifacts/X/file`) was keyed by the folder itself, which
removed seven study folders whole. Their kept files (33,644) were copied back from a worktree's
copy-on-write clone of `artifacts/` and checked. Afterwards the store kept every object that a symlink in the checkout or a digest in
`.artifact-store/{dependencies,manifests,runs}` or `data/` names (17,873 of 20,000) and removed
the other 2,127 (6.6 MB, 11 MB on disk) under the store's writer lock; no symlink dangles. The
catalogs were backed up the same day (`catalog_backup.py backup`, read back and hash-checked).
The generator cache's own GC
covers only its cache store; this covers `artifacts/` outside it and the private bucket.

How "what reads it" was measured, per file under `artifacts/` (paths resolved through
`infra/storage/path-moves.json`):

- **build**: inputs of the declared generators (`robot/body/bom/rebuild_chain.py --dry-run`),
  the selected manifest's `native_step` / `mesh`, the `reads` of every generator-cache entry
  (`.cache/work/gen-cache/entries/*/*.json`, all chain generators covered) and `robot/inputs.json`;
- **code**: any artifact path written in tracked `*.py`, `*.sh`, `*.yaml`, `*.json` (a folder
  path keeps everything under it; deliberately broad);
- **dashboard**: `assets` and clip `source_path` of the records in `data/results.sqlite`, split into
  current (selected design per section, newest record of each skill: 25 records) and older (99 records);
- **regenerable**: registered in `data/regenerable-artifacts.json` (recreated by
  `python -m infra.storage.regenerable_artifacts PATH`).

Local `artifacts/`: 2,770 top-level entries, 111,617 regular files, 8.21 GB, plus 229,013
symlinks into `.artifact-store/objects`, of which 156,815 dangle (the blob index
`data/blob-store.sqlite` holds 0 files, so their targets cannot be restored) and 72,198 resolve
into the 104 MB that is left of the store.

| Tier | What | Files | Size | Symlinks |
| --- | --- | ---: | ---: | ---: |
| 1 | whole study folders (second level) that nothing in build, code or current dashboard reads | 73,099 | 2.51 GB | 133,046 |
| 2 | files nothing reads inside folders that still hold read files (unreferenced 12,690, registered regenerable, old-dashboard 489) | 13,179 | 1.34 GB | 7,289 |
| hold | possible external inputs (names with source / capture / recording / vendor, iPhone and Spectacular rooms) | 5,590 | 0.05 GB | 88,515 |
| keep | `artifacts/robot/` (selected body and its chain) | 6,947 | 3.16 GB | 0 |
| keep | read files in other study folders (build, code, current dashboard) | 12,761 | 1.16 GB | 202 |

Tiers 1 and 2 together free **3.85 GB locally** (47 % of `artifacts/`), plus the part of the
104 MB store that only their symlinks reach. The exact path lists (8,590 tier-1 folders,
20,468 tier-2 entries) are kept outside the repository with the session notes; recreate them
with the same measurement before deleting, since the body changes daily.

| Folder | Files | MB | Symlinks | Reads | Kind | Tier 1 MB | Tier 2 MB | Kept MB |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `robot` | 6,947 | 3,158 | 0 | build, code, dashboard | current body; chain outputs recreatable by `rebuild_chain.py`, section outputs it reads as given | 0 | 0 | 3,158 |
| `cove50-arm-shell` | 3,255 | 1,131 | 0 | build: `r1` seated parts, `r3` 4 files, `r4` 1 STEP, `r5` whole (`robot/body/arm/common.py`) | old shell rounds; no producer in the tree | 0 | 1,033 | 98 |
| `cove50-engineering-competition` | 21,354 | 981 | 14,503 | build: `hand-radical` (links), `upper-body`, `waist-pelvis`, `elbow-studio`; code: `ankle-studio`; dashboard | studio rounds; hand, upper-body, waist-pelvis read as source (defect) | 165 | 270 | 547 |
| `cove50-strength` | 11,222 | 917 | 48 | code: one review JSON (`robot/model/build/compliance.py`) | `connector-studies` 902 MB, producer gone; the selected FDM rounds already moved to `artifacts/robot` | 915 | 1 | 0.5 |
| `runs` | 2,647 | 884 | 0 | current dashboard 12 MB | 3 hip-pelvis publications (483 MB, registered regenerable), r6 topopt runs 330 MB | 870 | 1 | 12 |
| `cove50-power-integration` | 7,794 | 228 | 397 | build: `internal-r33`; dashboard | electronics rounds | 18 | 19 | 191 |
| `cove50-sensor-integration` | 1,591 | 217 | 11 | build: `original-shell-layout`, `v4`, `v17`, `v19`, `v21` | sensor rounds read as source (defect) | 7 | 0 | 210 |
| `cove50-right-leg-shell` | 1,194 | 144 | 0 | code: `r14`, `r4/knee125/native` (`robot/body/leg/common.py`) | shell rounds r10-r12 unused | 75 | 0 | 69 |
| `cove50-compute-architecture` | 1,081 | 95 | 241 | build: `r23-powered-head`, `head-freeze`; dashboard | `r10-rock4d-port` 77 MB unused | 84 | 1 | 10 |
| `cove50-basic-skills-retry` | 9,032 | 86 | 23,882 | dashboard 7 MB; code (`control/skills/basic-skills`) | training retries | 78 | 1 | 7 |
| `ape-realistic-people` | 4,874 | 47 | 0 | nothing (one old dashboard text mention) | perception screens (json, py, logs, usda) | 47 | 0 | 0 |
| `cove50-fitted-shell` | 120 | 10 | 0 | build: two torso panels, hardware | fitted shell r1 | 0 | 5 | 5 |
| 2,758 other folders | 40,506 | 308 | 189,931 | mostly nothing; 1,593 hold no regular file | old runs and studies | 251 | 5 | 7 kept, 45 hold |

Private bucket (`<archive-bucket>`, listed 2026-11-23 with `infra/storage/archive_config.client()`):
22 objects, 61.3 MB, under two prefixes.

| Prefix | Objects | MB | What | Proposal |
| --- | ---: | ---: | --- | --- |
| `robot-learning-lab/results/videos/` | 15 | 24.5 | the clip of the newest record of each skill (skills 1-15) | keep (dashboard current state) |
| `robot-learning-lab/results/videos/` | 5 | 23.4 | clips of the perception records `iphone-room-3915`, `iphone-room-3921`, `iphone-room-perception`, `spectacular-room-world001`, `-world002` (not a skill, not a design selection) | delete if they are not current dashboard state |
| `robot-learning-lab/source/vendor-motors/` | 2 | 13.4 | vendor STEP of the STS3250 and ST3215 servos | keep (external input) |

No derived output is left in the bucket; at most **23.4 MB remotely** could go. The bucket
holds no `catalogs/` prefix, although `data/catalog-backup.json` (2026-11-13) points there:
the results and blob catalogs had no remote copy (closed by the backup below). That was a source-of-truth gap, not a
delete, and `catalog_backup.py backup` closes it.

Defects found on the way (not deletes): the selected body still reads study folders outside
`artifacts/robot/` as given inputs (`hand-radical`, `upper-body`, `waist-pelvis`,
`internal-r33`, the sensor rounds, `fitted-shell`, `electronics-integration`, the arm and leg
shell rounds); each should become a script plus its inputs. 1.37 GB under `artifacts/robot/`
is not read by any recorded build read and is kept only by folder-level code mentions; the
generator-cache GC is the place to judge it.
