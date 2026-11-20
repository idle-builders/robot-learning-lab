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
