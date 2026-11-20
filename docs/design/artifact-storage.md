# Artifact storage

Historical immutable files share a SHA-256 object store. Cold files can live in the existing private Infomaniak bucket with an exact-path restore index. The local object store deduplicates content; it does not by itself upload or evict it. This is storage maintenance, not a new engineering validation. The original migration retained failed runs, incomplete runs, reports, source snapshots, seals and logs; ongoing retention is selective (below).

## Ongoing retention

New producers use the [managed storage lifecycle](../operations/storage-lifecycle.md):
`infra/lab storage run NAME -- COMMAND` separates disposable work from native
evidence, automatically verifies/offloads closed evidence and evicts local bulk.
The local hourly retry worker processes only declared closed runs. Existing
producers can use `storage finish` or results publication with `--close-run`.

Use the existing private Infomaniak blob archive for large, completed evidence that is worth retaining but no longer needs to occupy local disk: datasets and sensor recordings, full episode/state traces, selected checkpoints, native CAD/simulation outputs and selected media. Keep code, compact reports/configurations, hashes, manifests and logical asset references local; keep active writers and the exact inputs/checkpoints needed for current work materialized. Retain full episodes and decisive failures required by the experiment, even when their bulk bytes move to blob storage. Local presence is not the definition of retained evidence.

Choose what is worth retaining before uploading. At task completion, delete unused derived outputs from local disk and remote storage when one documented script invocation cheaply recreates them from retained inputs. Verify regeneration, retain the script/configuration/input hashes and a compact regeneration record, and update asset/restore references before deletion. Do not upload these disposable outputs as backups. Keep original observations, full episodes, decisive failures and expensive or non-reproducible native evidence; having a script that reruns training, data collection or a costly solve does not establish cheap regeneration. Hash-verify remote copies before evicting local evidence that must be retained. Age or a rejected design alone does not establish that its evidence is unused. `.artifact-store` is local deduplication, not a remote backup or disposable cache.

Close a run by publishing its compact result and complete asset references through the results hub, using its existing upload flow for selected videos. Archive the explicit stable selection with the workflow below, verify remote recovery, then evict eligible local bytes. Record the selection/manifest, hashes, verified storage destination and restore prefix so the next reader can recover them. Before native replay, CAD work, dataset enumeration, tree capture or worker export, restore only the required run/dataset and its dependencies; evict unchanged restored copies when that work is finished. Runpod remains temporary compute and bounded transport; project blob credentials stay local.

Keep generated bulk out of Git: datasets, videos, meshes/CAD exports, full traces, checkpoints, caches and runtime databases belong in the appropriate artifact storage, with code and compact provenance in Git. Respect the repository ignore rules rather than force-adding those outputs. Back up runtime catalogs and the blob index through the existing recovery workflow; an indexed asset and a committed source file serve different purposes.

`data/regenerable-artifacts.json` records disposable expanded FEM exports whose original bytes have been reproduced from retained unit fields and load coefficients. The dashboard regenerates these on demand; native tools can use `.venv/bin/python -B -m infra.storage.regenerable_artifacts EXACT_PATH`. Each regeneration checks input hashes and the original output SHA-256. `infra/storage/prune_regenerable_artifacts.py --apply` verifies every registered recipe and compacts mixed packs before deleting their obsolete copies. After recovering an older blob-index snapshot, run `recover-index` to obtain the current pack ranges.

Gaussian room PLY exports are also registered for exact CPU regeneration from retained native checkpoints. The pinned gsplat 1.5.3 exporter and checkpoint hashes are recorded per recipe; no reconstruction training is repeated. Use the same `infra.storage.regenerable_artifacts EXACT_PATH` command. Temporary downloaded checkpoint packs and export verification copies are removed automatically. The 29 October cleanup proofs and private-pack compaction receipts are in `data/storage-cleanup-20261029/`.

Historical composed GLBs can also regenerate exactly from retained common/owned mesh inputs and a compact copy of their original scene layout. `infra/storage/glb_regeneration.py` verifies the regenerated binary chunk and whole output against their original hashes; it does not rerun CAD or physics. Required mesh inputs are retained privately and restored transiently during browser regeneration; explicit native work can materialize them locally. Use the same exact-path regeneration command above. Proofs, layout templates and the explicit release selection are in `data/storage-cleanup-20261029/reconciliation/`. For inspection, keep the working export in cache: `.venv/bin/python -B -m infra.storage.regenerable_artifacts EXACT_PATH --output .cache/inspection/assembly.glb`; remove that copy when finished.

## Cold files and local restoration

`data/blob-store.sqlite` maps cold files to compressed ranges in private blob packs. Each pack has a separate remote manifest with the original paths, byte counts and SHA-256 hashes. Archiving verifies the complete remote pack and manifest, restores sample members through the actual range-download path, then rechecks each unchanged local file before optional eviction. Sources changed during archiving remain local. Keep executable sources, compact records and narrowly identified active inputs local. Large closed records and selected evidence can remain cold; a selected studio does not require an entire folder to stay local.

The results dashboard reads cold assets and regeneration inputs in request-scoped `.cache/asset-requests/` directories, removed after delivery or failure. Browser inspection does not permanently restore a run. Archive listings retain cold assets. `infra.storage.artifact_store.validate_file` also restores indexed files before checking them. Native applications and code that read the filesystem directly need an explicit restoration first; XML mesh dependencies must be restored with their body. The dashboard's retrieval is not a virtual filesystem.

Tree snapshot, dependency capture and migration planning require evicted regular files to be restored first. They stop with the exact restore command instead of silently omitting files that are absent from a filesystem walk. Known individual files and CAS objects are retrieved when validation or materialization needs them.

```sh
# Inspect indexed and currently materialized bytes.
.venv/bin/python -B -m infra.storage.blob_store status

# Restore a specific file or a whole logical directory before native reanalysis.
.venv/bin/python -B -m infra.storage.blob_store restore artifacts/EXACT_RUN/
.venv/bin/python -B -m infra.storage.blob_store restore assets/real_indoor_scenes/EXACT_DATASET/

# CAS-linked inputs use their object path in the index. Opening through the
# dashboard or validate_file follows the logical link and restores that object.

# Evict unchanged, already archived material after the native work is finished.
# This prefix selects indexed regular files; CAS objects have their own paths.
.venv/bin/python -B -m infra.storage.blob_store evict artifacts/EXACT_RUN/

# Recover a lost index from the private pack manifests. No source files deleted.
.venv/bin/python -B -m infra.storage.blob_store recover-index
```

Restoration verifies the original SHA-256 and byte count and refuses to replace different local content. Credentials use the same Keychain/environment mechanism as the results hub. Packs and recovery manifests stay private in `<archive-bucket>`; no new storage provider or worker credential is introduced. Keep the local index and cleanup selections in backups; the remote pack manifests can reconstruct the index independently. Explicit native restores occupy local space until evicted again; browser requests use transient files.

Archive only an explicit, reviewed selection using `.venv/bin/python -B -m infra.storage.blob_store archive PLAN.json.gz --prune`. A selection has schema `robot-learning-lab.cold-selection.v1` and a `files` array of exact `path`, `bytes` and `mtime_ns` entries. The command checks these against the current files and refuses another concurrent archive writer. Without `--prune`, it uploads and verifies without deleting local files. This does not select an experiment by its scientific quality or change a retained assessment.

For a deduplicated run, select the physical `.artifact-store/objects/sha256/` payloads from its manifests, not its logical symlinks. Preserve the logical links and reference manifests locally. `restore RUN_PREFIX` follows those links to indexed objects, but `status` and `evict` select indexed physical paths by prefix: inspect or evict the exact object paths separately, checking that no active run still needs them materialized. Evicting only the logical run prefix does not release its shared CAS bytes.

Closed native observations can already be archived under another exact path, including a CAS object, while their original regular-file paths still occupy disk. `lookup` does not search for a regular file by content hash, and archiving without `--prune` retains local bytes. Before releasing those originals, verify every local SHA-256 and size against an unchanged verified pack, add the exact logical aliases to both its fully read-back remote manifest and the live index, and exercise recovery through those aliases. The 29 October reconciliation in `data/storage-cleanup-20261029/reconciliation/` records this operation for closed episode copies. Preserve current policy checkpoints and active input capsules.

## Git history recovery

The public history starts on 2026-11-20; the private history before it stays on the owner's computer (see `docs/operations/public-upstream.md`). An older bundle of the original history, from before generated bulk was removed from it on 2026-10-21, is in the private bucket. Its recovery manifest is `data/git-history-recovery-20261021.json`; `data/git-history-commit-map-20261021.txt` maps its commit ids to the later ones. Restore it into a fresh directory without touching the current repository:

```sh
.venv/bin/python -B infra/storage/artifact_archive.py restore \
  --manifest data/git-history-recovery-20261021.json \
  --destination /path/to/fresh-recovery \
  --endpoint <archive-endpoint> --region us-east-1 \
  --bucket <archive-bucket>
git clone --mirror \
  /path/to/fresh-recovery/.cache/robot-lab-original-history-20261021.bundle \
  /path/to/original-history.git
```

The restore checks the complete byte count and SHA-256 and refuses an existing destination. `data/storage-recovery-20261021.json` describes a whole-lab recovery snapshot of the same day (databases, Git bundle, working patches, untracked sources); restore it the same way with that manifest.

**Keep logical artifact references and their verified remote recovery routes together.** Local CAS payloads can be evicted after verified offload. A portable offline copy needs the referenced bytes materialized; a lean checkout retains the links, manifests and cold index. Unselected runs and raw variants can be discarded after their compact decision evidence is retained. Unused optional environments and large native tool installations were removed in the 29 October lean cleanup. Restore optional environments with `.venv/bin/python -B infra/storage/restore_local_environment.py ENVIRONMENT`; exact installed package lists and native installer/version references are in `data/storage-cleanup-20261029/lean/`. The main and control environments remain available. Restore optional native tools with `.venv/bin/python -B infra/storage/restore_native_tool.py kicad` (or `arm`, `bambu`, `prusa`); downloads are hash-checked before installation.

## How the content store works

The store is a small filesystem adapter: standard SHA-256, JSON and file operations, with native APFS `clonefile` for independent copies and an ordinary-copy fallback. Content objects live at `.artifact-store/objects/sha256/<first-two-digest-characters>/<remaining-characters>`. Completed runs keep their original logical paths as relative symlinks. Objects are read-only; never edit through these links. Hard links are not used. A writable tool input must be an independent materialized file, not another shared-object alias. (DVC and git-annex were considered for the migration and not installed.)

## Provenance and recursion boundaries

- Original run JSON, hash seals and source/output bytes are unchanged by migration. Historical nested paths remain available; the migration does not erase them.
- Migration plans are sealed JSON inventories outside run trees. They record every logical path, SHA-256, byte length, original mode and modification time. Apply attempts retain separate terminal results. Hashes detect changes; these are not signed attestations.
- `snapshot_file(source, destination)` stores immutable inputs once, creates a compatibility link, and appends the reference to `.artifact-store/runs/<hash-of-run-path>/references.jsonl`. The reference retains the logical source path, content hash and metadata. An existing destination is an error.
- `retain_dependency(upstream_run)` retains an immutable, hash-bound manifest of the upstream tree and its objects. New reports record that reference in `dependency_manifests`; they do not copy the upstream tree beneath another `source/`. Historical nested files can occur inside the dependency manifest without spawning more directory nesting. Historical list-shaped dependency declarations are retained with stable index labels and every child reference is verified; new reports use named mappings. `verify_dependency` checks its objects even if mutable upstream paths later change.
- New snapshot operations reject nested `source` destinations, store paths disguised as source inputs, directory symlinks and recursive upstream source capture. `snapshot_tree` is for direct asset/source trees without earlier artifact snapshots.
- No current producer writes into the store; it holds the historical runs migrated in October. New runs use the [storage lifecycle](../operations/storage-lifecycle.md). A historical run is finalized with the commands below. Active outputs remain regular writable files until closed.
- Private outputs outside the workspace, including test fixtures, use verified independent copies. Portable worker exports deliberately remain materialized and keep their strict no-symlink verification. This is a necessary portability exception, not evidence loss.

Use logical paths when opening files. Calling `.resolve()` on an XML or mesh file can replace its directory and suffix with the object-store name, breaking relative asset lookup or format selection. The adapted readers check logical confinement and accept only byte-verified, direct store links where an exception is needed. Do not relax a worker's extraction rules to accept arbitrary links.

## Migrate or finalize a completed run

Run from the workspace with the existing environment. Never target an active run or an open log writer. Use a fresh plan path each time; the commands refuse to overwrite it.

```sh
.venv/bin/python -B -m infra.storage.artifact_store plan artifacts/COMPLETED_RUN
# The command prints its manifest path. Use that same path below.
.venv/bin/python -B -m infra.storage.artifact_store apply .artifact-store/manifests/PLAN_ID/manifest.json
.venv/bin/python -B -m infra.storage.artifact_store verify .artifact-store/manifests/PLAN_ID/manifest.json
```

Plans stream SHA-256 over the inputs before replacement. Apply verifies original bytes again, verifies each stored object, then atomically replaces the legacy file with its relative link. It uses a single local writer lock. Stale inputs or corrupt objects fail closed; already processed files and the failed attempt remain. Repeating `apply` on the same sealed plan resumes safely. Replanning a stored run treats its valid CAS links as content entries, so it remains verifiable and materializable.

`verify` checks every inventoried historical path and every distinct expected object. It does not change an engineering failure into a pass. It also does not claim that today's mutable repository source still matches an older study: producer edits must remain visible as source drift.

The local deduplication adapter has no garbage collector. Explicit cold archival above is a separate operation and can evict verified objects while retaining their remote recovery record and logical links. Empty directories are retained in place during migration; file-based materialization does not promise to reproduce empty directories, extended attributes or original filesystem allocation. Original file bytes and recorded mode/time metadata are the restoration contract. The migration assumes completed inputs have no concurrent writers; do not use it as live recording storage.

## Materialize for native tools or portable exports

```sh
.venv/bin/python -B -m infra.storage.artifact_store materialize \
  .artifact-store/manifests/PLAN_ID/manifest.json /path/to/NEW_PRIVATE_DIRECTORY
# Add --writable when a native tool must modify the private copies.
```

The new directory contains manifest paths, e.g. `NEW_PRIVATE_DIRECTORY/artifacts/COMPLETED_RUN/...`. It must not exist. Files are verified after copying; native APFS clones remain independent when modified. Default restoration preserves recorded permissions, while `--writable` adds owner write permission. Never chmod or edit a shared store object to make a tool work. For a historical portable worker bundle, materialize its run and invoke its unchanged standalone preflight from the restored bundle directory.

To preserve a new dependency independently, use its `manifest_path` with the same materialization command. For backup/relocation of the complete lab, keep the artifacts and store under one common root so relative links continue to work. Restoring legacy bundles with absolute authored paths may still require their existing relocation workflow; deduplication does not rewrite those bytes.

## Evidence and limits

The storage report with the measured before/after figures, migration manifests and compatibility checks is in the private history: `git show 9bf48951f:docs/research/artifact-storage-report.md`. Store integrity is verified by SHA-256; read-only permissions guard against accidental writes, not against an owner deliberately changing permissions and bytes. Test corruption is detected rather than silently repaired.
