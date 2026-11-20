# Finished designs kept in Git history

The selected robot is the complete printed FDM body (`robot/design.json`, sources in `robot/body/`). Earlier designs that the owner accepted or that reached a finished, published state are kept in Git history at the exact paths and names they were built with. Intermediate rounds that were never marked finished are not kept.

These commits and tags are in the private history on the owner's computer, not in the public repository (`docs/operations/public-upstream.md`). Their sources are in commit `427b0e980` ("Snapshot sources of finished designs …") unless the table says otherwise. Their dashboard studies also have tags: `design/hip-direct-serial-low-roll`, `design/neck-double-moving-case-C`, `design/cove50-ankle-foot-fixed-roll`, `design/cove50-hand-symmetric-geared`, `design/cove50-bent-aluminium-upper-body`, `design/cove50-simple-sheet-leg-chains` (`git tag -l 'design/*'` lists every kept version). Their bulky outputs (STEP, GLB, renders, FEM fields) are in the private bucket and restore by path through `data/blob-store.sqlite` (see `docs/operations/storage-lifecycle.md`, "Fresh clone").

Every design below was rebuilt from its commit on 2026-11-12, from a clean checkout with inputs restored from the bucket. The last column says what that took and how the outputs compared with the recorded ones (STEP files always differ in their timestamp line, so "same geometry" means identical solids, volumes and bounds).

| Design | Status when finished | Sources, and how to rebuild | Rebuild result |
|---|---|---|---|
| Hip, direct serial low roll, rear mount | owner's desired round, 2026-10-21 (M-6hcvnd) | `427b0e980`: `studies/cove50-engineering-competition/hip-body-first-20261020/rear-mount-r3/build.py` | GLBs, parts and sample grids identical; STEPs same geometry |
| Neck/head, double moving case | owner's final solution, 2026-10-21 | `427b0e980`: sources in `…/neck-studio/minimal-serial-20261021/` (`finalize-20261021/` holds the acceptance): `mechanism/build.py`, `repair-camera-cap.py`, `lead/inspection-r3.py` | 33 part STLs and BOMs identical; final STEP and inspection GLB same geometry (within 1e-6) |
| Ankle/foot, fixed roll | owner-designated, 2026-10-21 | tag `history/ankle-foot-fixed-roll`: build scripts `…/ankle-studio/narrow-round27/`, then `direct-sole/build.py` from the repository root | neutral GLB and STLs identical; STEPs same geometry; pose STEPs and report come from a collision check not rerun |
| Hand, symmetric geared gripper | owner-decided, 2026-10-26 | `427b0e980`: `…/hand-radical-20261026/A/twin-crawl-wrist-seats-r1/` (build, interface and fastener checks, mesh, finalize, seal), then `lead-twin-crawl-r1/scene.py`; create the output folder first | everything identical, including the accepted report and assembly GLB |
| Sheet-metal waist, pelvis and legs (integration V16) | owner-authorized CAD rounds, 2026-10-21 to 10-23; the baseline the FDM body replaced | `427b0e980`: `…/waist-pelvis-20261021/lead/`; the designer packets `A/consistent-recess-v16`, `B/deeper-web-v16` are at tag `history/sheet-metal-leg-v16`. `A/build_r1.py`, `hold_native.py`, `lead/consistent-recess-v16/pink.py`, `B/yoke.py`, then `integrate.py --packet` | packets and integration identical or same geometry; BOM text differs in a few process fields edited at publication |
| Blexon bending sheets, legs and arms | supplier package, 2026-10-28 (T-pa3dgm) | `427b0e980`: `robot/blexon-leg-revision-20261028/`, `robot/blexon-arm-revision-20261028/` (READMEs list the steps; the leg pins the recorded V16 packets) | STLs and GLBs identical; STEPs same geometry |
| Fitted hard-shell patterns, right leg and arms | verified, 2026-10-29/30 (T-ewe9am, T-8dfa7r) | `427b0e980`: `robot/fitted-shell-cad/right-leg-pattern/`, `robot/fitted-shell-cad/arm-pattern/` (`reproduce.py` needs `.venv` at the root) | right leg 46 of 47 identical (zip: entry timestamps); arm passed its own byte comparisons |

The finished electronics rounds behind the selected electronics design (2026-10-26) are in commit `b46960e7b`:

| Design | Status when finished | Sources, and how to rebuild | Rebuild result |
|---|---|---|---|
| Internal power space r33 | owner-selected, 2026-10-26 | `robot/power-integration/cad/internal-r33-20261025/` and the helpers it imports; the last rounds re-run from the recorded r23 | assembly, parts and inspection GLB identical; STEPs same geometry |
| Head compute freeze and powered head | compute rounds the electronics design binds | `robot/compute-architecture/r23-powered-head-20261026/` (`build.py`, `package.py`); `head-freeze-20261026/` only writes selection pointers | all identical |
| Selected sensor CAD | `robot/electronics/sensing/selection.json` | tag `history/body-sensors-selection`: the builder `studies/cove50-sensor-integration/original-shell-layout/`, `build.py` (create the output folder first) | 188 of 189 identical; `interface.json` differs only in elapsed time |

Inputs that existed only on the laptop when these were rebuilt (neck reports, one ankle record, the hand's link tree, the right-leg shell records, the arm shell's parent round and source meshes) were archived to the bucket on 2026-11-12, so a fresh clone with the bucket can repeat every rebuild.

To work on one again, check it out next to the current tree and restore its inputs:

```bash
git worktree add ../robot-lab-sheet-metal 427b0e980
```

or bring single folders back into this checkout with `git checkout 427b0e980 -- <path>`. If you develop it further, give it a folder named by content and continue in place; do not reintroduce the numbered names (`docs/operations/git-and-bookkeeping.md`).

A fresh clone needs the private bucket settings (`data/private-archive.json` or `ROBOTLAB_ARCHIVE_BUCKET`) and the restored catalogs (`infra/storage/catalog_backup.py restore`).
