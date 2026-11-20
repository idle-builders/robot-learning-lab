# Codex workflows for the ape

For the selected assembled robot, resolve `robot/model/current-model.json`.
The shared [selected-model reference](ape-body-transfer/references/selected-model.md)
identifies the `cove50_description` package, corrected URDF, shell-free view,
native dynamics and sensor/actuator contracts. Installed skill directories are
symlinks to this tree, so these references apply to the installed skills too.

**Evaluation rule.** No skill decides anything by pass/fail. Collect clean metrics in native units with their conditions and distributions, judge the behaviour or setup qualitatively in words, and grade each relevant dimension on the 0–5 distance-to-viable scale with confidence and trend. Whenever evidence shows a gap, ask whether the solution or the requirement is wrong, record both, and put requirement changes to the user. This holds for design, mechanics, electronics, control, training and perception; see [evaluation principles](../../docs/design/evaluation-principles.md). Historical PASS/FAIL and GO/NO-GO verdicts stay as provenance.

Start with [ape-experiments](ape-experiments/SKILL.md) for learning experiments, or use the relevant focused skill directly:

- [ape-control](ape-control/SKILL.md): native PPO/BC, replay and physical skill composition.
- [ape-body-transfer](ape-body-transfer/SKILL.md): geometry, actuator/contact assumptions and paired control transfer.
- [ape-design-studio](ape-design-studio/SKILL.md): run joint design studios (three competing designers, graded evidence cards, user look reviews, requirement challenges).
- [ape-joint-bank](ape-joint-bank/SKILL.md): retrieve, stage and maintain structured actuator and joint options for CAD and motion design.
- [ape-inspector](ape-inspector/SKILL.md): publish and inspect reviewed results and retained runs in the single local dashboard.

These are agent instructions. The robot's policies are named in `control/*/registry.json`; their checkpoints are in the bucket.

**Versions, records and research for every track.** Git holds versions; file names never do. Edit a source and commit; never write `_v2`, `-r3`, `final` or dated copies of scripts, cards, configurations or study folders, and name new files by content. Record the commit hash wherever a result depends on a specific version. Process records (reviews, challenges, authorizations, dispatch and provider status, verifications, receipts, bindings, handoff packets, input snapshots, publication checks, locks and logs) go to the Git-ignored `.bookkeeping/<task-or-run-id>/`, never into `data/` or `studies/`. Research findings go into the matching topic document in `docs/research/` (updated in place, results only), never a note per run. See [git and bookkeeping](../../docs/operations/git-and-bookkeeping.md); `.githooks/pre-commit` refuses new numbered names and committed process records.

**Storage rule for every track.** Large completed evidence belongs in the existing private Infomaniak blob archive where it no longer needs local access. Keep active inputs and compact reports/configs/hashes/manifests local; keep generated bulk out of Git. Local `.artifact-store` deduplication does not upload anything. Select useful evidence, verify remote bytes and recovery before eviction, and restore the needed paths/dependencies before native tools or worker exports use them. Follow the shared [retention, archive and restore guide](../../docs/design/artifact-storage.md); selected videos still use the results hub's existing upload flow. Do not retain every cache or duplicate output merely because blob storage is available. After any publication or offload, run `.venv/bin/python infra/storage/catalog_backup.py backup` and commit `data/catalog-backup.json`, so a fresh clone can restore everything (see "Fresh clone" in `docs/operations/storage-lifecycle.md`).

Read the focused references when needed: [evaluation](ape-experiments/references/evidence-and-evaluation.md), [learning-method diagnosis](ape-control/references/learning-methods-and-failure-diagnosis.md), [perception results](../../docs/research/perception-and-sensing.md), and [dynamics/identification](ape-body-transfer/references/dynamics-and-identification.md). Unproven ape benefits remain explicit hypotheses in existing Idle tasks.

The source of truth is this directory (`.agents/skills/`), which is discovered directly by both Codex and Antigravity; `codex-skills` is a repository symlink for compatibility with existing scripts and notes, and `~/.codex/skills/` symlinks point here as well. The skills expect access to this repository and its retained artifacts. A link does not restore missing checkpoints, datasets or GPU environments.
