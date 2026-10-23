# Codex workflows for the ape

**Evaluation rule.** No skill decides anything by pass/fail. Collect clean metrics in native units with their conditions and distributions, judge the behaviour or setup qualitatively in words, and grade each relevant dimension on the 0–5 distance-to-viable scale with confidence and trend. Whenever evidence shows a gap, ask whether the solution or the requirement is wrong, record both, and put requirement changes to the user. This holds for design, mechanics, electronics, control, training and perception; see [evaluation principles](../docs/design/evaluation-principles.md). Historical PASS/FAIL and GO/NO-GO verdicts stay as provenance.

Start with [ape-experiments](ape-experiments/SKILL.md) for learning experiments, or use the relevant focused skill directly:

- [ape-control](ape-control/SKILL.md): native PPO/BC, replay and physical skill composition.
- [ape-perception](ape-perception/SKILL.md): teach/find, belief, social grounding and sensor evidence.
- [ape-body-transfer](ape-body-transfer/SKILL.md): geometry, actuator/contact assumptions and paired control transfer.
- [ape-design-studio](ape-design-studio/SKILL.md): run joint design studios (three competing designers, graded evidence cards, user look reviews, requirement challenges).
- [ape-joint-bank](ape-joint-bank/SKILL.md): retrieve, stage and maintain structured actuator and joint options for CAD and motion design.
- [ape-inspector](ape-inspector/SKILL.md): publish and inspect reviewed results and retained runs in the single local dashboard.
- [train-arena-groot](train-arena-groot/SKILL.md): explicit G1/Berkeley upstream reference work.

The [learning synthesis and Idle map](../docs/research/ape-learning-roadmap-20261006.md) explain the priorities and evidence. These are Codex instructions; `skills/ape/` holds the robot's retained policies.

**Storage rule for every track.** Large completed evidence belongs in the existing private Infomaniak blob archive where it no longer needs local access. Keep active inputs and compact reports/configs/hashes/manifests local; keep generated bulk out of Git. Local `.artifact-store` deduplication does not upload anything. Select useful evidence, verify remote bytes and recovery before eviction, and restore the needed paths/dependencies before native tools or worker exports use them. Follow the shared [retention, archive and restore guide](../docs/design/artifact-storage.md); selected videos still use the results hub's existing upload flow. Do not retain every cache or duplicate output merely because blob storage is available.

The [detailed learning audit](../docs/research/learning-audit-20261007/README.md) records section coverage and primary-source corrections. Read the focused references when needed: [evaluation](ape-experiments/references/evidence-and-evaluation.md), [learning-method diagnosis](ape-control/references/learning-methods-and-failure-diagnosis.md), [frames/time/belief](ape-perception/references/frames-time-and-belief.md), and [dynamics/identification](ape-body-transfer/references/dynamics-and-identification.md). Unproven ape benefits remain explicit hypotheses in existing Idle tasks.

The source of truth is this directory. On this machine these directories are symlinked into `~/.codex/skills/`, following the existing training-skill installation. To install in another checkout, link each desired directory into your Codex skills directory; inspect any existing same-name entry before replacing it. The skills expect access to this repository and its retained artifacts. A link does not restore missing checkpoints, datasets or GPU environments.
