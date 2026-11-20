# Sources and transfer contract

Paths below are relative to the repository root. Read only what the experiment needs.

For the selected assembly, use the [current model reference](selected-model.md)
and resolve `robot/model/current-model.json`. The selected robot has
22 active axes; the 21/19/14-axis intent studies below are historical comparisons.

| Question | Current evidence/code |
| --- | --- |
| What is the selected assembled transfer target? | `robot/model/current-model.json`: corrected full and shell-free `cove50_description` URDF, native MJCF and exact actuator/sensor/timing references. |
| What body is the policy for? | `control/skills/registry.json` and `control/wbc/registry.json` (`body` field), with `control/skills/profiles.json`; the September policy shelf (`skills/ape/`) is in commit `6fbb278e4`. |
| What did spacing change? | `docs/research/mechanical-design.md`: v7 spacing already breaks gait/step; v8 adds relocated case mass/collision; squat survives with changed depth. These are 35 cm scaffolds. |
| What is the 50 cm axis/drive question? | `docs/research/actuators-power-and-electronics.md`: servo catalog points and the 3.2 kg quasi-static gait screen; the October 21/19/14-axis envelope note is in Git history (commit `6fbb278e4`). No 50 cm body or continuous duty is established by this screen. |
| What was the historical bare-frame candidate? | `experiments/compact_morphology/ape_frame50/frame_contract.yaml` (commit `6fbb278e4`) and its README, under `T-59kag5` / `D-3ejdcp`: sketch/Cove anatomy, source-sized components and explicitly assumed dynamics. The contract is not an assembled body with control evidence. `T-xmnhsx` supplies the simulator export reused by `T-t28835`; `T-5dqypt` owns its first learned motion. Re-read live state before relying on this snapshot. |
| What was measured versus assumed? | `robot/model/physics-ledger.yaml`, `robot/actuators/catalog.yaml`, `docs/research/actuators-power-and-electronics.md`, `docs/research/actuators-power-and-electronics.md`; confirm evidence provenance and revision before use. |
| Where is the simulator interface? | `control/common/mjlab_ape/src/ape_walk/task.py` and `physics/candidate.py`; exact compiled contacts matter, not only config names. |
| Which morphology conclusions are historical? | `docs/research/low-level-control.md`, `docs/research/low-level-control.md`. |
| What needs to see during motion? | Use `ape-perception` for 50 cm camera/ray screens. Guessed rigid pivots are not selected assembly kinematics. |
| What real measurements remain? | `docs/research/low-level-control.md`, physical task `T-j8fmq8`; inspect live state before deciding prerequisites. |

For a reproducible comparison, retain model and mesh dependencies, ordered named joints, actuator law and limits, body mass/COM/inertia, collision/contact settings, control and physics periods, observation/reset/action offsets and normalized action scale, checkpoint/config hashes, and seed/start/object distributions. State any missing member instead of substituting a similarly named body.

The historical intent comparisons used 22→21 to remove `head_pitch`, 21→19 to remove both `shoulder_yaw` channels, and a 14-axis alternative that also removed lower-body/shoulder axes and restored pitch. These are not the selected assembly's joint map. Resolve its named axes from the current model and actuator contract. A new topology needs an explicit control contract and its own graded evidence. Conversely, a same-width controller can still be incompatible because joint order, reset offsets or history changed.

Offline pose playback answers where a shape intersects along that trace. A fresh policy-controlled simulation answers how behavior changes. Hardware measurements answer how the selected real mechanism behaves. Preserve these distinctions in task assessments and reports, and grade each level separately on the 0–5 distance-to-viable scale in the [evaluation principles](../../../../docs/design/evaluation-principles.md).
