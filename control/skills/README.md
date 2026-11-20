# Skills

Per-skill policies on the R3 body, kept until the whole-body controller supersedes them. [registry.json](registry.json) names the promoted policy of each skill by run id; [profiles.json](profiles.json) records how each is used (command, entry, observed behaviour, limits).

| Folder | Role |
|---|---|
| `walker/` | The R3 walker (walker200): policy adapter and task; base for walking, stance and the forward lines |
| `forward-walk/` | Walker contracts (model pointer, source BC actor) |
| `forward-heading`, `forward-heading-correction` | Sustained forward walking with heading correction (current: `forward-heading-correction`) |
| `forward-stop-learning`, `forward-entry-braking` | Walk-to-stop (current: `forward-entry-braking`) |
| `deep-squat-feedback/` | Squat, deep crouch and rise |
| `coordinated-arm-reference`, `coordinated-arm-feedback`, `coordinated-arm-feedback-next` | Bilateral arm raise and lower (current: `-next`; the zero-weight reference stays as comparison) |
| `independent-arm-feedback/` | Independent left and right arm motion |
| `supported-skill-transitions/` | Stance, posture and arm transitions |
| `basic-gait/` | Turn training (fit50) and the gait replay used by the forward lines |
| `basic-skills/` | Arm and posture donor components (command student, arm command GRU, crouch) |
| `low_level/`, `low_level_portfolio/`, `policy_transfer/`, `shared_locomotion/` | Earlier controllers the donors and the walker still import |

Each training folder has `setup/` with the configuration of its promoted run. Several setups project earlier rounds byte-exact (`forward-*`), so those rounds stay beside them. Skills without a policy on the current body: one-foot hold, short step, backward walk, side travel, reach and touch, acquire and place.
