# Selected assembled ape model

Resolve `robot/model/current-model.json` at the start of work on the
selected robot. Record that pointer's SHA and the exact referenced inputs in
the run. It is the model selector; `robot/design.json` identifies
the selected mechanical/electronics CAD source. Historical checkpoint body
overrides in the September shelf (`skills/ape/registry.json`, commit `6fbb278e4`) describe those checkpoints' source
embodiments, not the current transfer target.

The current ROS package is **`cove50_description`**. Its reviewed visual export
is `r3-urdf-visual-r1-20261026`, retaining 22 axes and nominal 3.8303 kg mass.
Resolve the pointer rather than copying this dated version into a new runner.

| Need | Resolve or use |
| --- | --- |
| Corrected full URDF | Pointer field `rigid_URDF` |
| Shell-free URDF | Pointer field `shell_hidden_URDF`; same inertials, mass, joints and collisions |
| Floating-base rigid simulation | Pointer field `rigid_MJCF`, with `actuators`, `sensors` and `timing` contracts |
| Provisional FEM simulation | Pointer field `compliant_MJCF`; flexibility is not encoded automatically by the rigid URDF |
| Motor/sensor/frame details | Pointer fields `actuators`, `sensors`, `timing`; adjacent R3 `frames.json` and source contracts |
| Complete URDF mesh package | `artifacts/robot/model/urdf/cove50-urdf-shell-hidden.zip` |
| Export recipe and ROS package files | `robot/model/urdf/` |
| Mesh/package hashes and picture | `data/urdf-inspection-r1-20261026/verification.json` |
| Fresh independent review | `data/urdf-inspection-r1-20261026/independent-review/review.md` |

The ZIP contains relative-path `cove50.urdf` and
`cove50-shell-hidden.urdf`, their `package://cove50_description/` ROS variants,
all 1,445 mesh dependencies and copied actuator/sensor/mass/timing/FEM sidecars.
Use the existing private archive to restore the exact bundle when cold, check
its recorded SHA and extract it to obtain those mesh paths. Git retains the
URDF/configuration and regeneration source; meshes are retained in the bundle.

All 22 motor-case visuals now retain the complete selected CAD surfaces. The
old 4,000-face reductions lost most case surface area despite similar bounds.
The shell-free URDF removes 45 appearance visuals, retaining shell mass and
physical contacts. Blue motor cases in its inspection PNG are a render-only
highlight; exported URDFs retain source colors.

Plain URDF import does not supply a floating base, apply SRDF exclusions, drive
the motors or reproduce sensor delays. Use the native model plus its Runtime
contract for those dynamics. The rigid R3 model uses 0.25 ms numerical steps,
20 ms policy commands and the recorded delay assumptions. The selected native
MJCF still has its original reduced visual meshes: the URDF visual repair does
not automatically repair camera observations rendered from that MJCF. State
which visual model an image-dependent experiment actually uses.

When transferring an existing policy, retain its checkpoint and source body,
then map named joints, observations, action units/history, resets and clocks
explicitly to this target. Loading a new body file alone is not that adapter.
Do not relabel an old checkpoint's body as this model or treat a shell-free
picture as evidence of walking. Unknown hip stops, installed motor/sensor
calibration, conditional mass and FEM assumptions remain in the review.
