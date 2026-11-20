# Body

The selected complete 3D-printed FDM body is `motor-refit/`: the assembly refitted to the locked three-tier motors (ST3025, STS3250, ST3215-C047; `robot/model/motor-map/`). Each part folder holds its build, FEM and publish scripts; outputs go to `artifacts/robot/body/<part>/`.

| Folder | Part |
|---|---|
| `leg/` | Foot-to-hip connectors (lower leg, upper leg and hip; combined, topology-optimised and space-truss families) |
| `arm/` | Shoulder-to-wrist connectors |
| `neck-head/` | Torso-to-head connectors and the head sensor/compute carrier |
| `torso/` | Centre-body torso, pelvis and the repeated 9-cell battery carrier |
| `assembly/` | The r1 complete body assembly from the part outputs (input to the refit) |
| `motor-refit/` | **The current body:** the assembly refitted to the selected servos (FEM, fit check) |
| `loads/` | Load cases used by the FEM |
| `bom/` | **The robot BOM, start here for hardware:** generated per line with files, producer scripts, print setup, hardware specs, assembly steps and open defects (`build.py`; hand-maintained `lines.json`, `defects.json`) |

Static FEM and collision sweeps may run locally, one heavy job at a time under `infra/tools/memguard.py`. Publish to the design dashboard with `source` naming the part folder; its commit is the version.
