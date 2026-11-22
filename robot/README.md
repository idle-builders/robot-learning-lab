# Robot

The current Cove-50 robot, as the files it is built from. Selection: [design.json](design.json). Outputs go to `artifacts/robot/…` and the private bucket; each part's folder names its build command.

| Folder | Contents |
|---|---|
| `body/` | The printed FDM structure by part: `leg`, `arm`, `neck-head`, `torso`, the complete `assembly`, the `motor-refit` of the assembly to the selected servos, and `loads` (load cases). |
| `actuators/` | Actuator catalog and parts list, joint bank, motor maps, source geometry, category survey, and the tier pick. |
| `electronics/` | Integration build, power (battery, design, CAD), sensing (selection, mount studies), head compute (`head.json`). |
| `shell/` | The ape look reference: the sketch and the shell build derived from it (`shell/sketch`). The printed body's exterior shells are built in `body/`. |
| `model/` | The simulation model: body + actuators → MJCF/URDF (`cove50_description`), built in stages (see its README). |

**Hardware starts at the BOM:** [body/bom/README.md](body/bom/README.md): what to print and buy, from which scripts and files, and the assembly order of the whole body, all regenerated from the selected body and the `<builder>.bom.yaml` definitions next to the builders.

**Inputs from earlier designs.** The current body build reads outputs of earlier designs (the sheet-metal baseline, shells, hand, sensor layout). Those designs are in Git history with tags (`git tag -l 'design/*'`); their outputs restore from the bucket by path.
