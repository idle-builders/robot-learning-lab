# Selected APE0 URDF with complete motor visuals

This export retains the selected R3 joint tree, all 22 axes, collisions, inertial
tensors, actuator/sensor contracts and nominal 3.8303 kg mass. It restores all
22 motor-case visual meshes from the exact selected CAD scene. The earlier
4,000-face reductions removed enough case surface to make motors look missing.
The original R3 URDF is retained as `cove50-source.urdf` for provenance.
`source-model.json` records that original release's paths and identities; it
is provenance, not a package-local model selector.

`cove50-shell-hidden.urdf` removes 45 shell/head appearance visuals only.
`cove50.urdf` retains the shell. Their corresponding `.ros.urdf` files use
`package://cove50_description/` mesh URLs for ROS. Both models retain the same
mass and collision geometry. Blue motor cases in the inspection PNG are a
rendering aid; exported URDF materials retain their source colors.

The Git files contain the URDF and configuration; generated meshes stay in the
private portable package, following repository storage policy. Unzip
`artifacts/robot/model/urdf/cove50-urdf-shell-hidden.zip`
to obtain a self-contained package with all mesh dependencies and both views.
That directory can be built as a ROS `ament_cmake` description package.
The archived package SHA and independent review are recorded under
`data/urdf-inspection-r1-20261026/`.

Regenerate from the retained selected CAD scene and R3 source package:

```sh
.venv-control/bin/python robot/model/urdf/build.py
```

This is an export and visualization repair, not new mechanics or a trained
walking policy. URDF alone does not implement servo torque-speed laws, sensor
delay, FEM flexibility or a floating simulation base. Use the R3 native MJCF
and Runtime contracts for those simulations. Hip travel, hardware servo gains,
motor thermal behavior and unfinished installation mass remain unmeasured
assumptions documented in the original R3 release.
