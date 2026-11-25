# Selected electronics composition

The current selection is recorded in `selected-electronics.json` and the Electronics section of `robot/design.json`.

The integrated study `cove50-electronics-integrated`, version `r1-20261026`, combines selected sensing V24 (unchanged V23 geometry), the accepted R23 compute head (`local-regulator-r2`) and torso power R33 with selected upper V5/lower V16 mechanics as context. It retains the exact source poses, native part dependencies, source evidence and replacement ledger. The unselected R24 head experiment is excluded.

The complete neutral scene contains 853 named parts. The full-detail GLB and native STEP/BREP paths are retained; the dashboard uses a smaller inspection GLB. Continuous motion is disabled for this composition. The seven shared neck parts, sensor modules and common torso carrier are counted once. The old head, spine and fan bridge are excluded.

The BOM is partial. It includes the installed electronics and nominal mounting parts with retained source observations. It does not establish complete robot mass, a landed purchase total, electrical continuity or physical qualification. The local head converter has open body supply/return endpoints; the remaining power functions, wiring and moving harness are unfinished. The aggregate 28 W sustained / 45 W peak useful electronics output reserve is counted once. No new runtime estimate is made.

Reproduction from the retained source assets:

```sh
.venv/bin/python -B robot/electronics/build/compose.py
.venv/bin/python -B robot/electronics/build/check_interfaces.py
.venv/bin/python -B robot/electronics/build/render.py
```

`compose.py --metadata-only` refreshes source inventories and BOM transfer without regenerating scene meshes. The composition has no dashboard record of its own (removed 2026-11-25): its parts are in the body (the assembly generator reads `assembly-parts.json`) and show on the body record, generated from the body definitions.

The requested universal policy-learning model is registered separately, dependent on this composition freeze. It must reconcile joint/action interfaces, mass/CoM/inertia, supplier versus identified actuator dynamics, sensor frames and timing, compute delay, and FEM-derived compliance where supported. Static FEM does not identify damping; visual CAD does not establish physical dynamics.
