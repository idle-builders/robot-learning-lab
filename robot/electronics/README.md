# Electronics

The electronics are parts of the body: the boards, cells, IMU, ADC, cameras, microphone and fan sit in the body manifest
(overlay `electronics_and_battery`), each claimed by a body definition (`robot/**/<builder>.bom.yaml`). Their geometry comes
from the frozen composition `artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json`, a `pinned` input of
`robot/body/assembly/assemble_complete_fdm_body.bom.yaml` (its producer's inputs are gone; the pin says which).

Kept here as sources the definitions cite: `sensing/selection.json` (sensor suppliers), `power/battery.json` (cells),
`power/design.json`, `head.json`, `selection.json` (read by `robot/model/build`), and the one native electronics builder,
`sensing/fsr_pucks.py` with its definition. `power/cad/zbuffer_renderer.cpp` is the renderer the assembly's figures compile.
