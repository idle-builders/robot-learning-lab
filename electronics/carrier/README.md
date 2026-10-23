# Common logic carrier

`main.ato` owns connectivity. `parts.json` owns selected part identities/pin maps/values, and generates `parts/parts.ato` inside each run. `layout.json` owns placements, outline, antenna cutout, holes and one explicit interrupt route. The KiCad library footprints are vendored under `parts/`; see its license and attribution.

Build from the repository root:

```sh
.venv/bin/python tools/build_carrier.py --output artifacts/my-carrier
```

The output must be new. This compiles Atopile, restores complete KiCad footprint geometry while preserving compiled nets, places parts, routes locally, imports the session, checks connections against firmware, runs independent DRC and a deliberate short, then exports a populated STEP and image. Exact sources/hashes and all failed runs are retained. Nothing is ordered, uploaded for manufacture or flashed.

This is a routed **logic carrier candidate**, not the complete power/controller system. The external power board, independent motor disconnect, IMU, protection and harness remain unfinished. The exported Pico W 3D body is a package proxy for Pico 2 W. See [design, evidence and limitations](../../docs/design/carrier-v1.md).
