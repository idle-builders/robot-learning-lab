# LSM6DSOTR sensor daughterboard

See [the subsystem contract](../../docs/design/imu.md). `main.ato` owns connectivity; exact part/pad definitions are in `parts.json`; `layout.json` owns the two-layer candidate routes. Footprints are vendored from the locally installed official KiCad 10.0.6 library. Their generic package models are CAD proxies, not supplier-qualified assembly drawings.

Run `.venv/bin/python tools/build_imu_board.py --output artifacts/my-imu` from the repo root. Keep the output new. No pull-ups are fitted here: the existing carrier supplies 4.7 kΩ. Two unique single-pad NC nets preserve solder pads without connecting the unused functions.

The board has passed scoped pad mapping and DRC, including a deliberate-short negative test. It has not been fabricated, mounted, calibrated or costed at production volume.
