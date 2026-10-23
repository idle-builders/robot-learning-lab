# Candidate 5 V logic supply

J2 now selects a JST XH output connector; J1/J3 retain JST PH. The optional `--verification electronics/logic-supply/vision-verification.yaml` profile exercises the camera-inclusive planning load. See [current power iteration](../../docs/design/vision-power.md) for rating conditions, retained results and limits.

The editable circuit is `main.ato`. `components.json` maps references to exact candidate parts in `parts.json`; `verification.yaml` owns voltage bounds, load probes and solver settings. Parts, connectors and their physical pad numbers compile through local Atopile. The simulation consumes those compiled nets.

From the repository root:

```sh
.venv/bin/python -m lab.logic_supply --output artifacts/my-logic-supply --case nominal
.venv/bin/python -m lab.logic_supply --output artifacts/my-logic-load --case load-step
.venv/bin/python -m lab.logic_supply --output artifacts/my-logic-all
```

Output must be a new directory. Omitting `--case` runs all probes, including the slow cold-start model. Selected subsets remain `partial` because omitted cases are `not_run`. Failed and incomplete transients remain failures even if the simulator exits with code 0. Files include compiled nets, BOM, source snapshots/hashes, tool versions, exact SPICE input, logs, waveform data and reports. A full run can take tens of minutes locally.

| Connector | Pin 1 | Pin 2 | Intended connection |
|---|---|---|---|
| J1 | GND | VIN | Candidate protected 3S battery, upstream of motor stop |
| J2 | GND | V5 | Logic carrier supply; rail allocation 0.6 A nominal, 1 A probe |
| J3 | GND | EN | Optional short-to-ground disable; not a motor emergency stop |

The enable divider is not a battery-protection circuit. JST connectors and a 3 A regulator do not establish a 3 A carrier/system rating. The generated PCB is unplaced and unrouted; no fabrication files or power-board envelope are qualified. See [circuit decisions and measured simulation scope](../../docs/design/logic-supply.md).

## Routed and packaged candidate

`tools/build_logic_board.py --output artifacts/my-logic-board` adds explicit routing, DRC/negative-short checks, pad-net parity, STEP, SVGs and a render. Run with `.venv/bin/python` from the repo root. `lab.logic_packaging` consumes a passed board and retained detailed robot assembly to check the battery, torso and connector-space allocations. See [current layout, frame and limits](../../docs/design/logic-board.md); the earlier unplaced-circuit scope above does not describe the separate routed-board builder.
