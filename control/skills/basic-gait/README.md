# Selected R3 gait replays

The stop/burst collector uses the exact candidate200 checkpoint and the frozen plan at `data/basic-skills-retry-20261027/stop-step/execution-plan.json`. It runs `.30→0` at2s/full6s, `.35→0` at4s/full8s, and `.25` pulses ending at.6s/.9s/full3s, with seeds142101/142102. Eight sequential visits total2000controls/160000physics steps and zero optimizer updates. These are requested counts; native reports retain actual completion and censoring.

`plan_replay.py` validates one named case against its plan and method, then calls the original native task, checkpoint initialization and full-episode recorder. It keeps the same actor throughout each switch and copies the motor ring, sensor packets, RNG and raw history around the command edit. `worker_gait.py` executes the frozen batch; `launch_gait.py` submits exactly once and restores every member with SHA/size verification, including the final receipt members. Run remote collection only after the campaign lead's explicit dispatch and reviewed IaC. The stop request has3600s totalTTL,1800s execution timeout and1650s worker budget; an absolute-expiry guard requires1650s plus120s margin before inputs/bootstrap/native work. Status/progress inspections are at least300s apart. No retry or watchdog is installed.

After native restoration, measure the stop batch with:

```sh
.venv-control/bin/python -B control/skills/basic-gait/measure_stop.py --run-dir artifacts/cove50-basic-skills-retry/stop-step-20261027/cuda-r1 --out data/basic-skills-retry-20261027/stop-step/measurements.json
```

The measurement script reads saved states and performs offline saved-weight inference only. It separates initial-heading travel from body-velocity error, full tails from pre-flag coverage, and contact acquisition from later touchdown/chatter. `render_stop.py --episode <visit-directory> --out <media-directory>` replays retained states with `mj_forward` and no integration. The immutable input snapshot and bundle-regeneration receipt preserve the exact submitted closure independently of later helper edits.

## Original turning collector

Preparation uses candidate 200 unchanged: actor81, critic96, raw24, saved normalizers and the original physical decoder. The six schedules distinguish moving yaw at vx0.25/yaw±0.3, pure yaw±0.4, and vx0.25→pure yaw±0.3 at0.9s. Units are m/s and rad/s.

Run source-only preparation:

```sh
.venv-control/bin/python -B control/skills/basic-gait/prepare.py
```

This reads the existing native replay for offline inference/decoder checks. Model construction, forward, stepping and integration are blocked during that check. It creates no native episode.

The subsequent assigned turning slot can collect one original, using a fresh output path:

```sh
.venv-control/bin/python -B control/skills/basic-gait/replay.py --profile moving-left --seed 141001 --device cpu --out artifacts/cove50-basic-skills-retry/turn-20261027/moving-left-141001
```

Do not run collection during preparation or concurrent arm work. The frozen proposed plan has two seed families and six conditions; it is a proposal for the lead's physical assignment.

Collection imports the original MJLab task, R3 motor/sensor boundary, checkpoint loader and full `Episodes` recorder. It changes command buffers before pre-action observation capture. MJLab's `get_observations()` returns a populated cache, so the harness clones that packet and replaces only actor69:72/value72:75 after validating the frozen layout and command dependencies. All noncommand features remain identical; no group recomputation, new noise sample, delay push or history update occurs. The same clone feeds actor, value and recorder. Robot state and queues never reset or copy at a switch.

Control0 still samples raw24 while the native boundary holds mapped home awaiting the first delayed packet. The standing recorder's different bootstrap is not used. Falls and resting tails continue through the requested horizon with original termination annotations. Offline preparation exercises the installed cache and wrapper methods on a populated fake cache, including the0.9s switch.

The retained turning campaign measured a full local CPU probe and a complete CUDA grid. Device/seed labels do not imply exact trajectory determinism; compare saved initial states and actual common prefixes. If the lead selects remote GPU collection, use the existing Runpod IaC with bounded expiry, preflight and sparse foreground checks.

The plan lists native-unit measurements and the recorder's limits. Its 50Hz motor snapshots do not establish integrated4kHz torque duty. No fitting, changed reward, model edit or rendering is included.
