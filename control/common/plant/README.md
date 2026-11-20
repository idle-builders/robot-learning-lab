# R3 batched boundary

`robot_cfg()` supplies one custom MJLab actuator in the selected 22-channel
actuator order. Use actual physics timestep 0.00025 s and policy decimation 80.
The task must set the scene's numerical options from the selected native model;
MJLab scene composition can otherwise override child model options.

`get_runtime(env)` returns the `BatchedRuntime` attached to robot actuator 0.
The action term calls `get_runtime(env).arm_physics_step()` from its
`apply_actions()` method, then writes named output-joint radians to `robot`
position targets. This is required: MJLab also writes actuators during full and
partial reset without integrating physics. Unarmed writes return prior torque
without advancing clocks, queues, sensors or random draws in any world.
Every pre-physics actuator call evaluates the source target envelope, delayed
command, motor curve, IMU estimate and delayed joint-feedback packet.

The physical boundary keeps accepted physical target history. Historical actor
raw output history, including removed or disabled channels, belongs to the task.
Do not reconstruct it from physical target history.

| Field | Shape and units |
| --- | --- |
| `last_values` | `[N,50]`: q22 rad, qd22 rad/s, gyro3 rad/s, estimated gravity3 unit vector |
| `previous_command` | `[N,22]` most recently accepted/clamped target rad |
| `target` | `[N,22]` motor target after command delay, rad |
| `torque`, `current` | `[N,22]` N m and estimated A |
| `observation(command)` | `[N,75]`: last_values, command3, previous_command22 |
| `valid` | `[N]` false until the first feedback packet arrives |
| `capture_time_s`, `imu_capture_time_s` | `[N]` actual joint/IMU acquisition clock, relative to that world's reset |
| `time_s`, `age_s` | `[N]` elapsed physics clock and joint packet age, s |

Reset clears only selected world histories. Initial motor target is the actual
reset q seen on the first actuator call, before any physics. The first requested
target arrives 80 physics ticks later under the nominal 20 ms delay. The first
100 Hz feedback packet is captured at t=0 and delivered at t=10 ms. The task must
declare how it handles invalid reset observations and retain/count any bootstrap
motion. No hidden simulation warmup occurs inside this boundary.

Defaults use 12 V and the scalar Runtime's non-peak mode: C047 rated torque cap
0.980665 N m, neck cap min(0.2 N m, digitized curve). `peak=True` enables the
scalar Runtime's peak mode; record it in each run. Neck voltage clips at 12 V.
These are provisional source simulation laws, not installed motor identification.

Production noise uses Torch random streams with the same source distributions.
It does not reproduce NumPy random bytes. The shared stream makes future draws
depend on reset ordering. Matched checks inject the original source bias/noise
draws to isolate deterministic algebra and acquisition/delivery timing.

Regenerate compact evidence locally:

```sh
.venv-control/bin/python control/common/plant/audit.py
```

The report is `data/walk-resume-r1-20261026/runtime/audit.json`. It compares source
Runtime with float64/float32 batched algebra across rail, delay, peak and partial
reset cases, then runs two 60.25 ms native CPU source episodes and compiles the
MJLab entity. The complete short native inputs are retained in
`native-input-trace.json`. This establishes no GPU throughput or walking result.
