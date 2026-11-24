# Selected APE0 model: continuous-servo resolution R3

R3 retains the selected V24/R23/R33 electronics and upper V5/lower V16 mechanics,
the R2 physical collision correction, all meshes, 22 axes, mass/inertia, travel
assumptions, motor gains and torque-speed laws. Its rigid native timestep is
0.25 ms. The unchanged external Runtime law is reevaluated at 4 kHz, while policy
commands remain 50 Hz with 20 ms command and 10 ms feedback delay.

The previous 1 ms numerical evaluation produced a growing alternating mode in
an unloaded ankle/knee. A matched complete 37.3 s comparison removed that chatter
at 0.25 ms with identical initial states and requested/accepted commands. The
body still fell during load transfer. This is a numerical correction to the
assumed continuous servo law, not a successful gait or general convergence study.

Actual digital servo cadence, gains, reflected rotor/gear inertia and thermal
capability remain unmeasured. The 1e-6 kg m² armature is a numerical assumption;
the motor case's shape inertia does not replace reflected rotor inertia. Do not
raise armature or add uncapped passive damping to hide the observed mode.

Sensor clocks accumulate their nominal periods and quantize to physics steps.
Record actual capture/delivery times; the nominal 208 Hz IMU has different
interval quantization at 0.25 ms than at 1 ms. A fresh task must use the declared
physics timestep and wire the 75-value observation explicitly. In the MJLab
adapter, 20 ms delay is 80 physics lags at the default resolution; pass the actual
task `physics_dt_s` when overriding it. Compilation does not establish batched
Runtime parity or GPU behavior.

Use URDF with SRDF, a floating base and the actuator/timing contract. Direct
MuJoCo URDF import alone creates a fixed-base 22-coordinate model and does not
apply SRDF exclusions. The native rigid MJCF has 29 position / 28 velocity
coordinates. Spawn from actual collider clearance: neutral torso Z is around
302.2 mm for 0.2 mm foot clearance, rather than the old CAD 300 mm datum.

Nominal mass is still 3.8303 kg, including 0.8138 kg conditional reference skin
and 0.150 kg unfinished installation allowance. The source engineering mass
sensitivity interval is 2.953–4.730 kg, not a statistical confidence interval.
Unfitted reference skin remains appearance/mass context without hard collision;
the real battery bridge and fitted thigh panel still require physical clearance.
Six source hip travel ranges remain unknown; ±0.35 rad is a diagnostic command
envelope. The compliant MJCF retains its separate provisional FEM reductions
and 2 µs numerical step; this release does not calibrate the flexible robot.

Regenerate into an empty output directory with `COVE50_MODEL_DIR` when retaining
the existing release:

```sh
.venv/bin/python -B robot/model/timestep/revise.py
.venv/bin/python -B robot/model/timestep/package.py
.venv/bin/python -B robot/model/timestep/portable.py
```

`revise.py` verifies the R2 seal and copies its exact physical inputs. It changes
only the rigid timestep and timing/release metadata, compares native physical
arrays and FK, and seals the new identities. R1/R2 and their checkpoints remain
historical, immutable evidence. This release alone does not select new mechanics
or establish useful walking, physical qualification or broad tuning readiness.
