# Simulation core and embedded target

The [motor-power sequence](../docs/design/target-power-sequence.md) now requests power before waiting for servo replies, verifies torque disabled before activation and removes the GPIO request before shutdown packets. The startup wait is nonblocking and unmeasured; independent hardware protection remains open. Tests execute this target source against fake peripherals, while default flashable builds remain uncalibrated.

`supervisor.c` is portable C11 used by MuJoCo and native fault tests. `actuator.c` wraps its bounds/expiry/fault checks for real servo position/speed commands. `wire.c` implements host framing and Feetech packet encoding. `target_main.c` supplies the Pico 2 W USB, UART, ADC, GPIO and watchdog adapter. No board has been flashed or physically tested.

Build locally from the repository root:

```sh
.venv/bin/python tools/build_firmware.py variants/pip.yaml --output artifacts/my-pip-firmware
```

Use `variants/moss.yaml` or `variants/scout.yaml` for the other concepts. The output directory must be new. Each build retains its generated joint/pin map, source snapshot and hashes, compiler logs, ELF/BIN/UF2 and report. A missing calibration map produces an image that refuses to arm. Optional `--calibration measured.json` requires every joint name with an integer encoder `zero` and `direction` of +1 or −1. This input is an operator assertion, not proof of measurement.

See [the target and transport contract](../docs/design/embedded-target.md) for setup, pins, packets, command units and limitations. The hardware adapter does not use simulation PD effort output, implement torque control or establish free-standing stop behavior.

The target now uses a [timed RLB1 receiver](../docs/design/firmware-frame-timeout.md): a 50 ms frame-assembly policy, optional expiry telemetry, and the SDK's 64-bit millisecond clock. This is separate from the 150 ms actuator guard and does not bound host-queued command age.
