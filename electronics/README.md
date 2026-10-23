# Electronics scope

Electrical interfaces and the power model are generated from the robot manifest into each run directory. The implemented local path is:

1. Atopile 0.15.8 compiles the logical power/signal interface module. It has no purchased components and its empty PCB output is not a robot board.
2. ngspice 47 evaluates a lumped motor-rail model against current estimated from the simulated trajectory and a deliberate overload.
3. KiCad 10.0.6 checks a separate two-connector power coupon. A shorted counterpart proves DRC failure detection. The coupon exports STEP using the manifest's board envelope.

The KiCad coupon is generated independently of Atopile and is a toolchain fixture. This does not yet demonstrate a populated Atopile-to-KiCad circuit round trip. No current rating, battery protection, regulator design or fabrication readiness is claimed.

The next board should be a supervisor/interface carrier around proven modules after selecting the actuator bus, logic levels and power domains. Keep `.ato` authoritative for electrical intent and KiCad authoritative for physical layout. Replace the coupon with that real design and add schematic parity only when a corresponding schematic exists.

`controller.yaml` now allocates a Pico 2 W pinout, two TTL bus-driver pairs and voltage-divider/interlock signals consumed by the embedded build. See [target contract](../docs/design/embedded-target.md). This is the input for the next populated carrier; it does not change the existing coupon into a controller board. IMU pins are reserved but no sensor or driver is selected.

The standalone [common carrier build](carrier/README.md) now implements a populated Atopile-to-KiCad-to-local-routing path. It retains compiled nets while restoring complete vendor footprint geometry, then checks firmware pins, DRC and a deliberate short. It exports the real candidate board and STEP; the older workbench coupon remains a separate tool fixture. The complete power board, protection and IMU are still open.

A separate [5 V logic-supply circuit](logic-supply/README.md) now contains exact candidate parts and compiles their physical-pad connections into a vendor-model ngspice probe. It is an unplaced/unrouted circuit experiment, with startup and solver limitations exposed in its reports. It does not replace the motor/protection model or carrier power allocation automatically.
