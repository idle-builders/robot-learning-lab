# Center-Body Torso Base Plate & Repeated 9-Cell Honeycomb Battery Carrier — Blank-Slate 3D-Printable FDM Architecture (`T-zdevbv`)

## Scope & Objective
Redesign the entire Cove-50 center-body torso, pelvis, 9-cell battery retention, and electronics carrier system (**24 legacy parts** spanning `Z = 194.5..387.5 mm` from the Hip Yaw/Pitch/Roll pelvis cradle through the 9× Molicel P50B 21700 battery pack and front/rear PCBs up to the Shoulder Pitch saddles and Neck/Head root cradle, totaling **251.82 g**) from scratch into a purpose-built, 100% home-lab 3D-printable FDM structural system (`PAHT-CF` / `PETG-CF`) with **56 pressed-in 304 SS compression sleeves** and asymmetric 3D bevels (`r_inner > r_outer`).

### Repeated `5 + 2 + 2` Vertical Hexagonal Honeycomb 9-Cell Battery Architecture, 9 Vertical Fin Slots & 9× Direct-on-Battery Aluminum Heat-Sink Fins
Instead of the legacy scattered vertical and horizontal battery cells held by 15 asymmetric clamps/straps (`Cell_capture_bridge_00..14`), all **9× Molicel P50B 21700 cells (`Ø21.5 × 70.15 mm`)** are arranged in a **symmetric, repeated `5 + 2 + 2` Vertical Hexagonal Honeycomb Lattice (`22.4 mm` center-to-center pitch, all 9 cells parallel to `+Z`)**:
- **Upper 7-Cell Honeycomb Magazine (`Stage 1`, `Z_center = 304.2 mm`):**
  - `5` Dorsal cells at `Y = +25.6 mm`, `X = -44.8, -22.4, 0.0, +22.4, +44.8 mm` (`Honeycomb_Cell_01..05`) — full-height `7.70 × 71.2 mm` rear (`+Y`) vertical fin-exposure slots
  - `2` Nested Central Spine cells at `Y = +6.2 mm`, `X = -11.2, +11.2 mm` (`Honeycomb_Cell_06..07`) — full-height `7.70 × 71.2 mm` front (`-Y`) vertical fin-exposure slots
- **Lower 2-Cell Coaxial Honeycomb Pelvis Column (`Stage 2`, `Z_center = 230.5 mm`):**
  - `2` Central Pelvis cells at `Y = +14.8 mm`, `X = -11.2, +11.2 mm` (`Honeycomb_Cell_08..09`) — full-height `7.70 × 70.4 mm` rear (`+Y`) vertical fin-exposure slots
- **9× Direct-on-Battery `6063-T5` Anodized Aluminum 4-Blade Vertical Heat-Sink Fin Strips (`Honeycomb_Fin_01..09`) + 3 Interstitial Cooling Chimneys:**
  - Each of the 9 cells has a `6.00 × 62.0 mm` 4-blade vertical `6063-T5` anodized aluminum heat-sink fin strip (`~3,120 mm²` direct convective area per cell) bonded directly onto the bare cylindrical `21700` cell can inside its `7.70 mm`-wide vertical window slot using `0.15 mm` electrically-insulating thermal tape (`k = 1.5 W/(m·K)`), completely bypassing the insulating `PAHT-CF` wall without touching the plastic carrier (`0.0000 mm³` overlap).
  - `3 × Ø4.8 mm` full-height vertical interstitial cooling chimneys (`Z = 266.8..341.6 mm`) drilled straight through the equilateral triangle centroids of the 7-cell upper honeycomb (`(-11.2, 19.133)`, `(0.0, 12.667)`, `(+11.2, 19.133)`) + `r = 9.0 mm` (`Ø18.0 mm`) open top/bottom cell chimney throats + 3D inter-cell cross-ventilation tunnels.

Because all 9 cylindrical honeycomb cell tubes are oriented vertically along `+Z` with continuous `1.70 mm` shared walls (`Ø21.9 mm` ID / `Ø25.3 mm` OD), closed top/bottom `r = 9.0 mm` retaining rings, and `45°` self-supporting chamfers/gussets, **both `Stage 1` and `Stage 2` print vertically on the build plate with zero internal honeycomb support material**.

## Print-Bed Fit (`256 × 256 × 256 mm` Standard & `350 × 350 × 350 mm` Large FDM Beds)
- **Stage 1 (`stage1_center_torso_spine_chassis`):** `114.90 × 48.50 × 120.70 mm` (`>= 135.3 mm` margin on a `256 × 256 mm` bed)
- **Stage 2 (`stage2_lower_pelvis_battery_cradle`):** `37.00 × 38.10 × 72.00 mm` (`>= 184.0 mm` margin on a `256 × 256 mm` bed)
- **Stage 3 (`stage3_front_power_electronics_carrier`):** `79.50 × 23.02 × 97.00 mm` (`>= 159.0 mm` margin on a `256 × 256 mm` bed)
- **Stage 4 (`stage4_rear_imu_adc_carrier`):** `36.00 × 4.60 × 57.00 mm` (`>= 199.0 mm` margin on a `256 × 256 mm` bed)
- **Full 4-Stage Assembly:** `114.90 × 73.84 × 193.00 mm` (`>= 63.0 mm` margin on a `256 × 256 mm` bed; all 4 stages nest simultaneously on a single build plate).

## Replaced Legacy Parts (24 Parts -> 4 Single-Solid Blank-Slate FDM Stages + 56 Steel Sleeves)
1. `Common_power_carrier_mounting_r33` + `Cell_capture_bridge_00..12` (14 parts, `85,090.15 mm³`, `222.83 g` AlSi10Mg LPBF + PA12) -> **Stage 1 (`stage1_center_torso_spine_chassis`, 22 sleeves, `91.65 g` in `truss_box`)**
2. `lower_cradle_lugs_224_with_waist_guard_r18` + `lower_cradle_lugs_252_retained_outer_r18` + `Cell_capture_bridge_13` + `Cell_capture_bridge_14` (4 parts, `11,032.39 mm³`, `10.75 g` PA12) -> **Stage 2 (`stage2_lower_pelvis_battery_cradle`, 6 sleeves, `25.70 g` in `topopt`)**
3. `Power_supervisor_PA12_open_tray` + `Main_precharge_PA12_open_tray` + `BMS_recessed_fastener_front_guard` + `Cell_07_removable_front_clamp_L` + `Cell_07_removable_front_clamp_R` (5 parts, `14,132.45 mm³`, `13.78 g` PA12) -> **Stage 3 (`stage3_front_power_electronics_carrier`, 18 sleeves, `19.82 g` in `truss_box`)**
4. `IMU_ADC_carrier_v17` (1 part, `4,004.31 mm³`, `4.46 g` PA12) -> **Stage 4 (`stage4_rear_imu_adc_carrier`, 10 sleeves, `7.60 g` in `topopt`)**
- **Combined 4-Stage Total Mass:** **`144.77 g`** (`137.28 g` PAHT-CF polymer + `7.49 g` across 56 304 SS sleeves) vs **`251.82 g`** baseline (**`-42.5%` / `-107.05 g` mass savings**).

## Coupled 9-Cell Honeycomb Thermal, Chimney & 9× Direct-on-Battery Aluminum Fin Qualification
- **10.8 A Nominal Walk (`3.6 A/cell`, `4.14 W` total):** Closed `31.75 °C` -> Chimney-Vented `29.29 °C` -> **+ 9× Battery Al Fins: `28.45 °C`** (pack $\Delta T = 0.16\text{ °C}$, polymer wall `28.09 °C`).
- **18.6 A All-Rated Locomotion (`6.2 A/cell`, `10.24 W` total):** Closed `41.29 °C` -> Chimney-Vented `34.55 °C` -> **+ 9× Battery Al Fins: `32.51 °C`** (**`-8.78 °C`** vs unvented, pack $\Delta T = 0.39\text{ °C}$, polymer wall `31.70 °C`).
- **36.0 A Heavy Dynamic Duty (`12.0 A/cell`, `30.32 W` total, 15-min full `2.4C` depletion):** Closed `55.34 °C` -> Chimney-Vented `46.18 °C` -> **+ 9× Battery Al Fins: `42.64 °C`** (**`-12.70 °C`** vs unvented, pack $\Delta T = 0.81\text{ °C} \ll 5.0\text{ °C}$, polymer wall `40.66 °C`, `+153.3 °C` HDT margin on `PAHT-CF`, `+33.3 °C` on `PETG-CF`; standard `PLA` disqualified).
- **54.0 A Extreme Stall Burst (`18.0 A/cell`, `64.15 W` for `60 s`):** Closed `46.00 °C` -> Chimney-Vented `38.93 °C` -> **+ 9× Battery Al Fins: `36.80 °C`** (pack $\Delta T = 0.41\text{ °C}$, polymer wall `35.50 °C`).
- **Radial Thermal Expansion Clearance:** `+0.0035 mm` cell expansion leaves **`0.200 mm` radial (`0.400 mm` diametral) clearance** in the `Ø21.9 mm` honeycomb bores.

## Pre-Existing Baseline Collisions & CAD Defects Resolved
- **`12.3228 mm³` overlap** between `Common_power_carrier_mounting_r33` and `Power_supervisor_PA12_open_tray` -> **`0.0000 mm³`**
- **`36.1912 mm³` (`8 × 4.5239 mm³`) overlap** between `IMU_ADC_carrier_v17` and `IMU_M2x6_0..3` / `ADC_M2x6_0..3` -> **`0.0000 mm³`**
- **17+ baseline sliver faces (`< 0.05 mm²`)** across the 24 legacy parts -> **`0` slivers, `0` short edges, `solid_count = 1`** across all 4 stages and both families.
