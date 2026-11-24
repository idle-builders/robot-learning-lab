# Lower-Leg 3D-Printable FDM Connector Redesign — Round Card (`2026-10-30`)

## 1. Objective & Scope

Redesign the three anatomical lower-leg structural connector stages of `APE0` (`R`/`L` bilateral symmetry across `Plane.YZ`) for home-lab **FDM 3D printing**, starting from the accepted complete-body mechanical baseline (`cove50-electronics-integrated-r1-20261026` / lower-body `cove50-simple-sheet-leg-chains-v16`):

1. **Stage 1 — Knee-to-Ankle Shin Connector (`R_shin` / `L_shin`)**:
   - **Replaces**: `R/L_B13R14_short_inner_shin_plate` ($7.13\text{ g}$), `R/L_B13R14_short_outer_shin_plate` ($7.13\text{ g}$), and 4 loose cut-tube spacers (`R/L_B13R14_raised_B11_{inner,outer}_cut_spacer_{0,1}`, $4 \times 0.07\text{ g}$) — baseline structural mass **$14.54\text{ g/leg}$** ($29.08\text{ g}$ bilateral).
   - **Fixed Interfaces**:
     - **Upper (Knee Pitch Output, axis $X$ through $(Y, Z) = (4.0, 123.0)\text{ mm}$)**: Mates to `R/L_C047_knee_main_horn` ($X = \pm 41.25\text{ mm}$ seat plane, $14\text{ mm}$ PCD M3 pattern around $(Y,Z)=(4,123)$ with $\varnothing 8.0\text{ mm}$ central boss clearance) and `R/L_C047_knee_rear_idler` ($X = \pm 78.75\text{ mm}$ seat plane, $14\text{ mm}$ PCD M3 pattern at $(Y,Z) \in \{(4,116), (4,130)\}$ with $\varnothing 8.0\text{ mm}$ central clearance).
     - **Lower (Ankle Pitch Motor Case, axis $X$ through $(Y, Z) = (4.0, 61.0)\text{ mm}$)**: Cradles `R/L_B13R14_raised_C047_ankle_pitch_case` at the inner ear plane $X = \pm 44.0\text{ mm}$ ($(Y,Z) \in \{(-6.25, 69.3), (14.25, 93.75)\}$) and outer ear plane $X = \pm 76.0\text{ mm}$ ($(Y,Z) \in \{(-6.25, 69.3), (14.25, 90.0)\}$), with $\varnothing 22.0\text{ mm}$ clearance windows around $(Y,Z)=(4.0, 61.0)$ for the ankle pitch main horn and rear idler.

2. **Stage 2 — Ankle Pitch-to-Roll Coupler (`R_ankle_pitch` / `L_ankle_pitch`)**:
   - **Replaces**: `R/L_B13R14_one_bend_output_link` ($10.01\text{ g/leg}$ Al 5052-H32 two-bend sheet yoke).
   - **Fixed Interfaces**:
     - **Upper Pitch Output ($Z = 61.0\text{ mm}$, $Y = 4.0\text{ mm}$, axis $X$)**: Mates to `R/L_B13R14_raised_C047_ankle_pitch_main_horn` ($X = \pm 79.2\text{ mm}$ seat plane, 4 M3 holes at $(Y,Z) \in \{(4,54), (4,68), (-3,61), (11,61)\}$, central bore $\varnothing 9.0\text{ mm}$) and `R/L_B13R14_raised_C047_ankle_pitch_rear_idler` ($X = \pm 41.95\text{ mm}$ seat plane, 2 M3 holes at $(Y,Z) \in \{(4,54), (4,68)\}$, central bore $\varnothing 9.0\text{ mm}$).
     - **Lower Roll Output ($X = \pm 75.0\text{ mm}$, $Z = 19.0\text{ mm}$, axis $Y$)**: Mates to `R/L_B13_C047_foot_roll_main_horn` ($Y = +18.0\text{ mm}$ seat plane, 4 M3 holes at $(X,Z) \in \{(\pm 68,19), (\pm 82,19), (\pm 75,12), (\pm 75,26)\}$, central bore $\varnothing 9.0\text{ mm}$).

3. **Stage 3 — Foot Roll-to-Sole Mount & Sensor Carrier (`R_foot` / `L_foot`)**:
   - **Replaces**: `R/L_B13R14_front_case_L` ($2.49\text{ g}$), `R/L_B13R14_rear_case_L` ($2.88\text{ g}$), `R/L_B13R14_plain_sole` ($40.63\text{ g}$), 4 cut spacers ($4 \times 0.07\text{ g}$), 4 M3x6 sole button screws ($4 \times 0.66\text{ g}$), and 4 M3 sole hex nuts ($4 \times 0.36\text{ g}$) — baseline structural + eliminated sole-joint hardware mass **$50.36\text{ g/leg}$** ($46.00\text{ g}$ structural Al + $0.28\text{ g}$ spacers + $4.08\text{ g}$ eliminated sole-to-bracket fasteners).
   - **Fixed Interfaces**:
     - **Foot Roll Motor Case (`R/L_B13_C047_foot_roll_case`)**: Front mounting ear face at $Y = +14.8\text{ mm}$ ($(X,Z) \in \{(\pm 46.0, 8.75), (\pm 66.7, 8.75)\}$ lower row + upper ear cradle support) and rear mounting ear face at $Y = -17.2\text{ mm}$ ($(X,Z) \in \{(\pm 42.25, 8.75), (\pm 66.7, 8.75)\}$ lower row + upper ear cradle support), with clearance windows around $(X,Z)=(\pm 75.0, 19.0)$ for the foot-roll main horn and rear idler.
     - **Ground Pad & Foot Tactile Sensor Datum ($Z = 2.0\text{ mm}$)**: Exact $60 \times 86\text{ mm}$ rounded-rectangle sole footprint ($R=22\text{ mm}$, centered at $(X,Y)=(\pm 67.0, -12.0)\text{ mm}$, $X \in [\pm 37, \pm 97]$, $Y \in [-55, 31]$) with an uninterrupted flat bottom surface at $Z = 2.0\text{ mm}$ supporting `R/L_sensor_pocket_pad_v19` and the 3 foot contact sensors (`toe_lateral`, `heel_lateral`, `mid_medial`).

---

## 2. Non-Negotiable Constraints

- **Motors, Horns, Joint Axes & Ranges Fixed**: No servo case, horn, idler, or joint axis is translated or rotated.
- **Not Inspired by Bent Sheet Metal**: Both competing FDM connector families are designed from scratch as true 3D load-bearing structures (3D space-truss / closed-box beams / monocoque ribbed cradles vs. blank-space 3D continuum topology optimization), never thickened sheet-bend developments.
- **Home-Lab Buildable**: Printable on a standard desktop FDM printer (e.g., Bambu Lab / Prusa with $0.4\text{ mm}$ hardened nozzle in PAHT-CF / PA12-CF / PETG-CF / PLA-CF) supplemented only by off-the-shelf hardware (standard cylindrical steel compression limit sleeves $\text{OD } 4.5 \times \text{ID } 3.2\text{ mm}$ at M3 horn seats, M2/M3 machine screws, steel washers, and M2 hex nuts / heat-set inserts).
- **Shell & Motion Envelope Fit**: Must clear the fixed motors/horns across the joint ranges and fit inside the lower-leg exterior shell (`full-shell-neutral.glb` and right-leg hard shell `r14-20261029`).
- **Evaluation Discipline**: No binary pass/fail gates. Report native-unit stress, deflection, interface compliance, and stiffness-per-unit-mass across materials and load cases, and grade each dimension on the 0–5 distance-to-viable scale.
