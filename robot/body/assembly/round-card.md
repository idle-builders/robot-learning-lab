# Complete Cove-50 FDM Ape Body Integration (`cove50-complete-fdm-body-r1`, `T-g6ns9t`, `M-qzbpak`)

## 1. Objective & Scope
Combine all four completed 3D-printable FDM structural subsystems into one coherent, source-verified Complete Cove-50 FDM Ape Body CAD assembly (`cove50-complete-fdm-body-r1`):
1. **Lower Body (`cove50-complete-leg-hip-fdm-combined-r10`, `T-f39h6b`, `M-y8acff`):**
   - 12 printed parts (6 stages × 2 legs: Stage 1 Shin, Stage 2 Ankle Pitch-to-Roll Coupler, Stage 3 Foot Monocoque, Stage 4 Thigh, Stage 5 Hip Pitch-to-Roll Link, Stage 6 Hip Yaw-to-Pitch Link)
   - 128 pressed-in `304 SS` compression sleeves (`64/leg`)
   - `187.64 g` bilateral (`157.20 g` PAHT-CF + `30.44 g` steel sleeves) vs `332.78 g` legacy V16 sheet-metal + carrier baseline (`-145.14 g`, `-43.61%`), min dry Tsai-Hill safety factor `2.86×`.
   - Includes smooth closed round thigh & shin hard covers (`R/L_thigh_front_hard_cover`, `R/L_thigh_back_hard_cover`, `R/L_shin_front_hard_cover`, `R/L_shin_back_hard_cover`) and black anodized hip C047 motors (`#171b21`).
2. **Upper Body / Arms & Geared Hands (`cove50-arm-fdm-combined-r12`, `T-zw7f7t`, `M-d34p89`):**
   - 8 primary printed arm structural parts (4 stages × 2 arms: Stage 1 Shoulder Pitch-to-Roll Bridge with outward-facing shoulder roll motor, Stage 2 Shoulder Roll-to-Yaw Coupler, Stage 3 Upper-Arm Yaw-to-Elbow Monocoque, Stage 4 Forearm Elbow-to-Wrist Monocoque) + 8 printed geared-hand parts (`R/L_wrist_hand_frame_half`, `R/L_wrist_hand_frame_cap`, `R/L_finger_right`, `R/L_finger_left`) = 16 printed arm/hand parts
   - 68 pressed-in `304 SS` compression sleeves (`34/arm`)
   - `74.74 g` bilateral 8-part primary arm structure (`59.30 g` PAHT-CF + `15.44 g` steel sleeves) vs `136.05 g` legacy sheet-metal + carrier baseline (`-61.31 g`, `-45.06%`), min dry Tsai-Hill safety factor `1.65×`.
   - Includes all 8 smooth closed round arm hard-shell covers (`R/L_shoulder_hard_cover`, `R/L_upper_arm_hard_cover`, `R/L_fore_front_hard_cover`, `R/L_fore_back_hard_cover`) with `0.00 mm³` collision across all 32 arm flexion poses.
3. **Neck & Head (`cove50-neck-head-fdm-combined-r1`, `T-zesmm6`, `M-8jkx67`):**
   - 4 printed parts (Stage 1 Neck Root Cradle, Stage 2 Neck Yaw-to-Pitch Link, Stage 3 Neck Pitch-to-Head U-Cradle, Stage 4 Shared Head Sensor/Compute/Mic Carrier)
   - 34 pressed-in `304 SS` compression sleeves (`6 + 8 + 6 + 14`)
   - `28.56 g` total (`21.84 g` PAHT-CF + `6.72 g` steel sleeves) vs `52.67 g` legacy sheet-metal + PA12 carrier baseline (`-24.11 g`, `-45.78%`), min dry Tsai-Hill safety factor `2.34×`.
4. **Center-Body Torso, Pelvis, 9-Cell Honeycomb Battery & Electronics Carrier (`cove50-torso-fdm-combined-r1`, `T-zdevbv`, `M-h3aaq6`):**
   - 4 printed parts (Stage 1 7-Cell Honeycomb Torso Spine, Stage 2 2-Cell Pelvis & 7-Cell Bottom Lid U-Clevis, Stage 3 Dual-Deck Front Power Frame, Stage 4 Dorsal IMU/ADC Carrier)
   - 76 pressed-in `304 SS` compression sleeves in the whole-body audit (`39 + 13 + 10 + 14`, `86` in standalone subsystem rollup)
   - 9 vertical honeycomb `21700` P50B battery cells + 9 top insulators + 9 direct-on-battery `6063-T5` Al 4-blade heat-sink fin strips (`28.44 g`)
   - Smooth 2-piece continuous 12-station NURBS Center Shell (`center_front_shell` + `center_back_shell` + `chest_indicator`) bolted directly to flat Stage 1 & Stage 2 mounts
   - `178.21 g` total (`164.08 g` PAHT-CF + `14.13 g` steel sleeves) vs `251.82 g` legacy AlSi10Mg LPBF + PA12 baseline (`-73.61 g`, `-29.23%`), min dry Tsai-Hill safety factor `1.55×`.

## 2. Cross-Subsystem Interface & Multi-DOF Articulation Verification
- **Torso Spine (`stage1_center_torso_spine_chassis`) ↔ Neck Root Cradle (`neck_root_cradle`):** Flush `Y = 25.00 mm` face-to-face mating plane (`0.0000 mm³` BRep overlap).
- **Torso Spine (`stage1_center_torso_spine_chassis`) ↔ Bilateral Shoulder Pitch-to-Roll (`R/L_shoulder_pitch_to_roll_combined`):** Coaxial `X = ±62.7 mm, Y = 5.0 mm, Z = 359.0 mm` C047 horn + rear passive idler shaft (`0.0000 mm³` overlap across `[-90°, +75°]` pitch and `[0°, +95°]` abduction).
- **Torso Spine (`stage1_center_torso_spine_chassis`) & Pelvis U-Clevis (`stage2_lower_pelvis_battery_cradle`) ↔ Bilateral Hip Yaw-to-Pitch (`R/L_hip_yaw_to_pitch_combined`):** Coaxial `X = ±38.5 mm, Y = 0.0 mm, Z = 235.5 mm` C047 horn + bottom passive idler shaft (`0.0000 mm³` overlap across `[-30°, +30°]` yaw, `[-58°, +45°]` pitch, and `[-30°, +25°]` roll, including all electronics, battery cells, and shells across 22 hip/leg flexion poses).
- **Kinematics & Controls:** `26` kinematic joints driven by `24` interactive DOF sliders (`12` lower-body hip/knee/ankle controls, `8` arm shoulder/elbow controls, `2` symmetric geared-hand finger controls, and `2` neck yaw/pitch controls) across `747` scene components.

## 3. Whole-Body Rollup Summary
- **28 Primary Printed FDM Structural Parts (+ 8 Printed Geared-Hand Parts = 36 Printed FDM Parts) + 306 Pressed-In `304 SS` Sleeves (`316` in subsystem rollup):** **`469.15 g`** primary 28-part structure (`402.42 g` PAHT-CF polymer + `66.73 g` steel sleeves) vs **`773.32 g`** legacy metal/sheet/PA12 baseline (**`-304.17 g`, `-39.33%`**).
- **Legacy Sheet-Metal Residual Volume:** **`0.0000 mm³`** (`0` legacy sheet-metal parts remaining).
- **Whole-Body Multi-Pose Exact BRep Overlap:** **`0.0000 mm³`** across all 8 whole-body poses (`neutral_0deg`, `walking_stride_right_lead`, `walking_stride_left_lead`, `deep_biped_squat`, `bilateral_forward_reach`, `wide_abduction_arm_spread`, `head_look_down_left_wide_stance`, `full_ape_crouch_and_gaze`), plus **`0.00 mm³`** across all 32 arm poses and 22 hip/pelvis/electronics/shell poses.
- **Home-Lab Print-Bed Fit:** `100%` of all `36` printed FDM parts fit a standard `256 × 256 × 256 mm` print bed without splitting (max span `121.85 mm`, leaving `134.15 mm` margin to `256 mm`), nesting onto `4` standard `256 × 256 mm` plates or `2` large `350 × 350 mm` plates.
