# Cove-50 Complete 4-Stage Blank-State 3D Generative Arm FDM Connector Study (`r12` — Locked-In Selected Arm FDM Connectors)

- **Idle Task:** `T-zw7f7t`
- **Idle Decision:** `M-d34p89`
- **Selected Result ID:** `cove50-arm-fdm-combined-r12` (`version`: `r12`)
- **Dashboard Studies (`r12`):**
  - `http://127.0.0.1:8805/design/?result=cove50-arm-fdm-combined-r12`
  - `http://127.0.0.1:8805/design/?result=cove50-arm-fdm-truss-box-r12`
  - `http://127.0.0.1:8805/design/?result=cove50-arm-fdm-topopt-r12`

## Summary of `r12` Overhauled Stage 1, Down-First Stage 3, Asymmetric Edge Beveling & Enforced 3-Gate Verification (`38` Connections/Arm = `76` Bilateral)
- **Stage 1 (`R/L_shoulder_pitch_to_roll`) Complete Overhaul:**
  - Replaced the thin `21.1 mm^2` neck bridge and eliminated the 4 floating shoulder hard-shell mounting holes/prongs/captive-nut bosses (`R/L_shoulder_front_339/351`, `R/L_shoulder_back_339/353`) that had no interior bracket on `R/L_shoulder_front` or `R/L_shoulder_back`.
  - Built a 3-component tangent-continuous C-channel bridge (`pitch_hub` flush across `X in [70.625, 73.625]` matching the exact `3.00 mm` 304 SS compression sleeves with zero counterbore crescent lips, `rear_cheek_web` across `X in [70.625, 135.68], Y in [22.02, 23.575]`, and `yoke_deck_lower` / `yoke_deck_upper` across `Z in [336.5, 347.9]` and `Z in [355.4, 359.0]`, `Y in [12.80, 22.02]`), raising the minimum structural bridge cross-section to `31.25 mm^2` (`topopt`) and `33.54 mm^2` (`truss_box`) and cutting Stage 1 peak deflection by `-46.3%` (`0.3150 mm` vs `0.5863 mm` baseline) at `-67.7%` mass (`9.18 g` vs `28.43 g`).
- **Stage 3 (`R/L_upper_arm_yaw_to_elbow`) Down-First Wrap-Around Routing:**
  - Re-sculpted Stage 3 so it drops straight down first beneath the upper-arm yaw output horn (`Z in [268, 305] mm`) with `X_max <= 144.8 mm` (`7.3 mm` tighter laterally than the old `152.1 mm` outward flare) and then wraps closely over the top of `upper__R_elbow_C047_case` for easy fabric covering, while cutting Stage 3 peak deflection by `-60.3%` (`1.4393 mm` vs `3.6300 mm` baseline).
- **Full 3D Asymmetric Edge Beveling Across All 24 Solids (`r_inner > r_outer`):**
  - Applied OpenCascade 3D `BRepFilletAPI_MakeFillet` across all 24 Stage 1–4 solids (`R/L` x 3 families) with larger concave inner fillets (`r_inner = 0.70–0.85 mm`) for stress relief and smaller convex outer roundings (`r_outer = 0.35–0.45 mm`) for touch-safe handling while preserving crisp flat steel-sleeve boss faces.
- **Mass, Stiffness & 3-Gate Verification (`r12`):**
  - **`37.37 g/arm` (`74.74 g` bilateral) in `combined` (`topopt` selected across all 4 stages) vs `68.02 g/arm` (`136.05 g` bilateral) metal baseline (`-30.65 g/arm` / `-61.30 g` bilateral, `-45.1%` mass)**.
  - **`2.4146 mm` 4-stage RSS deflection vs `3.9206 mm` metal baseline (`-38.4%`)**, **`1.65x–10.03x` (`PAHT-CF dry`) and `1.30x–7.80x` (`PETG-CF`) 3D FEM `p99` strength margins** under `4 kg @ 1.5g` recovery load (`18 N` distal hand load + joint torques).
  - **`0` BRep slivers, `0` short edges, `0` acute vertices, `1.0000` tangential continuity ratio**, and **`0.0000 mm^3` overlap across all 41 arm poses and 42 fastener/nut-slot rays per arm (`34` pressed-in 304 SS sleeves + `4` captive-M3 forearm hard-shell mounts = `38` bolted connections/arm = `76` bilateral)**.
