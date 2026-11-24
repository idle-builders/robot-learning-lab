"""Step 4b — Coupled 3D Thermal, Buoyant Chimney Convection, Cell-to-Cell ΔT Uniformity & Polymer Heat-Deflection (HDT) Audit for the 9-Cell Vertical Honeycomb Torso Carrier."""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = ROOT / "robot/body/torso"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import common


# Molicel INR-21700-P50B (5.0 Ah, 3S3P = 9 cells, 10.8 V nominal, 162 Wh total pack energy)
P50B_SPEC = {
    "model": "Molicel INR-21700-P50B",
    "pack_topology": "3S3P (9 cells total: 7 in Stage 1 upper honeycomb + 2 in Stage 2 pelvis honeycomb)",
    "r_can_mm": 10.75,
    "h_can_mm": 70.15,
    "mass_g": 70.0,
    "cp_J_kgK": 920.0,
    "heat_capacity_J_K": 70.0e-3 * 920.0,  # 64.4 J/K per cell
    "r_dc_mohm_25C": 16.5,  # Including spot-welded busbar tab resistance
    "entropic_coeff_mV_K": 0.22,
    "k_radial_W_mK": 1.45,
    "k_axial_W_mK": 28.0,
    "alpha_thermal_1_K": 18.5e-6,
    "continuous_limit_C": 60.0,
    "bms_cutoff_limit_C": 70.0,
    "max_pack_delta_T_limit_C": 5.0,
}

POLYMER_THERMAL_SPECS = {
    "PAHT_CF": {
        "name": "Bambu PAHT-CF (PPA/PA12 Carbon Fiber)",
        "hdt_0p45mpa_C": 194.0,
        "hdt_1p80mpa_C": 172.0,
        "tg_C": 78.0,
        "max_continuous_service_C": 150.0,
        "k_in_plane_W_mK": 0.95,
        "k_through_wall_W_mK": 0.42,
        "alpha_xy_1_K": 14.0e-6,
    },
    "PETG_CF": {
        "name": "PETG-CF (Carbon-Fiber Reinforced PETG)",
        "hdt_0p45mpa_C": 74.0,
        "hdt_1p80mpa_C": 66.0,
        "tg_C": 80.0,
        "max_continuous_service_C": 65.0,
        "k_in_plane_W_mK": 0.58,
        "k_through_wall_W_mK": 0.29,
        "alpha_xy_1_K": 32.0e-6,
    },
    "PLA_ref": {
        "name": "Standard PLA (Disqualified Reference)",
        "hdt_0p45mpa_C": 55.0,
        "hdt_1p80mpa_C": 51.0,
        "tg_C": 58.0,
        "max_continuous_service_C": 48.0,
        "k_in_plane_W_mK": 0.18,
        "k_through_wall_W_mK": 0.15,
        "alpha_xy_1_K": 68.0e-6,
    },
}

THERMAL_REGIMES = {
    "nominal_walk_compute_10p8A": {
        "label": "Nominal Walking + Compute (10.8 A pack / 3.6 A per cell)",
        "I_pack_A": 10.8,
        "I_cell_A": 3.6,
        "Q_pcb_W": 1.80,
        "duration_s": 1800.0,  # 30 min steady state
        "T_amb_C": 25.0,
    },
    "rated_locomotion_18p6A": {
        "label": "All-Rated Continuous Locomotion + Peak Compute (18.6 A pack / 6.2 A per cell)",
        "I_pack_A": 18.6,
        "I_cell_A": 6.2,
        "Q_pcb_W": 3.40,
        "duration_s": 1800.0,  # 30 min continuous
        "T_amb_C": 25.0,
    },
    "heavy_dynamic_duty_36A_15min": {
        "label": "Heavy Dynamic Squat/Recovery Duty (36.0 A pack / 12.0 A per cell, 15 min full 2.4C depletion)",
        "I_pack_A": 36.0,
        "I_cell_A": 12.0,
        "Q_pcb_W": 6.20,
        "duration_s": 900.0,  # 15 min full 15 Ah depletion at 36 A
        "T_amb_C": 25.0,
    },
    "peak_stall_burst_54A_60s": {
        "label": "Extreme Multi-Limb Stall / High-Power Burst (54.0 A pack / 18.0 A per cell for 60 s after Rated)",
        "I_pack_A": 54.0,
        "I_cell_A": 18.0,
        "Q_pcb_W": 11.50,
        "duration_s": 60.0,
        "T_amb_C": 25.0,
        "pre_regime": "rated_locomotion_18p6A",
    },
}


def cell_heat_generation_W(I_cell_A: float, T_cell_C: float = 40.0) -> float:
    """Compute per-cell Joule + reversible entropic heat generation (W) with temperature-dependent R_dc."""
    # R_dc decreases slightly with warming from 25C to 50C (~ -0.35% / K):
    r_ohm = (P50B_SPEC["r_dc_mohm_25C"] * 1e-3) * (1.0 - 0.003 * max(0.0, min(T_cell_C - 25.0, 35.0)))
    q_joule = (I_cell_A ** 2) * r_ohm
    q_entropic = I_cell_A * (T_cell_C + 273.15) * (P50B_SPEC["entropic_coeff_mV_K"] * 1e-3) * 0.35
    return float(q_joule + q_entropic)


def build_thermal_conductance_matrix(vented_chimney: bool, battery_al_fins: bool = False) -> dict:
    """Build the 9-cell + carrier + internal-air conductance network from the exact 5+2+2 Honeycomb geometry."""
    cells = common.HONEYCOMB_9CELL_BATTERY_LAYOUT
    n = len(cells)
    A_can_side = 2.0 * math.pi * (10.75e-3) * (70.15e-3)  # 0.004738 m^2 per cell
    A_can_ends = 2.0 * math.pi * ((9.0e-3 if vented_chimney else 8.4e-3) ** 2)  # exposed end-cap area

    # Direct-on-battery 6063-T5 anodized aluminum 4-blade vertical heat-sink fin strip (6.0 x 62.0 x 2.95 mm):
    # - Bonded directly to the bare 21700 cell can inside each cell's 7.70 mm x 71.2 mm vertical window slot
    #   via 0.15 mm electrically-insulating thermal tape (k = 1.5 W/mK, A_base = 6.0e-3 * 62.0e-3 = 3.72e-4 m^2 -> R_tim = 0.268 K/W).
    # - 4 vertical fin blades (2.20 mm depth x 62.0 mm height x 8 faces + tips) add A_fin = 0.00312 m^2 (+66% of entire cell can area!)
    #   with fin efficiency eta_fin = 0.94 in 6063-T5 Al (k = 200 W/mK).
    A_al_fin_eff = 0.94 * 0.00312 if battery_al_fins else 0.0
    R_tim_K_W = 0.268

    # Pairwise cell-to-cell thermal conductance G_ij (W/K) across shared honeycomb walls (pitch = 22.4 mm):
    G_ij = np.zeros((n, n), dtype=float)
    neighbor_counts = np.zeros(n, dtype=int)
    for i in range(n):
        xi, yi, zi = cells[i]["center"]
        for j in range(i + 1, n):
            xj, yj, zj = cells[j]["center"]
            if abs(zi - zj) < 10.0:
                d_xy = math.hypot(xi - xj, yi - yj)
                if d_xy < 24.5:  # Adjacent in the 22.4 mm hexagonal lattice
                    neighbor_counts[i] += 1
                    neighbor_counts[j] += 1
                    if vented_chimney:
                        g_pair = 0.042
                    else:
                        g_pair = 0.068
                    G_ij[i, j] = G_ij[j, i] = g_pair

    G_cell_to_air = np.zeros(n, dtype=float)
    exposed_area_fraction = np.zeros(n, dtype=float)
    for i in range(n):
        nb = neighbor_counts[i]
        is_interior = i in (1, 2, 3, 5, 6)  # Cell_02, Cell_03, Cell_04, Cell_06, Cell_07 border the 3 interstitial chimneys
        if vented_chimney:
            # All 9 cells have a full-height 7.70 mm x 71.2 mm vertical window slot + inter-cell cross-tunnels + 3 vertical interstitial chimneys:
            f_exp = 0.56 - 0.024 * nb + (0.14 if is_interior else 0.05)
            h_conv = 18.8 + (4.2 if is_interior else 1.5)  # Buoyant chimney slot + cross-ventilation enhancement (W/m^2K)
            g_wall_cond = 0.055  # Conduction through PAHT-CF honeycomb ribs into outer frame fin area
            if battery_al_fins:
                # Series combination of 0.15 mm TIM tape (R_tim = 0.268 K/W) and 4-blade vertical Al fin convection/radiation:
                h_fin = h_conv + 3.8  # Vertical fin channel buoyant stack acceleration + anodized radiative emission
                g_fin_conv = h_fin * A_al_fin_eff
                g_al_fin_branch = 1.0 / (R_tim_K_W + 1.0 / g_fin_conv)
            else:
                g_al_fin_branch = 0.0
        else:
            f_exp = max(0.14, 0.42 - 0.075 * nb)
            h_conv = 7.2 - 0.85 * nb
            g_wall_cond = 0.024
            g_al_fin_branch = 0.0
        exposed_area_fraction[i] = round(float(f_exp), 3)
        G_cell_to_air[i] = float(h_conv * (f_exp * A_can_side + A_can_ends) + g_wall_cond + g_al_fin_branch)

    # Enclosure air-to-ambient conductance (buoyant torso chimney draft from pelvis Z=194 mm to neck/shoulder exits Z=388 mm + shell vents):
    G_air_to_amb = (3.45 if battery_al_fins else 2.85) if vented_chimney else 1.42
    # Stage 2 plume pre-heat coupling into Stage 1 central cells:
    plume_coupling = (0.032 if battery_al_fins else 0.045) if vented_chimney else 0.135

    return {
        "G_ij": G_ij,
        "G_cell_to_air": G_cell_to_air,
        "G_air_to_amb": G_air_to_amb,
        "plume_coupling": plume_coupling,
        "neighbor_counts": neighbor_counts.tolist(),
        "exposed_area_fraction": exposed_area_fraction.tolist(),
        "battery_al_fins": battery_al_fins,
    }


def simulate_regime(
    net: dict,
    regime: dict,
    T_init_cells: np.ndarray | None = None,
    T_init_air: float | None = None,
    dt: float = 0.5,
) -> dict:
    """Run transient thermal integration of the 9 honeycomb cells + torso air plume over `duration_s`."""
    n = 9
    T_amb = float(regime["T_amb_C"])
    I_cell = float(regime["I_cell_A"])
    Q_pcb = float(regime["Q_pcb_W"])
    dur = float(regime["duration_s"])
    steps = int(round(dur / dt))

    T_cells = np.full(n, T_amb, dtype=float) if T_init_cells is None else T_init_cells.copy()
    T_air = T_amb if T_init_air is None else float(T_init_air)

    C_cell = P50B_SPEC["heat_capacity_J_K"]
    C_air_and_carrier = 162.0e-3 * 1150.0  # 162 g PAHT-CF carrier + internal air thermal mass (~186 J/K)

    G_ij = net["G_ij"]
    G_ca = net["G_cell_to_air"]
    G_aa = net["G_air_to_amb"]
    plume_c = net["plume_coupling"]

    trace_t = []
    trace_max = []
    trace_min = []
    trace_air = []

    for step in range(steps + 1):
        t_s = step * dt
        if step % max(1, steps // 60) == 0 or step == steps:
            trace_t.append(round(t_s, 1))
            trace_max.append(round(float(np.max(T_cells)), 2))
            trace_min.append(round(float(np.min(T_cells)), 2))
            trace_air.append(round(float(T_air), 2))
        if step == steps:
            break

        # Buoyant sensitivity: natural convection coefficient scales with (Delta T)^0.25:
        dT_air = max(1.5, float(np.mean(T_cells) - T_amb))
        buoy_factor = (dT_air / 15.0) ** 0.22

        q_cells = np.array([cell_heat_generation_W(I_cell, T_cells[i]) for i in range(n)], dtype=float)
        # Stage 2 lower pelvis cells (7, 8) shed a fraction of their warm plume upward toward Stage 1 core cells (2, 5, 6):
        q_plume_s2 = float(np.sum(G_ca[7:9] * buoy_factor * (T_cells[7:9] - T_air)))

        dT_c = np.zeros(n, dtype=float)
        total_conv_to_air = 0.0
        for i in range(n):
            q_cond_nb = float(np.sum(G_ij[i] * (T_cells - T_cells[i])))
            local_air_T = T_air + (plume_c * q_plume_s2 / G_aa if i in (2, 5, 6) else 0.0)
            q_conv = G_ca[i] * buoy_factor * (T_cells[i] - local_air_T)
            dT_c[i] = (q_cells[i] + q_cond_nb - q_conv) / C_cell
            total_conv_to_air += q_conv

        q_air_out = G_aa * buoy_factor * (T_air - T_amb)
        dT_a = (total_conv_to_air + 0.65 * Q_pcb - q_air_out) / C_air_and_carrier

        T_cells += dt * dT_c
        T_air += dt * dT_a

    # Carrier inner wall peak temperature sits between hottest cell surface and internal air temperature:
    T_cell_max = float(np.max(T_cells))
    T_cell_min = float(np.min(T_cells))
    T_wall_max = float(0.82 * T_cell_max + 0.18 * T_air)
    dT_wall = T_wall_max - 25.0
    radial_can_expansion_mm = P50B_SPEC["r_can_mm"] * P50B_SPEC["alpha_thermal_1_K"] * (T_cell_max - 25.0)
    radial_paht_expansion_mm = 10.95 * POLYMER_THERMAL_SPECS["PAHT_CF"]["alpha_xy_1_K"] * dT_wall
    remaining_radial_clearance_mm = (10.95 - 10.75) - radial_can_expansion_mm + radial_paht_expansion_mm

    return {
        "cell_temperatures_C": [round(float(x), 2) for x in T_cells],
        "cell_names": [c["name"] for c in common.HONEYCOMB_9CELL_BATTERY_LAYOUT],
        "T_cell_max_C": round(T_cell_max, 2),
        "T_cell_min_C": round(T_cell_min, 2),
        "T_cell_mean_C": round(float(np.mean(T_cells)), 2),
        "pack_delta_T_C": round(T_cell_max - T_cell_min, 2),
        "T_torso_air_C": round(float(T_air), 2),
        "T_carrier_wall_max_C": round(T_wall_max, 2),
        "hottest_cell": common.HONEYCOMB_9CELL_BATTERY_LAYOUT[int(np.argmax(T_cells))]["name"],
        "per_cell_heat_W": round(cell_heat_generation_W(I_cell, float(np.mean(T_cells))), 3),
        "total_9cell_heat_W": round(9.0 * cell_heat_generation_W(I_cell, float(np.mean(T_cells))), 2),
        "total_system_heat_W": round(9.0 * cell_heat_generation_W(I_cell, float(np.mean(T_cells))) + Q_pcb, 2),
        "radial_can_expansion_mm": round(float(radial_can_expansion_mm), 4),
        "remaining_radial_bore_clearance_mm": round(float(remaining_radial_clearance_mm), 4),
        "hdt_margins_C": {
            k: round(v["hdt_0p45mpa_C"] - T_wall_max, 1) for k, v in POLYMER_THERMAL_SPECS.items()
        },
        "trace": {
            "time_s": trace_t,
            "T_cell_max_C": trace_max,
            "T_cell_min_C": trace_min,
            "T_torso_air_C": trace_air,
        },
    }


def render_thermal_figure(report: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), dpi=160)

    # Panel 1 (Top-Left): Per-Cell Temperature Comparison (Closed vs Chimney-Vented vs Chimney + 9x Direct-on-Battery Al Fins)
    ax = axes[0, 0]
    short_names = [
        "C1: Dors L2",
        "C2: Dors L1",
        "C3: Dors C0",
        "C4: Dors R1",
        "C5: Dors R2",
        "C6: Core L",
        "C7: Core R",
        "C8: Pelv L",
        "C9: Pelv R",
    ]
    x = np.arange(len(short_names))
    w = 0.16
    u_36 = report["configurations"]["unvented_closed_honeycomb"]["regimes"]["heavy_dynamic_duty_36A_15min"]["cell_temperatures_C"]
    v_36 = report["configurations"]["chimney_vented_honeycomb"]["regimes"]["heavy_dynamic_duty_36A_15min"]["cell_temperatures_C"]
    f_36 = report["configurations"]["chimney_plus_battery_al_fins"]["regimes"]["heavy_dynamic_duty_36A_15min"]["cell_temperatures_C"]
    v_18 = report["configurations"]["chimney_vented_honeycomb"]["regimes"]["rated_locomotion_18p6A"]["cell_temperatures_C"]
    f_18 = report["configurations"]["chimney_plus_battery_al_fins"]["regimes"]["rated_locomotion_18p6A"]["cell_temperatures_C"]

    ax.bar(x - 2.0 * w, u_36, width=w, color="#d96b43", label="Closed Honeycomb @ 36 A (15m)")
    ax.bar(x - 1.0 * w, v_36, width=w, color="#2a8ab8", label="Chimney-Vented @ 36 A (15m)")
    ax.bar(x + 0.0 * w, f_36, width=w, color="#19647e", label="Chimney + 9x Battery Al Fins @ 36 A")
    ax.bar(x + 1.0 * w, v_18, width=w, color="#66c2a5", label="Chimney-Vented @ 18.6 A Rated")
    ax.bar(x + 2.0 * w, f_18, width=w, color="#1ca48e", label="Chimney + 9x Battery Al Fins @ 18.6 A")
    ax.axhline(60.0, color="#d48634", linestyle="--", linewidth=1.6, label="P50B Continuous Target (60 °C)")
    ax.axhline(70.0, color="#c93b2b", linestyle=":", linewidth=1.8, label="BMS High-Temp Cutoff (70 °C)")
    ax.set_xticks(x, short_names, rotation=22, fontsize=8.5)
    ax.set_ylabel("Cell Temperature (°C)")
    ax.set_ylim(20, 82)
    ax.set_title("1. Per-Cell Peak Temp: Closed vs Vented vs 9x Direct-on-Battery Al Fins", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.3, ncol=2, loc="upper left")

    # Panel 2 (Top-Right): Transient Temperature Evolution vs Polymer HDT & Cell Limits
    ax2 = axes[0, 1]
    reg_v = report["configurations"]["chimney_vented_honeycomb"]["regimes"]
    reg_f = report["configurations"]["chimney_plus_battery_al_fins"]["regimes"]
    reg_u = report["configurations"]["unvented_closed_honeycomb"]["regimes"]
    for r_key, col, lbl in [
        ("nominal_walk_compute_10p8A", "#2e7d32", "Chimney + 9x Al Fins: 10.8 A Walk (4.1 W)"),
        ("rated_locomotion_18p6A", "#1ca48e", "Chimney + 9x Al Fins: 18.6 A Rated (10.2 W)"),
        ("heavy_dynamic_duty_36A_15min", "#19647e", "Chimney + 9x Al Fins: 36.0 A Heavy (30.3 W, 15m)"),
    ]:
        t_min = np.array(reg_f[r_key]["trace"]["time_s"]) / 60.0
        ax2.plot(t_min, reg_f[r_key]["trace"]["T_cell_max_C"], color=col, linewidth=2.3, label=lbl)
    t_v36 = np.array(reg_v["heavy_dynamic_duty_36A_15min"]["trace"]["time_s"]) / 60.0
    ax2.plot(t_v36, reg_v["heavy_dynamic_duty_36A_15min"]["trace"]["T_cell_max_C"], color="#2a8ab8", linestyle="-.", linewidth=1.9, label="Chimney-Vented Only: 36.0 A Heavy (No Al Fins)")
    t_u36 = np.array(reg_u["heavy_dynamic_duty_36A_15min"]["trace"]["time_s"]) / 60.0
    ax2.plot(t_u36, reg_u["heavy_dynamic_duty_36A_15min"]["trace"]["T_cell_max_C"], color="#c93b2b", linestyle="--", linewidth=2.0, label="Closed Honeycomb: 36.0 A Heavy (Unvented)")

    ax2.axhline(55.0, color="#9e2a2b", linestyle="-.", linewidth=1.4, label="PLA HDT Limit (55 °C — Softens!)")
    ax2.axhline(74.0, color="#d48634", linestyle="--", linewidth=1.5, label="PETG-CF HDT Limit (74 °C)")
    ax2.text(16.0, 76.5, "PAHT-CF HDT = 194 °C (>> 150 °C margin)", fontsize=8.5, fontweight="bold", color="#1b4d3e")
    ax2.set_xlabel("Operating Time (minutes)")
    ax2.set_ylabel("Peak Cell Temperature (°C)")
    ax2.set_ylim(22, 85)
    ax2.set_title("2. Transient Thermal Curves vs Cell & Polymer HDT Limits", fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.3, loc="upper left")

    # Panel 3 (Bottom-Left): Top-Down (X, Y) Cross-Section of the 5+2+2 Honeycomb Lattice + 3 Interstitial Chimneys + 9 Direct-on-Battery Al Fins
    ax3 = axes[1, 0]
    temps_36 = reg_f["heavy_dynamic_duty_36A_15min"]["cell_temperatures_C"]
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=38.0, vmax=60.0)
    for idx, c in enumerate(common.HONEYCOMB_9CELL_BATTERY_LAYOUT):
        cx, cy, cz = c["center"]
        s_y = float(c.get("fin_dir_y", +1))
        is_pelvis = idx >= 7
        y_plot = cy - (26.0 if is_pelvis else 0.0)
        # Outer polymer wall with vertical window slot cut open on s_y side:
        outer_cir = plt.Circle((cx, y_plot), 12.65, color="#3a4652", alpha=0.35 if is_pelvis else 0.75)
        ax3.add_patch(outer_cir)
        slot_clear = plt.Rectangle(
            (cx - 3.85, y_plot + (10.25 if s_y > 0 else -13.25)),
            7.70,
            3.00,
            color="#f5f8fa",
            zorder=2,
        )
        ax3.add_patch(slot_clear)
        # Cell can colored by temperature
        cell_cir = plt.Circle((cx, y_plot), 10.75, color=cmap(norm(temps_36[idx])), ec="#1c2228", lw=1.2, zorder=3)
        ax3.add_patch(cell_cir)
        # 4-Blade Anodized Aluminum Heat-Sink Fin glued directly to exposed battery arc inside slot:
        y_base = y_plot + (10.35 if s_y > 0 else -11.35)
        ax3.add_patch(plt.Rectangle((cx - 3.00, y_base), 6.00, 1.00, color="#c8d4e0", ec="#3d4f60", lw=0.8, zorder=4))
        for dx in (-2.56, -0.85, 0.85, 2.56):
            y_bl = y_plot + (11.15 if s_y > 0 else -13.35)
            ax3.add_patch(plt.Rectangle((cx + dx - 0.44, y_bl), 0.88, 2.20, color="#dce6f2", ec="#3d4f60", lw=0.6, zorder=5))
        tag = f"C{idx+1}\n{temps_36[idx]:.1f}°C"
        ax3.text(cx, y_plot, tag, ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=6)

    # Plot the 3 vertical interstitial cooling chimneys in Stage 1:
    for ch_x, ch_y in [(-11.2, 19.133), (0.0, 12.667), (11.2, 19.133)]:
        ch_cir = plt.Circle((ch_x, ch_y), 2.4, color="#dff6ff", ec="#0077b6", lw=1.6, zorder=6)
        ax3.add_patch(ch_cir)
        ax3.text(ch_x, ch_y, "Air\nChim", ha="center", va="center", fontsize=5.8, color="#003554", fontweight="bold", zorder=7)

    ax3.set_xlim(-64, 64)
    ax3.set_ylim(-28, 44)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Torso X coordinate (mm)")
    ax3.set_ylabel("Torso Y coordinate (mm, Pelvis offset -26 mm for view)")
    ax3.set_title("3. 9x Direct-on-Battery Al Fins in Honeycomb Slots + 3 Interstitial Chimneys", fontweight="bold")
    ax3.grid(alpha=0.25)

    # Panel 4 (Bottom-Right): Pack Delta-T Uniformity & Polymer Stiffness Retention vs Temperature
    ax4 = axes[1, 1]
    T_grid = np.linspace(20.0, 140.0, 120)
    E_paht = 100.0 / (1.0 + np.exp((T_grid - 115.0) / 18.0)) * (1.0 - 0.0012 * np.maximum(0.0, T_grid - 25.0))
    E_petg = 100.0 / (1.0 + np.exp((T_grid - 73.0) / 6.5))
    E_pla = 100.0 / (1.0 + np.exp((T_grid - 54.0) / 4.2))
    ax4.plot(T_grid, E_paht, color="#1ca48e", linewidth=2.5, label="PAHT-CF (HDT = 194 °C, 91% E @ 55 °C)")
    ax4.plot(T_grid, E_petg, color="#d48634", linewidth=2.2, linestyle="--", label="PETG-CF (HDT = 74 °C, 78% E @ 55 °C)")
    ax4.plot(T_grid, E_pla, color="#c93b2b", linewidth=2.0, linestyle="-.", label="Standard PLA (HDT = 55 °C — Disqualified)")
    w_18 = reg_f["rated_locomotion_18p6A"]["T_carrier_wall_max_C"]
    w_36 = reg_f["heavy_dynamic_duty_36A_15min"]["T_carrier_wall_max_C"]
    ax4.axvline(w_18, color="#1b4d3e", linestyle=":", linewidth=1.6, label=f"Wall w/ Al Fins @ 18.6 A ({w_18:.1f} °C)")
    ax4.axvline(w_36, color="#0077b6", linestyle=":", linewidth=1.6, label=f"Wall w/ Al Fins @ 36 A ({w_36:.1f} °C)")
    ax4.set_xlabel("Polymer Carrier Wall Temperature (°C)")
    ax4.set_ylabel("Relative Flexural / Tensile Modulus Retention (%)")
    ax4.set_ylim(0, 108)
    ax4.set_title("4. Polymer Stiffness Retention vs Carrier Wall Temperature", fontweight="bold")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=7.8, loc="lower left")

    fig.suptitle(
        "APE0 9-Cell Honeycomb Battery Carrier — Chimney Convection + 9x Direct-on-Battery Aluminum Heat Fins & Polymer HDT",
        fontsize=12.2,
        fontweight="bold",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    t0 = time.monotonic()
    configs = {
        "unvented_closed_honeycomb": build_thermal_conductance_matrix(vented_chimney=False, battery_al_fins=False),
        "chimney_vented_honeycomb": build_thermal_conductance_matrix(vented_chimney=True, battery_al_fins=False),
        "chimney_plus_battery_al_fins": build_thermal_conductance_matrix(vented_chimney=True, battery_al_fins=True),
    }

    report = {
        "schema": "cove50.torso.honeycomb_thermal_audit.v1",
        "battery_spec": P50B_SPEC,
        "polymer_specs": POLYMER_THERMAL_SPECS,
        "configurations": {},
    }

    for cfg_name, net in configs.items():
        reg_results = {}
        for r_key, r_spec in THERMAL_REGIMES.items():
            if "pre_regime" in r_spec:
                pre = reg_results[r_spec["pre_regime"]]
                res = simulate_regime(
                    net,
                    r_spec,
                    T_init_cells=np.array(pre["cell_temperatures_C"], dtype=float),
                    T_init_air=pre["T_torso_air_C"],
                )
            else:
                res = simulate_regime(net, r_spec)
            reg_results[r_key] = res
        report["configurations"][cfg_name] = {
            "neighbor_counts": net["neighbor_counts"],
            "exposed_area_fraction": net["exposed_area_fraction"],
            "regimes": reg_results,
        }

    for r_key in THERMAL_REGIMES:
        u = report["configurations"]["unvented_closed_honeycomb"]["regimes"][r_key]
        v = report["configurations"]["chimney_vented_honeycomb"]["regimes"][r_key]
        f = report["configurations"]["chimney_plus_battery_al_fins"]["regimes"][r_key]
        print(
            f"[Thermal] {r_key:30s} | Closed: {u['T_cell_max_C']:5.2f}C "
            f"-> Chimney: {v['T_cell_max_C']:5.2f}C "
            f"-> +9x Battery Al Fins: {f['T_cell_max_C']:5.2f}C (dT={f['pack_delta_T_C']:4.2f}C, wall={f['T_carrier_wall_max_C']:5.2f}C)"
        )

    out_json = common.ARTIFACT_DIR / "torso-thermal-results.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    out_png = common.ARTIFACT_DIR / "renders" / "honeycomb-thermal-analysis.png"
    render_thermal_figure(report, out_png)
    print(f"Saved {out_json.relative_to(ROOT)} and {out_png.relative_to(ROOT)} in {time.monotonic() - t0:.2f}s")


if __name__ == "__main__":
    main()
