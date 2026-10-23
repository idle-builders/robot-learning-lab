"""Motor-side yaw transmission and bearing candidate for the exposed ape.

This is source-size geometry with a simple radial-load screen in the coupon
report, not a qualified shaft, printed mount or spline fit.
"""

from __future__ import annotations

import build123d as bd

from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip


def apply(hw: dict) -> None:
    # Runtime import avoids a module cycle: the roll coupon itself imports
    # pelvis_tube_candidate, while this same manufacturer STEP is used here.
    from experiments.compact_morphology.ape_frame50.roll_coupler_package import (
        installed_coupler,
    )

    fixed = hw["fixed"]
    # The old pitch spindle projected 3 mm inward past its medial bearing.
    # Its tip hit the larger motor-side bearing tube during a 7° side shift.
    # Shorten only that unsupported inboard stub; outboard end and thigh-root
    # join remain unchanged. Pitch-axis axial retention is still unqualified.
    hw["thigh"]["pitch_spindle"] = hip.cyl(3, 65.5,
                                            (42.5, 2, 219), (1, 0, 0))
    # Rotate the pinned two-solid goBILDA 25T-to-6 mm STEP from the roll
    # motor's +Y output to the yaw motor's +Z output. The spline entrance
    # lies 0.5 mm above the ST3250 case; the coupler occupies Z=198..215.
    fixed["yaw_motor_coupler_4001"] = (
        bd.Pos(27.5, 2, 198) * bd.Rot(90, 0, 0) *
        bd.Pos(-59, 60, -278.5) * installed_coupler())
    # Separate candidate 6 mm steel shaft, with a positive 8 mm shoulder
    # above the upper bearing and a 5.7 mm DIN 471-6 groove below the lower.
    fixed["yaw_motor_output"] = (
        hip.cyl(3, 67, (27.5, 2, 203)) +
        hip.cyl(4, 1, (27.5, 2, 252)) -
        hip.ring(3.01, 2.85, .7, (27.5, 2, 216.1)))
    fixed["yaw_motor_inner_spacer"] = hip.ring(4, 3, 22, (27.5, 2, 224))
    fixed["yaw_motor_lower_washer"] = hip.ring(4, 3, .8, (27.5, 2, 217))
    fixed["yaw_motor_axial_DIN471_6_ring"] = (
        hip.ring(6, 2.85, .7, (27.5, 2, 216.1)) -
        hip.box((27.5, -4.5, 216.45), (4, 4, 1)))
    # Catalog JTEKT F606 ZZ opposed-flange envelopes: 6x17x6 mm, flange
    # 19x1.2 mm. The previously assumed 6x13x3.5 W618/6 lacked a useful
    # rated-point radial-load margin for this overhung motor-side pulley.
    del fixed["yaw_motor_bearing_low_W618_6"]
    del fixed["yaw_motor_bearing_high_W618_6"]
    fixed["yaw_motor_bearing_low_F606_ZZ"] = (
        hip.ring(8.5, 3, 6, (27.5, 2, 218)) +
        hip.ring(9.5, 8.5, 1.2, (27.5, 2, 218)))
    fixed["yaw_motor_bearing_high_F606_ZZ"] = (
        hip.ring(8.5, 3, 6, (27.5, 2, 246)) +
        hip.ring(9.5, 8.5, 1.2, (27.5, 2, 250.8)))
    fixed["yaw_motor_housing_tube"] = hip.ring(10.5, 8.5, 31.6,
                                                (27.5, 2, 219.2))
