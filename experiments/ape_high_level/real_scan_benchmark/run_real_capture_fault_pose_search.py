"""Run the fixed teach/find policy through four lost visual-motion updates.

The owner camera still provides RGB, depth and person detections. Only the
visual frame-to-frame pose estimate is unavailable on calls 2–5. This models a
temporary localization fault, not a blank camera or an absent owner.
"""

from __future__ import annotations

import run_real_capture_colour_bank_search as colour_bank
import run_real_capture_owner_loop as owner


original_odometry_delta = owner.odometry_delta
odometry_calls = 0
BLACKOUT_CALLS = (2, 3, 4, 5)


def faulted_odometry_delta(*args, **kwargs):
    global odometry_calls
    odometry_calls += 1
    if odometry_calls in BLACKOUT_CALLS:
        return None, {"accepted": False,
                      "status": "injected_visual_motion_unavailable",
                      "fault_schedule": "odometry_calls_2_through_5",
                      "odometry_call": odometry_calls}
    return original_odometry_delta(*args, **kwargs)


owner.odometry_delta = faulted_odometry_delta


if __name__ == "__main__":
    colour_bank.baseline.main()
