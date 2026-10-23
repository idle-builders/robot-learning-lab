"""Infer a partly occluded cube from RGB by rendering known-geometry hypotheses.

The search sees RGB and the camera/robot pose. Every hypothetical cube position
overwrites the simulation cube state before rendering; native cube coordinates
are never used as the predicted goal. This remains a virtual-camera sandbox.
"""

from __future__ import annotations

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.calibrated_virtual_rgb_goal import (
    CalibratedVirtualRgbGoal,
)


def orange_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    return (red > 100) & (red > green * 1.25) & (green > blue * 1.25)


class TemplateMatchedVirtualRgbGoal(CalibratedVirtualRgbGoal):
    """Refine only the reach-entry goal with exact virtual scene geometry."""

    def _match(self, robot_qpos: np.ndarray, rgb: np.ndarray,
               coarse_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        target = orange_mask(rgb)
        candidate_qpos = robot_qpos.copy()
        candidate_qpos[28:31] = [coarse_xy[0], coarse_xy[1], .01]
        candidate_qpos[31:35] = [1, 0, 0, 0]
        best: tuple[int, float, np.ndarray, int] | None = None
        rendered_count = 0
        tied_best = 0
        for step_mm, offsets_mm in ((5, range(-20, 21, 5)),
                                    (1, range(-5, 6))):
            center = coarse_xy if best is None else best[2]
            for dx in offsets_mm:
                for dy in offsets_mm:
                    candidate_xy = center + np.asarray([dx, dy]) / 1000
                    candidate_qpos[28:30] = candidate_xy
                    self.data.qpos[:] = candidate_qpos
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                    self.renderer.update_scene(self.data, camera="lower_rgb", scene_option=self.option)
                    visible = orange_mask(self.renderer.render())
                    rendered_count += 1
                    disagreement = int(np.count_nonzero(visible ^ target))
                    distance = float(np.linalg.norm((candidate_xy - coarse_xy) * 1000))
                    key = (disagreement, distance)
                    if best is None or key < (best[0], best[1]):
                        tied_best = 1 if best is None or disagreement < best[0] else tied_best + 1
                        best = (disagreement, distance, candidate_xy.copy(),
                                int(np.count_nonzero(visible)))
                    elif disagreement == best[0]:
                        tied_best += 1
        if best is None:
            raise RuntimeError("No rendered toy hypothesis")
        return best[2], {"template_pixels_disagree": best[0],
                         "template_visible_pixels": best[3],
                         "template_candidate_count": rendered_count,
                         "template_best_score_tie_count": tied_best,
                         "template_prior_distance_mm": round(best[1], 4)}

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        event["template_applied"] = False
        if not event["accepted"] or abs(time_s - 7.5) > 1e-6:
            return event, rgb
        if native_qpos.shape != (35,):
            raise ValueError("Unexpected native scene pose")
        coarse = np.asarray(event["estimated_goal_world_xyz_m"], dtype=float)
        matched_xy, evidence = self._match(native_qpos, rgb, coarse[:2])
        updated = coarse.copy()
        updated[:2] = matched_xy
        event.update(evidence)
        event["template_applied"] = True
        event["pre_template_goal_world_xyz_m"] = coarse.tolist()
        event["estimated_goal_world_xyz_m"] = updated.tolist()
        event["template_correction_xy_mm"] = np.round((matched_xy - coarse[:2]) * 1000, 4).tolist()
        event["template_assumptions"] = "exact synthetic camera, robot pose, and upright 20 mm cube model"
        if event["used_for_control"]:
            self.last_goal_world = updated
        return event, rgb
