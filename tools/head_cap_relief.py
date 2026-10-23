"""Conservative local relief derived from a bound failed cap-motion study."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from build123d import Box, Compound, Pos, import_step

from lab.leg_mechanism import overlap, transform_shape
from tools.check_camera_support import CameraQueries, sha
from tools.compose_forearm_enclosure import verify_report


def expanded_box(shape, extra_mm):
    box = CameraQueries().bounds(shape)
    lo, hi = np.array(list(box.min), dtype=float), np.array(list(box.max), dtype=float)
    return Pos(*((lo + hi) / 2)) * Box(*(hi - lo + 2 * extra_mm))


def relieve(covers, envelopes, robot, parts, name, margin_mm=1.5):
    """Remove the dilation of each local interference region, not a whole part.

    If E contains C dilated by margin m, and B contains E intersect target,
    then C' = C minus expand(B,m+epsilon) has a conservative envelope
    E' = E minus expand(B,epsilon). Subsequent removals cannot restore a hit.
    Bounding boxes may remove excess stock; connectedness is checked by caller.
    This uses sampled seed poses, not a continuous swept-volume proof.
    """
    source = Path(f"artifacts/{name}-head-cap-stock-v2")
    seed = verify_report(source)
    assembly = f"artifacts/{name}-distal-grip-integrated-v1/assembly.step"
    if seed["source_bindings"][assembly] != sha(assembly):
        raise ValueError("Relief source assembly differs")
    if seed.get("depth_mm") != 8 or seed["moving_clearance_mm"] != margin_mm:
        raise ValueError("Relief seed must be the checked 8 mm / 1.5 mm study")
    if (source / "resolved-config.json").read_text() != robot.config.model_dump_json(
        indent=2
    ):
        raise ValueError("Relief source configuration differs")
    bindings = {str(source / "report.json"): sha(source / "report.json")}
    for key, shape in covers.items():
        path = source / f"{key}.step"
        old = import_step(path)
        if (
            abs(shape.volume - old.volume) > 0.02
            or overlap(shape, old) < shape.volume - 0.02
        ):
            raise ValueError("Regenerated cap differs from relief seed")
        bindings[str(path)] = sha(path)
    original_volumes = {key: s.volume for key, s in covers.items()}
    neutral = robot.forward_kinematics()
    lookup = {p.name: p for p in parts}
    cuts = []
    for row in seed["motion"]:
        if not row["conservative_clearance_collisions"]:
            continue
        fk = robot.forward_kinematics(row["pose_rad"], enforce_limits=False)
        for hit in row["conservative_clearance_collisions"]:
            key, target_name = hit["cover"], hit["part"]
            if not covers[key].solids() or not envelopes[key].solids():
                continue
            target = lookup[target_name]
            if target.owner == "head":
                raise ValueError("Relief cannot remove head-owned mating interfaces")
            located = replace(
                target,
                shape=transform_shape(
                    target.shape,
                    np.linalg.inv(fk["head"])
                    @ fk[target.owner]
                    @ np.linalg.inv(neutral[target.owner]),
                ),
            )
            region = Compound(envelopes[key].solids()) & Compound(
                located.shape.solids()
            )
            if not region.solids() or region.volume <= 0.02:
                continue
            for solid in region.solids():
                # Retain the existing conservative bound method; a loose box
                # can reject material feasibility but cannot assert a false gap.
                local_target = transform_shape(
                    target.shape, np.linalg.inv(neutral[target.owner])
                )
                owner_to_head = np.linalg.inv(fk["head"]) @ fk[target.owner]
                outer_target = transform_shape(
                    expanded_box(local_target, margin_mm + 0.05), owner_to_head
                )
                inner_target = transform_shape(
                    expanded_box(local_target, 0.05), owner_to_head
                )
                cut = expanded_box(solid, margin_mm + 0.05) & outer_target
                inner_cut = expanded_box(solid, 0.05) & inner_target
                before = covers[key].volume
                covers[key] -= cut
                envelopes[key] -= inner_cut
                cuts.append(
                    dict(
                        pose=row["name"],
                        pose_rad=row["pose_rad"],
                        cover=key,
                        target=target_name,
                        interference_envelope_volume_mm3=solid.volume,
                        removed_mm3=before - covers[key].volume,
                        solid_count=len(covers[key].solids()),
                    )
                )
                print(
                    name,
                    "relief",
                    key,
                    target_name,
                    "remaining",
                    round(covers[key].volume, 2),
                    "solids",
                    len(covers[key].solids()),
                    flush=True,
                )
    return (
        covers,
        envelopes,
        dict(
            source_bindings=bindings,
            method="Intersection of head-frame local-region box and owner-aligned obstacle box; both expanded by margin + 0.05 mm",
            clearance_mm=margin_mm,
            numeric_allowance_mm=0.05,
            cuts=cuts,
            original_volumes_mm3=original_volumes,
            remaining_volumes_mm3={key: s.volume for key, s in covers.items()},
            continuous_sweep_status="not_run",
        ),
    )
