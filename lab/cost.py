from collections import Counter


def estimate(robot):
    c = robot.config
    k = c.cost
    counts = Counter(j.family for j in robot.joints)
    actuator = [
        sum(count * k.actuator_usd[family][i] for family, count in counts.items())
        for i in range(2)
    ]
    parts = [actuator[i] + k.other_parts_usd[i] for i in range(2)]
    fixed = [v / c.production_units for v in k.development_tooling_total_usd]
    total = [
        round(
            (
                parts[i]
                + k.assembly_test_usd[i]
                + k.logistics_packaging_usd[i]
                + k.support_operations_usd[i]
                + fixed[i]
            )
            * (1 + k.contingency_fraction),
            2,
        )
        for i in range(2)
    ]
    return {
        "status": "estimate",
        "evidence": k.evidence,
        "production_units": c.production_units,
        "actuator_counts": dict(counts),
        "actuator_order_quantities": {
            f: n * c.production_units for f, n in counts.items()
        },
        "actuator_cost_usd": actuator,
        "parts_cost_usd": parts,
        "fixed_cost_per_robot_usd": fixed,
        "total_nonprofit_price_range_usd": total,
        "target_usd": 1000,
        "planning_range_vs_target": (
            "below_target"
            if total[1] <= 1000
            else "above_target"
            if total[0] > 1000
            else "straddles_target"
        ),
        "target_feasibility": "unverified pending supplier quotes and manufacturing validation",
        "exclusions": [
            "customer taxes",
            "outbound delivery",
            "existing host computer",
            "ongoing training/inference",
        ],
    }
