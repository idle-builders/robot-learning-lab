"""Screen (not a concept): why a front flap turning k x hip pitch about the pitch axis was not built.

A cylindrical front flap about the pitch axis (x 12..90, 3 mm thick) on an interpolated link that turns k of the hip
pitch on top of the yaw link, at a few radii and arc spans, k = 0.5 (and 0.33 / 0.67). The pitch sweep lists what it
hits: in front of the pitch axis the thigh at -90 fills the arc up to ~30-35 deg above horizontal (its front face is
27 mm ahead of the axis and becomes its top), while the torso behind the belly wall is full (battery cradle, hip-yaw
connector, front electronics), so a flap at k < 1 has nowhere to be at deep flexion.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/moving-compliant/half_angle_check.py
"""
import numpy as np

import mc_common as mc
from mc_common import kit

HIDE = ("center_front_shell", "R_thigh_front_hard_cover", "R_thigh_back_hard_cover")


def cyl_flap(r0, r1, a0, a1, x0=12.0, x1=90.0):
    """Shell sector about the pitch axis; angle 0 = straight forward (-y), +90 = up."""
    def fn(u, v, w):
        a = np.radians(a0 + (a1 - a0) * u); r = r0 + (r1 - r0) * w
        return x0 + (x1 - x0) * v, 2.0 - r * np.cos(a), 219.0 + r * np.sin(a)
    return mc.mapped_cube(fn, (16, 4, 1))


def main():
    body = kit.base()
    vb = mc.VBody(body)
    base = mc.trimmed_torso(body, 250.0, None) + mc.chamfered_thigh_covers(body)
    out = {}
    for k in (0.33, 0.5, 0.67):
        vb.virtual[f"k{k}"] = mc.frac_link(k)
        for (r0, a0, a1) in ((46.0, -8.0, 35.0), (55.0, -15.0, 15.0), (60.0, -20.0, 5.0)):
            flap = kit.Concept("R_half_angle_flap", cyl_flap(r0, r0 + 3.0, a0, a1), f"R_k{k}", (0.9, 0.8, 0.7))
            sw = mc.sweep(vb, base + [flap], HIDE, "R_hip_pitch", [-90, -75, -60, -45, -30])
            key = f"k={k} r={r0:.0f}..{r0 + 3:.0f} arc {a0:+.0f}..{a1:+.0f} deg"
            out[key] = {str(p): {n: v for n, v in d.items() if "half_angle" in n} for p, d in sw.items()}
            worst = {p: round(sum(v for n, v in d.items() if "half_angle" in n), 0) for p, d in sw.items()}
            print(key, worst, flush=True)
    mc.save_json(out, mc.HERE / "half-angle-check.json")


if __name__ == "__main__":
    main()
