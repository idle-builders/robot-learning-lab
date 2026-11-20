"""Reference numbers for lens C: today's shells (uncut belly/back shells, original thigh covers) measured with the
same kit.interference grid, so each concept's contact-free ranges can be read against the body as it is.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/moving-compliant/reference_body.py
"""
import mc_common as mc
from mc_common import kit

NAMES = ("center_front_shell", "center_back_shell", "R_thigh_front_hard_cover", "R_thigh_back_hard_cover")


def main():
    body = kit.base()
    parts = []
    for n in NAMES:
        p = body.parts[n]; m = kit._mesh(body, n)
        if not m.is_watertight:
            m = kit.tm(kit.mf(m))
        parts.append(kit.Concept(n if n.startswith("R_") else "ref_" + n, m, p["owning_link"]))
    res = kit.interference(body, parts, hide=NAMES)
    res["concept"] = "reference: today's shells, no change"
    mc.save_json(res, mc.HERE / "reference-body-interference.json")
    for j, v in res["ranges"].items():
        print(j, v["contact_free_deg"], v["first_contact"])
    print(res["combined_poses_with_contact"], "/", res["combined_poses"])
    for w in res["worst_combined"][:4]:
        print(w)


if __name__ == "__main__":
    main()
