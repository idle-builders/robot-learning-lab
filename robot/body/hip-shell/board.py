"""One comparison board of the diverge-round hip shell concepts (T-czv2d8).

Crops the standing and -90 flexion panels out of each concept render and adds the measured numbers
from each lens's CARD.md / interference JSON. Run: .venv/bin/python -B robot/body/hip-shell/board.py
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
BANDS = {"front": (318, 622), "3/4": (628, 932), "side": (938, 1242)}
# (render, label, numbers) ; numbers: kit grid, contact-free pitch / roll / yaw, combined poses with contact of 420
ROWS = [
    ("moving-compliant/sliding-briefs.png", "today's body (reference)", None),
    ("converging/b2_sketch_crease-look.png", "B2 sketch V briefs, fixed",
     "pitch -30..+25 · roll -30..+15 · yaw ±20\nfixed parts only · 363/420 combined\nfront visible 710 mm² (today 1524)"),
    ("converging/b3_side_walls-look.png", "B3 sloped wedge + side skirt, fixed",
     "pitch -65..+30 · roll -30..+15 · yaw ±20\nfixed parts only · 310/420 combined\nfront 2030 / side 878 mm² visible"),
    ("converging/b1_axis_wedge-look.png", "B1 sloped wedge for -90, fixed",
     "pitch -80..+30 · roll -30..+15 · yaw ±20\nfixed parts only · 297/420 combined\nfront 2556 mm² visible (1.7x today)"),
    ("concentric/a2-coaxial-drum.png", "A2 drum about pitch axis",
     "pitch -15..0 · roll -15..0 · yaw 0\ncap on pitch link · 415/420 combined\ncloses the sketch look best standing"),
    ("concentric/a3-ball-between-servos.png", "A3 ball about hip centre",
     "pitch -90..+40 · roll -15..0 · yaw ±20\nneeds pelvis pocket + connector reroute\n334/420 combined (pocketed)"),
    ("moving-compliant/hinged-tassets.png", "C2 spring-closed flaps",
     "pitch -40..+15 · roll ±15 · yaw ±20\n4 hinged flaps per hip · 356/420\nflaps stick out 55 mm at -90"),
    ("moving-compliant/sliding-briefs.png", "C1 briefs panel slides up",
     "pitch -15..+20 · roll ±15 · yaw ±20\nsliding panel in a channel · 385/420\nchannel hits electronics at -90"),
    ("moving-compliant/fabric-core.png", "C3 segmented core + stretch fabric",
     "pitch -90..+40 · roll ±15 · yaw ±20\nfabric + side panels · 302/420 (fewest)\nfabric strain: seat 25-71 %, inner 150-250 %"),
]

def panels(img, band):
    a = np.asarray(img.convert("L"))[:, band[0]:band[1]]
    rows = (a < 235).mean(1) > 0.25
    runs, start = [], None
    for i, r in enumerate(list(rows) + [False]):
        if r and start is None: start = i
        if not r and start is not None:
            if i - start > 150: runs.append((start, i))
            start = None
    return [img.crop((band[0], s, band[1], e)).resize((300, 288)) for s, e in runs]

def main():
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    cols = [("standing", "front"), ("standing", "3/4"), ("-90 flexion", "front"), ("-90 flexion", "side")]
    W, H, TXT = 300, 288, 330
    head = 40
    board = Image.new("RGB", (TXT + W * 4, head + H * (len(ROWS) - 1)), "white")
    d = ImageDraw.Draw(board)
    for j, (p, v) in enumerate(cols):
        d.text((TXT + j * W + 10, 10), f"{p} · {v}", fill="black", font=big)
    y = head
    for src, label, nums in ROWS[1:]:
        img = Image.open(HERE / src)
        got = {v: panels(img, BANDS[v]) for v in ("front", "3/4", "side")}
        for j, (p, v) in enumerate(cols):
            k = 0 if p == "standing" else 1
            board.paste(got[v][k], (TXT + j * W, y))
        d.text((10, y + 12), label, fill="black", font=big)
        d.multiline_text((10, y + 50), nums, fill=(60, 60, 60), font=font, spacing=6)
        d.line((0, y, board.width, y), fill=(180, 180, 180), width=2)
        y += H
    out = HERE / "concept-board.png"
    board.save(out)
    print(out, board.size)

if __name__ == "__main__":
    main()
