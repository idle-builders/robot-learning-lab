"""Render V75's saved candidate screen as a readable map and result table."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import owner_deferred_wait_pose_feasibility as v75


def main():
    path = v75.OUT / "report.json"
    report = json.loads(path.read_text())
    width, height = 1450, 930
    image = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    try:
        title = ImageFont.truetype("DejaVuSans.ttf", 24)
        body = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title = body = small = ImageFont.load_default()
    x0, x1, y0, y1 = .4, 2.45, -5.7, -4.05
    left, top, map_width, map_height = 85, 120, 770, 650

    def xy(point):
        return (round(left+(point[0]-x0)*map_width/(x1-x0)),
                round(top+(y1-point[1])*map_height/(y1-y0)))

    draw.text((75, 32), "V75 | The current waiting-pose menu has no robust choice",
              fill=(18, 27, 35), font=title)
    draw.text((75, 68),
              "Evaluator-only six-path screen, 20 ms historical-body geometry; all 16 eligible poses breach 5 cm",
              fill=(55, 63, 70), font=body)
    draw.rectangle((left, top, left+map_width, top+map_height),
                   outline=(50, 60, 68), width=2)
    for x_tick in np.arange(.5, 2.46, .25):
        x_screen, _ = xy((x_tick, y0))
        draw.line((x_screen, top, x_screen, top+map_height),
                  fill=(220, 225, 229))
        draw.text((x_screen-16, top+map_height+7), f"{x_tick:.2f}",
                  fill=(50, 58, 66), font=small)
    for y_tick in np.arange(-5.5, -4.0, .25):
        _, y_screen = xy((x0, y_tick))
        draw.line((left, y_screen, left+map_width, y_screen),
                  fill=(220, 225, 229))
        draw.text((left-57, y_screen-8), f"{y_tick:.2f}",
                  fill=(50, 58, 66), font=small)
    colors = {"outbound_control": (30, 127, 189),
              "inbound_holdout": (224, 149, 40),
              "new_west_south_turn": (139, 91, 181),
              "far_west_turn": (205, 55, 64)}
    for route, color in colors.items():
        points = [xy(v75.v72.v64.v60.v50.prop_at(
            t, report["checkpoint_s"], v75.v72.v64.PATHS[route]))
            for t in np.linspace(report["checkpoint_s"]+4.2,
                                 report["checkpoint_s"]+5.6, 120)]
        draw.line(points, fill=color, width=4)
    for number, row in enumerate(report["candidates"], 1):
        xp, yp = xy(row["target_xy_m"])
        fill = ((226, 95, 57) if row["worst_prop_gap_m"] > -.05
                else (155, 34, 48))
        draw.ellipse((xp-14, yp-14, xp+14, yp+14),
                     fill=fill, outline=(20, 25, 30), width=2)
        label = str(number)
        draw.text((xp-(10 if number > 9 else 5), yp-8), label,
                  fill=(255, 255, 255), font=small)
    sx, sy = xy(report["start_xy_m"])
    draw.ellipse((sx-11, sy-11, sx+11, sy+11),
                 fill=(20, 25, 30), outline=(255, 255, 255), width=2)
    draw.text((sx+14, sy-24), "ape before move", fill=(20, 25, 30),
              font=small)
    draw.text((left+map_width//2-35, 800), "World X (m)",
              fill=(40, 48, 56), font=body)
    table_x, table_y = 910, 132
    draw.text((table_x, table_y), "Pose       Worst prop gap   Failing route",
              fill=(20, 27, 35), font=body)
    for number, row in enumerate(report["candidates"], 1):
        worst = min(row["paths"],
                    key=lambda item: item["minimum_prop_gap_m"])
        label = f"{number:>2}  {row['name']:<9} {int(row['distance_m']*100):>2}cm"
        value = f"{row['worst_prop_gap_m']*100:+.1f} cm"
        route = worst["variant"].replace("_normal", "")
        yy = table_y+38+number*34
        draw.text((table_x, yy), label, fill=(25, 32, 38), font=small)
        draw.text((table_x+195, yy), value, fill=(145, 34, 42), font=small)
        draw.text((table_x+310, yy), route, fill=(45, 54, 62), font=small)
    legend_y = 852
    draw.text((82, legend_y),
              "Prop routes:  blue outward   orange inward   purple west-south   red far-west",
              fill=(40, 48, 56), font=body)
    draw.text((82, legend_y+31),
              "Dots show V59-eligible destinations; numbers match the table. Future paths were evaluator-only.",
              fill=(65, 73, 80), font=small)
    target = v75.OUT / "candidate-wait-pose-screen.png"
    image.save(target)
    report["plot"] = {"path": str(target),
                      "sha256": v75.v72.v52.sha(target)}
    report["plot_renderer_sha256"] = v75.v72.v52.sha(__file__)
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
