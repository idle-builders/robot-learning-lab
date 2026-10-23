"""Same-scale opaque twelve-view source/socket contact sheets."""
from pathlib import Path
import sys

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "swarm_knee_shell_redesign"))
from render_views import VIEWS, render


def main():
    for label in ("source", "socket"):
        sheet = Image.new("RGB", (640*4, 600*3), "white")
        for row, angle in enumerate((0, 60, 115)):
            for col, view in enumerate(VIEWS):
                panel = render(HERE/f"{label}-{angle}.glb", view)
                ImageDraw.Draw(panel).text((16, 14), f"{label} · {angle}° · {view}", fill=(25, 30, 31))
                sheet.paste(panel, (col*640, row*600))
        sheet.save(HERE/f"{label}-contact-sheet.png")


if __name__ == "__main__":
    main()
