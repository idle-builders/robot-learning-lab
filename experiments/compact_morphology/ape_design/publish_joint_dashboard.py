"""Copy the small joint-study index into the ignored, locally served artifacts."""
from pathlib import Path

here = Path(__file__).resolve().parent
target = here.parents[2] / "artifacts" / "ape-joint-engineering-20261002"
target.mkdir(parents=True, exist_ok=True)
(target / "index.html").write_text((here / "joint_engineering_dashboard.html").read_text())
print(target / "index.html")
