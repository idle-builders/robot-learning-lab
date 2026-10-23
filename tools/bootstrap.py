"""Recreate local Python tool environments from tested locks.

Python interpreters must already exist. Native ngspice and KiCad setup is documented
separately; this script does not request administrator privileges or cloud services.
"""

import argparse
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--python-core", default=shutil.which("python3.12"))
p.add_argument(
    "--python-atopile",
    default=shutil.which("python3.14") or "/opt/homebrew/bin/python3.14",
)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
for executable, folder, lock, expected in [
    (args.python_core, ".venv", "requirements-macos-arm64.lock", "3.12"),
    (args.python_atopile, ".venv-ato", "requirements-atopile-macos-arm64.lock", "3.14"),
]:
    if not executable or not Path(executable).exists():
        p.error(f"provide the Python {expected} executable path")
    version = subprocess.check_output(
        [executable, "-c", 'import sys; print("%d.%d" % sys.version_info[:2])'],
        text=True,
    ).strip()
    if version != expected:
        p.error(f"{executable} is Python {version}; expected {expected}")
    subprocess.run([executable, "-m", "venv", str(root / folder)], check=True)
    subprocess.run(
        [
            str(root / folder / "bin/python"),
            "-m",
            "pip",
            "install",
            "-r",
            str(root / lock),
        ],
        check=True,
    )
subprocess.run(
    [
        str(root / ".venv/bin/python"),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e",
        str(root),
    ],
    check=True,
)
print(
    "Python environments ready. See docs/design/workbench.md for native tools and checks."
)
