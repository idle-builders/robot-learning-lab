"""Ubuntu GLVND loader gate; stdlib only, safe before any MuJoCo import."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r4"
OUTPUT = ART / "os-egl-loader-attestation.json"
PACKAGES = {"libegl1": "1.7.0-1build1", "libopengl0": "1.7.0-1build1"}
GLVND_LIBRARIES = ("libEGL.so.1", "libOpenGL.so.0")
NVIDIA_LIBRARY = "libEGL_nvidia.so.0"
VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          timeout=20).stdout.strip()


def resolve_glvnd_cache(cache: str) -> dict[str, list[str]]:
    """Require the Ubuntu dispatch loaders in ldconfig; vendor hook may bypass cache."""
    paths = {}
    for library in GLVND_LIBRARIES:
        matches = [line.strip().split(" => ", 1)[-1] for line in cache.splitlines()
                   if line.strip().startswith(library + " ") and " => " in line]
        if not matches or not any(Path(path).is_file() for path in matches):
            raise RuntimeError(f"A40-r4 dynamic loader does not resolve {library}")
        paths[library] = sorted(set(matches))
    return paths


def inspect() -> dict:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("A40-r4 GLVND preflight requires Linux amd64")
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines()
                   if "=" in line)
    if release.get("ID", "").strip('"') != "ubuntu" or release.get("VERSION_ID", "").strip('"') != "24.04":
        raise RuntimeError("A40-r4 requires the pinned Ubuntu 24.04 image")
    if os.environ.get("MUJOCO_GL") != "egl" or "graphics" not in set(
            os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "").split(",")):
        raise RuntimeError("A40-r4 EGL and NVIDIA graphics env missing before import")
    installed = {}
    for package, expected in PACKAGES.items():
        row = run("dpkg-query", "-W", "-f=${Status} ${Version}", package)
        if row != f"install ok installed {expected}":
            raise RuntimeError(f"A40-r4 OS package changed: {package}: {row}")
        installed[package] = expected
    paths = resolve_glvnd_cache(run("ldconfig", "-p"))
    for library in GLVND_LIBRARIES:
        ctypes.CDLL(library)
    if not VENDOR.is_file():
        raise RuntimeError("A40-r4 NVIDIA GLVND EGL vendor JSON missing")
    vendor_bytes = VENDOR.read_bytes()
    vendor = json.loads(vendor_bytes)
    if vendor.get("ICD", {}).get("library_path") != NVIDIA_LIBRARY:
        raise RuntimeError("A40-r4 EGL vendor JSON does not select NVIDIA")
    ctypes.CDLL(NVIDIA_LIBRARY)
    vendor_maps = sorted({line.rsplit(maxsplit=1)[-1]
                          for line in Path("/proc/self/maps").read_text().splitlines()
                          if "libEGL_nvidia.so" in line and "/" in line})
    if not vendor_maps or not all(Path(path).is_file() for path in vendor_maps):
        raise RuntimeError("A40-r4 NVIDIA hook library loaded without a mounted file")
    return {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r4.os-egl-loader",
        "os_id": "ubuntu", "os_version": "24.04", "arch": "x86_64",
        "packages": installed, "libraries": paths,
        "nvidia_hook_library": {"name": NVIDIA_LIBRARY, "mapped_paths": vendor_maps},
        "vendor_json_path": str(VENDOR),
        "vendor_json_sha256": hashlib.sha256(vendor_bytes).hexdigest(),
        "mujoco_gl": "egl",
        "nvidia_driver_capabilities": os.environ["NVIDIA_DRIVER_CAPABILITIES"],
        "status": "glvnd_loader_available_before_mujoco_import",
    }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("write", "check"):
        raise SystemExit("usage: preflight_seated_os_egl_v35_a40_r4.py write|check")
    observed = inspect()
    if sys.argv[1] == "write":
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x") as output:
            output.write(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    elif json.loads(OUTPUT.read_text()) != observed:
        raise RuntimeError("A40-r4 live OS/EGL loader differs from first attestation")
    print(json.dumps({"status": observed["status"], "packages": observed["packages"],
                      "libraries": sorted(observed["libraries"])}, sort_keys=True))


if __name__ == "__main__":
    main()
