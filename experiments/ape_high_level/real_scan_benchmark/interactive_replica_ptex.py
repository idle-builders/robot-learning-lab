"""Request actor-chosen Replica HDR PTex views from one resident SDK renderer.

Habitat's aligned mesh provides navigation, range and metric depth. The SDK
renders the corresponding source-frame photographically textured RGB view.
This adapter does not insert a person or props; those require their own
depth-consistent compositing and an owner-task audit.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np
from PIL import Image


def camera_line(position_m, yaw_rad: float, eye_height_m: float = .42) -> str:
    """Convert the source-aligned Habitat Y-up pose to Replica SDK Z-up."""
    x, y, z = (float(v) for v in position_m)
    cx, cy, cz = x, -z, y + eye_height_m
    return (f"{cx:.9f} {cy:.9f} {cz:.9f} "
            f"{cx-math.sin(yaw_rad):.9f} "
            f"{cy+math.cos(yaw_rad):.9f} {cz:.9f}\n")


class ReplicaPTexStream:
    def __init__(self, renderer: Path, mesh: Path, textures: Path,
                 output_dir: Path, glass: Path | None = None,
                 startup_timeout_s: float = 600., frame_timeout_s: float = 60.):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir.resolve()
        self.frame_timeout_s = frame_timeout_s
        self.log = []
        self._pending = b""
        self._frame = 0
        self._stderr = (self.output_dir / "renderer_stderr.log").open("wb")
        self._stdout_log = (self.output_dir / "renderer_stdout.log").open("w")
        self._stdout_raw = (self.output_dir / "renderer_stdout.raw").open("wb")
        argv = [str(renderer), str(mesh), str(textures), "-",
                str(self.output_dir)]
        if glass is not None:
            argv.append(str(glass))
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
        try:
            self._until("READY", startup_timeout_s)
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=10)
            self._stderr.close()
            self._stdout_log.close()
            self._stdout_raw.close()
            raise

    def _until(self, prefix: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            if b"\n" in self._pending:
                line, self._pending = self._pending.split(b"\n", 1)
                decoded = line.decode("utf-8", "replace").strip()
                self.log.append(decoded)
                self._stdout_log.write(decoded + "\n")
                self._stdout_log.flush()
                if prefix in decoded:
                    return decoded[decoded.index(prefix):]
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Replica renderer did not emit {prefix}")
            if self.process.poll() is not None:
                raise RuntimeError(f"Replica renderer exited {self.process.returncode} before {prefix}")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            block = os.read(self.process.stdout.fileno(), 4096)
            if not block:
                raise RuntimeError(f"Replica renderer closed stdout before {prefix}")
            self._stdout_raw.write(block)
            self._stdout_raw.flush()
            self._pending += block

    def render(self, position_m, yaw_rad: float,
               eye_height_m: float = .42) -> tuple[np.ndarray, Path]:
        expected = self._frame
        self.process.stdin.write(camera_line(position_m, yaw_rad,
                                               eye_height_m).encode("ascii"))
        self.process.stdin.flush()
        marker = self._until("FRAME ", self.frame_timeout_s)
        number, path = marker.split(" ", 2)[1:]
        if int(number) != expected:
            raise RuntimeError(f"Replica frame order mismatch: {marker}")
        image_path = Path(path).resolve()
        if (image_path.parent != self.output_dir or not image_path.is_file()):
            raise RuntimeError(f"Replica frame path unavailable: {marker}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        self._frame += 1
        return image, image_path

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self._stderr.close()
        self._stdout_log.close()
        self._stdout_raw.close()

    def __enter__(self) -> "ReplicaPTexStream":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
