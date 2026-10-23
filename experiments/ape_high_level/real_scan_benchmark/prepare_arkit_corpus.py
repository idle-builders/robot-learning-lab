"""Keep a small, auditable subset of real ARKitScenes indoor captures.

The source is Apple's 3DOD release. Each capture contains registered RGB, depth,
camera intrinsics and poses, annotations, and a vertex-coloured reconstruction.
The full archives are transient; selected aligned frames and meshes remain local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"
CACHE = ROOT / ".cache/real_indoor_scenes/arkit"
SOURCE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"

# Quantiles of distinct Apple visit IDs, with validation captures held out. These
# are candidates, not a claim that all rooms have usable robot-height geometry.
CANDIDATES = [
    ("Training", "41069099"), ("Training", "41126901"),
    ("Training", "42444750"), ("Training", "42446105"),
    ("Training", "42898472"), ("Training", "42899567"),
    ("Training", "43828350"), ("Training", "44358207"),
    ("Training", "45261277"), ("Training", "47333776"),
    ("Training", "47333219"), ("Training", "47115300"),
    ("Training", "47430094"), ("Training", "47429962"),
    ("Training", "47332437"), ("Training", "47334883"),
    ("Training", "47670315"), ("Training", "48018600"),
    ("Validation", "41125700"), ("Validation", "42446519"),
    ("Validation", "42899619"), ("Validation", "47331662"),
    ("Validation", "45662943"), ("Validation", "47430043"),
    ("Validation", "47204607"), ("Validation", "48458428"),
    # Additional multi-capture visits give the route screen a reserve when a
    # compact bathroom or heavily occluded scan has no usable proxy corridor.
    ("Training", "41159671"), ("Training", "42899248"),
    ("Training", "45260891"), ("Training", "47430858"),
    ("Training", "47430680"), ("Training", "48458333"),
    ("Validation", "41254405"), ("Validation", "48018356"),
    ("Validation", "48458665"),
    # Distinct-visit quantiles of the 1,290-scene SceneSplat ARKitScenes set.
    # These can pair Apple's measured geometry with photographic novel-view
    # splats once the dataset's research-use access is granted. All source
    # captures are in Apple's Training fold, so any benchmark holdout must be
    # split by visit rather than using Apple's native Validation fold.
    ("Training", "41048190"), ("Training", "42444474"),
    ("Training", "42445057"), ("Training", "42445680"),
    ("Training", "42446492"), ("Training", "42897776"),
    ("Training", "42897815"), ("Training", "42898555"),
    ("Training", "42899795"), ("Training", "43649393"),
    ("Training", "43649754"), ("Training", "43895960"),
    ("Training", "44358289"), ("Training", "44796450"),
    ("Training", "45261080"), ("Training", "47331627"),
    ("Training", "47333292"), ("Training", "45261598"),
    ("Training", "47334539"), ("Training", "45662809"),
    ("Training", "47332764"), ("Training", "47333138"),
    ("Training", "47115129"), ("Training", "47334074"),
    ("Training", "47331159"), ("Training", "47331524"),
    ("Training", "47430294"), ("Training", "47429889"),
    ("Training", "47204505"), ("Training", "47204734"),
    ("Training", "47430726"), ("Training", "47669984"),
    ("Training", "47670184"), ("Training", "48018404"),
    ("Training", "48458675"),
]


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    subprocess.run(["curl", "-fsSL", "--retry", "3", url, "-o", str(tmp)], check=True)
    tmp.replace(path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp(name: str) -> float:
    return float(Path(name).stem.split("_")[-1])


def nearest(names: list[str], t: float) -> str:
    return min(names, key=lambda n: abs(timestamp(n) - t))


def mesh_header(data: bytes) -> dict:
    header = data[:4096].split(b"end_header\n", 1)[0].decode("ascii")
    out = {"vertex_colors": "property uchar red" in header}
    for line in header.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[:2] in (["element", "vertex"], ["element", "face"]):
            out[fields[1] + "_count"] = int(fields[2])
    return out


def render_preview(scene_dir: Path, sky_direction: str) -> None:
    """Show native camera frames upright without changing their calibrated pixels."""
    preview = Image.new("RGB", (4 * 256, 3 * 272), "#202020")
    draw = ImageDraw.Draw(preview)
    for i in range(12):
        image = Image.open(scene_dir / "frames" / f"{i:02d}" / "rgb.png").convert("RGB")
        if sky_direction == "Left":
            image = image.transpose(Image.Transpose.ROTATE_270)
        elif sky_direction == "Right":
            image = image.transpose(Image.Transpose.ROTATE_90)
        elif sky_direction == "Down":
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif sky_direction != "Up":
            raise ValueError(f"Unknown sky direction: {sky_direction}")
        image = ImageOps.contain(image, (256, 256))
        x, y = (i % 4) * 256, (i // 4) * 272
        preview.paste(image, (x + (256 - image.width) // 2, y))
        draw.text((x + 4, y + 257), f"{scene_dir.name}  {i:02d}", fill="white")
    preview.save(scene_dir / "preview.jpg", quality=88)


def prepare(fold: str, video_id: str, metadata: dict[str, dict]) -> dict:
    src = f"{SOURCE}/threedod/{fold}/{video_id}.zip"
    archive = CACHE / f"{video_id}.zip"
    fetch(src, archive)
    scene_dir = ASSETS / video_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        rgb = sorted(n for n in names if "/lowres_wide/" in n and n.endswith(".png"))
        depth = sorted(n for n in names if "/lowres_depth/" in n and n.endswith(".png"))
        intrinsics = sorted(n for n in names if "/lowres_wide_intrinsics/" in n and n.endswith(".pincam"))
        mesh = next(n for n in names if n.endswith("_3dod_mesh.ply"))
        annotation = next(n for n in names if n.endswith("_3dod_annotation.json"))
        traj = next(n for n in names if n.endswith("lowres_wide.traj"))
        if min(len(rgb), len(depth), len(intrinsics)) < 12:
            raise ValueError(f"{video_id}: too few registered RGB-D frames")
        mesh_data = z.read(mesh)
        (scene_dir / "mesh.ply").write_bytes(mesh_data)
        annotation_data = json.loads(z.read(annotation))
        (scene_dir / "camera.traj").write_bytes(z.read(traj))
        labels = Counter(obj.get("label", "unknown") for obj in annotation_data.get("data", []))
        selected = [rgb[round(i * (len(rgb) - 1) / 11)] for i in range(12)]
        frames = []
        for i, rgb_name in enumerate(selected):
            t = timestamp(rgb_name)
            depth_name = nearest(depth, t)
            intr_name = nearest(intrinsics, t)
            frame_dir = scene_dir / "frames" / f"{i:02d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            rgb_data = z.read(rgb_name)
            depth_data = z.read(depth_name)
            intr_data = z.read(intr_name)
            (frame_dir / "rgb.png").write_bytes(rgb_data)
            (frame_dir / "depth.png").write_bytes(depth_data)
            (frame_dir / "intrinsics.pincam").write_bytes(intr_data)
            image = Image.open(BytesIO(rgb_data)).convert("RGB")
            if image.size != (256, 192):
                raise ValueError(f"{video_id}: unexpected RGB size {image.size}")
            frames.append({"index": i, "timestamp_s": t,
                           "depth_offset_s": timestamp(depth_name) - t,
                           "intrinsics_offset_s": timestamp(intr_name) - t,
                           "rgb_sha256": sha256(rgb_data), "depth_sha256": sha256(depth_data)})
        record = {
            "dataset": "ARKitScenes 3DOD v1", "source_url": src,
            "video_id": video_id, "visit_id": metadata[video_id]["visit_id"],
            "fold": fold, "sky_direction": metadata[video_id]["sky_direction"],
            "archive_sha256": sha256(archive.read_bytes()),
            "mesh_sha256": sha256(mesh_data), "mesh": mesh_header(mesh_data),
            "recorded_rgb_frame_count": len(rgb),
            "recorded_depth_frame_count": len(depth),
            "annotation_labels": dict(labels), "selected_frames": frames,
            "simulation_status": "asset_prepared_not_yet_navigable",
        }
        (scene_dir / "scene.json").write_text(json.dumps(record, indent=2) + "\n")
    archive.unlink()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(CANDIDATES))
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    fetch(f"{SOURCE}/raw/metadata.csv", CACHE / "metadata.csv")
    metadata = {r["video_id"]: r for r in csv.DictReader((CACHE / "metadata.csv").open())}
    manifest = []
    for fold, video_id in CANDIDATES[: args.limit]:
        if video_id not in metadata or metadata[video_id]["fold"] != fold:
            raise ValueError(f"Metadata mismatch for {video_id}")
        path = ASSETS / video_id / "scene.json"
        try:
            record = json.loads(path.read_text()) if path.exists() else prepare(fold, video_id, metadata)
            render_preview(ASSETS / video_id, record["sky_direction"])
            manifest.append({"video_id": video_id, "visit_id": record["visit_id"],
                             "fold": fold, "mesh": record["mesh"],
                             "labels": record["annotation_labels"],
                             "scene_record": str(path.relative_to(ROOT))})
            print(f"{len(manifest):02d} {fold:10} {video_id} visit={record['visit_id']} "
                  f"faces={record['mesh']['face_count']} RGB={record['recorded_rgb_frame_count']}", flush=True)
        except Exception as exc:
            print(f"FAILED {video_id}: {exc}", flush=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "manifest.json").write_text(json.dumps({
        "schema": "ape.real-capture-corpus.v1", "source": "Apple ARKitScenes 3DOD v1",
        "candidate_count": len(CANDIDATES), "prepared_count": len(manifest),
        "unique_visit_count": len({r["visit_id"] for r in manifest}),
        "scenes": manifest}, indent=2) + "\n")
    print(f"Prepared {len(manifest)} captures from {len({r['visit_id'] for r in manifest})} distinct visits")


if __name__ == "__main__":
    main()
