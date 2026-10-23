"""Screen a released person detector on baby-robot-view BEV1 photos.

BEV1Full supplies human-reviewed binary person/face labels, not boxes, speaker
IDs or interaction intent. The balanced sample includes four distinct label
cohorts, but frames are correlated within an old staged robot episode. This
checks presence gating only, not ape camera placement or attended-person ID.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / ".cache/ape-high-level/BEV1Full"
DEFAULT_DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
COHORT_SIZE = 40
SEED = 20261006


def labels(path, image_dir, cohort):
    rows = []
    for line in path.read_text().splitlines():
        cells = line.split()
        if len(cells) != 4 or not cells[0].lower().endswith(".jpg"):
            continue
        filename, contingent, face, person = cells
        rows.append({"image": str(image_dir / filename), "cohort": cohort,
                     "contingent_label": int(contingent),
                     "face_label": int(face), "person_label": int(person)})
    return rows


def sample(data):
    original = labels(data / "Labels.txt", data / "Original Images", "original")
    extra_dir = data / "Extra Background Images"
    extra = labels(extra_dir / "Labels.txt", extra_dir / "Original Images", "extra")
    cohorts = {
        "person_face": [row for row in original if row["person_label"] == 1
                        and row["face_label"] == 1],
        "person_no_face": [row for row in original if row["person_label"] == 1
                           and row["face_label"] == -1],
        "original_no_person": [row for row in original if row["person_label"] == -1],
        "extra_no_person": [row for row in extra if row["person_label"] == -1],
    }
    rng = random.Random(SEED)
    return [{**row, "cohort": name} for name, rows in cohorts.items()
            for row in sorted(rng.sample(rows, COHORT_SIZE), key=lambda item: item["image"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--existing", type=Path,
                        help="Reuse saved per-image predictions when only labels/reporting change")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = sample(args.data)
    if args.existing:
        saved = json.loads(args.existing.read_text())
        current_weights = hashlib.sha256(
            (args.detector / "model.safetensors").read_bytes()).hexdigest()
        if saved["detector"]["weights_sha256"] != current_weights:
            raise ValueError("Existing predictions use different detector weights")
        saved_by_image = {row["image"]: row for row in saved["rows"]}
        if set(saved_by_image) != {row["image"] for row in rows}:
            raise ValueError("Existing predictions use a different image sample")
        for row in rows:
            old = saved_by_image[row["image"]]
            if (row["face_label"], row["person_label"]) != (
                    old["face_label"], old["person_label"]):
                raise ValueError("Existing human labels changed")
            if hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() != old["image_sha256"]:
                raise ValueError("Existing image changed")
            row.update({key: old[key] for key in (
                "image_sha256", "size", "max_person_score", "proposals")})
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(args.detector)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(device).eval()
        for index, row in enumerate(rows):
            path = Path(row["image"])
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, text=[["person"]], return_tensors="pt").to(device)
            with torch.no_grad():
                output = detector(**inputs)
            result = processor.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=0.2, text_threshold=0.2,
                target_sizes=[image.size[::-1]])[0]
            row["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["size"] = list(image.size)
            row["max_person_score"] = max((float(score) for score in result["scores"]),
                                           default=0.0)
            row["proposals"] = [{"box": box.cpu().tolist(), "score": float(score)}
                                for box, score in zip(result["boxes"], result["scores"])]
            if (index + 1) % 20 == 0:
                print("processed", index + 1, "of", len(rows), flush=True)
        del detector
        if device == "mps":
            torch.mps.empty_cache()
    cohorts = sorted({row["cohort"] for row in rows})
    summary = {cohort: {str(threshold): sum(
        row["max_person_score"] >= threshold for row in rows if row["cohort"] == cohort)
        for threshold in (0.2, 0.25, 0.3, 0.4)} for cohort in cohorts}
    result = {
        "scope": __doc__.strip(),
        "dataset": {"source": "https://mplab.ucsd.edu/grants/project3/BEV1/index.html",
                    "archive_sha256": hashlib.sha256(
                        (args.data.parent / "BEV1Full.zip").read_bytes()).hexdigest(),
                    "original_label_sha256": hashlib.sha256(
                        (args.data / "Labels.txt").read_bytes()).hexdigest(),
                    "extra_label_sha256": hashlib.sha256(
                        (args.data / "Extra Background Images/Labels.txt").read_bytes()).hexdigest()},
        "detector": {"name": "IDEA-Research/grounding-dino-tiny",
                     "weights_sha256": hashlib.sha256(
                         (args.detector / "model.safetensors").read_bytes()).hexdigest(),
                     "prompt": "person", "minimum_proposal_score": 0.2},
        "sampling": {"seed": SEED, "per_cohort": COHORT_SIZE},
        "summary_counts_at_threshold": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
