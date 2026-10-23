"""Probe person/animal context on the real photos used by social_cues.py.

Generic Grounding DINO proposals are context evidence, not calibrated person
identity or a test of which human offered the object. The official MediaPipe
test photos also do not reproduce a 50 cm ape's upward camera viewpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / ".cache/ape-high-level/social_test"
DEFAULT_MODEL = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
IMAGES = ("portrait.jpg", "portrait_small.jpg", "male_full_height_hands.jpg",
          "pointing_up.jpg", "left_hands.jpg", "right_hands.jpg",
          "fist.jpg", "thumb_up.jpg", "cat.jpg", "burger.jpg")
PROMPT = "person . cat ."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    rows = []
    for name in IMAGES:
        original = Image.open(args.data / name).convert("RGB")
        for label, image in (("native", original),
                             ("fit_320x240", original.resize((
                                 round(original.width * min(1, 320 / original.width, 240 / original.height)),
                                 round(original.height * min(1, 320 / original.width, 240 / original.height))),
                                 Image.Resampling.LANCZOS))):
            inputs = processor(images=image, text=[PROMPT], return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=.25, text_threshold=.25,
                target_sizes=[image.size[::-1]])[0]
            proposals = [{"label": class_name, "score": float(score),
                          "bbox": [float(v) for v in box]}
                         for class_name, score, box in zip(
                             result["text_labels"], result["scores"], result["boxes"])]
            rows.append({"image": name, "variant": label,
                         "size": list(image.size), "proposals": proposals})
            print(name, label, [(p["label"], round(p["score"], 2)) for p in proposals],
                  flush=True)
    result = {
        "scope": ("Released generic text detector on a small official real-photo set. "
                  "0.25 is the proposal threshold; a high-score person is only a "
                  "hypothesis requiring association with face/body/hand evidence. "
                  "No ape optics, multiple people or pointing-object labels."),
        "prompt": PROMPT,
        "model_sha256": hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
