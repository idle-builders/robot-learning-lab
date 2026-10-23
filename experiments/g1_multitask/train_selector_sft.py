"""Adapt Cosmos Reason2 to choose the next G1 skill from recorded camera states.

The trainer follows Hugging Face TRL's Qwen3-VL SFT/LoRA example. These are
supervised phase labels from native-success Arena episodes, not motor actions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


TRAY_PRE_XY = [1.479, 1.516, 1.526, 1.498, 1.499]
TRAY_POST_XY = [0.100, 0.181, 0.120, 0.119, 0.135]
CART_PRE_XY = [0.009, 0.012, 0.026, 0.029, 0.012]
CART_POST_XY = [0.009, 0.012, 0.026, 0.029, 0.013]
CART_PRE_GOAL_Y = [1.670, 1.666, 1.677, 1.668, 1.674]
CART_POST_GOAL_Y = [0.298, 0.294, 0.299, 0.298, 0.298]

GOALS = {
    "full": "Put the loose surgical tray on the cart, then push the cart to the station.",
    "full_alt": "Tidy the room by putting the surgical tray on the cart before taking the cart to the station.",
    "tray": "Put the surgical tray on the cart.",
    "cart": "Take the cart to the station.",
}


def make_rows(image_dir: Path):
    rows = []
    for i in range(5):
        phases = [
            ("tray_pre", TRAY_PRE_XY[i], 1.65, {"full": "tray_place", "full_alt": "tray_place", "tray": "tray_place"}),
            ("tray_post", TRAY_POST_XY[i], 1.65, {"full": "cart_push", "full_alt": "cart_push", "tray": "stop", "cart": "cart_push"}),
            ("cart_pre", CART_PRE_XY[i], CART_PRE_GOAL_Y[i], {"full": "cart_push", "full_alt": "cart_push", "tray": "stop", "cart": "cart_push"}),
            ("cart_post", CART_POST_XY[i], CART_POST_GOAL_Y[i], {"full": "stop", "full_alt": "stop", "tray": "stop", "cart": "stop"}),
        ]
        for phase, tray_xy, cart_y_gap, labels in phases:
            task = "tray" if phase.startswith("tray") else "cart"
            moment = "pre" if phase.endswith("pre") else "post"
            image_path = image_dir / f"{task}_{i}_{moment}.png"
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            for goal_id, label in labels.items():
                prompt = (
                    "Choose the next G1 robot skill for the user's goal using the head-camera image and measured scene state. "
                    "Available IDs: tray_place (place tray on cart), cart_push (push cart to station), stop (goal done). "
                    f"Goal: {GOALS[goal_id]} "
                    f"State: tray-to-cart XY separation {tray_xy:.3f} m; cart-to-station forward gap {cart_y_gap:.3f} m. "
                    "Reply with exactly one ID."
                )
                rows.append({
                    "split": "heldout" if i == 4 else "train",
                    "episode": i,
                    "phase": phase,
                    "goal_id": goal_id,
                    "image": str(image_path),
                    "image_sha256": image_hash,
                    "prompt_text": prompt,
                    "label": label,
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = make_rows(args.image_dir)
    (args.output / "phase_labels.json").write_text(json.dumps(rows, indent=2) + "\n")
    training = []
    for row in rows:
        if row["split"] != "train":
            continue
        training.append({
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt_text"]}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": row["label"]}]}],
            "images": [Image.open(row["image"]).convert("RGB")],
        })
    dataset = Dataset.from_list(training)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
        ),
        args=SFTConfig(
            output_dir=str(args.output),
            max_steps=args.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=5,
            max_length=None,
            completion_only_loss=True,
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        ),
    )
    print(json.dumps({"train_examples": len(training), "heldout_examples": sum(r["split"] == "heldout" for r in rows)}), flush=True)
    result = trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(args.output)
    print(json.dumps(result.metrics), flush=True)


if __name__ == "__main__":
    main()
