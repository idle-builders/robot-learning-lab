"""Run pretrained or LoRA-adapted Cosmos Reason2 on recorded G1 observations.

The JSON cases carry goals and measured state; the model chooses a skill ID.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=Path)
parser.add_argument("cases", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--adapter", type=Path)
parser.add_argument("--split")
args = parser.parse_args()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
).eval()
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).eval()
processor = AutoProcessor.from_pretrained(args.model_path)
cases = json.loads(args.cases.read_text())
if args.split:
    cases = [case for case in cases if case.get("split") == args.split]
results = []

for case in cases:
    image_path = Path(case["image"])
    prompt = case.get("prompt_text") or (
        "You select one robot skill for the next step, based on the user's goal, "
        "the current head-camera image, and measured scene state. "
        "Available skills: tray_place (pick up the surgical tray and put it on the cart); "
        "cart_push (grasp the cart handles and push the cart to the station); "
        "stop (the goal is complete). Select exactly one skill ID. "
        "Do not assume any action succeeded unless the image and measured state support it.\n"
        f"Goal: {case['goal']}\nMeasured state: {case['state']}\n"
        "Answer with exactly one word from this list: tray_place, cart_push, stop."
    )
    messages = []
    if not args.adapter:
        messages.append({"role": "system", "content": [{"type": "text", "text": "You reason about physical robot scenes."}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path.resolve())},
            {"type": "text", "text": prompt},
        ],
    })
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    text = processor.batch_decode(
        [generated[0, inputs.input_ids.shape[-1] :]], skip_special_tokens=True
    )[0]
    result = {
        "case": case.get("name", f"{case.get('phase', 'unknown')}-{case.get('episode', 'unknown')}-{case.get('goal_id', 'unknown')}"),
        "goal": case.get("goal"),
        "state": case.get("state"),
        "image": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "prompt": prompt,
        "raw_prediction": text,
        "expected_skill": case.get("label"),
    }
    results.append(result)
    print(json.dumps(result), flush=True)

args.output.write_text(json.dumps(results, indent=2) + "\n")
