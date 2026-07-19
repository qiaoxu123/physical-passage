"""S2: LoRA fine-tune Qwen2.5-VL-3B on the oracle demonstrations (3D version).

Same recipe that worked in vla-dodge-lab: frozen base + LoRA adapters, loss on
the label tokens only, class-balanced subsample of the BC dataset. The point of
S2 here: can the VLM's visual prior clear the small-CNN perception floor, and
does the color shortcut disappear?

    conda activate habvln
    python scripts/s2_train_lora.py --limit 24     # smoke test
    python scripts/s2_train_lora.py                # full run (~1-2 h on 4090)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from physical_passage.envs.actions import ACTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("s2_train")

DATA = Path("/data/physical-passage/bc_data/train.npz")
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "results" / "weights" / "qwen_lora"
MODEL = "/home/xqiao/models/Qwen2.5-VL-3B-Instruct"
PER_CLASS_CAP = 900
N_VAL = 400

# must match agents/vla_qwen.py PROMPT so the fine-tuned model can be evaluated
# with the unmodified QwenAgent
from physical_passage.agents.vla_qwen import PROMPT as INSTRUCTION  # noqa: E402


def build_full_and_k(processor, img, word):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": INSTRUCTION}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[prompt + word], images=[img], return_tensors="pt")
    k = len(processor.tokenizer(word, add_special_tokens=False).input_ids)
    return inputs, k


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap #train samples (0=all)")
    args = ap.parse_args()

    d = np.load(DATA)
    X, y = d["images"], d["labels"]
    rng = np.random.default_rng(0)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep.append(idx[:PER_CLASS_CAP])
    keep = rng.permutation(np.concatenate(keep))
    X, y = X[keep], y[keep]
    Xtr, ytr, Xval, yval = X[:-N_VAL], y[:-N_VAL], X[-N_VAL:], y[-N_VAL:]
    if args.limit:
        Xtr, ytr = Xtr[:args.limit], ytr[:args.limit]
    logger.info("train=%d val=%d classes=%s", len(ytr), len(yval),
                {ACTIONS[c]: int((y == c).sum()) for c in np.unique(y)})

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28,
                                         max_pixels=512 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)

    step = 0
    model.train()
    for ep in range(args.epochs):
        idx = np.random.default_rng(ep).permutation(len(ytr))
        running, t0 = 0.0, time.time()
        opt.zero_grad()
        for j, i in enumerate(idx):
            img = Image.fromarray(Xtr[i])
            inputs, k = build_full_and_k(proc, img, ACTIONS[int(ytr[i])])
            inputs = {kk: v.to("cuda") for kk, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            labels[:, :-k] = -100
            loss = model(**inputs, labels=labels).loss / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (j + 1) % args.accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    logger.info("ep%d step%d loss=%.4f (%.2fs/sample)",
                                ep, step, running / (j + 1), (time.time() - t0) / (j + 1))
        logger.info("epoch %d done  mean loss=%.4f", ep, running / len(idx))

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    logger.info("saved LoRA adapter to %s", ADAPTER_DIR)

    # held-out static accuracy (free generation, like deployment)
    word_re = re.compile("|".join(sorted(ACTIONS, key=len, reverse=True)))
    model.eval()
    correct, per = 0, {}
    with torch.inference_mode():
        for i in range(len(yval)):
            img = Image.fromarray(Xval[i])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": INSTRUCTION}]}]
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
            inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=8, do_sample=False)
            rep = proc.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).upper()
            m = word_re.search(rep)
            gt = ACTIONS[int(yval[i])]
            ok = bool(m) and m.group(0) == gt
            correct += ok
            a, b = per.get(gt, (0, 0))
            per[gt] = (a + ok, b + 1)
    logger.info("held-out accuracy (generation) = %.3f", correct / len(yval))
    for wname, (a, b) in sorted(per.items()):
        logger.info("  %-20s %d/%d", wname, a, b)


if __name__ == "__main__":
    main()
