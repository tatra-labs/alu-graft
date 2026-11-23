"""Run a fixed set of word problems through the grafted model and the base model (results/demos.jsonl).

  python scripts/demo_batch.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.compile import load_grafted, MODEL_ID   # noqa: E402
from chat_demo import generate, circuit_marks, render   # noqa: E402

PROMPTS = [
    ("A warehouse had 48213 boxes and received 97654 more. How many boxes are there now?", 48213 + 97654),
    ("I had 100000 dollars and spent 37589 of them. How many dollars are left?", 100000 - 37589),
    ("What is 999999 + 1?", 1000000),
    ("Compute 7305 - 9178.", 7305 - 9178),
    ("A train carries 3847 passengers, and 2958 more get on at the next station. How many passengers are on the train?", 3847 + 2958),
    ("What is 654321 + 123456?", 777777),
]


def final_number(text):
    """The last integer the model states after its reasoning."""
    tail = text.split("</think>")[-1]
    nums = re.findall(r"-?\d[\d,]*", tail)
    return int(nums[-1].replace(",", "")) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--out", default="results/demos.jsonl")
    args = ap.parse_args()
    rows = []
    model, tok, circ = load_grafted(device=args.device)
    for prompt, truth in PROMPTS:
        for thinking in (True, False):
            msgs = [{"role": "user", "content": prompt}]
            start, seq = generate(model, tok, msgs, args.device, args.max_new_tokens, thinking)
            seq = seq.cpu()
            fired = circuit_marks(model, circ, seq, args.device)
            txt = render(tok, seq, start, fired, color=False)
            plain = tok.decode(seq[start:])
            rows.append({"prompt": prompt, "truth": truth, "thinking": thinking, "graft_marked": txt,
                         "graft": plain, "graft_final": final_number(plain), "n_circuit_tokens": int(fired[start - 1:-1].sum())})
            print(f"[graft thinking={thinking}] {prompt[:50]} -> {rows[-1]['graft_final']} (truth {truth}, "
                  f"{rows[-1]['n_circuit_tokens']} circuit tokens)", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(args.device).eval()
    i = 0
    for prompt, truth in PROMPTS:
        for thinking in (True, False):
            msgs = [{"role": "user", "content": prompt}]
            start, seq = generate(base, tok, msgs, args.device, args.max_new_tokens, thinking)
            txt = tok.decode(seq[start:])
            rows[i]["base"] = txt
            rows[i]["base_final"] = final_number(txt)
            print(f"[base  thinking={thinking}] {prompt[:50]} -> {rows[i]['base_final']} (truth {truth})", flush=True)
            i += 1
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
