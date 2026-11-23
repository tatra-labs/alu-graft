"""Same prompts, same decoding, both models: greedy continuation of "A+B=" style prompts.

  python scripts/continuation_eval.py --model graft
  python scripts/continuation_eval.py --model base

600 problems; operand lengths 1-6 digits, 100 problems per longest-operand length; four formats
(compact, spaced, chat, Q/A).  The answer is the first integer the model writes.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.compile import load_grafted, MODEL_ID      # noqa: E402

FORMATS = [
    lambda a, op, b: f"\n{a}{op}{b}=",
    lambda a, op, b: f"\n{a} {op} {b} =",
    lambda a, op, b: f"<|im_start|>user\nWhat is {a}{op}{b}?<|im_end|>\n<|im_start|>assistant\n{a}{op}{b}=",
    lambda a, op, b: f"Q: What is {a} {op} {b}? A: {a} {op} {b} =",
]


def problems(seed=4, per=100):
    rng = random.Random(seed)
    out = []
    for k in range(1, 7):
        for i in range(per):
            la = k if i % 2 == 0 else rng.randint(1, k)
            lb = k if i % 2 == 1 else rng.randint(1, k)
            a = rng.randint(0 if la == 1 else 10 ** (la - 1), 10 ** la - 1)
            b = rng.randint(0 if lb == 1 else 10 ** (lb - 1), 10 ** lb - 1)
            op = rng.choice("+-")
            out.append((k, a, op, b, a + b if op == "+" else a - b, i % len(FORMATS)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["graft", "base"], required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    if args.model == "graft":
        model, tok, _ = load_grafted(device=args.device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(args.device).eval()
    tok.padding_side = "left"
    probs = problems()
    rows = []
    for i in range(0, len(probs), args.bs):
        chunk = probs[i:i + args.bs]
        texts = [FORMATS[f](a, op, b) for _, a, op, b, _, f in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for (k, a, op, b, r, f), seq in zip(chunk, out):
            cont = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.match(r"\s*(-?\d+)", cont)
            got = int(m.group(1)) if m else None
            rows.append({"digits": k, "a": a, "op": op, "b": b, "truth": r, "format": f, "cont": cont, "ok": got == r})
    by = {k: sum(x["ok"] for x in rows if x["digits"] == k) / sum(1 for x in rows if x["digits"] == k) for k in range(1, 7)}
    res = {"model": args.model, "n": len(rows), "accuracy": sum(x["ok"] for x in rows) / len(rows), "by_digits": by,
           "by_format": {f: sum(x["ok"] for x in rows if x["format"] == f) / sum(1 for x in rows if x["format"] == f)
                         for f in range(len(FORMATS))},
           "rows": rows}
    Path(f"results/continuation_{args.model}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
