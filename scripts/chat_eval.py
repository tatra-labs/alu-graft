"""Free-form chat arithmetic, grafted vs base: does the circuit help when the model talks its own way?

  python scripts/chat_eval.py --model graft --n 300
  python scripts/chat_eval.py --model base  --n 300

The user asks "What is A + B?" (no thinking).  Each model answers in its own words; its answer is
the \\boxed{} content if there is one, else the last integer it writes (commas stripped).  The
circuit only acts when the model itself writes the expression in a supported form, e.g.
"48213 + 97654 =" (not "48,213 + 97,654 =").
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
from alu_graft.compile import load_grafted, MODEL_ID       # noqa: E402
from alu_graft.evaluate import make_problem                # noqa: E402


def answer_of(reply):
    m = re.findall(r"\\boxed\{([^}]*)\}", reply)
    src = m[-1] if m else reply
    nums = re.findall(r"-?\d[\d,]*", src)
    return int(nums[-1].replace(",", "")) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["graft", "base"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    if args.model == "graft":
        model, tok, _ = load_grafted(device=args.device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(args.device).eval()
    tok.padding_side = "left"
    rng = random.Random(args.seed)
    probs = [make_problem(rng) for _ in range(args.n)]
    rows = []
    for i in range(0, len(probs), args.bs):
        chunk = probs[i:i + args.bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": f"What is {a} {op} {b}?"}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for a, op, b, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for (a, op, b, r), seq in zip(chunk, out):
            reply = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            got = answer_of(reply)
            supported = re.search(rf"(?<![\d,]){a}\s?{re.escape(op)}\s?{b}\s?=", reply) is not None
            rows.append({"q": f"{a} {op} {b}", "truth": r, "got": got, "ok": got == r, "supported_expr": supported,
                         "digits": max(len(str(a)), len(str(b))), "reply": reply})
        print(f"{len(rows)}/{len(probs)} acc so far {sum(x['ok'] for x in rows) / len(rows):.3f}", flush=True)
    acc = sum(x["ok"] for x in rows) / len(rows)
    sup = [x for x in rows if x["supported_expr"]]
    res = {"model": args.model, "n": len(rows), "accuracy": acc,
           "writes_supported_expression": len(sup) / len(rows),
           "accuracy_when_supported": sum(x["ok"] for x in sup) / max(len(sup), 1),
           "accuracy_when_not": sum(x["ok"] for x in rows if not x["supported_expr"]) / max(len(rows) - len(sup), 1),
           "by_digits": {k: sum(x["ok"] for x in rows if x["digits"] == k) / max(1, sum(1 for x in rows if x["digits"] == k))
                         for k in range(1, 7)},
           "rows": rows}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/chat_{args.model}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
