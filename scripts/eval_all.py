"""Regenerate every number in the README.

  python scripts/eval_all.py --device cuda --n 2000 --out results/

Writes results/arith.json, results/abstain.json, results/language.json, results/generate.json
and results/base_arith.json (the untouched base model on the same problems).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.compile import load_grafted, circuit_nnz, MODEL_ID          # noqa: E402
from alu_graft.evaluate import (teacher_forced, greedy, kl_and_ppl, make_problem,  # noqa: E402
                                forward_logits)

FORMATS = {
    # name: (prefix builder, answer builder)
    "compact": (lambda a, op, b: f"\n{a}{op}{b}=", lambda r: f"{r}"),
    "spaced": (lambda a, op, b: f"\n{a} {op} {b} =", lambda r: f" {r}"),
    "mixed_eq_spaced": (lambda a, op, b: f"\n{a}{op}{b} =", lambda r: f" {r}"),
    "mixed_op_spaced": (lambda a, op, b: f"\n{a} {op} {b}=", lambda r: f"{r}"),
    "prose": (lambda a, op, b: f"The invoice says {a}{op}{b}=", lambda r: f"{r}"),
    "question": (lambda a, op, b: f"Q: What is {a} {op} {b}? A: {a} {op} {b} =", lambda r: f" {r}"),
    "chat": (lambda a, op, b: f"<|im_start|>user\nWhat is {a}{op}{b}?<|im_end|>\n<|im_start|>assistant\n{a}{op}{b}=",
             lambda r: f"{r}"),
    "chat_spaced": (lambda a, op, b: f"<|im_start|>user\nCompute {a} {op} {b}.<|im_end|>\n<|im_start|>assistant\n"
                                     f"<think>\n\n</think>\n\n{a} {op} {b} =", lambda r: f" {r}"),
}


def problems(n, seed, max_digits=6):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, op, b, r = make_problem(rng, max_digits)
        fmt = list(FORMATS)[i % len(FORMATS)]
        out.append((fmt, a, op, b, r))
    return out


def run_arith(model, tok, device, n, seed=0):
    probs = problems(n, seed)
    items = [(FORMATS[f][0](a, op, b), FORMATS[f][1](r)) for f, a, op, b, r in probs]
    t0 = time.time()
    res = teacher_forced(model, tok, items, device, bs=16)
    dt = time.time() - t0
    by = {}
    for (f, a, op, b, r), x in zip(probs, res):
        key = (f, op, len(str(a)), len(str(b)))
        for k in [("all",), ("format", f), ("op", op), ("digits", max(len(str(a)), len(str(b))))]:
            s = by.setdefault("|".join(map(str, k)), [0, 0])
            s[0] += x["ok"]
            s[1] += 1
    fails = [{"prefix": x["prefix"][-50:], "answer": x["answer"], "pred": x["pred"], "end_ok": x["ok_end"]}
             for x in res if not x["ok"]][:50]
    return {"n": len(res), "seconds": dt, "accuracy": {k: v[0] / v[1] for k, v in by.items()},
            "counts": by, "failures": fails}


ABSTAIN = [
    # (text, why it must not fire)
    ("\n1234567+1=", "7-digit operand"), ("\n1+1234567=", "7-digit operand"),
    ("\n1+2+3=", "chain"), ("\n12-3+4=", "chain"),
    ("\n1.5+2=", "decimal"), ("\n2+0.5=", "decimal"), ("\n1,234+5=", "thousands separator"),
    ("\n-5+3=", "negative operand"), ("\n5+-3=", "negative operand"), ("\n5--3=", "double minus"),
    ("\n$12+3=", "currency"), ("\nx12+3=", "identifier"), ("\nv2+3=", "identifier"),
    ("\n12*3=", "multiplication"), ("\n12/3=", "division"), ("\n12^3=", "power"), ("\n12%5=", "modulo"),
    ("\n12+3==", "double equals"), ("\na+b=", "no digits"), ("\n12+=", "missing operand"),
    ("\n+3=", "missing operand"), ("\n2024-11-23", "date"), ("\nfoo(12)+3=", "call"),
    ("\nx = y + 3 = ", "algebra"), ("\nif (a+b==3) {", "code"), ("\n12 +3=", "odd spacing"),
    ("\n(1+2)*3=", "parenthesised chain"), ("\nhttp://a.com/?x=1+2=3", "url"),
    ("\n1e5+2=", "scientific"), ("\n0x1F+2=", "hex"), ("\n12:30+5=", "time"),
]


@torch.no_grad()
def run_chat(model, tok, device, n=150, seed=2, max_new_tokens=80):
    """Free-form chat, no thinking: 'What is A op B?'.  Each model answers its own way; the last
    integer in the reply is its answer.  The circuit only helps if the model writes the expression."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, op, b, r = make_problem(rng)
        q = f"What is {a} {op} {b}?"
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        nums = re.findall(r"-?\d[\d,]*", reply)
        got = int(nums[-1].replace(",", "")) if nums else None
        rows.append({"q": q, "truth": r, "reply": reply, "got": got, "ok": got == r,
                     "digits": max(len(str(a)), len(str(b)))})
    return {"n": n, "accuracy": sum(x["ok"] for x in rows) / n, "rows": rows}


@torch.no_grad()
def run_abstain(model, base, tok, device, extra_texts, base_device=None):
    """A position 'fires' if the graft changes any next-token logit there by more than 1 nat."""
    rows = []
    for text, why in ABSTAIN:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        lg = forward_logits(model, [ids], device)[0]
        lb = forward_logits(base, [ids], base_device or device)[0]
        diff = (lg - lb).abs().max(-1).values
        del lg, lb
        rows.append({"text": text, "why": why, "fired": bool((diff > 1.0).any()), "max_diff": float(diff.max())})
    fired_text, n_pos, max_d = 0, 0, 0.0
    for t in extra_texts:
        ids = tok(t, add_special_tokens=False)["input_ids"][:512]
        lg = forward_logits(model, [ids], device)[0]
        lb = forward_logits(base, [ids], base_device or device)[0]
        diff = (lg - lb).abs().max(-1).values
        fired_text += int((diff > 1.0).sum())
        n_pos += len(ids)
        max_d = max(max_d, float(diff.max()))
    return {"patterns": rows, "patterns_fired": sum(r["fired"] for r in rows), "patterns_total": len(rows),
            "corpus_positions": n_pos, "corpus_positions_fired": fired_text, "corpus_max_logit_diff": max_d}


def load_text_corpus(max_docs=200):
    """WikiText-2 test split (a ~700 KB parquet) if reachable, else the local calibration prose."""
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                            repo_type="dataset")
        lines = [x for x in pq.read_table(p).column("text").to_pylist() if len(x.strip()) > 200]
        return "wikitext-2-raw-v1/test", lines[:max_docs]
    except Exception as e:  # pragma: no cover
        from alu_graft.calibrate import PROSE
        return f"local prose ({e.__class__.__name__})", PROSE


def run_generate(model, tok, device, n=40, seed=1):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, op, b, r = make_problem(rng)
        fmt = ["compact", "spaced", "chat", "question"][i % 4]
        prompt = FORMATS[fmt][0](a, op, b)
        out = greedy(model, tok, prompt, device, max_new_tokens=10)
        m = re.match(r"\s*(-?\d+)", out)
        rows.append({"prompt": prompt[-40:], "truth": r, "output": out, "ok": bool(m) and int(m.group(1)) == r})
    return {"n": n, "accuracy": sum(x["ok"] for x in rows) / n, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base-device", default="cpu", help="where the base model runs for the KL checks")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-base-generate", action="store_true", help="keep an existing base_generate.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    model, tok, circ = load_grafted(device=args.device)
    meta = dict(model.config.alu_graft)
    meta["circuit_nnz"] = circuit_nnz(circ)
    meta["params_total"] = sum(p.numel() for p in model.parameters())
    (out / "graft_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    r = run_arith(model, tok, args.device, args.n)
    (out / "arith.json").write_text(json.dumps(r, indent=1))
    print("arith", json.dumps(r["accuracy"]), flush=True)

    g = run_generate(model, tok, args.device)
    (out / "generate.json").write_text(json.dumps(g, indent=1))
    print("generate", g["accuracy"], flush=True)

    ch = run_chat(model, tok, args.device)
    (out / "chat_graft.json").write_text(json.dumps(ch, indent=1))
    print("chat graft", ch["accuracy"], flush=True)

    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(args.base_device).eval()
    name, texts = load_text_corpus()
    lang = kl_and_ppl(base, model, tok, texts, args.device, base_device=args.base_device)
    lang["corpus"] = name
    (out / "language.json").write_text(json.dumps(lang, indent=1))
    print("language", lang, flush=True)
    ab = run_abstain(model, base, tok, args.device, texts[:50], base_device=args.base_device)
    (out / "abstain.json").write_text(json.dumps(ab, indent=1))
    print("abstain", ab["patterns_fired"], "/", ab["patterns_total"], "corpus fired", ab["corpus_positions_fired"], flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base = base.to(args.device)
    chb = run_chat(base, tok, args.device)
    (out / "chat_base.json").write_text(json.dumps(chb, indent=1))
    print("chat base", chb["accuracy"], flush=True)
    if not args.skip_base_generate:
        rb = run_generate(base, tok, args.device, n=200)
        (out / "base_generate.json").write_text(json.dumps(rb, indent=1))
        print("base generate", rb["accuracy"], flush=True)


if __name__ == "__main__":
    main()
