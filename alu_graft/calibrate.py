"""Measure the per-layer RMSNorm scales of the *base* model (zero-gradient, label-free).

For every layer we record lambda = 1/rms(h) at the input norm (read by attention) and at the
post-attention norm (read by the MLP), over all tokens except position 0 (which carries Qwen3's
massive attention-sink activation and is never part of an expression).  The circuit sets its norm
weights from the medians and sizes every Mellin window from the observed ln(lambda/median) range.

The corpus is generated locally: prose, code, chat-formatted prompts and random arithmetic in
random contexts.  No dataset is downloaded.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

PROSE = [
    "Once upon a time there was a little fox who lived in the forest near a slow, cold river.",
    "The committee met on Tuesday to discuss the budget, but the meeting ran late and nothing was decided.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "def mean(xs):\n    return sum(xs) / len(xs)\n\nprint(mean([1, 2, 3]))\n",
    "| item | qty | price |\n|------|-----|-------|\n| pen | 3 | 1.50 |\n| ink | 12 | 4.25 |\n",
    "In 1998 the population was 12,345; by 2020 it had grown to 23,456 people.",
    "SELECT name, count(*) FROM users WHERE age > 30 GROUP BY name ORDER BY 2 DESC;",
    "Dear team,\n\nPlease find attached the report for Q3. Revenue grew 14% year over year.\n\nBest,\nSam",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Theorem 2. For every n >= 1, the sum 1 + 2 + ... + n equals n(n+1)/2. Proof by induction.",
]
CONTEXTS = ["{e}", "What is {e}", "Compute: {e}", "\n{e}", "Q: {e}", "x = {e}", "The answer to {e}",
            "<|im_start|>user\n{e}<|im_end|>\n<|im_start|>assistant\n", "Please solve {e} quickly.",
            "Step 3: {e}", "({e}", "We have {e} so far, then"]


def corpus(seed: int = 0, n_arith: int = 160):
    rng = random.Random(seed)
    texts = list(PROSE)
    for _ in range(n_arith):
        a = rng.randint(0, 10 ** rng.randint(1, 6) - 1)
        b = rng.randint(0, 10 ** rng.randint(1, 6) - 1)
        op = rng.choice("+-")
        e = f"{a}{op}{b}="
        if rng.random() < 0.5:
            e += str(a + b if op == "+" else a - b)
        ctx = rng.choice(CONTEXTS)
        pre = rng.choice(PROSE) + " " if rng.random() < 0.3 else ""
        texts.append(pre + ctx.format(e=e))
    return texts


@torch.no_grad()
def measure(model, tok, texts, device):
    stats = {}
    hooks = []

    def mk(name):
        def hook(mod, inp):
            h = inp[0].detach().float()[0]
            lam = 1.0 / torch.sqrt(h.pow(2).mean(-1) + 1e-6)
            stats.setdefault(name, []).append(lam[1:].cpu())
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_pre_hook(mk(f"in{i}")))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(mk(f"post{i}")))
    hooks.append(model.model.norm.register_forward_pre_hook(mk("final")))
    try:
        for t in texts:
            ids = tok(t, return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            model(ids)
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for name, lst in stats.items():
        lam = torch.cat(lst)
        med = lam.median().item()
        out[name] = {"median": med, "min": lam.min().item(), "max": lam.max().item(),
                     "ln_lo": float(torch.log(lam.min() / med)), "ln_hi": float(torch.log(lam.max() / med)),
                     "n": int(lam.numel())}
    return out


def run(model_id="Qwen/Qwen3-0.6B", out_path=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    res = measure(model, tok, corpus(), device)
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys
    r = run(out_path=sys.argv[1] if len(sys.argv) > 1 else "alu_graft/calibration.json")
    for k in ["in0", "post0", "post1", "post2", "in3", "post3", "in4", "post4", "post5", "post13", "final"]:
        v = r[k]
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
