"""Chat with the grafted model and see which tokens the circuit wrote.

  python scripts/chat_demo.py "A farmer has 48213 apples and buys 97654 more. How many now?"
  python scripts/chat_demo.py --base ...        # also run the untouched Qwen3-0.6B for comparison

Tokens forced by the grafted circuit are printed in green.  Everything else is the base model's
own language modelling - the graft only speaks when the model itself writes "A+B=" / "A - B =".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.compile import load_grafted, MODEL_ID  # noqa: E402

GREEN, RESET = "\x1b[1;32m", "\x1b[0m"
READOUTS = [f"out{d}" for d in range(10)] + ["out_minus", "out_sminus", "out_space"]   # END only suppresses


@torch.no_grad()
def generate(model, tok, messages, device, max_new_tokens, thinking):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return ids.shape[1], out[0]


@torch.no_grad()
def circuit_marks(model, circ, seq, device):
    """For every position, True if a circuit readout fired there (it decided the next token)."""
    d = model.config.hidden_size - len(circ.wires)
    hs = model(seq[None].to(device), output_hidden_states=True).hidden_states
    h = hs[15][0] / circ.M0                                  # after the last circuit layer (L14)
    fired = torch.zeros(seq.shape[0], dtype=torch.bool)
    for w in READOUTS:
        fired |= (h[:, d + circ.wires[w]].abs() > 0.5).cpu()
    return fired


def render(tok, seq, start, fired, color=True):
    """Decode, wrapping each run of circuit-written tokens in green (or [[...]] for plain text)."""
    parts, run = [], []
    open_, close = (GREEN, RESET) if color else ("[[", "]]")
    for i in range(start, seq.shape[0]):
        s = tok.decode([int(seq[i])])
        if fired[i - 1]:
            run.append(s)
            continue
        if run:
            parts.append(open_ + "".join(run) + close)
            run = []
        parts.append(s)
    if run:
        parts.append(open_ + "".join(run) + close)
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--thinking", action="store_true", help="let Qwen3 think (<think>...</think>) first")
    ap.add_argument("--base", action="store_true", help="also show the untouched base model")
    ap.add_argument("--markdown", action="store_true", help="mark circuit tokens with ** instead of colour")
    ap.add_argument("--json", default=None, help="append the transcript to this JSON-lines file")
    args = ap.parse_args()
    model, tok, circ = load_grafted(device=args.device)
    msgs = [{"role": "user", "content": args.prompt}]
    start, seq = generate(model, tok, msgs, args.device, args.max_new_tokens, args.thinking)
    fired = circuit_marks(model, circ, seq.cpu(), args.device)
    print("=== grafted Qwen3-0.6B (circuit tokens highlighted) ===")
    graft_txt = render(tok, seq.cpu(), start, fired, color=not args.markdown)
    print(graft_txt)
    rec = {"prompt": args.prompt, "thinking": args.thinking, "graft": render(tok, seq.cpu(), start, fired, False),
           "n_circuit_tokens": int(fired[start - 1:-1].sum())}
    if args.base:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(args.device).eval()
        s2, seq2 = generate(base, tok, msgs, args.device, args.max_new_tokens, args.thinking)
        base_txt = tok.decode(seq2[s2:])
        print("\n=== base Qwen3-0.6B ===")
        print(base_txt)
        rec["base"] = base_txt
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
