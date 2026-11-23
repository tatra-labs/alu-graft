"""Read the circuit's wires (logical values) out of the residual stream, per layer and position."""
from __future__ import annotations

import torch


@torch.no_grad()
def wires(model, tok, circ, text, names, device="cpu", layers=None):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states          # (n_layers+1) x (1, T, dn); hs[i] = input of layer i
    d = model.config.hidden_size - len(circ.wires)
    toks = [tok.decode([i]) for i in ids[0]]
    res = {}
    L = len(hs) - 1
    for n in names:
        i = d + circ.wires[n]
        res[n] = [(hs[l][0, :, i] / circ.M0).cpu() for l in range(L + 1)]
    return toks, res, out.logits[0].float().cpu()


def show(model, tok, circ, text, names, layer, device="cpu"):
    """Print wire values after `layer` (i.e. the input of layer+1) for every position."""
    toks, res, _ = wires(model, tok, circ, text, names, device)
    print("tok      " + " ".join(f"{t!r:>8}" for t in toks))
    for n in names:
        v = res[n][layer + 1]
        print(f"{n:8s} " + " ".join(f"{x:8.3f}" for x in v.tolist()))
