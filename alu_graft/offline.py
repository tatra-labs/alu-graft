"""Build the circuit without the base model: everything it needs from Qwen3-0.6B is tiny and is
vendored in alu_graft/data/ (token classes, a few token ids, the q_norm/k_norm vectors of the three
layers that host circuit heads).  Used by the simulator, the tests and CI.

    python -m alu_graft.offline --export     # regenerate alu_graft/data/ from the real model
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import program

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HEAD_LAYERS = (0, 3, 10)
N_ROWS = 151936
N_LAYERS = 28

# a tiny tokenizer for strings built from digits, + - = space newline (matches Qwen3's BPE)
MINI = {"+": 10, "-": 12, "=": 28, " +": 488, " -": 481, " =": 284, " ": 220, "\n": 198,
        **{str(d): 15 + d for d in range(10)}, "Q": 48, ":": 25, "(": 7, ")": 8, ".": 13, "$": 3,
        ",": 11, "*": 9, "x": 87}


def mini_encode(s: str):
    """Greedy tokenisation of digit/operator strings (spaces attach to a following operator)."""
    out, i = [], 0
    while i < len(s):
        if s[i] == " " and i + 1 < len(s) and s[i + 1] in "+-=":
            out.append(MINI[s[i:i + 2]])
            i += 2
        else:
            out.append(MINI[s[i]])
            i += 1
    return out


def export():
    from transformers import AutoTokenizer
    from .compile import load_base
    sd, cfg, _ = load_base()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    DATA.mkdir(exist_ok=True)
    classes = program.token_classes(tok, sd["model.embed_tokens.weight"].shape[0])
    (DATA / "token_classes.json").write_text(json.dumps({k: {str(t): v for t, v in d.items()} for k, d in classes.items()}))
    (DATA / "token_ids.json").write_text(json.dumps(program.token_ids(tok)))
    qk = {L: {"q": sd[f"model.layers.{L}.self_attn.q_norm.weight"].float().tolist(),
              "k": sd[f"model.layers.{L}.self_attn.k_norm.weight"].float().tolist()} for L in HEAD_LAYERS}
    (DATA / "qk_norms.json").write_text(json.dumps(qk))
    for k, v in MINI.items():
        assert tok(k, add_special_tokens=False)["input_ids"] == [v], k
    for s in ["123+456=", "12 + 3 = 15", "5 - 7 = -2", "\n9-8="]:
        assert mini_encode(s) == tok(s, add_special_tokens=False)["input_ids"], s


def load_data():
    classes = json.loads((DATA / "token_classes.json").read_text())
    ids = json.loads((DATA / "token_ids.json").read_text())
    qk_raw = json.loads((DATA / "qk_norms.json").read_text())
    qk = {}
    for L in range(N_LAYERS):
        r = qk_raw.get(str(L))
        if r is None:   # layers without circuit heads: any vector works (never used)
            qk[L] = (torch.ones(128, dtype=torch.float64), torch.ones(128, dtype=torch.float64))
        else:
            qk[L] = (torch.tensor(r["q"], dtype=torch.float64), torch.tensor(r["k"], dtype=torch.float64))
    return classes, ids, qk


def build_offline(**kw):
    classes, ids, qk = load_data()
    calib = json.loads((HERE / "calibration.json").read_text())
    circ = program.build(None, calib, qk, N_ROWS, classes=classes, ids=ids, **kw)
    return circ, calib, qk


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        export()
        print("exported to", DATA)
