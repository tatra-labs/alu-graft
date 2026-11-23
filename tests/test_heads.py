"""The offset heads: RoPE convention, and the margins of the shipped designs over 40,960 tokens."""
import json
import math

import pytest
import torch

from alu_graft.heads import CACHE, offset_logits, evaluate, _inv_freq
from alu_graft.offline import load_data

USED = {  # design key -> (layer, targets, minimum acceptable margin in nats)
    "L0_1-2_sNone_d40960": (0, [1, 2], 12.0),
    "L0_3-4_sNone_d40960": (0, [3, 4], 12.0),
    "L0_5-6_sNone_d40960": (0, [5, 6], 12.0),
    "L0_7-8_sNone_d40960": (0, [7, 8], 12.0),
    "L3_2-3-4-5-6-7-8_sNone_d40960": (3, [2, 3, 4, 5, 6, 7, 8], 9.0),
    "L10_0-1-2-3-4-5-6-7-8_sNone_d40960": (10, [0, 1, 2, 3, 4, 5, 6, 7, 8], 8.0),
}


def test_rope_convention_matches_transformers():
    pytest.importorskip("transformers")
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    g = torch.Generator().manual_seed(0)
    q = torch.randn(128, generator=g, dtype=torch.float64)
    k = torch.randn(128, generator=g, dtype=torch.float64)
    wq = torch.rand(128, generator=g, dtype=torch.float64) * 3
    wk = torch.rand(128, generator=g, dtype=torch.float64) * 3
    T = 12
    pos = torch.arange(T, dtype=torch.float64)
    emb = torch.cat([pos[:, None] * _inv_freq()[None]] * 2, -1)
    qh = wq * q / q.pow(2).mean().sqrt()
    kh = wk * k / k.pow(2).mean().sqrt()
    Q = qh.expand(1, 1, T, 128)
    K = kh.expand(1, 1, T, 128)
    Qr, Kr = apply_rotary_pos_emb(Q, K, emb.cos()[None], emb.sin()[None])
    ref = (Qr[0, 0, T - 1] @ Kr[0, 0].T) / math.sqrt(128)          # query at T-1, keys 0..T-1
    mine = offset_logits(q, k, wq, wk, torch.arange(T)).flip(0)     # delta = T-1-s
    assert torch.allclose(ref, mine, atol=1e-9)


@pytest.mark.parametrize("key", sorted(USED))
def test_shipped_designs_have_margin(key):
    layer, targets, need = USED[key]
    designs = json.loads(CACHE.read_text())
    assert key in designs
    d = designs[key]
    _, _, qk = load_data()
    wq, wk = qk[layer]
    ev = evaluate(torch.tensor(d["q"], dtype=torch.float64), torch.tensor(d["k"], dtype=torch.float64),
                  wq, wk, targets, 40960)
    worst = min(e["margin"] for e in ev)
    leak = max(e["leak"] for e in ev)
    assert worst >= need, (key, worst)
    assert leak < 0.05, (key, leak)
