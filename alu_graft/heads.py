"""Design query/key vectors for "offset heads" that survive Qwen3's QK-norm.

Qwen3 normalises every query and key per head (RMSNorm over the 128-dim head, with a learned
weight vector shared by all heads of a layer) *before* RoPE.  So a head whose query and key are
built from a constant wire has a fixed logit magnitude, independent of the token's residual scale.
The only freedom is the direction of the pre-norm vectors.

`design(layer, targets)` finds one key vector k (shared, as in a GQA group) and one query vector
q_j per target offset d_j such that, after q_norm/k_norm and RoPE, query j puts its largest logit
on relative offset d_j and beats every other offset 0..dmax by the largest possible margin.

Optionally (`second=0`) offset 0 (the token itself) must be the clear runner-up.  Then a head
whose target lies before the start of the sequence reads the current token instead of a random
neighbour, which the circuit treats as "no expression here".
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

THETA = 1e6
HEAD_DIM = 128
CACHE = Path(__file__).resolve().parent / "head_designs.json"


def _inv_freq():
    return THETA ** (-torch.arange(HEAD_DIM // 2, dtype=torch.float64) / (HEAD_DIM // 2))


def offset_logits(q, k, wq, wk, deltas):
    """Attention logit as a function of relative offset delta = t_query - t_key.

    q: (..., 128) pre-norm query, k: (128,) or (..., 128) pre-norm key, wq/wk: q_norm/k_norm weights.
    Returns (..., len(deltas)).  Matches transformers' rotate-half RoPE convention (checked in tests).
    """
    ang = deltas.to(torch.float64)[:, None] * _inv_freq().to(deltas.device)[None, :]
    cos, sin = torch.cos(ang).to(q.dtype).to(q.device), torch.sin(ang).to(q.dtype).to(q.device)
    qh = wq * q / q.pow(2).mean(-1, keepdim=True).sqrt()
    kh = wk * k / k.pow(2).mean(-1, keepdim=True).sqrt()
    a = qh[..., :64] * kh[..., :64] + qh[..., 64:] * kh[..., 64:]
    b = qh[..., 64:] * kh[..., :64] - qh[..., :64] * kh[..., 64:]
    return (a @ cos.T - b @ sin.T) / math.sqrt(HEAD_DIM)


def _optimise(wq, wk, targets, dmax, second, steps, restarts, seed, device):
    deltas = torch.arange(0, dmax + 1, device=device)
    J = len(targets)
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(restarts, J, HEAD_DIM, generator=g).to(device).requires_grad_()
    k = torch.randn(restarts, 1, HEAD_DIM, generator=g).to(device).requires_grad_()
    tgt = torch.tensor(targets, device=device)
    ar = torch.arange(J, device=device)
    mask = torch.ones(J, dmax + 1, dtype=torch.bool, device=device)
    mask[ar, tgt] = False
    if second is not None:
        mask2 = mask.clone()
        mask2[:, second] = False
    opt = torch.optim.Adam([q, k], lr=0.05)

    def margins(s, soft_t=None):
        st = s[:, ar, tgt]
        if soft_t is None:
            comp = s.masked_fill(~mask, -1e9).max(-1).values
        else:
            comp = torch.logsumexp(soft_t * s.masked_fill(~mask, -1e9), -1) / soft_t
        m = st - comp
        if second is None:
            return m
        s0 = s[:, :, second]
        if soft_t is None:
            comp2 = s.masked_fill(~mask2, -1e9).max(-1).values
        else:
            comp2 = torch.logsumexp(soft_t * s.masked_fill(~mask2, -1e9), -1) / soft_t
        m2 = s0 - comp2
        # targets equal to `second` have no runner-up requirement
        m2 = torch.where(tgt[None] == second, torch.full_like(m2, 1e3), m2)
        return torch.minimum(m, m2 + 0.0)

    for it in range(steps):
        s = offset_logits(q, k, wq, wk, deltas)
        t = 2.0 + 10.0 * it / steps
        m = margins(s, t)
        loss = -(-torch.logsumexp(-4 * m, -1) / 4).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = offset_logits(q, k, wq, wk, deltas)
        m = margins(s)
        worst = m.min(-1).values
        r = int(worst.argmax())
    return q[r].detach().cpu(), k[r, 0].detach().cpu(), float(worst[r])


def design(wq, wk, targets, *, dmax=4096, second=None, steps=1200, restarts=24, seed=0, key=None):
    """Return (q: (J,128) tensor, k: (128,) tensor, worst_margin_nats).  Cached on disk by `key`."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if key is not None and key in cache:
        c = cache[key]
        return torch.tensor(c["q"], dtype=torch.float64), torch.tensor(c["k"], dtype=torch.float64), c["margin"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, m = _optimise(wq.float().to(device), wk.float().to(device), list(targets), dmax, second,
                        steps, restarts, seed, device)
    # rescale so the pre-norm vectors have unit RMS (only the direction matters)
    q = q / q.pow(2).mean(-1, keepdim=True).sqrt()
    k = k / k.pow(2).mean().sqrt()
    if key is not None:
        cache[key] = {"q": q.tolist(), "k": k.tolist(), "margin": m, "targets": list(targets),
                      "dmax": dmax, "second": second}
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, CACHE)
    return q.double(), k.double(), m


def evaluate(q, k, wq, wk, targets, dmax):
    """Exact margins and total leak (sum of softmax weight off-target, all offsets present)."""
    deltas = torch.arange(0, dmax + 1)
    s = offset_logits(q.double(), k.double(), wq.double(), wk.double(), deltas)
    out = []
    for j, t in enumerate(targets):
        row = s[j]
        other = torch.cat([row[:t], row[t + 1:]])
        out.append({"target": t, "margin": float(row[t] - other.max()),
                    "leak": float(torch.exp(other - row[t]).sum())})
    return out
