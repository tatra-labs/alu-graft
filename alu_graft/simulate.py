"""Run the circuit alone, without the 0.6B base model.

The circuit only ever sees the base model through the RMSNorm scale lambda of each (layer,
position).  This simulator replaces the base model by *random* lambdas drawn independently per
layer and position from the calibrated range (and optionally beyond it), and executes exactly the
same circuit weights: the extra attention heads (with Qwen3's q_norm / k_norm and RoPE), the extra
SwiGLU neurons, and the paired readouts.  Random independent scales are harsher than the real
model, where scales are smooth and correlated.

It is ~1000x cheaper than the full model, so it can check hundreds of thousands of problems on a
CPU, and it runs in CI.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .compile import E_R, DELTA_NATS
from .heads import _inv_freq, HEAD_DIM


class CircuitSim:
    def __init__(self, circ, calib, qk_norms, n_layers=28, device="cpu", dtype=torch.float32, vocab_ids=None):
        self.c = circ
        self.calib = calib
        self.dev = device
        self.dt = dtype
        K = len(circ.wires)
        self.K = K
        idx = circ.wires
        self.idx = idx
        # only the token ids that will be simulated get embedding rows (the full table is 151936 x K)
        if vocab_ids is None:
            from .offline import MINI
            vocab_ids = sorted(set(MINI.values()))
        self.vocab = list(vocab_ids)
        self.local = {t: i for i, t in enumerate(self.vocab)}
        self.lut = torch.full((max(self.vocab) + 1,), -1, dtype=torch.long)
        for t, i in self.local.items():
            self.lut[t] = i
        V = len(self.vocab)
        E = torch.zeros(V, K, dtype=torch.float64)
        for w, vals in circ.emb.items():
            for t, i in self.local.items():
                x = vals.get(t, vals.get(str(t), 0.0)) if vals else 0.0
                if x:
                    E[i, idx[w]] = circ.M0 * x
        RE = torch.zeros(V, K, dtype=torch.float64)
        for p, n, signs in circ.readouts:
            for t, sgn in signs.items():
                if t in self.local:
                    RE[self.local[t], idx[p]] = E_R * sgn
                    RE[self.local[t], idx[n]] = E_R * sgn
        self.E = (E + RE).to(device, dtype)
        self.readout_E = RE.to(device, dtype)
        C_F = DELTA_NATS / (E_R * calib["final"]["median"] * circ.M0)
        self.fn = torch.zeros(K, dtype=torch.float64)
        for p, n, _ in circ.readouts:
            self.fn[idx[p]] = C_F
            self.fn[idx[n]] = -C_F
        self.fn = self.fn.to(device, dtype)
        self.layers = []
        for L in range(n_layers):
            heads = circ.heads.get(L, [])
            neurons = circ.neurons.get(L, [])
            lay = {"L": L}
            if heads:
                nkv = max(h.kv_slot for h in heads) + 1
                Wq = torch.zeros(2 * nkv * HEAD_DIM, K, dtype=torch.float64)
                Wk = torch.zeros(nkv * HEAD_DIM, K, dtype=torch.float64)
                Wv = torch.zeros(nkv * HEAD_DIM, K, dtype=torch.float64)
                Wo = torch.zeros(K, 2 * nkv * HEAD_DIM, dtype=torch.float64)
                done = set()
                for h in heads:
                    qh = 2 * h.kv_slot + h.q_slot
                    for w, vec in h.q_terms.items():
                        Wq[qh * HEAD_DIM:(qh + 1) * HEAD_DIM, idx[w]] += torch.as_tensor(vec, dtype=torch.float64)
                    if h.kv_slot not in done:
                        done.add(h.kv_slot)
                        for w, vec in h.k_terms.items():
                            Wk[h.kv_slot * HEAD_DIM:(h.kv_slot + 1) * HEAD_DIM, idx[w]] += torch.as_tensor(vec, dtype=torch.float64)
                        for vd, terms in h.v_terms.items():
                            for w, cf in terms.items():
                                Wv[h.kv_slot * HEAD_DIM + vd, idx[w]] += cf
                    for vd, terms in h.out.items():
                        for w, cf in terms.items():
                            Wo[idx[w], qh * HEAD_DIM + vd] += circ.M0 * cf
                wq, wk = qk_norms[L]
                lay.update(nkv=nkv, Wq=Wq.to(device, dtype), Wk=Wk.to(device, dtype), Wv=Wv.to(device, dtype),
                           Wo=Wo.to(device, dtype), wq=wq.to(device, dtype), wk=wk.to(device, dtype))
            if neurons:
                N = len(neurons)
                G = torch.zeros(N, K, dtype=torch.float64)
                U = torch.zeros(N, K, dtype=torch.float64)
                Dn = torch.zeros(K, N, dtype=torch.float64)
                for j, nr in enumerate(neurons):
                    for w, cf in nr.gate.items():
                        G[j, idx[w]] += cf
                    for w, cf in nr.up.items():
                        U[j, idx[w]] += cf
                    for w, cf in nr.down.items():
                        Dn[idx[w], j] += circ.M0 * cf
                lay.update(G=G.to(device, dtype), U=U.to(device, dtype), Dn=Dn.to(device, dtype))
            self.layers.append(lay)
        self.inv = _inv_freq().to(device)

    def _lam(self, name, shape, gen, spread):
        c = self.calib[name]
        lo, hi = c["ln_lo"] * spread, c["ln_hi"] * spread
        u = torch.rand(shape, generator=gen, dtype=torch.float64) * (hi - lo) + lo
        return torch.exp(u).to(self.dev, self.dt)        # rho = lambda / median

    @torch.no_grad()
    def run(self, ids, *, seed=0, spread=1.0, pos0=0, return_wires=False, final_spread=1.0):
        """ids: (B, T) long global token ids (must be in self.vocab).  Returns the logit shifts the
        circuit adds, (B, T, len(self.vocab)) over self.vocab, and optionally the final circuit
        residual in logical units."""
        gen = torch.Generator().manual_seed(seed)
        B, T = ids.shape
        loc = self.lut[ids]
        assert (loc >= 0).all(), "token outside the simulated vocabulary"
        h = self.E[loc]                                     # (B, T, K) absolute
        pos = torch.arange(pos0, pos0 + T, device=self.dev)
        mask = torch.full((T, T), float("-inf"), device=self.dev, dtype=self.dt).triu(1)
        M0 = self.c.M0
        for lay in self.layers:
            L = lay["L"]
            if "Wq" in lay:
                rho = self._lam(f"in{L}", (B, T, 1), gen, spread)
                x = rho * h / M0
                nkv = lay["nkv"]
                q = (x @ lay["Wq"].T).view(B, T, 2 * nkv, HEAD_DIM)
                k = (x @ lay["Wk"].T).view(B, T, nkv, HEAD_DIM)
                v = (x @ lay["Wv"].T).view(B, T, nkv, HEAD_DIM)
                q = lay["wq"] * q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
                k = lay["wk"] * k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
                q = self._rope_bt(q, pos)
                k = self._rope_bt(k, pos)
                k = k.repeat_interleave(2, dim=2)
                v = v.repeat_interleave(2, dim=2)
                att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(HEAD_DIM) + mask
                att = att.softmax(-1)
                o = torch.einsum("bhts,bshd->bthd", att, v).reshape(B, T, -1)
                h = h + o @ lay["Wo"].T
            if "G" in lay:
                rho = self._lam(f"post{L}", (B, T, 1), gen, spread)
                x = rho * h / M0
                a = F.silu(x @ lay["G"].T) * (x @ lay["U"].T)
                h = h + a @ lay["Dn"].T
        # the final norm only sets how loud the readout is (+300 nats at the median scale); it is
        # kept at the calibrated range so that `spread` stresses the computation, not the volume
        rho_f = self._lam("final", (B, T, 1), gen, final_spread) * self.calib["final"]["median"]
        normed = self.fn * rho_f * h                          # only readout dims are non-zero
        shift = normed @ self.readout_E.T                     # (B, T, V) logit shifts
        if return_wires:
            return shift, h / M0
        return shift

    def _rope_bt(self, x, pos):
        ang = pos.to(torch.float64)[:, None] * self.inv[None, :]          # (T, 64)
        cos = torch.cat([ang.cos(), ang.cos()], -1).to(self.dt)[None, :, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1).to(self.dt)[None, :, None, :]
        x1, x2 = x[..., :HEAD_DIM // 2], x[..., HEAD_DIM // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin


def circuit_answer(shift_row, tok_ids_of_interest, thresh=100.0):
    """Decode what the circuit forces at one position: a token id, 'END', or None (silent)."""
    vals = shift_row
    best = int(vals.argmax())
    if vals[best] > thresh:
        return best
    if vals.min() < -thresh:
        return "END"
    return None
