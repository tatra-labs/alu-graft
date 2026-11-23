"""The three scale-free primitives, evaluated as real SwiGLU neurons at random RMSNorm scales."""
import math
import random

import torch
import torch.nn.functional as F

from alu_graft.builder import Circuit


def mlp(c, layer, x, rho, dtype=torch.float32):
    """Run one layer's circuit neurons on logical wire values x (dict) seen at scale rho."""
    idx = c.wires
    K = len(idx)
    v = torch.zeros(K, dtype=dtype)
    for w, val in x.items():
        v[idx[w]] = val
    u = rho * v
    ns = c.neurons[layer]
    G = torch.zeros(len(ns), K, dtype=dtype)
    U = torch.zeros(len(ns), K, dtype=dtype)
    Dn = torch.zeros(K, len(ns), dtype=dtype)
    for j, nr in enumerate(ns):
        for w, cf in nr.gate.items():
            G[j, idx[w]] += cf
        for w, cf in nr.up.items():
            U[j, idx[w]] += cf
        for w, cf in nr.down.items():
            Dn[idx[w], j] += cf
    out = Dn @ (F.silu(G @ u) * (U @ u))
    return {w: out[i] for w, i in idx.items()}


def test_mellin_ratio_is_scale_free():
    c = Circuit(4, M0=1.0, beta=3000.0)
    c.wires_("R", "v", "out")
    c.mellin(0, {"v": 1.0}, {"R": 1.0}, {"out": 1.0}, (-2.0, 2.0))
    rng = random.Random(0)
    worst = 0.0
    for _ in range(300):
        rho = math.exp(rng.uniform(-2.0, 2.0))
        val = rng.choice([0.0, 1.0, -1.0, 9.0, 0.37, -7.5, 19.0])
        got = mlp(c, 0, {"R": 1.0, "v": val}, rho, torch.float64)["out"].item()
        worst = max(worst, abs(got - val) / max(1.0, abs(val)))
    assert worst < 1e-5, worst


def test_mellin_zero_ref_gives_exact_zero():
    c = Circuit(4, M0=1.0, beta=3000.0)
    c.wires_("R", "v", "out")
    c.mellin(0, {"v": 1.0}, {"R": 1.0}, {"out": 1.0}, (-2.0, 2.0))
    assert mlp(c, 0, {"R": 0.0, "v": 5.0}, 1.0)["out"].item() == 0.0


def _threshold_circuit(tau):
    c = Circuit(4, M0=1.0, beta=3000.0)
    c.wires_("R", "f", "raw", "S", "out", "rl_raw", "rl")
    c.stage_ref(1, "S")
    c.heaviside(1, {"f": 1.0}, tau, {"raw": 1.0})
    c.relu(1, {"f": 1.0}, {"rl_raw": 1.0})
    c.mellin(2, {"raw": 1.0}, {"S": 1.0}, {"out": 1.0}, (-4.0, 2.5))
    c.mellin(2, {"rl_raw": 1.0}, {"S": 1.0}, {"rl": 1.0}, (-4.0, 2.5))
    return c


def test_heaviside_and_relu_two_layers():
    tau = 2.5
    c = _threshold_circuit(tau)
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(-5, 9)
        noise = rng.uniform(-0.3, 0.3)
        f = n + noise
        r1, r2 = math.exp(rng.uniform(-1.1, 0.8)), math.exp(rng.uniform(-1.1, 0.8))
        h1 = mlp(c, 1, {"R": 1.0, "f": f}, r1, torch.float64)
        x2 = {"R": 1.0, "raw": h1["raw"].item(), "S": h1["S"].item(), "rl_raw": h1["rl_raw"].item()}
        h2 = mlp(c, 2, x2, r2, torch.float64)
        want = 1.0 if n > tau else 0.0
        assert abs(h2["out"].item() - want) < 1e-4, (f, h2["out"].item())
        if n >= 0:
            assert abs(h2["rl"].item() - max(f, 0.0)) < 1e-2 + 1e-3 * abs(f), (f, h2["rl"].item())
        elif n <= -1:
            assert abs(h2["rl"].item()) < 1e-3


def test_off_ramps_are_bitwise_zero_in_fp32():
    """Below threshold the SiLU underflows, so a dead flag is exactly 0.0, not 1e-30."""
    c = _threshold_circuit(2.5)
    for f in [-5.0, 0.0, 1.0, 2.0, 2.0 + 0.1]:
        for rho in [0.33, 1.0, 2.2]:
            h1 = mlp(c, 1, {"R": 1.0, "f": f}, rho, torch.float32)
            assert h1["raw"].item() == 0.0, (f, rho, h1["raw"].item())
