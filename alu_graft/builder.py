"""A tiny "circuit assembler" for grafting exact logic into a pretrained Qwen3 checkpoint.

Everything the circuit computes lives on K extra residual dimensions ("wires") appended to the
model's hidden size.  Wire values are stored as M0 * x, where x is the *logical* value (a digit,
a 0/1 flag, ...) and M0 is tiny so the wires barely change any RMSNorm denominator.

Every read of the residual goes through an RMSNorm, which multiplies all wires at a position by
the same unknown positive factor.  Per layer we set the norm weight on circuit dims to
1/(median_lambda * M0), so a layer sees rho * x with rho = lambda/median_lambda in roughly [0.4, 2].
The circuit is written so that rho never matters:

* ``mellin``   - the scale-free ratio unit.  A bank of SwiGLU neurons with log-spaced gains on a
                 positive reference wire r and a payload wire v outputs v/r exactly up to ~1e-4
                 (Mellin-transform quadrature: sum_k SiLU(-a_k r) (-a_k v) ~ (pi^2/12) v/r / Delta).
* ``heaviside`` - two SiLU ramps at tau-0.1 and tau+0.1 whose difference saturates at 0.2*beta*rho^2
                 when f > tau + 0.1 and is exactly -0.0 when f < tau - 0.4 (the SiLU underflows).
                 For integer-valued f and half-integer tau it tolerates input errors up to ~0.3.
* ``relu``     - one SiLU ramp, beta*rho^2*f for integer f >= 0, exactly 0 for f <= -1.
* ``stage_ref`` - one neuron that writes the matching 0.2*beta*rho^2 reference for a ramp layer.

A ramp layer followed by a Mellin layer therefore computes an exact threshold or ReLU of any
linear function of absolute wires, in two transformer layers, independent of the residual scale.
Attention heads ("gathers") copy wires from another position; the copy carries that position's
scale, so it is always normalised by a Mellin against a reference copied by the same head.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

DELTA = 0.5            # Mellin grid spacing in log-gain (quadrature error ~1e-6)
W_LO, W_HI = -5.5, 3.5  # every r~ must see grid points w = k*DELTA + ln r~ covering [W_LO, W_HI]
MELLIN_C = 12.0 * DELTA / math.pi ** 2
HW = 0.2               # width of the Heaviside ramp (centred on tau)


@dataclass
class Head:
    layer: int
    kv_slot: int
    q_slot: int                      # 0 or 1 inside the GQA pair of kv_slot
    q_terms: dict                    # wire -> 128-vector (pre-norm query, per unit of rho*x)
    k_terms: dict                    # wire -> 128-vector
    v_terms: dict                    # (vdim) -> {wire: coef}
    out: dict                        # vdim -> {wire: coef}
    note: str = ""


@dataclass
class Neuron:
    gate: dict
    up: dict
    down: dict
    note: str = ""


class Circuit:
    def __init__(self, n_layers: int, M0: float, beta: float):
        self.n_layers = n_layers
        self.M0 = M0
        self.beta = beta
        self.wires: dict[str, int] = {}
        self.emb: dict[str, dict[int, float]] = {}       # wire -> {token_id: logical value}
        self.heads: dict[int, list[Head]] = defaultdict(list)
        self.neurons: dict[int, list[Neuron]] = defaultdict(list)
        self.readouts = []                                # (p_wire, n_wire, {token_id: sign})
        self.notes = []

    # ----------------------------------------------------------------- wires
    def wire(self, name: str) -> str:
        if name not in self.wires:
            self.wires[name] = len(self.wires)
        return name

    def wires_(self, *names):
        return [self.wire(n) for n in names]

    def embed(self, name: str, values: dict[int, float]):
        self.wire(name)
        self.emb[name] = values

    # ----------------------------------------------------------------- MLP primitives
    def _neuron(self, layer, gate, up, down, note=""):
        for d in (gate, up, down):
            for w in d:
                assert w in self.wires, f"unknown wire {w}"
        self.neurons[layer].append(Neuron(dict(gate), dict(up), dict(down), note))

    def mellin(self, layer: int, payload: dict, ref: dict, out: dict, ln_ref_range, note=""):
        """out_wire += coef * payload(x)/ref(x).  ln_ref_range = (lo, hi) of ln ref(rho*x) seen by
        this layer's MLP.  ref must be > 0 wherever the result is used."""
        lo, hi = ln_ref_range
        k_lo = math.floor((W_LO - hi) / DELTA)
        k_hi = math.ceil((W_HI - lo) / DELTA)
        for k in range(k_lo, k_hi + 1):
            a = math.exp(k * DELTA)
            self._neuron(layer,
                         gate={w: -a * c for w, c in ref.items()},
                         up={w: -a * c for w, c in payload.items()},
                         down={w: c * MELLIN_C for w, c in out.items()},
                         note=note)

    def stage_ref(self, layer: int, out: str, R: str = "R"):
        b = self.beta
        self._neuron(layer, gate={R: HW * b}, up={R: 1.0}, down={out: 1.0 / (HW * b)}, note="stage_ref")

    def heaviside(self, layer: int, f: dict, tau: float, out: dict, R: str = "R", note=""):
        """out += coef * rho^2 * [f > tau]   (f integer-valued, tau half-integer)."""
        b = self.beta
        g1 = {w: b * c for w, c in f.items()}
        g1[R] = g1.get(R, 0.0) - b * (tau - HW / 2)
        g2 = {w: b * c for w, c in f.items()}
        g2[R] = g2.get(R, 0.0) - b * (tau + HW / 2)
        down = {w: c / (HW * b) for w, c in out.items()}
        self._neuron(layer, gate=g1, up={R: 1.0}, down=down, note=note)
        self._neuron(layer, gate=g2, up={R: -1.0}, down=down, note=note)

    def relu(self, layer: int, f: dict, out: dict, R: str = "R", note=""):
        """out += coef * rho^2 * relu(f)   (f integer-valued)."""
        b = self.beta
        self._neuron(layer, gate={w: b * c for w, c in f.items()}, up={R: 1.0},
                     down={w: c / b for w, c in out.items()}, note=note)

    # ----------------------------------------------------------------- attention
    def head(self, h: Head):
        for d in (h.q_terms, h.k_terms):
            for w in d:
                assert w in self.wires, w
        for vd, terms in h.v_terms.items():
            for w in terms:
                assert w in self.wires, w
        for vd, terms in h.out.items():
            for w in terms:
                assert w in self.wires, w
        self.heads[h.layer].append(h)

    # ----------------------------------------------------------------- readout
    def readout(self, p_wire: str, n_wire: str, token_signs: dict[int, float]):
        self.wire(p_wire)
        self.wire(n_wire)
        self.readouts.append((p_wire, n_wire, dict(token_signs)))

    # ----------------------------------------------------------------- stats
    def stats(self):
        kv = {L: (max(h.kv_slot for h in hs) + 1 if hs else 0) for L, hs in self.heads.items()}
        return {
            "wires": len(self.wires),
            "max_kv_slots": max(kv.values()) if kv else 0,
            "neurons_per_layer": {L: len(v) for L, v in sorted(self.neurons.items())},
            "max_neurons": max((len(v) for v in self.neurons.values()), default=0),
        }


def lin(*pairs, **named):
    """Build a linear functional {wire: coef} from (wire, coef) pairs; repeated wires add up."""
    d: dict[str, float] = {}
    for w, c in pairs:
        d[w] = d.get(w, 0.0) + c
    for w, c in named.items():
        d[w] = d.get(w, 0.0) + c
    return {w: c for w, c in d.items() if c != 0.0}
