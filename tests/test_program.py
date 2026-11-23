"""The compiled arithmetic program: dataflow, size, and behaviour in the circuit simulator."""
import random

import pytest
import torch

from alu_graft.offline import build_offline, mini_encode
from alu_graft.simulate import CircuitSim


@pytest.fixture(scope="module")
def built():
    circ, calib, qk = build_offline()
    return circ, CircuitSim(circ, calib, qk)


def test_dataflow_reads_only_earlier_writes(built):
    """Every wire a head (layer L) reads was written by an MLP of a layer < L or is an embedding
    feature; every wire a neuron (layer L) reads was written by attention of layer <= L or an MLP
    of a layer < L.  Each wire has a single writing layer."""
    circ, _ = built
    written = {w: (-1, "emb") for w in circ.emb}
    for L in range(circ.n_layers):
        for h in circ.heads.get(L, []):
            for t in h.out.values():
                for w in t:
                    assert written.get(w, (L, "attn")) == (L, "attn"), w
                    written[w] = (L, "attn")
        for nr in circ.neurons.get(L, []):
            for w in nr.down:
                assert written.get(w, (L, "mlp")) == (L, "mlp"), w
                written[w] = (L, "mlp")
    order = {"emb": 0, "attn": 1, "mlp": 2}
    for p, n, _ in circ.readouts:
        assert p in written and n in circ.wires

    def stage(w):
        L, kind = written[w]
        return (L, order[kind])

    for L in range(circ.n_layers):
        for h in circ.heads.get(L, []):
            reads = list(h.q_terms) + list(h.k_terms) + [w for t in h.v_terms.values() for w in t]
            for w in reads:
                assert stage(w) < (L, 1), (L, w, written[w])
        for nr in circ.neurons.get(L, []):
            for w in list(nr.gate) + list(nr.up):
                assert stage(w) < (L, 2), (L, w, written[w])


def test_size(built):
    circ, _ = built
    st = circ.stats()
    assert st["wires"] < 450
    assert st["max_kv_slots"] == 4
    assert st["max_neurons"] < 2000


def _check(sim, prefix, answer, seed=0, spread=1.0):
    p, a = mini_encode(prefix), mini_encode(answer)
    ids = torch.tensor([[198] * 4 + p + a])
    shift = sim.run(ids, seed=seed, spread=spread, pos0=1000)[0]
    s = 4 + len(p)
    digits = [sim.local[15 + d] for d in range(10)]
    for k in range(len(a) + 1):
        row = shift[s - 1 + k]
        if k < len(a):
            if int(row.argmax()) != sim.local[a[k]] or row.max() < 100:
                return False
        elif row[digits].max() > -100:
            return False
    return True


@pytest.mark.parametrize("prefix,answer", [
    ("12+3=", "15"), ("999+1=", "1000"), ("5-7=", "-2"), ("999999+999999=", "1999998"),
    ("1000-1000=", "0"), ("0+0=", "0"), ("12-345678=", "-345666"), ("100000-1=", "99999"),
    ("12 + 3 =", " 15"), ("5 - 7 =", " -2"), ("123456 + 654321 =", " 777777"), ("4 - 4 =", " 0"),
    ("12+3 =", " 15"), ("12 + 3=", "15"), ("Q:7-9=", "-2"), ("(31+11=", "42"), ("x = 1 + 2 =", " 3"),
    ("x=12+3=", "15"), ("12 +3=", "15"),
])
def test_examples(built, prefix, answer):
    _, sim = built
    assert _check(sim, "\n" + prefix if prefix[0].isdigit() else prefix, answer)


def test_random_problems(built):
    _, sim = built
    rng = random.Random(5)
    for i in range(150):
        la, lb = rng.randint(1, 6), rng.randint(1, 6)
        a = rng.randint(0 if la == 1 else 10 ** (la - 1), 10 ** la - 1)
        b = rng.randint(0 if lb == 1 else 10 ** (lb - 1), 10 ** lb - 1)
        op = rng.choice("+-")
        r = a + b if op == "+" else a - b
        if i % 2:
            assert _check(sim, f"\n{a}{op}{b}=", f"{r}", seed=i), (a, op, b)
        else:
            assert _check(sim, f"\n{a} {op} {b} =", f" {r}", seed=i), (a, op, b)


@pytest.mark.parametrize("text", [
    "\n1234567+1=", "\n1+1234567=", "\n1+2+3=", "\n12-3+4=", "\n1.5+2=", "\n2+0.5=", "\n1,234+5=",
    "\n$12+3=", "\nx12+3=", "\n12*3=", "\n3*4+5=", "\n12+3)=", "\n+3=", "\n12+ =",
    "\n1234567 + 1 =", "\n1 + 2 + 3 =", "\n12 ++ 3 =", "\n12+3.5=", "\n12:30+5=",
    "\n3 * 4 + 5 =", "\n(1+2)*(3+4=",
])
def test_abstains(built, text):
    """No logit shift anywhere: the circuit stays silent on everything it does not support."""
    _, sim = built
    ids = torch.tensor([[198] * 3 + mini_encode(text) + [15, 16, 17]])
    shift = sim.run(ids, seed=1)[0]
    assert shift.abs().max() < 1.0, (text, float(shift.abs().max()))


def test_silent_on_digit_soup(built):
    """Long random sequences of digits, operators and spaces with no '=' never trigger it."""
    _, sim = built
    rng = random.Random(0)
    vocab = [15 + d for d in range(10)] + [10, 12, 220, 488, 481, 198, 13, 11]
    ids = torch.tensor([[rng.choice(vocab) for _ in range(400)] for _ in range(4)])
    shift = sim.run(ids, seed=2)
    assert shift.abs().max() < 1.0


def test_chain_after_an_answer_is_silent(built):
    """'1+2=3+4=': the first '=' is a real expression (fires 3); the second must abstain."""
    _, sim = built
    ids = torch.tensor([[198] * 3 + mini_encode("\n1+2=3+4=")])
    shift = sim.run(ids, seed=4)[0]
    assert int(shift[3 + 4].argmax()) == sim.local[18] and shift[3 + 4].max() > 100   # '=' -> '3'
    assert shift[-1].abs().max() < 1.0
