"""Check the circuit on many problems with the base model replaced by random RMSNorm scales.

  python scripts/sim_check.py --n 100000 --spread 1.0
  python scripts/sim_check.py --exhaustive 2          # every pair of 1-2 digit operands, + and -

A problem passes if, at every step of the answer, the circuit forces exactly the next answer token
(+>100 nats) and after the last token it suppresses the digits (END).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.offline import build_offline, mini_encode  # noqa: E402
from alu_graft.simulate import CircuitSim                 # noqa: E402

STYLES = {
    "compact": (lambda a, op, b: f"\n{a}{op}{b}=", lambda r: f"{r}"),
    "spaced": (lambda a, op, b: f"\n{a} {op} {b} =", lambda r: f" {r}"),
    "eq_spaced": (lambda a, op, b: f"\n{a}{op}{b} =", lambda r: f" {r}"),
    "op_spaced": (lambda a, op, b: f"\n{a} {op} {b}=", lambda r: f"{r}"),
    "after_colon": (lambda a, op, b: f"Q:{a}{op}{b}=", lambda r: f"{r}"),
}


def check(sim, items, digit_ids, seed, spread, bs=512, pos0_max=30000):
    """items: list of (prefix_ids, answer_ids). Returns list of bools."""
    ok = []
    rng = random.Random(seed)
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        # left-pad with newlines so every sequence ends at the same column (causal: padding on the
        # left only adds context, like real text before the expression)
        T = max(len(p) + len(a) for p, a in chunk)
        ids = torch.full((len(chunk), T), 198, dtype=torch.long)
        starts = []
        for r, (p, a) in enumerate(chunk):
            s = T - len(p) - len(a)
            ids[r, s:] = torch.tensor(p + a)
            starts.append(s + len(p))
        pos0 = rng.randint(0, pos0_max)
        shift = sim.run(ids, seed=seed + i, spread=spread, pos0=pos0)      # (B, T, V)
        for r, (p, a) in enumerate(chunk):
            s = starts[r]
            good = True
            for k in range(len(a) + 1):
                row = shift[r, s - 1 + k]
                if k < len(a):
                    tgt = sim.local[a[k]]
                    if row[tgt] < 100 or (row.max() != row[tgt]):
                        good = False
                        break
                    others = row.clone()
                    others[tgt] = 0
                    if others.abs().max() > 1e-3 * row[tgt]:
                        good = False
                        break
                else:
                    if row[digit_ids].max() > -100 or row.max() > 1.0:
                        good = False
            ok.append(good)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--exhaustive", type=int, default=0, help="all operand pairs up to this many digits")
    ap.add_argument("--spread", type=float, default=1.0, help="scale the calibrated ln-lambda range")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 is fastest for this workload)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    circ, calib, qk = build_offline()
    sim = CircuitSim(circ, calib, qk)
    print(f"built circuit ({len(circ.wires)} wires) in {time.time() - t0:.1f}s", flush=True)
    digit_ids = torch.tensor([sim.local[15 + d] for d in range(10)])
    rng = random.Random(args.seed)
    probs = []
    if args.exhaustive:
        hi = 10 ** args.exhaustive
        for a in range(hi):
            for b in range(hi):
                for op in "+-":
                    probs.append((a, op, b))
    else:
        for _ in range(args.n):
            la, lb = rng.randint(1, 6), rng.randint(1, 6)
            a = rng.randint(0 if la == 1 else 10 ** (la - 1), 10 ** la - 1)
            b = rng.randint(0 if lb == 1 else 10 ** (lb - 1), 10 ** lb - 1)
            probs.append((a, rng.choice("+-"), b))
    styles = list(STYLES)
    items, meta = [], []
    for i, (a, op, b) in enumerate(probs):
        st = styles[i % len(styles)]
        r = a + b if op == "+" else a - b
        items.append((mini_encode(STYLES[st][0](a, op, b)), mini_encode(STYLES[st][1](r))))
        meta.append((st, a, op, b, r))
    ok = check(sim, items, digit_ids, args.seed, args.spread)
    acc = sum(ok) / len(ok)
    by = {}
    for (st, a, op, b, r), g in zip(meta, ok):
        for key in (st, op, f"digits{max(len(str(a)), len(str(b)))}"):
            s = by.setdefault(key, [0, 0])
            s[0] += g
            s[1] += 1
    fails = [m for m, g in zip(meta, ok) if not g]
    res = {"n": len(ok), "exact": acc, "spread": args.spread, "seconds": time.time() - t0,
           "by": {k: v[0] / v[1] for k, v in by.items()}, "failures": fails[:50], "n_fail": len(fails)}
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
    for f in fails[:10]:
        print("FAIL", f)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
