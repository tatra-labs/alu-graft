"""Export the grafted circuit (the actual extra weights) to JSON for the in-browser simulator.

  python scripts/export_web.py            -> docs/circuit.json

Only the circuit's own parameters are exported (the base model is not needed to run them: the
circuit sees the base model only through RMSNorm scales, which the page sets to the calibrated
median or jitters at random).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.offline import build_offline, MINI   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "circuit.json"


def r(x, n=7):
    return float(f"{x:.{n}g}")


def main():
    circ, calib, qk = build_offline()
    idx = circ.wires
    vocab = sorted(set(MINI.values()))
    emb = {}
    for t in vocab:
        row = {}
        for w, vals in circ.emb.items():
            x = vals.get(t, 0.0) if vals else 0.0
            if x:
                row[idx[w]] = x
        emb[t] = row
    heads = {}
    for L, hs in circ.heads.items():
        heads[L] = []
        for h in hs:
            heads[L].append({
                "kv": h.kv_slot, "qs": h.q_slot,
                "q": {idx[w]: [r(v) for v in vec.tolist()] for w, vec in h.q_terms.items()},
                "k": {idx[w]: [r(v) for v in vec.tolist()] for w, vec in h.k_terms.items()},
                "v": {vd: {idx[w]: c for w, c in t.items()} for vd, t in h.v_terms.items()},
                "o": {vd: {idx[w]: c for w, c in t.items()} for vd, t in h.out.items()},
                "note": h.note})
    neurons = {}
    for L, ns in circ.neurons.items():
        neurons[L] = [[[[idx[w], r(c)] for w, c in nr.gate.items()],
                       [[idx[w], r(c)] for w, c in nr.up.items()],
                       [[idx[w], r(c)] for w, c in nr.down.items()]] for nr in ns]
    qkw = {L: {"q": [r(v) for v in qk[L][0].tolist()], "k": [r(v) for v in qk[L][1].tolist()]}
           for L in heads}
    ranges = {k: [v["ln_lo"], v["ln_hi"]] for k, v in calib.items()}
    readouts = [[idx[p], idx[n], {str(t): s for t, s in signs.items()}] for p, n, signs in circ.readouts]
    show = ["a0", "a1", "a2", "a3", "a4", "a5", "m1", "m2", "m3", "m4", "m5", "m6", "VALID", "OPP", "OPM",
            "C1", "C2", "C3", "C4", "C5", "C6", "BR6", "r0_eq", "r1_eq", "r2_eq", "r3_eq", "r4_eq", "r5_eq",
            "r6_eq", "T1_eq", "T2_eq", "T3_eq", "T4_eq", "T5_eq", "T6_eq", "E", "ISDIG"]
    data = {"K": len(idx), "M0": circ.M0, "beta": circ.beta, "mini": MINI, "emb": emb, "heads": heads,
            "neurons": neurons, "qk": qkw, "ranges": ranges, "readouts": readouts,
            "wires": {w: idx[w] for w in show if w in idx}, "notes": circ.notes,
            "readout_names": [p for p, _, _ in circ.readouts]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    n = sum(len(v) for v in neurons.values())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {len(idx)} wires, {n} neurons")


if __name__ == "__main__":
    main()
