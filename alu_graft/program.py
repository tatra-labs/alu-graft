"""The arithmetic program: exact signed results of  A+B=  and  A-B=  for 1-6 digit A, B.

Both the compact form the tokenizer sees in prompts ("123+456=") and the spaced form the model
writes in its own reasoning ("123 + 456 = 579") are recognised; the answer is emitted in the
matching style ("579" / " 579", "-2" / " -2").

Layer plan (only the circuit's extra heads / neurons are listed; the base model is untouched):

  L0  attn  static offset heads copy the token features of offsets 1..8
      mlp   Mellin-normalise the copies                  -> absolute window flags / digits
  L1  mlp   ramps: digit runs, operator/terminator tests, answer-position tests J_j
  L2  mlp   Mellin                                       -> b digits, A-side validity, J_j, ...
  L3  attn  '=' reads the operand A (digits stored at the operator token, offset |B|+1 or +2)
      mlp   Mellin                                       -> a digits at '='
  L4  mlp   ramps: VALID, per-digit generate/propagate (add) and less/greater (subtract)
  L5  mlp   Mellin
  L6  mlp   ramps: carry-lookahead (3 chains: A+B, A-B, B-A)
  L7  mlp   Mellin
  L8  mlp   ramps: result digits r_p, "result has more than i digits" T_i, sign / style flags
  L9  mlp   Mellin                                       -> results at '=' (exactly 0 elsewhere)
  L10 attn  every answer position (offset j after '=') reads the results of its '='
      mlp   Mellin
  L11 mlp   ramps: which token comes at offset j (digit e, '-', ' -', ' ', or END)
  L12 mlp   Mellin (+ '-' / ' -' / ' ' / END readouts)
  L13 mlp   ramps: one-hot of the digit
  L14 mlp   Mellin into the digit readouts

The readout wires feed the final norm with huge weights, paired so that the tied input embedding
cancels exactly: logit(token) += ~300 nats when the circuit fires, and exactly 0 otherwise.
"""
from __future__ import annotations

from .builder import Circuit, Head, lin
from .heads import design

D = 6          # max digits per operand
WIN = 8        # window offsets gathered at L0
NOUT = 8       # max answer offset j (0..8)
MARGIN = 2.0   # extra ln-range for every Mellin window (survives 2x the calibrated log-range)


# --------------------------------------------------------------------------- token features
def token_ids(tok):
    one = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    ids = {str(d): one(str(d))[0] for d in range(10)}
    for s in ["-", " -", " ", "+", " +", "=", " ="]:
        t = one(s)
        assert len(t) == 1, (s, t)
        ids[s] = t[0]
    return ids


def token_classes(tok, n_rows):
    """Return dict feature -> {token_id: value}, for the embedding rows of the circuit."""
    feats = {k: {} for k in ("DG", "VAL", "PL", "MI", "EQ", "EQS", "SP", "CL", "NB", "SOFT")}
    special = set(tok.all_special_ids)
    for t in getattr(tok, "added_tokens_decoder", {}):
        special.add(int(t))
    n_tok = len(tok)
    for tid in range(n_rows):
        if tid >= n_tok:
            continue
        if tid in special:
            feats["CL"][tid] = 1.0
            continue
        s = tok.decode([tid])
        st = s.strip()
        if st and (st[-1] in "+-*/^%\u00d7\u00f7" or st[-1].isdigit()):
            feats["NB"][tid] = 1.0          # an operand preceded by "<this> " is part of a chain
        if s in (" ", ":", "=", " =", "(", "["):
            feats["SOFT"][tid] = 1.0        # clean only if the token before it is not a digit/operator
        if len(s) == 1 and s in "0123456789":
            feats["DG"][tid] = 1.0
            if s != "0":
                feats["VAL"][tid] = float(s)
            continue
        if s in ("+", " +"):
            feats["PL"][tid] = 1.0
        elif s in ("-", " -"):
            feats["MI"][tid] = 1.0
        elif s in ("=", " ="):
            feats["EQ"][tid] = 1.0
            feats["CL"][tid] = 1.0          # "x=12+3=" : an '=' may precede an operand
            if s == " =":
                feats["EQS"][tid] = 1.0
        elif s == " ":
            feats["SP"][tid] = 1.0
            feats["CL"][tid] = 1.0
        elif s and (s[-1].isspace() or s in ("(", "[", ":", '"', "'")):
            feats["CL"][tid] = 1.0
    return feats


def _rng(calib, *terms, margin=MARGIN):
    """ln-range of a product of rho factors: terms are (name, power)."""
    lo = sum(p * calib[n]["ln_lo"] for n, p in terms)
    hi = sum(p * calib[n]["ln_hi"] for n, p in terms)
    return (lo - margin, hi + margin)


def build(tok, calib, qk_norms, n_rows, *, M0=2.0 ** -13, beta=3000.0, dmax=40960, classes=None, ids=None):
    """qk_norms: {layer: (q_norm_weight, k_norm_weight)} as float64 tensors.
    `classes` / `ids` (from token_classes / token_ids) may be passed instead of a tokenizer."""
    c = Circuit(n_layers=28, M0=M0, beta=beta)
    W = c.wire
    tid = ids if ids is not None else token_ids(tok)

    def times(terms, s):
        return [(x, v * s) for x, v in terms]

    def neg_(terms):
        return times(terms, -1.0)

    # ------------------------------------------------------------------ embedding features
    c.embed("R", {t: 1.0 for t in range(n_rows)})
    for k, v in (classes if classes is not None else token_classes(tok, n_rows)).items():
        c.embed(k, {int(t): float(x) for t, x in v.items()})
    AT0 = [("DG", 1.0), ("MI", 1.0), ("SP", 1.0)]   # this token can be part of an answer
    ISOP = [("PL", 1.0), ("MI", 1.0)]

    # ------------------------------------------------------------------ L0: static window
    feats = ["R", "DG", "VAL", "PL", "MI", "EQ", "CL", "SP", "NB", "SOFT"]
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
    wq, wk = qk_norms[0]
    for slot, offs in enumerate(pairs):
        q, k, m = design(wq, wk, list(offs), dmax=dmax, steps=800, restarts=8,
                         key=f"L0_{'-'.join(map(str, offs))}_sNone_d{dmax}")
        c.notes.append(f"L0 heads {offs}: margin {m:.2f} nats")
        for qs, d in enumerate(offs):
            for f in feats:
                W(f"g{d}_{f}")
            c.head(Head(layer=0, kv_slot=slot, q_slot=qs,
                        q_terms={"R": q[qs]}, k_terms={"R": k},
                        v_terms={i: {f: 1.0} for i, f in enumerate(feats)},
                        out={i: {f"g{d}_{f}": 1.0} for i, f in enumerate(feats)},
                        note=f"window offset {d}"))
    need = {"DG": range(1, 8), "VAL": range(1, D + 1), "CL": range(2, 8), "PL": range(2, 9),
            "MI": range(2, 9), "SP": range(2, 8), "EQ": range(1, 9), "NB": range(3, 9),
            "SOFT": range(2, 8)}
    r0 = _rng(calib, ("post0", 1), ("in0", 1))
    for f, ds in need.items():
        for d in ds:
            c.mellin(0, {f"g{d}_{f}": 1.0}, {f"g{d}_R": 1.0}, {W(f"w{d}_{f}"): 1.0}, r0, note=f"w{d}_{f}")
    c.mellin(0, {"g1_DG": 1.0, "g1_MI": 1.0, "g1_SP": 1.0}, {"g1_R": 1.0}, {W("w1_AT"): 1.0}, r0, note="w1_AT")

    def w(d, f):
        return f"w{d}_{f}"

    def AT(d):
        return [(w(1, "AT"), 1.0)] if d == 1 else [(w(d, "DG"), 1.0), (w(d, "MI"), 1.0), (w(d, "SP"), 1.0)]

    def runsum(k):              # number of digits among offsets 1..k
        return [(w(d, "DG"), 1.0) for d in range(1, k + 1)]

    # ------------------------------------------------------------------ L1 ramps / L2 Mellin
    L = 1
    W("S1")
    c.stage_ref(L, "S1")
    raw1 = []
    for k in range(1, D + 1):
        # digit run value at offset k (0 if the run 1..k is broken): b digits at '=', a digits at the operator
        f = lin((w(k, "VAL"), 1.0), *times(runsum(k), 10.0), ("R", -10.0 * k))
        c.relu(L, f, {W(f"m{k}_raw"): 1.0}, note=f"m{k}")
        # same, but only at operator tokens (this is what '=' will fetch)
        fo = lin(*f.items(), *times(ISOP, 10.0), ("R", -10.0))
        c.relu(L, fo, {W(f"mo{k}_raw"): 1.0}, note=f"mo{k}")
        raw1 += [f"m{k}", f"mo{k}"]
    # A is valid at an operator token if a run of k<=6 digits is preceded by a clean token.  A "soft"
    # clean token (' ', ':', '=', '(', '[') only counts if the token before it is not a digit or an
    # operator, so "1 + 2 + 3 =", "12:30+5=" and "(1+2)*(3+4=" abstain while "x = 1 + 2 =" works.
    W("AVo_raw")
    for k in range(1, D + 1):
        c.heaviside(L, lin(*runsum(k), (w(k + 1, "CL"), 1.0), (w(k + 1, "SOFT"), -1.0), *ISOP), k + 1.5,
                    {"AVo_raw": 1.0}, note=f"AVo k={k}")
        c.heaviside(L, lin(*runsum(k), (w(k + 1, "SOFT"), 1.0), (w(k + 2, "NB"), -1.0), *ISOP), k + 1.5,
                    {"AVo_raw": 1.0}, note=f"AVo k={k} after soft-clean token")
    raw1.append("AVo")
    # at '=': B = run of k digits, then the operator at offset k+1 ("3+4=") or k+2 ("3 + 4 =")
    W("OPP_raw"), W("OPM_raw")
    for k in range(1, D + 1):
        W(f"OSc{k}_raw"), W(f"OSs{k}_raw")
        for op, wire in (("PL", "OPP_raw"), ("MI", "OPM_raw")):
            c.heaviside(L, lin(*runsum(k), (w(k + 1, op), 1.0), ("EQ", 1.0)), k + 1.5,
                        {f"OSc{k}_raw": 1.0, wire: 1.0}, note=f"{op}c k={k}")
            c.heaviside(L, lin(*runsum(k), (w(k + 1, "SP"), 1.0), (w(k + 2, op), 1.0), ("EQ", 1.0)), k + 2.5,
                        {f"OSs{k}_raw": 1.0, wire: 1.0}, note=f"{op}s k={k}")
        raw1 += [f"OSc{k}", f"OSs{k}"]
    raw1 += ["OPP", "OPM"]
    # answer positions: '=' at offset j, and offsets 0..j-1 are answer tokens (digit, '-', ' -', ' ')
    for j in range(1, NOUT + 1):
        f = lin((w(j, "EQ"), 1.0), *AT0, *[x for d in range(1, j) for x in AT(d)])
        c.heaviside(L, f, j + 0.5, {W(f"J{j}_raw"): 1.0}, note=f"J{j}")
        raw1.append(f"J{j}")
    r2 = _rng(calib, ("post2", 1), ("post1", 2))
    for n in raw1:
        c.mellin(2, {f"{n}_raw": 1.0}, {"S1": 1.0}, {W(n): 1.0}, r2, note=n)

    # ------------------------------------------------------------------ L3: '=' fetches A from the operator
    wq, wk = qk_norms[3]
    targets = list(range(2, D + 3))                  # operator at offset 2..8
    q, k_, m = design(wq, wk, targets, dmax=dmax, steps=800, restarts=8,
                      key=f"L3_{'-'.join(map(str, targets))}_sNone_d{dmax}")
    c.notes.append(f"L3 operand fetch: margin {m:.2f} nats")
    qt = {}
    for kk in range(1, D + 1):
        qt[f"OSc{kk}"] = q[targets.index(kk + 1)]
        qt[f"OSs{kk}"] = q[targets.index(kk + 2)]
    vals = [("ref", dict(ISOP))] + [(f"a{i}", {f"mo{i + 1}": 1.0}) for i in range(D)] + [("AV", {"AVo": 1.0})]
    for n, _ in vals:
        W(f"f3_{n}")
    c.head(Head(layer=3, kv_slot=0, q_slot=0, q_terms=qt, k_terms={"R": k_},
                v_terms={i: dict(t) for i, (_, t) in enumerate(vals)},
                out={i: {f"f3_{n}": 1.0} for i, (n, _) in enumerate(vals)}, note="operand fetch"))
    r3 = _rng(calib, ("post3", 1), ("in3", 1))
    for n, _ in vals[1:]:
        c.mellin(3, {f"f3_{n}": 1.0}, {"f3_ref": 1.0}, {W(n): 1.0}, r3, note=n)
    A = [f"a{i}" for i in range(D)]
    B = [f"m{i + 1}" for i in range(D)]          # b_i = digit run value at offset i+1 from '='

    # ------------------------------------------------------------------ L4 ramps / L5 Mellin
    L = 4
    W("S4")
    c.stage_ref(L, "S4")
    raw4 = ["VALID"]
    c.heaviside(L, lin(AV=1, OPP=1, OPM=1), 1.5, {W("VALID_raw"): 1.0}, note="VALID")
    for i in range(D):
        s = lin((A[i], 1), (B[i], 1))
        c.heaviside(L, s, 9.5, {W(f"G{i}_raw"): 1.0, W(f"P{i}_raw"): -1.0}, note=f"G{i}")
        c.heaviside(L, s, 8.5, {f"P{i}_raw": 1.0}, note=f"P{i}")
        c.heaviside(L, lin((B[i], 1), (A[i], -1)), 0.5, {W(f"LT{i}_raw"): 1.0}, note=f"LT{i}")
        c.heaviside(L, lin((A[i], 1), (B[i], -1)), 0.5, {W(f"GT{i}_raw"): 1.0}, note=f"GT{i}")
        raw4 += [f"G{i}", f"P{i}", f"LT{i}", f"GT{i}"]
    r5 = _rng(calib, ("post5", 1), ("post4", 2))
    for n in raw4:
        c.mellin(5, {f"{n}_raw": 1.0}, {"S4": 1.0}, {W(n): 1.0}, r5, note=n)

    # ------------------------------------------------------------------ L6 ramps / L7 Mellin: carry lookahead
    L = 6
    W("S6")
    c.stage_ref(L, "S6")
    raw6 = []

    def eqd(k):  # a_k == b_k as a linear function of flags
        return [("R", 1.0), (f"LT{k}", -1.0), (f"GT{k}", -1.0)]

    for i in range(1, D + 1):
        for chain, gen, prop in (("C", "G", None), ("BR", "LT", "EQ"), ("BQ", "GT", "EQ")):
            out = W(f"{chain}{i}_raw")
            for j in range(i):
                terms = [(f"{gen}{j}", 1.0)]
                for kk in range(j + 1, i):
                    terms += [(f"P{kk}", 1.0)] if prop is None else eqd(kk)
                c.heaviside(L, lin(*terms), (i - j) - 0.5, {out: 1.0}, note=f"{chain}{i}<-{j}")
            raw6.append(f"{chain}{i}")
    r7 = _rng(calib, ("post7", 1), ("post6", 2))
    for n in raw6:
        c.mellin(7, {f"{n}_raw": 1.0}, {"S6": 1.0}, {W(n): 1.0}, r7, note=n)

    def car(ch, i):
        return [] if i == 0 or i > D else [(f"{ch}{i}", 1.0)]

    # linear digit expressions (absolute wires)
    s_dig, d_dig, e_dig = [], [], []
    for p in range(D + 1):
        ap = [(A[p], 1.0)] if p < D else []
        bp = [(B[p], 1.0)] if p < D else []
        s_dig.append(lin(*ap, *bp, *car("C", p), *times(car("C", p + 1), -10.0)))
        d_dig.append(lin(*ap, *neg_(bp), *neg_(car("BR", p)), *times(car("BR", p + 1), 10.0)) if p < D else {})
        e_dig.append(lin(*bp, *neg_(ap), *neg_(car("BQ", p)), *times(car("BQ", p + 1), 10.0)) if p < D else {})

    # ------------------------------------------------------------------ L8 ramps / L9 Mellin: results at '='
    L = 8
    W("S8")
    c.stage_ref(L, "S8")
    NEG = f"BR{D}"                      # A < B
    sel_add = [("R", 2.0), ("OPP", -1.0), ("VALID", -1.0)]                          # 0 iff '+', valid
    sel_sub = [("R", 2.0), ("OPM", -1.0), ("VALID", -1.0), (NEG, 1.0)]              # 0 iff '-', valid, A>=B
    sel_neg = [("R", 3.0), ("OPM", -1.0), ("VALID", -1.0), (NEG, -1.0)]             # 0 iff '-', valid, A<B
    raw8 = []
    for p in range(D + 1):
        out = W(f"r{p}_raw")
        for dig, sel in ((s_dig, sel_add), (d_dig, sel_sub), (e_dig, sel_neg)):
            if dig[p]:
                c.relu(L, lin(*dig[p].items(), *times(sel, -20.0)), {out: 1.0}, note=f"r{p}")
        raw8.append(f"r{p}")
    for i in range(1, D + 1):
        out = W(f"T{i}_raw")
        for dig, sel in ((s_dig, sel_add), (d_dig, sel_sub), (e_dig, sel_neg)):
            tot = lin(*[kv for p in range(i, D + 1) for kv in dig[p].items()])
            if tot:
                c.heaviside(L, lin(*tot.items(), *times(sel, -100.0)), 0.5, {out: 1.0}, note=f"T{i}")
        raw8.append(f"T{i}")
    # style flags.  U: digits start at j=0 ("=579").  Otherwise a prefix token comes first:
    #   MC: '-' ("=-2"),  SM: ' -' (" = -2"),  VE - SM: ' ' (" = 579")
    c.heaviside(L, lin(("VALID", 2.0), ("EQS", -2.0), ("OPM", -1.0), (NEG, -1.0)), 0.5, {W("U_raw"): 1.0}, note="U")
    c.heaviside(L, lin(("VALID", 1.0), ("EQS", -1.0), ("OPM", 1.0), (NEG, 1.0)), 2.5, {W("MC_raw"): 1.0}, note="MC")
    c.heaviside(L, lin(("VALID", 1.0), ("EQS", 1.0), ("OPM", 1.0), (NEG, 1.0)), 3.5, {W("SM_raw"): 1.0}, note="SM")
    c.heaviside(L, lin(("VALID", 1.0), ("EQS", 1.0)), 1.5, {W("VE_raw"): 1.0}, note="VE")
    raw8 += ["U", "MC", "SM", "VE"]
    r9 = _rng(calib, ("post9", 1), ("post8", 2))
    for n in raw8:
        c.mellin(9, {f"{n}_raw": 1.0}, {"S8": 1.0}, {W(f"{n}_eq"): 1.0}, r9, note=n)

    # ------------------------------------------------------------------ L10: answer positions fetch the results
    wq, wk = qk_norms[10]
    targets = list(range(0, NOUT + 1))
    q, k_, m = design(wq, wk, targets, dmax=dmax, steps=800, restarts=8,
                      key=f"L10_{'-'.join(map(str, targets))}_sNone_d{dmax}")
    c.notes.append(f"L10 result fetch: margin {m:.2f} nats")
    vals = [("ref", {"EQ": 1.0}), ("VALID", {"VALID": 1.0})] + [(n, {f"{n}_eq": 1.0}) for n in raw8]
    for n, _ in vals:
        W(f"f10_{n}")
    qt = {"EQ": q[0]}
    for j in range(1, NOUT + 1):
        qt[f"J{j}"] = q[j]
    c.head(Head(layer=10, kv_slot=0, q_slot=0, q_terms=qt, k_terms={"R": k_},
                v_terms={i: dict(t) for i, (_, t) in enumerate(vals)},
                out={i: {f"f10_{n}": 1.0} for i, (n, _) in enumerate(vals)}, note="result fetch"))
    r10 = _rng(calib, ("post10", 1), ("in10", 1))
    for n, _ in vals[1:]:
        c.mellin(10, {f"f10_{n}": 1.0}, {"f10_ref": 1.0}, {W(f"{n}_a"): 1.0}, r10, note=n)

    # ------------------------------------------------------------------ L11 ramps / L12 Mellin: which token now
    L = 11
    W("S11")
    c.stage_ref(L, "S11")
    Jw = {0: [("EQ", 1.0)]}
    for j in range(1, NOUT + 1):
        Jw[j] = [(f"J{j}", 1.0)]
    Uc = [("U_a", 1.0)]
    Sc = [("VALID_a", 1.0), ("U_a", -1.0)]          # shifted: a prefix token at j=0

    def T(i):
        if i <= 0:
            return [("R", 1.0)]
        if i > D:
            return []
        return [(f"T{i}_a", 1.0)]

    def F(j, p):  # [n == p + j + 1]
        return lin(*T(p + j), *neg_(T(p + j + 1)))

    W("E_raw"), W("ISDIG_raw"), W("ISEND_raw"), W("OMC_raw"), W("OSM_raw"), W("OSP_raw")
    for j in range(0, NOUT + 1):
        for p in range(D + 1):
            for case, fj in ((Uc, F(j, p)), (Sc, F(j - 1, p) if j >= 1 else {})):
                if not fj:
                    continue
                f = lin((f"r{p}_a", 1.0), ("R", -60.0), *times(Jw[j], 20.0), *times(fj.items(), 20.0),
                        *times(case, 20.0))
                c.relu(L, f, {"E_raw": 1.0}, note=f"e j={j} p={p}")
        # digit here?  unshifted: j < n  (T_j);  shifted: 1 <= j <= n  (T_{j-1})
        if j <= D:
            c.heaviside(L, lin(*Jw[j], *Uc, *T(j)), 2.5, {"ISDIG_raw": 1.0}, note=f"isdig U {j}")
        if 1 <= j <= D + 1:
            c.heaviside(L, lin(*Jw[j], *Sc, *T(j - 1)), 2.5, {"ISDIG_raw": 1.0}, note=f"isdig S {j}")
        # end of answer?  unshifted: j == n ; shifted: j == n + 1
        if j >= 1:
            c.heaviside(L, lin(*Jw[j], *Uc, *T(j - 1), *neg_(T(j))), 2.5, {"ISEND_raw": 1.0}, note=f"end U {j}")
        if j >= 2:
            c.heaviside(L, lin(*Jw[j], *Sc, *T(j - 2), *neg_(T(j - 1))), 2.5, {"ISEND_raw": 1.0}, note=f"end S {j}")
    c.heaviside(L, lin(*Jw[0], ("MC_a", 1.0)), 1.5, {"OMC_raw": 1.0}, note="'-'")
    c.heaviside(L, lin(*Jw[0], ("SM_a", 1.0)), 1.5, {"OSM_raw": 1.0}, note="' -'")
    c.heaviside(L, lin(*Jw[0], ("VE_a", 1.0), ("SM_a", -1.0)), 1.5, {"OSP_raw": 1.0}, note="' '")

    r12 = _rng(calib, ("post12", 1), ("post11", 2))
    c.mellin(12, {"E_raw": 1.0}, {"S11": 1.0}, {W("E"): 1.0}, r12, note="E")
    c.mellin(12, {"ISDIG_raw": 1.0}, {"S11": 1.0}, {W("ISDIG"): 1.0}, r12, note="ISDIG")
    c.readout("out_minus", "out_minus_n", {tid["-"]: 1.0})
    c.readout("out_sminus", "out_sminus_n", {tid[" -"]: 1.0})
    c.readout("out_space", "out_space_n", {tid[" "]: 1.0})
    c.readout("out_end", "out_end_n", {tid[str(d)]: -1.0 for d in range(10)})
    c.mellin(12, {"OMC_raw": 1.0}, {"S11": 1.0}, {"out_minus": 1.0}, r12, note="out -")
    c.mellin(12, {"OSM_raw": 1.0}, {"S11": 1.0}, {"out_sminus": 1.0}, r12, note="out ' -'")
    c.mellin(12, {"OSP_raw": 1.0}, {"S11": 1.0}, {"out_space": 1.0}, r12, note="out ' '")
    c.mellin(12, {"ISEND_raw": 1.0}, {"S11": 1.0}, {"out_end": 1.0}, r12, note="out END")

    # ------------------------------------------------------------------ L13 ramps / L14 Mellin: digit one-hot
    L = 13
    W("S13")
    c.stage_ref(L, "S13")
    for d in range(10):
        f = lin(("E", 1.0), ("R", -float(d) - 20.0), ("ISDIG", 20.0))
        out = W(f"O{d}_raw")
        c.heaviside(L, f, -0.5, {out: 1.0}, note=f"O{d}")
        c.heaviside(L, f, 0.5, {out: -1.0}, note=f"O{d}")
    r14 = _rng(calib, ("post14", 1), ("post13", 2))
    for d in range(10):
        c.readout(f"out{d}", f"out{d}_n", {tid[str(d)]: 1.0})
        c.mellin(14, {f"O{d}_raw": 1.0}, {"S13": 1.0}, {f"out{d}": 1.0}, r14, note=f"out{d}")
    return c
