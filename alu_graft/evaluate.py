"""Evaluation: exact arithmetic, abstention / false triggers, and language preservation."""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

DIGITS = "0123456789"


def answer_tokens(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def forward_logits(model, ids_list, device, bs=32, reduce=None):
    """Right-padded batched forward.  Returns a list with, per sequence, the (T, V) float32 logits
    on CPU, or reduce(logits) if given (use it: full logits are 0.6 MB per token)."""
    out = []
    pad = 0
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i:i + bs]
        T = max(len(x) for x in chunk)
        x = torch.full((len(chunk), T), pad, dtype=torch.long)
        for r, ids in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
        lg = model(x.to(device)).logits.float()
        for r, ids in enumerate(chunk):
            row = lg[r, :len(ids)]
            out.append(reduce(row).cpu() if reduce is not None else row.cpu())
        del lg
    return out


def make_problem(rng, max_digits=6, ops="+-", len_a=None, len_b=None):
    la = len_a or rng.randint(1, max_digits)
    lb = len_b or rng.randint(1, max_digits)
    a = rng.randint(0 if la == 1 else 10 ** (la - 1), 10 ** la - 1)
    b = rng.randint(0 if lb == 1 else 10 ** (lb - 1), 10 ** lb - 1)
    op = rng.choice(ops)
    return a, op, b, (a + b if op == "+" else a - b)


TEMPLATES = ["{e}", "\n{e}", "What is {e}", "Compute: {e}", " {e}", "The total is {e}",
             "<|im_start|>user\n{e}<|im_end|>\n<|im_start|>assistant\n{e}", "x = {e}"]


def teacher_forced(model, tok, items, device, bs=32):
    """items: list of (prefix_text, answer_text).  Greedy decoding reproduces the answer and then
    stops emitting digits  <=>  argmax at every answer step equals the answer token and the argmax
    after the last answer token is neither a digit nor '-'.  Returns list of dicts."""
    digit_ids = {tok.convert_tokens_to_ids(c) for c in DIGITS} | {tok.convert_tokens_to_ids("-")}
    seqs, spans = [], []
    for prefix, ans in items:
        p = answer_tokens(tok, prefix)
        a = answer_tokens(tok, ans)
        seqs.append(p + a)
        spans.append((len(p), len(a)))
    preds = forward_logits(model, seqs, device, bs, reduce=lambda lg: lg.argmax(-1))
    res = []
    for (prefix, ans), ids, (lp, la), pr in zip(items, seqs, spans, preds):
        pred = pr[lp - 1: lp + la].tolist()                 # predictions for answer tokens + the one after
        ok_ans = pred[:la] == ids[lp:lp + la]
        ok_end = pred[la] not in digit_ids
        res.append({"prefix": prefix, "answer": ans, "ok": ok_ans and ok_end, "ok_answer": ok_ans,
                    "ok_end": ok_end, "pred": tok.decode(pred[:la])})
    return res


@torch.no_grad()
def greedy(model, tok, prompt, device, max_new_tokens=12):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:])


@torch.no_grad()
def kl_and_ppl(base, graft, tok, texts, device, max_len=512, base_device=None):
    """Mean per-token KL(base || graft) and perplexities on a list of texts."""
    base_device = base_device or device
    kl_sum, n, nll_b, nll_g, max_abs = 0.0, 0, 0.0, 0.0, 0.0
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :max_len].to(device)
        if ids.shape[1] < 2:
            continue
        lb = base(ids.to(base_device)).logits.float()[0].to(device)
        lg = graft(ids).logits.float()[0]
        pb, pg = F.log_softmax(lb, -1), F.log_softmax(lg, -1)
        kl_sum += (pb.exp() * (pb - pg)).sum(-1).sum().item()
        tgt = ids[0, 1:]
        nll_b += -pb[:-1].gather(-1, tgt[:, None]).sum().item()
        nll_g += -pg[:-1].gather(-1, tgt[:, None]).sum().item()
        max_abs = max(max_abs, (lb - lg).abs().max().item())
        n += ids.shape[1] - 1
    return {"tokens": n, "kl_per_token": kl_sum / max(n, 1), "ppl_base": math.exp(nll_b / max(n, 1)),
            "ppl_graft": math.exp(nll_g / max(n, 1)), "max_abs_logit_diff": max_abs}
