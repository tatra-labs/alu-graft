"""Materialise a Circuit into the weights of a wider Qwen3 checkpoint (no gradients anywhere).

The result is an ordinary `Qwen3ForCausalLM` with a larger hidden size, more attention heads and a
larger MLP.  It loads with vanilla `transformers` - no custom code, no hooks, no tools.

Width extension: the base model's d=1024 residual dims are kept bit-for-bit; K circuit dims are
appended.  Base weights never read circuit dims and circuit weights never read base dims, so the
two only touch through the shared RMSNorm denominators.  Every base norm weight is multiplied by
sqrt(d/(d+K)), which makes the base computation exact when the circuit dims are zero, and off by
O(|h_circ|^2/|h_base|^2) ~ 1e-9 otherwise (circuit wires are stored at M0 = 2^-13 scale).
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from . import program

MODEL_ID = "Qwen/Qwen3-0.6B"
HERE = Path(__file__).resolve().parent
E_R = 1e-6           # embedding entry of a readout wire (tiny: the input copy cancels in pairs anyway)
DELTA_NATS = 300.0   # logit shift of a firing readout at the median final-norm scale


def _snapshot(model_id=MODEL_ID):
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"])


def load_base(model_id=MODEL_ID):
    path = _snapshot(model_id)
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(load_file(f))
    cfg = json.loads(Path(path, "config.json").read_text())
    return {k: v.float() for k, v in sd.items()}, cfg, path


def build_circuit(tok, sd, cfg, calib, **kw):
    L = cfg["num_hidden_layers"]
    qk = {i: (sd[f"model.layers.{i}.self_attn.q_norm.weight"].double(),
              sd[f"model.layers.{i}.self_attn.k_norm.weight"].double()) for i in range(L)}
    return program.build(tok, calib, qk, sd["model.embed_tokens.weight"].shape[0], **kw)


def materialise(circ, sd, cfg, calib):
    d = cfg["hidden_size"]
    hd = cfg["head_dim"]
    nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    ff = cfg["intermediate_size"]
    nL = cfg["num_hidden_layers"]
    K = len(circ.wires)
    dn = d + K
    st = circ.stats()
    Hkv = st["max_kv_slots"]
    Hq = 2 * Hkv
    assert (nq + Hq) // (nkv + Hkv) == nq // nkv
    N = st["max_neurons"]
    M0 = circ.M0
    comp = math.sqrt(d / dn)
    idx = {w: d + i for i, w in circ.wires.items()} if False else {w: d + i for w, i in circ.wires.items()}

    out = {}
    V = sd["model.embed_tokens.weight"].shape[0]
    E = torch.zeros(V, dn)
    E[:, :d] = sd["model.embed_tokens.weight"]
    for w, vals in circ.emb.items():
        if not vals:
            continue
        ids = torch.tensor(list(vals.keys()))
        E[ids, idx[w]] = M0 * torch.tensor(list(vals.values()), dtype=torch.float32)
    for p, n, signs in circ.readouts:
        for t, s in signs.items():
            E[t, idx[p]] = E_R * s
            E[t, idx[n]] = E_R * s
    out["model.embed_tokens.weight"] = E

    for li in range(nL):
        pre = f"model.layers.{li}."
        g_in = 1.0 / (calib[f"in{li}"]["median"] * M0)
        g_post = 1.0 / (calib[f"post{li}"]["median"] * M0)
        # norms
        for name, g in (("input_layernorm", g_in), ("post_attention_layernorm", g_post)):
            wt = torch.empty(dn)
            wt[:d] = sd[pre + name + ".weight"] * comp
            wt[d:] = g
            out[pre + name + ".weight"] = wt
        out[pre + "self_attn.q_norm.weight"] = sd[pre + "self_attn.q_norm.weight"]
        out[pre + "self_attn.k_norm.weight"] = sd[pre + "self_attn.k_norm.weight"]
        # attention
        Wq = torch.zeros((nq + Hq) * hd, dn)
        Wk = torch.zeros((nkv + Hkv) * hd, dn)
        Wv = torch.zeros((nkv + Hkv) * hd, dn)
        Wo = torch.zeros(dn, (nq + Hq) * hd)
        Wq[: nq * hd, :d] = sd[pre + "self_attn.q_proj.weight"]
        Wk[: nkv * hd, :d] = sd[pre + "self_attn.k_proj.weight"]
        Wv[: nkv * hd, :d] = sd[pre + "self_attn.v_proj.weight"]
        Wo[:d, : nq * hd] = sd[pre + "self_attn.o_proj.weight"]
        kv_written = {}
        for h in circ.heads.get(li, []):
            qh = nq + 2 * h.kv_slot + h.q_slot
            kh = nkv + h.kv_slot
            for w, vec in h.q_terms.items():
                Wq[qh * hd:(qh + 1) * hd, idx[w]] += torch.as_tensor(vec, dtype=torch.float32)
            key = (h.kv_slot,)
            sig = (tuple(sorted((w, tuple(torch.as_tensor(v).tolist())) for w, v in h.k_terms.items())),
                   tuple(sorted((vd, tuple(sorted(t.items()))) for vd, t in h.v_terms.items())))
            if key in kv_written:
                assert kv_written[key] == sig, "heads sharing a KV slot must share k and v"
            else:
                kv_written[key] = sig
                for w, vec in h.k_terms.items():
                    Wk[kh * hd:(kh + 1) * hd, idx[w]] += torch.as_tensor(vec, dtype=torch.float32)
                for vd, terms in h.v_terms.items():
                    for w, cf in terms.items():
                        Wv[kh * hd + vd, idx[w]] += cf
            for vd, terms in h.out.items():
                for w, cf in terms.items():
                    Wo[idx[w], qh * hd + vd] += M0 * cf
        out[pre + "self_attn.q_proj.weight"] = Wq
        out[pre + "self_attn.k_proj.weight"] = Wk
        out[pre + "self_attn.v_proj.weight"] = Wv
        out[pre + "self_attn.o_proj.weight"] = Wo
        # MLP
        Wg = torch.zeros(ff + N, dn)
        Wu = torch.zeros(ff + N, dn)
        Wd = torch.zeros(dn, ff + N)
        Wg[:ff, :d] = sd[pre + "mlp.gate_proj.weight"]
        Wu[:ff, :d] = sd[pre + "mlp.up_proj.weight"]
        Wd[:d, :ff] = sd[pre + "mlp.down_proj.weight"]
        for j, nr in enumerate(circ.neurons.get(li, [])):
            r = ff + j
            for w, cf in nr.gate.items():
                Wg[r, idx[w]] += cf
            for w, cf in nr.up.items():
                Wu[r, idx[w]] += cf
            for w, cf in nr.down.items():
                Wd[idx[w], r] += M0 * cf
        out[pre + "mlp.gate_proj.weight"] = Wg
        out[pre + "mlp.up_proj.weight"] = Wu
        out[pre + "mlp.down_proj.weight"] = Wd

    fn = torch.zeros(dn)
    fn[:d] = sd["model.norm.weight"] * comp
    C_F = DELTA_NATS / (E_R * calib["final"]["median"] * M0)
    for p, n, _ in circ.readouts:
        fn[idx[p]] = C_F
        fn[idx[n]] = -C_F
    out["model.norm.weight"] = fn

    new_cfg = dict(cfg)
    new_cfg.update(hidden_size=dn, intermediate_size=ff + N, num_attention_heads=nq + Hq,
                   num_key_value_heads=nkv + Hkv, head_dim=hd, torch_dtype="float32", dtype="float32")
    new_cfg["alu_graft"] = {"base_model": MODEL_ID, "circuit_wires": K, "extra_heads": Hq,
                            "extra_kv_heads": Hkv, "extra_neurons": N, "M0": M0, "beta": circ.beta,
                            "readout_nats": DELTA_NATS, "notes": circ.notes}
    return out, new_cfg


def circuit_nnz(circ):
    """Non-zero parameters the circuit adds (what a sparse sidecar would store)."""
    n = sum(len(v) for v in circ.emb.values()) + 2 * sum(len(s) for _, _, s in circ.readouts)
    for hs in circ.heads.values():
        for h in hs:
            n += 128 * (len(h.q_terms) + len(h.k_terms))
            n += sum(len(t) for t in h.v_terms.values()) + sum(len(t) for t in h.out.values())
    for ns in circ.neurons.values():
        for nr in ns:
            n += len(nr.gate) + len(nr.up) + len(nr.down)
    return n


def graft(model_id=MODEL_ID, calib_path=HERE / "calibration.json", **kw):
    """Return (state_dict, config_dict, circuit) of the grafted model."""
    from transformers import AutoTokenizer
    sd, cfg, path = load_base(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    calib = json.loads(Path(calib_path).read_text())
    circ = build_circuit(tok, sd, cfg, calib, **kw)
    new_sd, new_cfg = materialise(circ, sd, cfg, calib)
    return new_sd, new_cfg, circ


def load_grafted(device="cpu", **kw):
    """Build the grafted model in memory (fp32) and return (model, tokenizer, circuit)."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    sd, cfg, circ = graft(**kw)
    c = {k: v for k, v in cfg.items() if k not in ("alu_graft", "architectures", "transformers_version")}
    config = Qwen3Config(**c)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]          # tied
    model.load_state_dict(sd, strict=True, assign=True)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    model.model.rotary_emb = Qwen3RotaryEmbedding(config=config)   # buffers were on meta
    model.tie_weights()
    model = model.to(device=device, dtype=torch.float32).eval()
    model.config.alu_graft = cfg["alu_graft"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tok, circ


def save(out_dir, **kw):
    from transformers import AutoTokenizer
    model, tok, circ = load_grafted(**kw)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return circ
