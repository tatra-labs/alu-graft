"""Compile the circuit into Qwen3-0.6B and save a standard Hugging Face checkpoint.

  python scripts/graft.py --out qwen3-0.6b-alu

The result loads with vanilla transformers (no trust_remote_code, no hooks):

  from transformers import AutoModelForCausalLM, AutoTokenizer
  m = AutoModelForCausalLM.from_pretrained("qwen3-0.6b-alu", dtype="float32")
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alu_graft.compile import save, circuit_nnz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qwen3-0.6b-alu")
    args = ap.parse_args()
    t0 = time.time()
    circ = save(args.out)
    info = circ.stats()
    info["circuit_nnz"] = circuit_nnz(circ)
    print(json.dumps(info, indent=1))
    print(f"saved {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
