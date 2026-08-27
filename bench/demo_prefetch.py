# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers>=4.44", "huggingface-hub>=0.24", "torch"]
# ///
"""Pre-download everything the demo needs from HuggingFace, and check that
the project weights are where the demo expects them.

WHY PRE-FETCH
-------------
The demo loads three things from the network on first use: the shared
mHuBERT-147 backbone, the fusion text encoder, and (optionally) nothing else.
Doing that live, in front of an audience, on conference wifi, is the single
most avoidable way for a demo to fail. Run this once beforehand; afterwards the
demo works with the network unplugged.

WHAT IT DOES NOT DO
-------------------
It does not fetch the project's own weights. Those live in Google Drive
(`CLEAR/Phase 1/ASR-300/...`, `CLEAR/ser_runs/...`) and in the fusion benchmark
zip; download them by hand and point --weights at the folder. This script only
verifies they are present and loadable, so a missing file is found now rather
than during the demo.

Usage:
    python demo_prefetch.py --weights ./demo_weights
    python demo_prefetch.py --weights ./demo_weights --encoder bert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The ONE frozen backbone. Both the ASR adapter and the SER adapter read from
# it -- that sharing is the architectural claim of the project, and it also
# means this 380 MB download happens once, not twice.
BACKBONE = "utter-project/mHuBERT-147"

ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # ~90 MB, 384-dim
    "bert": "bert-base-uncased",                          # ~440 MB, 768-dim CLS
}

# What each component needs on disk, and where the demo will look for it.
EXPECTED = {
    "asr": ("config.json", "adapter.pt", "head.pt"),
    "ser": ("config.json", "adapter.pt", "head.pt"),
}


def fetch(repo: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        p = snapshot_download(repo)
        print(f"  OK   {repo}\n       -> {p}")
        return True
    except Exception as exc:
        print(f"  FAIL {repo}: {type(exc).__name__}: {exc}")
        return False


def check_weights(root: Path) -> bool:
    ok = True
    for part, names in EXPECTED.items():
        d = root / part
        if not d.is_dir():
            print(f"  MISSING  {d}/  -- create it and copy the {part.upper()} run's "
                  f"{', '.join(names)} into it")
            ok = False
            continue
        for n in names:
            f = d / n
            if not f.is_file():
                print(f"  MISSING  {f}")
                ok = False
            elif f.stat().st_size == 0:
                print(f"  EMPTY    {f}")
                ok = False
        cfg = d / "config.json"
        if cfg.is_file():
            try:
                c = json.loads(cfg.read_text())
                print(f"  OK   {part}: ws={c.get('ws')} lora={c.get('lora_layers')} "
                      f"r={c.get('lora_r')}")
            except Exception as exc:
                print(f"  BAD  {cfg}: not valid JSON ({exc})")
                ok = False

    # Optional but worth naming, because forgetting it silently changes results:
    # the demo should decode with the SAME tuned alpha/beta the results table
    # used, not pyctcdecode's defaults.
    lp = root / "asr" / "lm_params_clean.json"
    if lp.is_file():
        p = json.loads(lp.read_text())
        print(f"  OK   decoder params: alpha={p.get('alpha')} beta={p.get('beta')} "
              f"beam={p.get('beam_width')}")
    else:
        print(f"  note {lp} absent -- the demo will fall back to untuned decoder "
              "settings, which are not the ones the reported numbers used")

    # The shipped winner is named WINNER_*.pt; BEST_*.pt was the earlier
    # convention and setup_weights.py still copies both, so accept either.
    fdir = root / "fusion"
    fus = (sorted(fdir.glob("WINNER_*.pt")) + sorted(fdir.glob("BEST_*.pt"))) if fdir.is_dir() else []
    if fus:
        print(f"  OK   fusion checkpoint(s): {[f.name for f in fus]}")
    else:
        print(f"  MISSING  {root}/fusion/WINNER_*.pt -- run setup_weights.py, or "
              "take it from the fusion benchmark's checkpoints/ directory")
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="./demo_weights",
                    help="folder holding asr/, ser/ and fusion/ subfolders")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="minilm",
                    help="the text encoder the winning fusion config used")
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()

    print("1. HuggingFace models (cached in ~/.cache/huggingface, fetched once)")
    hf_ok = True
    if not args.skip_hf:
        hf_ok &= fetch(BACKBONE)
        hf_ok &= fetch(ENCODERS[args.encoder])
    else:
        print("  skipped")

    print(f"\n2. Project weights under {args.weights}")
    w_ok = check_weights(Path(args.weights))

    print()
    if hf_ok and w_ok:
        print("READY -- the demo can run offline.")
        sys.exit(0)
    print("NOT READY -- fix the lines above. Doing it now is cheap; doing it "
          "during the demo is not.")
    sys.exit(2)


if __name__ == "__main__":
    main()
