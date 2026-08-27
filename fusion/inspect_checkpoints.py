"""List the fusion benchmark's saved checkpoints and say which one to deploy.

The benchmark tags every trained model
`{phase}/{variant}/{encoder}/{method}/seed{n}` and saves it as
`checkpoints/{tag}.pt`. Because the tag contains slashes, that is a NESTED
directory tree, not flat filenames -- so "BEST_intermediate_attn is missing" is
usually a question about the layout, not about a missing file.

Two things this settles:

  * WHICH files exist, with the val macro-F1 and the config recorded inside each
    payload -- rather than inferring the config from a path.
  * WHICH to deploy. The benchmark's own results table ranks configs under
    ORACLE emotion; the demo runs on real SER output, so the ranking that
    matters is the one measured under the noisy regime. Picking the oracle
    winner would deploy `early`, which drops 0.11 macro-F1 the moment class
    weighting changes and a further 0.13 under real SER.

Usage:
    python inspect_checkpoints.py                       # searches ./runs
    python inspect_checkpoints.py --root path/to/run_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured on the FULL variant, class-weighted, with simulated real SER emotion
# at test time (phase p3) -- i.e. the regime the demo actually runs in.
DEPLOY_RANKING = [
    ("intermediate_attn", "bert", 0.5259),
    ("intermediate_film", "minilm", 0.5186),
    ("intermediate", "bert", 0.5163),
    ("intermediate", "minilm", 0.5098),
    ("intermediate_attn", "minilm", 0.5083),
]


def load_payload(p: Path):
    try:
        import torch

        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--deep", action="store_true",
                    help="open every .pt to read its recorded meta (slower)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Point --root at the run directory "
                         "(the one containing results_table.csv and checkpoints/).")

    pts = sorted(root.rglob("*.pt"))
    if not pts:
        raise SystemExit(
            f"No .pt under {root}. If the benchmark reported "
            "'checkpoints_saved: 0' in manifest.json, the save step failed -- most "
            "likely FileNotFoundError, because the tag contains slashes and "
            "torch.save does not create parent directories. In that case the heads "
            "must be retrained; they are small, so a single-config rerun is minutes, "
            "not hours.")

    print(f"{len(pts)} checkpoint file(s) under {root}\n")
    rows = []
    for p in pts:
        rel = p.relative_to(root)
        meta, val = {}, None
        if args.deep:
            pay = load_payload(p)
            if "_error" in pay:
                print(f"  UNREADABLE {rel}: {pay['_error']}")
                continue
            meta = pay.get("meta", {}) or {}
            val = pay.get("val_f1")
        rows.append((str(rel), p.stat().st_size, meta, val))

    for rel, size, meta, val in rows:
        extra = ""
        if meta:
            extra = (f"  method={meta.get('method')} encoder={meta.get('encoder')}"
                     f" variant={meta.get('variant')}")
        if val is not None:
            extra += f" val_f1={val:.4f}"
        print(f"  {size / 1024:8.1f} KB  {rel}{extra}")

    print("\nDeploy ranking measured under the REAL-SER regime (p3, weighted, full):")
    for m, e, f1 in DEPLOY_RANKING:
        hit = [r for r, *_ in rows if m in r and e in r]
        mark = "  <- present" if hit else "  (not found here)"
        print(f"  {m}/{e}: {f1:.4f}{mark}")
    print("\nTake the highest one that is present. Do NOT take `early` even though "
          "it tops the oracle table: 0.5917 oracle -> 0.4810 once class weighting "
          "changes -> 0.4610 under real SER. The demo runs under real SER.")


if __name__ == "__main__":
    main()
