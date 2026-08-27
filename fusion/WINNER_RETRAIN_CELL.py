# =============================================================================
# PASTE AS A SINGLE NEW CELL in fusion_benchmark.ipynb, AFTER the cell
# that defines C / build_splits / get_embeddings / DEVICE / SEED_UNIVERSE.
#
# WHY RETRAIN AT ALL
# ------------------
# The saved checkpoints are unusable as provenance. Every method tags its
# checkpoint `{method}_{encoder}_{seed}` -- no phase, no variant -- and
# CHECKPOINT_SINK is a dict, so each phase OVERWRITES the previous one under the
# same key. RUN_PLAN order is p1 -> p2 -> p3 -> p4, therefore every surviving
# file is from **p4_serboth_weighted**: the noise-trained configuration, which
# the benchmark itself showed does not work (0.5018 vs 0.5822 for the same
# method trained on oracle emotion). Deploying that file would ship the worst of
# the six measured cells.
#
# This cell retrains ONLY the winning configuration, keeps all three seeds,
# selects between them on VALIDATION macro-F1 (never test), and writes one file
# whose name and payload both say exactly what it is.
# =============================================================================

import json, time, hashlib
from pathlib import Path
import numpy as np
import torch

# --- NOT STANDALONE ---------------------------------------------------------
# This cell reuses the notebook's own loaders instead of reimplementing them,
# because a second copy of the split logic is a second thing that can drift out
# of sync with the one that produced the reported numbers. The price is that it
# must run in a kernel where the earlier cells have already run.
#
# Checking here turns a bare `NameError: name 'build_splits' is not defined`
# into a list of which cell to run first.
_needs = {
    "C": "the %%writefile cell that imports common as C",
    "build_splits": "the driver cell (defines build_splits / SEED_UNIVERSE)",
    "get_embeddings": "the driver cell",
    "get_token_embeddings": "the driver cell (token cache, attn needs it)",
    "DEVICE": "the setup cell",
    "CACHE_DIR": "the setup cell",
    "TOK_CACHE_DIR": "the setup cell",
    "RUNS_DIR": "the setup cell",
    "DATASET_PATH": "the setup cell",
}
_missing = {k: v for k, v in _needs.items() if k not in dir()}
if _missing:
    raise NameError(
        "This cell is not standalone. Missing: "
        + ", ".join(f"{k} (from {v})" for k, v in _missing.items())
        + ".\nRun the notebook's setup + %%writefile + driver cells first, then "
          "re-run this one. It deliberately does NOT redefine them: a second copy "
          "of the split logic could drift from the one that produced the results.")

try:
    from attn import run_intermediate_attn
except ImportError as _e:
    raise ImportError(
        f"{_e} -- the %%writefile cell has not written benchmark_modules to disk, "
        "or sys.path does not include them. Run that cell first.") from _e

# The winner, chosen on the DEPLOYMENT regime (real SER emotion at test time,
# class-weighted) rather than on the oracle table. `early` tops the oracle
# ranking at 0.5917 and then falls to 0.4810 the moment class weighting changes
# and 0.4610 under real SER; the demo runs under real SER.
WIN_METHOD   = "intermediate_attn"
WIN_ENCODER  = "bert"          # attn is bert-only; minilm raises in the encoder
WIN_VARIANT  = "full"          # more data; the deploy ranking was computed on it
WIN_REGIME   = "oracle"        # train on clean emotion...
WIN_WEIGHTED = True            # ...with class weighting  == phase p2
SEEDS        = [0, 1, 2]

# Where to put it. On Colab the mounted Drive path; falls back to the run dir
# with a loud message rather than silently writing somewhere nobody will look.
DRIVE_DIR = Path("/content/drive/MyDrive/CLEAR/Phase 1/fusion_winner")

# Measured in the benchmark for this exact cell of the grid. Carried into the
# payload so the demo can state its own expected accuracy without anyone having
# to find the CSV again.
BENCH_REFERENCE = {
    "p2_oracle_weighted/full":  0.5822,   # what this head was trained as
    "p3_sertest_weighted/full": 0.5259,   # same head, scored with real SER  <- demo
    "p4_serboth_weighted/full": 0.5018,   # trained on noise too; did not help
}

print(f"retraining {WIN_METHOD}/{WIN_ENCODER}/{WIN_VARIANT} "
      f"regime={WIN_REGIME} weighted={WIN_WEIGHTED} seeds={SEEDS}")

C.EMOTION_REGIME  = WIN_REGIME
C.CLASS_WEIGHTING = WIN_WEIGHTED
sink = {}
C.CHECKPOINT_SINK = sink

splits = build_splits(WIN_VARIANT)
emb    = get_embeddings(splits, WIN_ENCODER, DEVICE, cache_dir=CACHE_DIR)
# attn reads token-level features itself, but priming the cache here means the
# three seeds do not each pay for the encoder pass.
_ = get_token_embeddings(splits, WIN_ENCODER, DEVICE, cache_dir=TOK_CACHE_DIR)


per_seed = {}
for s in SEEDS:
    t0 = time.time()
    C.set_seed(s)
    y_true, y_pred = run_intermediate_attn(splits, emb, s, DEVICE, WIN_ENCODER)
    m = C.compute_metrics(y_true, y_pred)
    tag = f"{WIN_METHOD}_{WIN_ENCODER}_{s}"
    entry = sink.get(tag)
    if entry is None:
        raise RuntimeError(f"no checkpoint offered for {tag} -- C.CHECKPOINT_SINK "
                           "was not honoured; check that this cell set it before "
                           "calling the method")
    per_seed[s] = {"val_f1": float(entry["val_f1"]),
                   "test_macro_f1": float(m["macro_f1"]),
                   "test_acc": float(m["acc"]),
                   "state_dict": entry["state_dict"]}
    print(f"  seed {s}: val_f1 {entry['val_f1']:.4f} | test macro-F1 "
          f"{m['macro_f1']:.4f} acc {m['acc']:.4f}  ({time.time() - t0:.0f}s)")

# Selection on VALIDATION. Picking the best test score would be selecting on the
# number being reported, which is how a seed's good luck becomes a headline.
#
# (Checked before writing this: `intermediate_attn` is NOT in SWEPT_CORES -- only
# `late` and `intermediate` were hyperparameter-swept -- so the defaults used
# here are the same ones that produced the reported 0.5822. And `_run_attn_core`
# never reads its `emb` argument; it fetches token features itself from
# `C.TOKEN_CACHE_DIR`, so passing the pooled `emb` is equivalent to the driver
# passing `tok`.)
best_seed = max(per_seed, key=lambda s: per_seed[s]["val_f1"])
best = per_seed[best_seed]
vals = [per_seed[s]["test_macro_f1"] for s in SEEDS]
print(f"\nselected seed {best_seed} on val_f1={best['val_f1']:.4f} "
      f"(its test macro-F1 {best['test_macro_f1']:.4f}; "
      f"seed spread {min(vals):.4f}..{max(vals):.4f})")

# Deployment score for THIS EXACT head, not the benchmark's average.
#
# regime="ser_test" only noises the emotion feature of the TEST split, so
# training is bit-for-bit the run above: same seed, same oracle train/val
# emotion. Re-running it therefore re-derives the same weights and scores them
# the way the demo will see them. If val_f1 comes back different, training is
# not deterministic and the two numbers do not describe one model -- so that is
# checked rather than assumed.
print(f"\nre-running seed {best_seed} under regime=ser_test for the deployment score")
C.EMOTION_REGIME = "ser_test"
C.set_seed(best_seed)
_sink2 = {}
C.CHECKPOINT_SINK = _sink2
_yt, _yp = run_intermediate_attn(splits, emb, best_seed, DEVICE, WIN_ENCODER)
_m_deploy = C.compute_metrics(_yt, _yp)
_v2 = _sink2.get(f"{WIN_METHOD}_{WIN_ENCODER}_{best_seed}", {}).get("val_f1")
_deterministic = _v2 is not None and abs(_v2 - best["val_f1"]) < 1e-6
print(f"  deployment macro-F1 (real SER at test): {_m_deploy['macro_f1']:.4f} "
      f"| oracle test {best['test_macro_f1']:.4f} "
      f"| cost {_m_deploy['macro_f1'] - best['test_macro_f1']:+.4f}")
if not _deterministic:
    print(f"  !! val_f1 differs on re-run ({best['val_f1']:.4f} vs "
          f"{_v2 if _v2 is not None else 'n/a'}) -- training is NOT deterministic, "
          "so the deployment number above is a DIFFERENT model's score. Report it "
          "as indicative, not as this checkpoint's.")
C.EMOTION_REGIME = WIN_REGIME

payload = {
    "state_dict": best["state_dict"],
    "meta": {
        # Everything the filename used to leave ambiguous.
        "method": WIN_METHOD, "encoder": WIN_ENCODER, "variant": WIN_VARIANT,
        "regime": WIN_REGIME, "class_weighting": WIN_WEIGHTED,
        "phase_equivalent": "p2_oracle_weighted",
        "seed": best_seed, "seeds_trained": SEEDS,
        "selected_on": "validation macro-F1",
        "val_f1": best["val_f1"],
        "test_macro_f1": best["test_macro_f1"],
        "test_acc": best["test_acc"],
        "deployment_macro_f1_ser_test": float(_m_deploy["macro_f1"]),
        "deployment_measured_on_same_weights": bool(_deterministic),
        "per_seed": {s: {k: v for k, v in d.items() if k != "state_dict"}
                     for s, d in per_seed.items()},
        "benchmark_reference": BENCH_REFERENCE,
        "deployment_note": (
            "Trained on ORACLE emotion; the demo feeds it real SER output. The "
            "measured cost of that gap is -0.052 macro-F1 (0.5822 -> 0.5259), and "
            "training under simulated SER noise did NOT recover it (2 of 12 cells "
            "improved, mean -0.009) because the injected noise is "
            "label-independent and therefore irreducible."),
        "labels": getattr(C, "LABELS", ["normal", "borderline", "anomaly"]),
        "dataset": str(DATASET_PATH),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
}

name = f"WINNER_{WIN_METHOD}_{WIN_ENCODER}_{WIN_VARIANT}_p2_seed{best_seed}.pt"
local = Path(RUNS_DIR) / name
local.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, local)
sha = hashlib.sha1(local.read_bytes()).hexdigest()[:8]
print(f"\nwrote {local}  ({local.stat().st_size / 1024:.1f} KB, sha1:{sha})")

if DRIVE_DIR.parent.exists():
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    dst = DRIVE_DIR / name
    torch.save(payload, dst)
    (DRIVE_DIR / (name + ".json")).write_text(json.dumps(payload["meta"], indent=2,
                                                         default=str))
    print(f"mirrored to {dst}\n  + {name}.json (the meta alone, readable without torch)")
else:
    print(f"!! {DRIVE_DIR.parent} does not exist -- Drive is not mounted, so the "
          f"file exists ONLY at {local}. Mount Drive and re-run this cell, or copy "
          "it by hand; the benchmark output has already been lost once this way.")

C.CHECKPOINT_SINK = None   # back to no-op so later sweeps do not accumulate models
