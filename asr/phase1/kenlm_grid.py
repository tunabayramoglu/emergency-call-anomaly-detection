# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "jiwer", "pyctcdecode", "kenlm"]
# ///
"""
Phase-1 — KenLM beam search decode + α/β grid.

TWO STAGES:
  1) in marimo (with ablation_engine loaded): dump_dev_logits(...) -> dev_logits.npz
     The backbone and head run once, writing each utterance's logits and reference.
  2) HERE (standalone): read the npz, build the KenLM decoder, sweep α/β.
     torch and the backbone are NOT needed, only numpy, pyctcdecode, kenlm and jiwer.

KenLM stayed off during the ablation. It is a fixed offset, it does not change the
ranking, and it costs decode time. It is enabled once, after the winner is chosen.

Usage:
    python kenlm_grid.py --selftest
    python kenlm_grid.py --grid --npz /marimo/runs/FINAL/dev_logits.npz \
        --lm /marimo/lm_work/3-gram.pruned.1e-7.arpa
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from itertools import groupby
from pathlib import Path

import numpy as np


# ---- SHARED NORMALIZER (§7.8) -------------------------------------------
# If the reference and the hypothesis do not go through the SAME transform, WER is
# inflated. The old benchmark used a "shared normalizer". Here a single function is
# applied to the reference and to every hypothesis, so jiwer only sees normalised strings.
_NORM_RE = re.compile(r"[^A-Z' ]+")


def normalize(s: str) -> str:
    """upper -> '|' becomes space -> keep only A-Z, apostrophe and space -> collapse spaces and strip."""
    s = s.upper().replace("|", " ")
    s = _NORM_RE.sub(" ", s)
    return " ".join(s.split())


# ---- AUTOMATIC GOOGLE DRIVE PUSH, EMBEDDED in this notebook (NO outside dependency) -
# Cross-script imports are not allowed (each notebook is uploaded on its own). put
# OVERWRITES, and each run gets its own subfolder. Skips silently without credentials or when ECAD_GDRIVE=0.

def push_gdrive(local_dir, phase, skip=("last.pt",), require_summary=True):
    local_dir = Path(local_dir)
    if os.environ.get("ECAD_GDRIVE", "1") == "0":
        return False
    if require_summary and not (local_dir / "summary.json").exists():
        print(f"[gdrive] run not finished ({local_dir.name}), skipped")
        return False
    try:
        try:
            from gdrivefs import GoogleDriveFileSystem
        except Exception:
            from gdrive_fsspec import GoogleDriveFileSystem
        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                   auth_kwargs={"use_local_webserver": False})
    except Exception as e:
        print(f"[gdrive] push skipped ({type(e).__name__})")
        return False
    root = f"CLEAR/Phase {phase}/runs/{local_dir.name}"
    files = [f for f in sorted(local_dir.rglob("*"))
             if f.is_file() and f.name not in set(skip)]
    ok = 0
    for f in files:
        remote = f"{root}/{f.relative_to(local_dir).as_posix()}"
        try:
            fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
            (fs.put_file if hasattr(fs, "put_file") else fs.put)(str(f), remote)
            ok += 1
        except Exception:
            pass
    print(f"[gdrive] {local_dir.name}: {ok}/{len(files)} -> {root}")
    return ok == len(files)

# ---- vocab, the SAME order as ablation_engine ---------------------------------
# 0-25 A-Z - 26 apostrophe - 27 | (word separator) - 28 [UNK] - 29 [PAD] (blank)
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")


def build_vocab():
    v = {c: i for i, c in enumerate(CHARS)}
    v["|"], v["[UNK]"], v["[PAD]"] = len(v), len(v) + 1, len(v) + 2
    return v


def vocab_to_labels(vocab):
    """The pyctcdecode label list. Index aligned, blank is '', word separator is ' '.

    | -> ' ' (this is how pyctcdecode expects the space)
    [PAD] -> '' (the single CTC blank; pyctcdecode does not allow duplicates)
    [UNK] -> '⁇' (a unique placeholder; rare, and normalize() strips it from ref and hyp)"""
    labels = [""] * len(vocab)
    for tok, i in vocab.items():
        if tok == "|":
            labels[i] = " "
        elif tok == "[PAD]":
            labels[i] = ""
        elif tok == "[UNK]":
            labels[i] = "⁇"          # normalize() drops anything outside [A-Z' ], so this cannot affect WER
        else:
            labels[i] = tok
    return labels


# ============================================================================
# GREEDY (numpy). A cross-check without pyctcdecode, verifies the npz is correct
# ============================================================================


def greedy_decode(logits, vocab):
    """logits [T, V] -> text. CTC collapse, then blank and unk are dropped."""
    blank, unk = vocab["[PAD]"], vocab["[UNK]"]
    i2c = {i: c for c, i in vocab.items()}
    ids = logits.argmax(-1)
    out = [i2c[k] for k, _ in groupby(ids.tolist()) if k not in (blank, unk)]
    return "".join(out).replace("|", " ").strip()


# ============================================================================
# LM — downloaded when missing (LibriSpeech 3-gram pruned, openslr.org/11)
# ============================================================================

LM_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"


def ensure_lm(path):
    p = Path(path)
    if p.exists():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    gz = p.with_suffix(p.suffix + ".gz")
    import gzip
    import urllib.request

    print(f"[LM] downloading -> {gz}")
    urllib.request.urlretrieve(LM_URL, gz)
    print("[LM] unpacking...")
    with gzip.open(gz, "rb") as f, open(p, "wb") as o:
        o.write(f.read())
    gz.unlink()
    print(f"[LM] ready -> {p}")
    return str(p)


# ============================================================================
# GRID - alpha (LM weight) x beta (word bonus)
# ============================================================================


def _load_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    # flexible key, different dump versions
    logits = z["logits"] if "logits" in z else z["arr_0"]
    refs = list(z["refs"]) if "refs" in z else list(z["texts"])
    logits = [np.asarray(l, np.float32) for l in logits]
    return logits, [str(r) for r in refs]


def run_grid(npz_path, lm_path, alphas, betas, beam_width=100, out_json=None):
    import jiwer
    from pyctcdecode import build_ctcdecoder

    logits, refs = _load_npz(npz_path)
    refs = [normalize(r) for r in refs]          # §7.8: put the reference through the same normalizer
    vocab = build_vocab()
    labels = vocab_to_labels(vocab)
    print(f"[GRID] {len(logits)} utterance · beam={beam_width}")

    # greedy reference (the baseline without an LM)
    g_hyp = [normalize(greedy_decode(l, vocab)) for l in logits]
    g_wer, g_cer = jiwer.wer(refs, g_hyp), jiwer.cer(refs, g_hyp)
    # W/C = WER divided by CER. Phase-1 finding: augmentation does not move it, only the LM does.
    print(f"[GREEDY] WER {g_wer*100:.2f} · CER {g_cer*100:.2f} · W/C {g_wer/g_cer:.2f}")

    lm = ensure_lm(lm_path)
    results = []
    best = (float("inf"), None, None, None, None)   # wer, a, b, cer, ratio
    for a in alphas:
        for b in betas:
            dec = build_ctcdecoder(labels, kenlm_model_path=lm, alpha=a, beta=b)
            t0 = time.perf_counter()
            hyp = [normalize(dec.decode(l, beam_width=beam_width)) for l in logits]
            wer, cer = jiwer.wer(refs, hyp), jiwer.cer(refs, hyp)
            ratio = wer / cer if cer else 0.0
            dt = time.perf_counter() - t0
            results.append({"alpha": a, "beta": b, "wer": wer, "cer": cer,
                            "wc_ratio": ratio, "secs": dt})
            flag = ""
            if wer < best[0]:
                best = (wer, a, b, cer, ratio)
                flag = " *"
            print(f"  α={a:.2f} β={b:.2f} | WER {wer*100:.2f} CER {cer*100:.2f} "
                  f"W/C {ratio:.2f} | {dt:.0f}s{flag}")

    bw, ba, bb, bc, br = best
    print(f"\n[BEST] alpha={ba} beta={bb} -> WER {bw*100:.2f} - CER {bc*100:.2f} - W/C {br:.2f} "
          f"(greedy {g_wer*100:.2f}, gain {(g_wer-bw)*100:.2f} points)")
    summary = {"greedy_wer": g_wer, "greedy_cer": g_cer, "greedy_wc": g_wer / g_cer,
               "best": {"alpha": ba, "beta": bb, "wer": bw, "cer": bc, "wc_ratio": br},
               "grid": results, "beam_width": beam_width, "n": len(logits)}
    if out_json:
        Path(out_json).write_text(json.dumps(summary, indent=2))
        print(f"[SAVED] {out_json}")
        # the grid result lands in the run folder, so push the same run to Drive (Phase 1)
        push_gdrive(Path(out_json).parent, phase=1, require_summary=False)
    return summary


# ============================================================================
# SELFTEST — numpy plus a fake decoder (pyctcdecode/kenlm not needed)
# ============================================================================


def selftest():
    ok = True

    def chk(n, c, e=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n} {e}")

    vocab = build_vocab()
    labels = vocab_to_labels(vocab)

    print("[vocab -> pyctcdecode labels]")
    chk("30 labels", len(labels) == 30, f"({len(labels)})")
    chk("A index 0", labels[0] == "A")
    chk("Z index 25", labels[25] == "Z")
    chk("apostrophe index 26", labels[26] == "'")
    chk("| -> space", labels[27] == " ")
    chk("[UNK] -> '⁇' (not blank)", labels[28] == "⁇")
    chk("[PAD] -> blank ''", labels[29] == "")
    chk("there is exactly one space", labels.count(" ") == 1)
    chk("there is exactly one blank ''", labels.count("") == 1)
    chk("labels are unique (a pyctcdecode requirement)", len(labels) == len(set(labels)))

    print("[greedy decode (CTC collapse)]")
    V = len(vocab)

    def onehot(seq):
        L = np.full((len(seq), V), -10.0, np.float32)
        for t, i in enumerate(seq):
            L[t, i] = 10.0
        return L

    P, U = vocab["[PAD]"], vocab["[UNK]"]
    # "CAT" = C A T, with a repeat and a blank in between
    seq = [vocab["C"], vocab["C"], P, vocab["A"], vocab["T"], vocab["T"]]
    chk("repeat plus blank collapse", greedy_decode(onehot(seq), vocab) == "CAT")
    # "A B" with the word separator
    seq2 = [vocab["A"], vocab["|"], vocab["B"]]
    chk("| -> space", greedy_decode(onehot(seq2), vocab) == "A B")
    # UNK is dropped
    seq3 = [vocab["H"], U, vocab["I"]]
    chk("[UNK] is dropped", greedy_decode(onehot(seq3), vocab) == "HI")
    chk("empty -> ''", greedy_decode(onehot([P, P]), vocab) == "")

    print("[npz loading (flexible key)]")
    import tempfile

    d = Path(tempfile.mkdtemp())
    L1, L2 = onehot(seq), onehot(seq2)
    np.savez(d / "a.npz", logits=np.array([L1, L2], dtype=object),
             refs=np.array(["CAT", "A B"]))
    lg, rf = _load_npz(d / "a.npz")
    chk("2 utterance", len(lg) == 2 and rf == ["CAT", "A B"])
    np.savez(d / "b.npz", arr_0=np.array([L1], dtype=object),
             texts=np.array(["CAT"]))
    lg2, rf2 = _load_npz(d / "b.npz")
    chk("old keys (arr_0/texts)", len(lg2) == 1 and rf2 == ["CAT"])

    print("[grid logic - fake decoder]")
    # mimic pyctcdecode, does it pick the best alpha/beta correctly?
    grid = [{"alpha": a, "beta": b, "wer": 0.10 - 0.01 * (a == 0.5) - 0.005 * (b == 1.0)}
            for a in (0.0, 0.5) for b in (0.0, 1.0)]
    best = min(grid, key=lambda r: r["wer"])
    chk("best is α=0.5 β=1.0", best["alpha"] == 0.5 and best["beta"] == 1.0,
        f"(WER {best['wer']:.3f})")

    print("[shared normalizer (§7.8)]")
    chk("lower -> UPPER", normalize("the cat") == "THE CAT")
    chk("punctuation is dropped", normalize("HELLO, WORLD!") == "HELLO WORLD")
    chk("multiple spaces collapse", normalize("A   B") == "A B")
    chk("| -> space plus strip", normalize(" A|B ") == "A B")
    chk("apostrophe is kept", normalize("don't") == "DON'T")
    chk("digits are dropped", normalize("ROOM 12") == "ROOM")
    chk("ref==hyp idempotent", normalize(normalize("Foo, bar")) == normalize("Foo, bar"))

    print("[LM URL]")
    chk("openslr 3-gram", "3-gram.pruned.1e-7" in LM_URL)

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--lm", default="/marimo/lm_work/3-gram.pruned.1e-7.arpa")
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--alphas", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--betas", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    if args.grid:
        if not args.npz:
            raise SystemExit("--npz is required (run dump_dev_logits in marimo first)")
        alphas = [float(x) for x in args.alphas.split(",")]
        betas = [float(x) for x in args.betas.split(",")]
        run_grid(args.npz, args.lm, alphas, betas, args.beam,
                 args.out or str(Path(args.npz).parent / "kenlm_grid.json"))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
