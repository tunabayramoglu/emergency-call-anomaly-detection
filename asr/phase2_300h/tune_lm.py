# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch", "torchaudio", "transformers>=4.44", "peft>=0.11", "jiwer",
#     "numpy", "datasets==5.0.0", "soundfile==0.14.0",
#     "pyctcdecode", "kenlm",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchaudio = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
"""
ASR -- tune the CTC beam-search decoder (alpha, beta, beam_width).

WHY THIS EXISTS
---------------
eval_asr.py built its decoder with `alpha=0.5, beta=1.0` hardcoded and the
pyctcdecode default beam width. Nobody ever checked those numbers. They are the
LM weight and the word-insertion bonus, and they are the two most sensitive
knobs in the whole decode path -- more sensitive, at this scale, than upgrading
the 3-gram to a 4-gram.

They also cannot be shared blindly between the 100h and 300h models. alpha
trades the acoustic posterior against the LM; a better-trained acoustic model
produces sharper, better-calibrated posteriors and therefore wants a LOWER
alpha. Using one guessed value for both systems does not merely leave WER on the
table, it leaves a DIFFERENT amount on the table for each row of the results
table, which is worse than a fair comparison at a suboptimal point.

PROTOCOL -- the part that keeps the numbers honest
--------------------------------------------------
`--tune-on other` (default) fits on LibriSpeech dev-other, which is NOT reported.
`--tune-on clean` fits on dev-clean, which IS reported -- and which is exactly
what the 100h baseline did. Run both: the clean row is like-for-like with the
published 5.1%, the other row is what the model is worth without peeking.

ONE parameter set is chosen and applied to BOTH reported columns. Tuning
separately per test set would give a better-looking L2-ARCTIC number that means
nothing, because at deployment there is no oracle telling you which domain the
call came from. The expected consequence -- that LibriSpeech-tuned parameters
are slightly wrong for accented speech -- is part of the finding, not a bug.

Every model gets its own tuning run: `--run-dir` is what identifies the system,
and the output json records it, so a params file can never be silently reused
for the wrong checkpoint.

COST
----
The acoustic forward pass runs ONCE. Log-probs are cached in RAM and every grid
point re-decodes those same arrays, so the grid costs CPU beam search only --
`decoder.reset_params()` mutates the existing decoder instead of reloading the
LM, which would otherwise dominate the runtime.

Usage:
    python tune_lm.py --run-dir /marimo/runs/FINAL_300h \
        --lm /marimo/lm/3-gram.pruned.1e-7.arpa --n 500 --out lm_params.json

    # apply the same protocol to the 100h baseline so both rows match:
    python tune_lm.py --run-dir /marimo/runs/baseline_100h --lm ... \
        --out lm_params_100h.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_asr import (_decode_audio_array, _no_decode, greedy_decode,
                      load_our_model, log, sync_to_drive, wer_normalize)

# TWO tuning sets, because the 100h baseline and good practice disagree and the
# honest answer is to report both.
#
#   "other" -- dev-other. NOT a reported set, so parameters fitted here are not
#              fitted to anything we publish. Methodologically clean.
#   "clean" -- dev-clean. This is what the BASELINE did: ablation_engine.py's
#              `dump_dev_logits` exports dev-clean logits and kenlm_grid.py
#              --grid sweeps alpha/beta over them, and the published 5.1% WER is
#              dev-clean. So the baseline's headline number was produced with the
#              decoder tuned on the set it reports.
#
# Tuning only on dev-other and then comparing our dev-clean number against the
# baseline's would pit a clean number against a flattered one and UNDERSTATE the
# 300h model. Running both settles it: the "clean" row is like-for-like with the
# baseline, the "other" row is what the model is worth without peeking, and the
# gap between them measures how much the baseline's protocol flattered it.
TUNE_SPLITS = {
    "other": ("openslr/librispeech_asr", "other", "validation"),
    "clean": ("openslr/librispeech_asr", "clean", "validation"),
}

# Coarse grid. alpha 0 = ignore the LM entirely, which is the greedy-equivalent
# anchor and a useful sanity row: if the best alpha comes out at 0 the LM is
# hurting and something upstream (vocab mismatch, wrong normalisation) is wrong.
ALPHA_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
BETA_GRID = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
BEAM_GRID = [50, 100, 200]


def load_tuning_rows(limit: int, which: str = "other") -> list[dict]:
    from datasets import load_dataset

    repo, conf, split = TUNE_SPLITS[which]
    # decode=False, for the same reason eval_asr.py needs it: datasets v5 hands
    # Audio decoding to torchcodec, which cannot load on this stack.
    ds = _no_decode(load_dataset(repo, conf, split=split))
    n = min(limit, len(ds)) if limit else len(ds)
    # Fixed stride rather than the first n rows: LibriSpeech splits are ordered
    # by speaker, so rows[:500] would be a handful of speakers and the tuned
    # alpha would be fitted to their voices.
    step = max(1, len(ds) // n)
    idx = list(range(0, len(ds), step))[:n]
    rows = [{"audio": ds[i]["audio"], "text": ds[i]["text"]} for i in idx]
    log(f"[tune] {len(rows)} utterances from {repo}/{conf}/{split} "
        f"(every {step}th of {len(ds)}, so speakers are not clustered)")
    return rows


def compute_logprobs(rows, run_dir: Path, device: str):
    """Run the acoustic model ONCE; return (log-prob arrays, refs, greedy hyps)."""
    import torch

    bb, head, ws, vocab = load_our_model(run_dir, device)
    lps, refs, greedy = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, r in enumerate(rows):
            w = _decode_audio_array(r["audio"])
            X = torch.from_numpy(w).unsqueeze(0).to(device)
            am = torch.ones_like(X, dtype=torch.long)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                o = bb(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([o.hidden_states[L] for L in ws], 2)
                logits = head(hs.float())[0]
            lp = logits.log_softmax(-1).float().cpu().numpy()
            lps.append(lp)
            refs.append(wer_normalize(r["text"]))
            greedy.append(wer_normalize(greedy_decode(logits.log_softmax(-1).cpu(), vocab)))
            if (i + 1) % 100 == 0:
                log(f"  [tune] logits {i + 1}/{len(rows)} "
                    f"({time.perf_counter() - t0:.0f}s)")
    log(f"[tune] acoustic pass done in {time.perf_counter() - t0:.0f}s -- the grid "
        f"below re-uses these arrays and never touches the GPU again")
    return lps, refs, greedy, vocab


def vocab_to_labels(v: dict) -> list[str]:
    labels = [""] * len(v)
    for tok, i in v.items():
        if tok == "|":
            labels[i] = " "
        elif tok == "[PAD]":
            labels[i] = ""
        elif tok == "[UNK]":
            labels[i] = "?"
        else:
            labels[i] = tok
    return labels


def _wer_with_se(refs, hyps) -> tuple[float, float]:
    """WER plus a bootstrap-free standard error over utterances.

    Without an error bar a grid search happily reports a 0.1-point "win" that is
    pure sampling noise on 500 utterances, and that fake win then gets locked
    into the reported numbers. The SE here is over per-utterance error counts,
    normalised by total reference length -- the standard cheap approximation.
    """
    import jiwer

    overall = jiwer.wer(refs, hyps)
    per, lens = [], []
    for r, h in zip(refs, hyps):
        nw = max(1, len(r.split()))
        m = jiwer.process_words([r], [h])
        per.append(m.substitutions + m.deletions + m.insertions)
        lens.append(nw)
    per, lens = np.asarray(per, float), np.asarray(lens, float)
    n = len(per)
    # delta method on the ratio of sums
    tot_e, tot_l = per.sum(), lens.sum()
    var = (np.var(per - overall * lens, ddof=1) * n) / (tot_l ** 2) if n > 1 else 0.0
    return overall, math.sqrt(max(var, 0.0))


def tune(lps, refs, greedy, vocab, lm_path: str, jobs: int) -> dict:
    import multiprocessing
    from pyctcdecode import build_ctcdecoder

    g_wer, g_se = _wer_with_se(refs, greedy)
    log(f"[tune] greedy reference point: WER {100 * g_wer:.2f}% (+/- {100 * g_se:.2f})")

    # pyctcdecode only WARNS when the kenlm bindings are missing, then fails deep
    # inside build_ctcdecoder with `NameError: name 'kenlm' is not defined`, which
    # names neither the real problem nor the fix. Check here instead.
    try:
        import kenlm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"KenLM python bindings are not installed ({exc}).\n"
            "  pip install kenlm\n"
            "  pip install https://github.com/kpu/kenlm/archive/master.zip   # if the above has no wheel\n"
            "Without them pyctcdecode cannot load an ARPA file. The greedy column "
            "of the results table does not need KenLM, so eval_asr.py can still be "
            "run with --lm omitted."
        ) from exc

    log(f"[tune] loading KenLM once: {lm_path}")
    t0 = time.perf_counter()
    decoder = build_ctcdecoder(vocab_to_labels(vocab), kenlm_model_path=lm_path,
                               alpha=ALPHA_GRID[0], beta=BETA_GRID[0])
    log(f"[tune] LM loaded in {time.perf_counter() - t0:.0f}s")

    results = []
    def run(alpha, beta, beam):
        """Decode the cached log-probs at one grid point.

        THE POOL IS CREATED HERE, NOT ONCE OUTSIDE THE LOOP.

        `reset_params` mutates the decoder in the PARENT. pyctcdecode keeps the
        loaded LanguageModel in a class-level cache and, under a `fork` pool, the
        children inherit whatever that cache held AT FORK TIME -- so a pool forked
        before the loop decodes every grid point with the FIRST alpha/beta, and
        all 70 rows come back byte-identical. That is exactly what happened:
        70 points, one WER, 15.30% for every combination.

        Forking after the reset costs one fork per grid point. Fork is
        copy-on-write, so the ~1 GB of loaded LM is not duplicated, and the LM is
        still loaded only ONCE overall -- which was the reason for hoisting the
        pool in the first place.
        """
        decoder.reset_params(alpha=alpha, beta=beta)
        if jobs > 1:
            with multiprocessing.get_context("fork").Pool(jobs) as _pool:
                hyps = decoder.decode_batch(_pool, lps, beam_width=beam)
        else:
            hyps = [decoder.decode(lp, beam_width=beam) for lp in lps]
        return [wer_normalize(h) for h in hyps]

    # Stage 1: alpha x beta at a single mid beam. Beam width interacts far
    # more weakly with alpha/beta than they do with each other, so searching
    # the full 3-D product would triple the cost for almost no information.
    base_beam = BEAM_GRID[len(BEAM_GRID) // 2]
    total = len(ALPHA_GRID) * len(BETA_GRID)
    k = 0
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            k += 1
            t1 = time.perf_counter()
            hyps = run(alpha, beta, base_beam)
            wer, se = _wer_with_se(refs, hyps)
            results.append({"alpha": alpha, "beta": beta, "beam": base_beam,
                            "wer": wer, "wer_se": se,
                            "secs": time.perf_counter() - t1})
            log(f"  [{k:>3}/{total}] alpha {alpha:>4} beta {beta:>5} beam {base_beam} "
                f"-> WER {100 * wer:.2f}% (+/- {100 * se:.2f})")

    best = min(results, key=lambda r: r["wer"])

    # Stage 1b: REFINE around the coarse winner.
    #
    # Only meaningful when the tuning set can resolve the difference. On n=500 the
    # coarse grid put FOURTEEN points within one SE of the best (spread 0.35 pts,
    # SE 0.50) -- a finer grid there picks whichever point the noise favoured, and
    # that choice does not transfer to dev-clean. So the refinement runs, but its
    # winner must beat the coarse winner by more than 1 SE to be adopted. Raise
    # --n to make it bite: SE scales as 1/sqrt(n), so n=1500 gives ~0.29.
    _cw = min(results, key=lambda r: r["wer"])
    _fa = [round(_cw["alpha"] + d, 3) for d in (-0.1, -0.05, 0.05, 0.1)
           if _cw["alpha"] + d > 0]
    _fb = [round(_cw["beta"] + d, 3) for d in (-0.25, 0.25)]
    log(f"  [refine] around alpha={_cw['alpha']} beta={_cw['beta']}: "
        f"alpha={_fa} x beta={_fb}")
    _done = {(r["alpha"], r["beta"]) for r in results if r["beam"] == base_beam}
    for alpha in _fa + [_cw["alpha"]]:
        for beta in _fb + [_cw["beta"]]:
            if (alpha, beta) in _done:
                continue
            _done.add((alpha, beta))
            t1 = time.perf_counter()
            hyps = run(alpha, beta, base_beam)
            wer, se = _wer_with_se(refs, hyps)
            results.append({"alpha": alpha, "beta": beta, "beam": base_beam,
                            "wer": wer, "wer_se": se, "stage": "refine",
                            "secs": time.perf_counter() - t1})
            log(f"  [refine] alpha {alpha:>5} beta {beta:>5} -> "
                f"WER {100 * wer:.2f}% (+/- {100 * se:.2f})")

    _rw = min(results, key=lambda r: r["wer"])
    if _rw.get("stage") == "refine" and _rw["wer"] > _cw["wer"] - _cw["wer_se"]:
        log(f"  [refine] refined best {100 * _rw['wer']:.2f}% does not beat the "
            f"coarse winner {100 * _cw['wer']:.2f}% by more than 1 SE "
            f"({100 * _cw['wer_se']:.2f}) -- keeping the coarse point. A smaller "
            "number here would be noise, not skill, and would not transfer.")
        results = [r for r in results if r.get("stage") != "refine"]

    # Stage 2: beam width at the winning alpha/beta.
    for beam in BEAM_GRID:
        if beam == base_beam:
            continue
        t1 = time.perf_counter()
        hyps = run(best["alpha"], best["beta"], beam)
        wer, se = _wer_with_se(refs, hyps)
        results.append({"alpha": best["alpha"], "beta": best["beta"], "beam": beam,
                        "wer": wer, "wer_se": se, "secs": time.perf_counter() - t1})
        log(f"  [beam] alpha {best['alpha']} beta {best['beta']} beam {beam:>4} "
            f"-> WER {100 * wer:.2f}% (+/- {100 * se:.2f})")


    # A grid where nothing moves is not a flat optimum, it is a broken sweep --
    # and it is invisible unless something checks. This is the detector for the
    # bug above (and any future one that silently ignores the parameters).
    _uniq = {round(r["wer"], 6) for r in results}
    if len(_uniq) == 1 and len(results) > 1:
        raise SystemExit(
            f"All {len(results)} grid points returned the SAME WER "
            f"({100 * results[0]['wer']:.2f}%). alpha and beta are not reaching the "
            "decoder, so this is a bug, not a flat surface. Most likely cause: a "
            "multiprocessing pool forked BEFORE reset_params, whose children keep "
            "pyctcdecode's cached LanguageModel from fork time. Re-run with "
            "--jobs 1 to confirm."
        )

    # Beam width is NOT chosen by plain argmin. Decode cost grows with the beam,
    # so a wider beam has to earn its keep: pick the smallest beam whose WER is
    # within 1 SE of the best beam. Without this rule a 0.1-point difference that
    # is pure sampling noise buys a permanently slower decoder -- and in the
    # synthetic check of this function the argmin did exactly that, picking a
    # beam whose apparent win came from noise.
    ab_best = min(results, key=lambda r: r["wer"])
    same_ab = [r for r in results
               if r["alpha"] == ab_best["alpha"] and r["beta"] == ab_best["beta"]]
    beam_best = min(same_ab, key=lambda r: r["wer"])
    cheap = [r for r in sorted(same_ab, key=lambda r: r["beam"])
             if r["wer"] <= beam_best["wer"] + beam_best["wer_se"]]
    best = cheap[0] if cheap else beam_best
    if best["beam"] != beam_best["beam"]:
        log(f"  [beam] beam {beam_best['beam']} was nominally best "
            f"({100 * beam_best['wer']:.2f}%) but beam {best['beam']} is within 1 SE "
            f"({100 * best['wer']:.2f}%) -- taking the cheaper one")

    # A best point sitting on the edge of the grid means the grid was too narrow
    # and the true optimum is outside it. Saying so is the difference between a
    # tuned decoder and a decoder that merely ran a loop.
    warnings = []
    if best["alpha"] in (ALPHA_GRID[0], ALPHA_GRID[-1]):
        warnings.append(f"best alpha {best['alpha']} is on the grid BOUNDARY "
                        f"({ALPHA_GRID[0]}..{ALPHA_GRID[-1]}) -- widen ALPHA_GRID")
    if best["beta"] in (BETA_GRID[0], BETA_GRID[-1]):
        warnings.append(f"best beta {best['beta']} is on the grid BOUNDARY "
                        f"({BETA_GRID[0]}..{BETA_GRID[-1]}) -- widen BETA_GRID")
    if best["beam"] == BEAM_GRID[-1]:
        warnings.append(f"best beam {best['beam']} is the largest tried -- a wider "
                        "beam may still help, at linear decode cost")
    if best["alpha"] == 0.0:
        warnings.append("best alpha is 0, i.e. the LM is NOT helping at all. Suspect a "
                        "vocab/normalisation mismatch between the CTC labels and the "
                        "LM's training text before accepting this result")

    # How much of the reported gain survives the error bar?
    margin = g_wer - best["wer"]
    combined_se = math.sqrt(g_se ** 2 + best["wer_se"] ** 2)
    significant = margin > 2 * combined_se

    return {"best": best, "greedy": {"wer": g_wer, "wer_se": g_se},
            "grid": results, "warnings": warnings,
            "gain_vs_greedy": margin,
            "gain_is_significant_2se": bool(significant),
            "alpha_grid": ALPHA_GRID, "beta_grid": BETA_GRID, "beam_grid": BEAM_GRID}


def _selfstamp() -> str:
    """Fingerprint of THIS file, printed at startup.

    The notebook writes these modules from embedded blobs. If the module cell has
    not been re-run, the script on disk is an older version than the notebook --
    and the only symptom is a traceback whose line numbers do not match the code
    you are reading, which is a genuinely confusing way to lose ten minutes. The
    module cell prints the same hashes after writing; if they differ, re-run it.
    """
    import hashlib

    p = Path(__file__).resolve()
    h = hashlib.sha1(p.read_bytes()).hexdigest()[:8]
    return f"{p.name} sha1:{h} mtime:{time.strftime('%H:%M:%S', time.localtime(p.stat().st_mtime))}"


def main() -> None:
    log(f"[src] {_selfstamp()}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--n", type=int, default=500,
                    help="tuning utterances; more = less noise, linearly slower")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tune-on", choices=sorted(TUNE_SPLITS), default="other",
                    help="'other' = dev-other, not a reported set (clean protocol); "
                         "'clean' = dev-clean, matching what the 100h baseline did")
    ap.add_argument("--out", default="lm_params.json")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)

    rows = load_tuning_rows(args.n, args.tune_on)
    if args.tune_on == "clean":
        log("[tune] !! tuning on dev-clean, which eval_asr.py also REPORTS. This "
            "matches the 100h baseline's protocol (kenlm_grid.py swept alpha/beta "
            "over dev-clean logits and 5.1% is a dev-clean number), so the two rows "
            "stay like-for-like -- but the resulting dev-clean WER is optimistic "
            "for BOTH systems and must be labelled as such.")
    lps, refs, greedy, vocab = compute_logprobs(rows, run_dir, device)
    res = tune(lps, refs, greedy, vocab, args.lm, args.jobs)

    b = res["best"]
    out = {
        # Provenance: these params belong to ONE checkpoint and ONE LM. Recording
        # both is what stops a params file being reused for the wrong system.
        "run_dir": str(run_dir),
        "lm_path": str(args.lm),
        "tuned_on": "/".join(TUNE_SPLITS[args.tune_on]),
        "tuned_on_n": len(rows),
        "tuned_on_a_reported_set": args.tune_on == "clean",
        "protocol": ("matches the 100h baseline (tuned on the reported set)"
                     if args.tune_on == "clean" else
                     "strict (tuned on dev-other, which is never reported)"),
        "alpha": b["alpha"], "beta": b["beta"], "beam_width": b["beam"],
        "tune_wer": b["wer"], "tune_wer_se": b["wer_se"],
        "greedy_wer": res["greedy"]["wer"],
        "gain_vs_greedy": res["gain_vs_greedy"],
        "gain_is_significant_2se": res["gain_is_significant_2se"],
        "warnings": res["warnings"],
        "grid": res["grid"],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    log("")
    log("=" * 68)
    log(f"BEST  alpha={b['alpha']}  beta={b['beta']}  beam={b['beam']}")
    log(f"      tuning WER {100 * b['wer']:.2f}% vs greedy "
        f"{100 * res['greedy']['wer']:.2f}%  "
        f"(gain {100 * res['gain_vs_greedy']:.2f} points, "
        f"{'SIGNIFICANT' if res['gain_is_significant_2se'] else 'WITHIN NOISE'} at 2 SE)")
    log(f"      old hardcoded values were alpha=0.5 beta=1.0 beam=100")
    for w in res["warnings"]:
        log(f"  !!  {w}")
    log(f"written to {args.out}")
    log("=" * 68)
    log(f"These numbers are the TUNING set ({args.tune_on}). They are not a result "
        "-- the result is dev-clean / L2-ARCTIC in eval_asr.py using these params.")

    # kenlm aborts the process during teardown:
    #   util/mmap.cc:138 SyncOrThrow ... Cannot allocate memory / Failed to sync mmap
    #   Fatal Python error: Aborted   (while garbage-collecting)
    # It happens AFTER the results are written, but it still returns a non-zero
    # exit code, and the caller cannot tell "crashed on exit" from "crashed before
    # producing anything" -- which is why the second tuning run never started.
    # Everything is flushed and on disk here, so skip the destructors rather than
    # let a teardown bug masquerade as a real failure.
    # Mirror BEFORE os._exit: molab is ephemeral and this json is the only record
    # of which decoder produced the reported numbers.
    sync_to_drive([args.out], run_dir.name)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
