# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "marimo",
#     "numpy<2.0.0",
#     "jiwer",
#     "pyctcdecode",
#     "pypi-kenlm",
#     # --- to pull from Drive, add the SAME package that provides gdrive_fsspec
#     #     in the overnight deps here, whatever it is named, e.g. "gdrivefs".
#     #     Not needed if you are not using Drive and upload the npz by hand.
# ]
# ///
#
# KenLM grid (Stage 2). NO TORCH. SINGLE FILE, no external modules.
#
# WHY A SEPARATE NOTEBOOK ON PYTHON 3.11:
#   the prebuilt Cython output of kenlm does not compile on Python 3.13 (removed C-APIs:
#   _PyGen_SetStopIterationValue, _PyLong_AsByteArray, _PyGC_FINALIZED). On 3.11 it
#   builds cleanly, no Cython needed, exactly as before. Because the header pins
#   <3.12, uv and molab open this notebook with 3.11.
#
#   Input: runs/<run>/dev_logits.npz produced by ablation_engine, read without torch.
#   molab: upload, enter the npz path, press "Run grid".

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="KenLM grid")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # KenLM beam decode plus alpha/beta grid (Stage 2)

        Reads `runs/<run>/dev_logits.npz`, produced by `dump_dev_logits(run)` inside
        `ablation_engine` (no torch needed), sweeps α/β and reports the best WER.

        > **Pinned to Python 3.11** (header `<3.12`): kenlm does not compile on 3.13; on 3.11 it
        > builds cleanly as before, no Cython needed. This notebook carries no torch, so it opens fast.
        """
    )
    return (mo,)


@app.cell
def _():
    import gzip
    import re
    import time
    import urllib.request
    from itertools import groupby
    from pathlib import Path

    import numpy as np

    LM_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
    CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

    def build_vocab():
        v = {c: i for i, c in enumerate(CHARS)}
        v["|"], v["[UNK]"], v["[PAD]"] = len(v), len(v) + 1, len(v) + 2
        return v

    def vocab_to_labels(vocab):
        # [PAD] -> '' is the single CTC blank; [UNK] -> '⁇' is unique (normalize() strips it).
        # Making both '' raises a "duplicate entries" error in pyctcdecode.
        labels = [""] * len(vocab)
        for tok, i in vocab.items():
            if tok == "|":
                labels[i] = " "
            elif tok == "[UNK]":
                labels[i] = "⁇"
            elif tok == "[PAD]":
                labels[i] = ""
            else:
                labels[i] = tok
        return labels

    # §7.8: reference AND hypothesis go through the same normalizer
    _NORM = re.compile(r"[^A-Z' ]+")

    def normalize(s):
        s = s.upper().replace("|", " ")
        return " ".join(_NORM.sub(" ", s).split())

    def greedy_decode(logits, vocab):
        blank, unk = vocab["[PAD]"], vocab["[UNK]"]
        i2c = {i: c for c, i in vocab.items()}
        ids = logits.argmax(-1)
        return "".join(i2c[k] for k, _ in groupby(ids.tolist())
                       if k not in (blank, unk)).replace("|", " ").strip()

    def load_npz(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        logits = z["logits"] if "logits" in z else z["arr_0"]
        refs = list(z["refs"]) if "refs" in z else list(z["texts"])
        return [np.asarray(l, np.float32) for l in logits], [str(r) for r in refs]

    def fetch_from_drive(run, local_path, remote_root="CLEAR/Phase 1/runs"):
        """Download runs/<run>/dev_logits.npz from Drive when it is not present locally.
        An isolated molab notebook has no token cache, so the FIRST call triggers headless
        OAuth: a URL is printed to the console, you approve it in the browser and paste the
        code back, and it is cached from then on."""
        p = Path(local_path)
        if p.exists():
            print(f"[drive] already present locally, download skipped: {p}")
            return str(p)
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem   # alternative module name
        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                   auth_kwargs={"use_local_webserver": False})
        remote = f"{remote_root}/{run}/dev_logits.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[drive] downloading: {remote} -> {p}")
        (fs.get_file if hasattr(fs, "get_file") else fs.get)(remote, str(p))
        print(f"[drive] downloaded ({p.stat().st_size/1e6:.0f} MB)")
        return str(p)

    def ensure_lm(path):
        p = Path(path)
        if p.exists():
            return str(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        gz = p.with_suffix(p.suffix + ".gz")
        print("[LM] downloading…")
        urllib.request.urlretrieve(LM_URL, gz)
        with gzip.open(gz, "rb") as f, open(p, "wb") as o:
            o.write(f.read())
        gz.unlink()
        return str(p)

    def run_grid(npz_path, lm_path, alphas, betas, beam=100):
        import jiwer
        from pyctcdecode import build_ctcdecoder

        logits, refs_raw = load_npz(npz_path)
        refs = [normalize(r) for r in refs_raw]
        vocab = build_vocab()
        labels = vocab_to_labels(vocab)

        g_hyp = [normalize(greedy_decode(l, vocab)) for l in logits]
        g_wer, g_cer = jiwer.wer(refs, g_hyp), jiwer.cer(refs, g_hyp)
        # W/C = WER divided by CER (Phase-1: augmentation does not change it, only the LM does)
        print(f"[GREEDY] WER {g_wer*100:.2f} · CER {g_cer*100:.2f} · W/C {g_wer/g_cer:.2f}")

        lm = ensure_lm(lm_path)
        rows, best = [], (1e9, None, None, None, None)   # wer, a, b, cer, ratio
        for a in alphas:
            for b in betas:
                dec = build_ctcdecoder(labels, kenlm_model_path=lm, alpha=a, beta=b)
                t0 = time.perf_counter()
                hyp = [normalize(dec.decode(l, beam_width=beam)) for l in logits]
                w, c = jiwer.wer(refs, hyp), jiwer.cer(refs, hyp)
                r = w / c if c else 0.0
                rows.append({"alpha": a, "beta": b, "wer": w, "cer": c, "wc_ratio": r})
                if w < best[0]:
                    best = (w, a, b, c, r)
                print(f"  α={a:.2f} β={b:.2f} | WER {w*100:.2f} CER {c*100:.2f} "
                      f"W/C {r:.2f} | {time.perf_counter()-t0:.0f}s{' *' if w == best[0] else ''}")
        print(f"[BEST] alpha={best[1]} beta={best[2]} -> WER {best[0]*100:.2f} - "
              f"CER {best[3]*100:.2f} · W/C {best[4]:.2f} "
              f"(greedy {g_wer*100:.2f}, gain {(g_wer-best[0])*100:.2f} points)")
        return {"greedy_wer": g_wer, "greedy_cer": g_cer,
                "best": {"alpha": best[1], "beta": best[2], "wer": best[0],
                         "cer": best[3], "wc_ratio": best[4]}, "grid": rows}

    return run_grid, fetch_from_drive, normalize, greedy_decode, build_vocab


@app.cell
def _(mo):
    mo.md(r"""## Settings""")
    return


@app.cell
def _(mo):
    run_ui = mo.ui.text(value="FINAL", label="run name -> runs/<run>/dev_logits.npz")
    fetch_ui = mo.ui.checkbox(
        value=True, label="Pull from Drive (isolated notebook, downloads when missing locally)")
    remote_ui = mo.ui.text(value="CLEAR/Phase 1/runs", full_width=True,
                           label="Drive remote root")
    lm_ui = mo.ui.text(value="lm_work/3-gram.pruned.1e-7.arpa", full_width=True,
                       label="LM path (downloaded from OpenSLR when missing)")
    alphas_ui = mo.ui.text(value="0.3,0.5,0.7,0.9", label="α (LM weight)")
    betas_ui = mo.ui.text(value="0.5,1.0,1.5,2.0", label="β (word bonus)")
    run_btn = mo.ui.run_button(label="Run the grid")
    mo.vstack([run_ui, fetch_ui, remote_ui, lm_ui, alphas_ui, betas_ui, run_btn,
               mo.md("*The first pull triggers Drive OAuth once (isolated environment, "
                     "no token cache) — just like the first time in the overnight run.*")])
    return run_ui, fetch_ui, remote_ui, lm_ui, alphas_ui, betas_ui, run_btn


@app.cell
def _(alphas_ui, betas_ui, fetch_from_drive, fetch_ui, lm_ui, mo, remote_ui,
      run_btn, run_grid, run_ui):
    if not run_btn.value:
        _out = mo.md("Enter the run name and press **Run the grid**.")
    else:
        _run = run_ui.value.strip()
        _npz = f"runs/{_run}/dev_logits.npz"
        _a = [float(x) for x in alphas_ui.value.split(",")]
        _b = [float(x) for x in betas_ui.value.split(",")]
        try:
            if fetch_ui.value:
                fetch_from_drive(_run, _npz, remote_ui.value.strip())
            _s = run_grid(_npz, lm_ui.value.strip(), _a, _b)
            _bt = _s["best"]
            _rows = "\n".join(
                f"| {r['alpha']} | {r['beta']} | {r['wer']*100:.2f} | "
                f"{r['cer']*100:.2f} | {r['wc_ratio']:.2f} |" for r in _s["grid"])
            _out = mo.md(
                f"### Result\n"
                f"- greedy: WER **{_s['greedy_wer']*100:.2f}** · CER "
                f"{_s['greedy_cer']*100:.2f} · W/C {_s['greedy_wer']/_s['greedy_cer']:.2f}\n"
                f"- **best: α={_bt['alpha']} β={_bt['beta']} → WER "
                f"{_bt['wer']*100:.2f} · CER {_bt['cer']*100:.2f} · W/C {_bt['wc_ratio']:.2f}** "
                f"(gain {(_s['greedy_wer']-_bt['wer'])*100:.2f} points)\n\n"
                f"| α | β | WER | CER | W/C |\n|---|---|---|---|---|\n{_rows}")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


if __name__ == "__main__":
    app.run()
