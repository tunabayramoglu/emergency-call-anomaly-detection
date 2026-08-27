# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers>=4.44",
#     "datasets==5.0.0",
#     "peft>=0.11",
#     "soundfile==0.14.0",
#     "huggingface-hub==1.24.0",
#     "numpy",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
#
# Phase-2 - ANOMALY SIGNAL test (Option 2b in the diagram). SINGLE FILE, no external modules.
#   molab : upload, run the cells in order, press "Run anomaly"
#   local : marimo edit anomaly_flag.py

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    app_title="Anomaly signal",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Phase-2 - Anomaly signal (Option 2b, no fusion, direct class mismatch)

        Goal: test the *anomaly flag* at the very end of the architecture end to end, **without training** a SER head.

        - **Voice emotion** ("how it is said") = the **ground-truth label** we already have (CREMA-D/RAVDESS).
        - **Text emotion** ("what is said") = the **FINAL ASR transcript** -> an HF text-emotion classifier.
        - **Anomaly** = voice_emotion ≠ text_emotion.

        > **Honesty note (for the report):** CREMA-D and RAVDESS use **fixed, neutral** sentences,
        > so text emotion is almost always neutral and the anomaly signal degenerates into a *"the voice is not neutral"* detector.
        > It **validates** the pipeline but does **not measure** the real discriminative power of the mismatch signal.
        > That is why the output also prints the number of unique transcripts and the text-emotion distribution.

        The ASR inference code (vocab, backbone, head, decode) is **identical** to `ablation_engine`, copied here.
        """
    )
    return (mo,)


@app.cell
def _():
    # ====================================================================
    # ENGINE - all constants and functions (one cell, no outside dependencies)
    # ====================================================================
    import io
    import json
    import os
    import time
    from itertools import groupby
    from pathlib import Path

    import numpy as np

    _ROOT = Path("/marimo") if Path("/marimo").exists() else Path.cwd()
    os.environ.setdefault("ECAD_OUT_ROOT", str(_ROOT / "runs_anomaly"))
    os.environ.setdefault("ECAD_BACKBONE", "utter-project/mHuBERT-147")
    OUT_ROOT = Path(os.environ["ECAD_OUT_ROOT"])
    BACKBONE = os.environ["ECAD_BACKBONE"]

    def log(*a):
        print(*a, flush=True)

    # ---- Google Drive push, embedded. Silent if there are no credentials or ECAD_GDRIVE=0 ----
    def push_gdrive(local_dir, phase=2, skip=("last.pt",), require_summary=True):
        local_dir = Path(local_dir)
        if os.environ.get("ECAD_GDRIVE", "1") == "0":
            return False
        if require_summary and not (local_dir / "summary.json").exists():
            log(f"[gdrive] run not finished ({local_dir.name}), skipped")
            return False
        try:
            try:
                from gdrivefs import GoogleDriveFileSystem
            except Exception:
                from gdrive_fsspec import GoogleDriveFileSystem
            fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                       auth_kwargs={"use_local_webserver": False})
        except Exception as e:
            log(f"[gdrive] push skipped ({type(e).__name__})")
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
        log(f"[gdrive] {local_dir.name}: {ok}/{len(files)} -> {root}")
        return ok == len(files)

    # ---- emotion labels + data (CREMA-D + RAVDESS) ----
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
    CREMAD_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fear",
                  "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
    RAVDESS_MAP = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
                   "05": "angry", "06": "fear", "07": "disgust"}   # 08=surprised is DROPPED
    CREMAD_REPOS = ["confit/cremad", "myleslinder/crema-d", "AbstractTTS/CREMA-D"]
    RAVDESS_REPOS = ["confit/ravdess", "narad/ravdess", "xbgoose/ravdess"]

    def label_from_cremad(fname):
        parts = Path(fname).stem.split("_")
        return CREMAD_MAP.get(parts[2]) if len(parts) >= 3 else None

    def label_from_ravdess(fname):
        parts = Path(fname).stem.split("-")
        return RAVDESS_MAP.get(parts[2]) if len(parts) >= 3 else None

    def fname_of(row):
        for k in ("file", "filename", "path", "audio_id", "id", "name"):
            v = row.get(k)
            if isinstance(v, str) and v:
                return v
        a = row.get("audio")
        if isinstance(a, dict) and a.get("path"):
            return a["path"]
        return ""

    def decode_audio(cell, sr):
        import soundfile as sf

        if isinstance(cell, dict) and cell.get("array") is not None:
            w, s = np.asarray(cell["array"], np.float32), cell.get("sampling_rate", sr)
        elif isinstance(cell, dict) and cell.get("bytes"):
            w, s = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif isinstance(cell, dict) and cell.get("path"):
            w, s = sf.read(cell["path"], dtype="float32")
        else:
            return None
        w = np.asarray(w, np.float32)
        if w.ndim > 1:
            w = w.mean(1)
        if int(s) != sr:
            w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr / s)),
                          np.arange(len(w)), w).astype(np.float32)
        return w

    def first_working(repos):
        from datasets import load_dataset

        for r in repos:
            try:
                ds = load_dataset(r, split="train", streaming=True)
                next(iter(ds))
                return r
            except Exception as e:
                log(f"[DATA] {r} did not work ({type(e).__name__})")
        return None

    def iter_clips(sr=16_000, max_secs=8.0, limit=None):
        from datasets import load_dataset

        maxlen = int(max_secs * sr)
        n = 0
        for repos, labeler, name in ((CREMAD_REPOS, label_from_cremad, "cremad"),
                                     (RAVDESS_REPOS, label_from_ravdess, "ravdess")):
            repo = first_working(repos)
            if not repo:
                log(f"[DATA] warning: could not download {name}, skipping")
                continue
            ds = load_dataset(repo, split="train", streaming=True)
            for row in ds:
                lab = labeler(fname_of(row))
                if lab is None:
                    continue
                w = decode_audio(row.get("audio"), sr)
                if w is None or len(w) < sr // 4:
                    continue
                yield lab, w[:maxlen]
                n += 1
                if limit and n >= limit:
                    return

    # ---- ASR inference, byte-for-byte the same as ablation_engine ----
    CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

    def build_vocab():
        v = {c: i for i, c in enumerate(CHARS)}
        v["|"], v["[UNK]"], v["[PAD]"] = len(v), len(v) + 1, len(v) + 2
        return v

    def ctc_decode(ids, i2c, blank, unk):
        return "".join(i2c.get(k, "") for k, _ in groupby(ids)
                       if k not in (blank, unk)).replace("|", " ").strip()

    def load_arch(final_dir):
        d = Path(final_dir)
        raw = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        ws = tuple(sorted(int(x) for x in raw.get("ws", (9, 10, 11, 12))))
        lora_layers = raw.get("lora_layers") or list(range(1, max(ws) + 1))
        return {"ws": ws, "lora_r": int(raw.get("lora_r", 16)),
                "lora_alpha": int(raw.get("lora_alpha", 32)),
                "hid": int(raw.get("hid", 768)),
                "lora_layers": tuple(int(i) for i in lora_layers)}

    def make_head():
        import torch
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self, n, dim, V, dropout=0.0, aux_idx=None):
                super().__init__()
                self.n, self.aux_idx = n, aux_idx
                self.layer_w = nn.Parameter(torch.zeros(n))
                self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(),
                                         nn.Dropout(dropout), nn.Linear(dim, V))
                self.aux = nn.Linear(dim, V) if aux_idx is not None else None

            def forward(self, x):
                w = self.layer_w.softmax(0)
                f = (x * w[None, None, :, None]).sum(2)
                a = self.aux(x[:, :, self.aux_idx, :]) if self.aux is not None else None
                return self.net(f), a

        return Head

    def load_asr(final_dir, dev):
        import torch
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import HubertModel

        d = Path(final_dir)
        arch = load_arch(d)
        vocab = build_vocab()
        try:
            bb = HubertModel.from_pretrained(BACKBONE, attn_implementation="sdpa")
        except Exception:
            bb = HubertModel.from_pretrained(BACKBONE)
        bb = bb.to(dev)
        lc = LoraConfig(r=arch["lora_r"], lora_alpha=arch["lora_alpha"],
                        target_modules=["q_proj", "v_proj"], bias="none",
                        layers_to_transform=[i - 1 for i in arch["lora_layers"]])
        bb = inject_adapter_in_model(lc, bb)
        if (d / "adapter.pt").exists():
            bb.load_state_dict(torch.load(d / "adapter.pt", map_location=dev), strict=False)
            log(f"[ASR] adapter loaded <- {d/'adapter.pt'}")
        else:
            log(f"[ASR] warning: no adapter.pt ({d}), bare backbone, the transcript will be broken")
        bb.eval()
        Head = make_head()
        head = Head(len(arch["ws"]), arch["hid"], len(vocab)).to(dev)
        sd = torch.load(d / "head.pt", map_location=dev)
        if "net.2.weight" in sd and sd["net.2.weight"].shape[0] != sd["net.0.weight"].shape[0]:
            sd["net.3.weight"] = sd.pop("net.2.weight")
            sd["net.3.bias"] = sd.pop("net.2.bias")
        head.load_state_dict(sd, strict=False)
        head.eval()
        return bb, head, bb._get_feat_extract_output_lengths, arch, vocab

    def transcribe_all(clips, final_dir, dev=None):
        import torch

        dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        bb, head, flen, arch, vocab = load_asr(final_dir, dev)
        blank, unk = vocab["[PAD]"], vocab["[UNK]"]
        i2c = {v: k for k, v in vocab.items()}
        ws = arch["ws"]
        out, t0 = [], time.perf_counter()
        with torch.no_grad():
            for i, (lab, w) in enumerate(clips):
                x = torch.from_numpy(np.asarray(w, np.float32))[None].to(dev)
                with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    o = bb(x, output_hidden_states=True)
                    hs = torch.stack([o.hidden_states[L] for L in ws], 2)
                logits = head(hs.float())[0]
                xl = int(flen(torch.tensor([x.shape[1]], device=dev))[0])
                ids = logits[0, :xl].argmax(-1).cpu().tolist()
                out.append((lab, ctc_decode(ids, i2c, blank, unk)))
                if (i + 1) % 500 == 0:
                    log(f"  [asr] {i+1} clips ({time.perf_counter()-t0:.0f}s)")
        log(f"[ASR] {len(out)} transcripts ({time.perf_counter()-t0:.0f}s)")
        return out

    # ---- text emotion (HF zero-shot) -> the 6 classes ----
    TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    TEXT2CLS = {"anger": "angry", "angry": "angry", "disgust": "disgust", "fear": "fear",
                  "joy": "happy", "happiness": "happy", "happy": "happy", "neutral": "neutral",
                  "sadness": "sad", "sad": "sad", "surprise": "neutral"}

    def map_text_emotion(raw):
        return TEXT2CLS.get(raw.strip().lower(), "neutral")

    def text_emotions(transcripts, model_name=TEXT_MODEL, dev=None):
        import torch
        from transformers import pipeline

        uniq = sorted({(t or "").strip() for t in transcripts})
        log(f"[TXT] {len(transcripts)} transcripts · {len(uniq)} unique "
            f"(fixed-sentence data, few unique transcripts expected)")
        device = 0 if (dev == "cuda" or (dev is None and torch.cuda.is_available())) else -1
        clf = pipeline("text-classification", model=model_name, top_k=1, device=device)
        cache = {}
        for t in uniq:
            if not t:
                cache[t] = "neutral"
                continue
            r = clf(t)[0]
            r = r[0] if isinstance(r, list) else r
            cache[t] = map_text_emotion(r["label"])
        return cache

    # ---- anomaly (Option 2b) + report ----
    def anomaly_report(pairs, text_map, run="anomaly_2b", text_model=TEXT_MODEL):
        n = len(pairs)
        records, mism = [], 0
        conf = {v: {t: 0 for t in EMOTIONS} for v in EMOTIONS}
        txt_dist = {e: 0 for e in EMOTIONS}
        voice_dist = {e: 0 for e in EMOTIONS}
        per_voice = {e: [0, 0] for e in EMOTIONS}
        for vlab, tr in pairs:
            tpred = text_map.get((tr or "").strip(), "neutral")
            flag = vlab != tpred
            mism += int(flag)
            conf[vlab][tpred] += 1
            txt_dist[tpred] += 1
            voice_dist[vlab] += 1
            per_voice[vlab][1] += 1
            per_voice[vlab][0] += int(flag)
            records.append({"voice": vlab, "text": tpred, "anomaly": flag, "transcript": tr})
        summary = {
            "run": run, "mode": "option_2b_direct_mismatch", "text_model": text_model,
            "n": n, "anomaly_rate": mism / n if n else 0.0,
            "agreement_rate": 1 - (mism / n) if n else 0.0,
            "n_unique_transcripts": len({(t or '').strip() for _, t in pairs}),
            "voice_distribution": voice_dist, "text_emotion_distribution": txt_dist,
            "per_voice_mismatch": {e: {"mismatch": m, "total": t, "rate": (m / t if t else 0.0)}
                                   for e, (m, t) in per_voice.items()},
            "confusion_voice_x_text": conf,
        }
        return summary, records

    def fetch_final_from_drive(final_dir, remote_root="CLEAR/Phase 1/runs"):
        """Download the FINAL checkpoint (adapter.pt + head.pt + config.json) from Drive
        when it is not present locally. An isolated notebook has no token cache, so the FIRST call
        triggers headless OAuth. A URL is printed, you approve it in the browser and paste the code back, then it is cached."""
        d = Path(final_dir)
        run = d.name
        need = [f for f in ("adapter.pt", "head.pt", "config.json")
                if not (d / f).exists()]
        if not need:
            log(f"[drive] FINAL is complete locally, download skipped: {d}")
            return
        try:
            try:
                from gdrive_fsspec import GoogleDriveFileSystem
            except Exception:
                from gdrivefs import GoogleDriveFileSystem
            fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                       auth_kwargs={"use_local_webserver": False})
        except Exception as e:
            log(f"[drive] no drive or credentials ({type(e).__name__}), download skipped")
            return
        d.mkdir(parents=True, exist_ok=True)
        for f in need:
            remote = f"{remote_root}/{run}/{f}"
            log(f"[drive] downloading: {remote}")
            try:
                (fs.get_file if hasattr(fs, "get_file") else fs.get)(remote, str(d / f))
            except Exception as e:
                log(f"[drive] ⚠ {f} did not download ({type(e).__name__})")

    def run_anomaly(final_dir, limit=None, run="anomaly_2b", text_model=TEXT_MODEL,
                    fetch=True, remote_root="CLEAR/Phase 1/runs"):
        if fetch:
            log(f"[0/3] checking/downloading the FINAL checkpoint from Drive…")
            fetch_final_from_drive(final_dir, remote_root)
        log(f"[1/3] loading clips (limit={limit})...")
        clips = list(iter_clips(limit=limit))
        log(f"[1/3] {len(clips)} clips.")
        if not clips:
            raise RuntimeError(
                "0 clips loaded. The likely cause is that the HF mirror rows carry no file name, "
                "so the label could not be derived, or there is no network access. Pick a reachable "
                "mirror from CREMAD_REPOS / RAVDESS_REPOS.")
        log("[2/3] ASR transcription…")
        pairs = transcribe_all(clips, final_dir)
        log("[3/3] text emotion + anomaly…")
        tmap = text_emotions([t for _, t in pairs], text_model)
        summary, records = anomaly_report(pairs, tmap, run, text_model)
        outdir = OUT_ROOT / run
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        push_gdrive(outdir, phase=2)
        log(f"[SAVED] {outdir}")
        return summary

    def report_md(s):
        lines = [f"### Anomaly report — `{s['run']}`",
                 f"- clips: **{s['n']}** · unique transcripts: **{s['n_unique_transcripts']}**",
                 f"- **ANOMALY rate: {s['anomaly_rate']*100:.1f}%** - agreement: {s['agreement_rate']*100:.1f}%",
                 "", "**Text-emotion distribution** (a neutral-heavy one is the fixed-sentence effect):", "",
                 "| emotion | count |", "|---|---|"]
        for e in EMOTIONS:
            lines.append(f"| {e} | {s['text_emotion_distribution'][e]} |")
        lines += ["", "**Anomaly rate by voice emotion:**", "",
                  "| voice emotion | anomaly/total | rate |", "|---|---|---|"]
        for e in EMOTIONS:
            pv = s["per_voice_mismatch"][e]
            lines.append(f"| {e} | {pv['mismatch']}/{pv['total']} | {pv['rate']*100:.1f}% |")
        return "\n".join(lines)

    return EMOTIONS, OUT_ROOT, build_vocab, ctc_decode, load_arch, map_text_emotion, \
        push_gdrive, report_md, run_anomaly, anomaly_report


@app.cell
def _(mo):
    mo.md(r"""## 1 · Quick verification (no GPU, no network)""")
    return


@app.cell
def _(EMOTIONS, anomaly_report, build_vocab, ctc_decode, load_arch,
      map_text_emotion, mo):
    def _selftest():
        ok, rows = True, []

        def chk(n, c):
            nonlocal ok
            ok &= bool(c)
            rows.append(f"| {'✅' if c else '❌'} | {n} |")

        v = build_vocab()
        i2c = {i: c for c, i in v.items()}
        chk("30 tokens", len(v) == 30)
        chk("repeat+blank collapse -> CAT",
            ctc_decode([v["C"], v["C"], v["[PAD]"], v["A"], v["T"], v["T"]],
                       i2c, v["[PAD]"], v["[UNK]"]) == "CAT")
        chk("| -> space", ctc_decode([v["A"], v["|"], v["B"]], i2c, v["[PAD]"], v["[UNK]"]) == "A B")
        chk("joy->happy", map_text_emotion("joy") == "happy")
        chk("surprise->neutral", map_text_emotion("surprise") == "neutral")
        chk("no config -> ws 9-12", load_arch("/does/not/exist")["ws"] == (9, 10, 11, 12))
        pairs = [("angry", "ITS ELEVEN OCLOCK"), ("neutral", "ITS ELEVEN OCLOCK"),
                 ("sad", "I AM SO HAPPY"), ("happy", "I AM SO HAPPY")]
        tmap = {"ITS ELEVEN OCLOCK": "neutral", "I AM SO HAPPY": "happy"}
        s, recs = anomaly_report(pairs, tmap)
        chk("2/4 anomaly", abs(s["anomaly_rate"] - 0.5) < 1e-9)
        chk("angry->neutral anomaly", recs[0]["anomaly"] is True)
        chk("neutral->neutral agreement", recs[1]["anomaly"] is False)
        return ok, rows

    _ok, _rows = _selftest()
    mo.md(("**SELFTEST PASSED**" if _ok else "**SELFTEST FAILED**")
          + "\n\n| | check |\n|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""## 2 · Settings and run""")
    return


@app.cell
def _(mo):
    final_dir_ui = mo.ui.text(value="runs/FINAL", full_width=True,
                              label="FINAL ASR run folder (adapter.pt + head.pt + config.json)")
    fetch_ui = mo.ui.checkbox(value=True,
                              label="Pull FINAL from Drive (isolated notebook, downloads when missing locally)")
    remote_ui = mo.ui.text(value="CLEAR/Phase 1/runs", full_width=True,
                           label="Drive remote root (FINAL lives here)")
    limit_ui = mo.ui.number(0, 25000, value=200,
                            label="clip limit (for a quick smoke test, 0 means all)")
    run_btn = mo.ui.run_button(label="Run anomaly")
    mo.vstack([final_dir_ui, fetch_ui, remote_ui, limit_ui, run_btn,
               mo.md("*The first pull triggers Drive OAuth once (isolated environment). "
                     "If 0 clips come back it is a mirror or file-name problem.*")])
    return final_dir_ui, fetch_ui, remote_ui, limit_ui, run_btn


@app.cell
def _(fetch_ui, final_dir_ui, limit_ui, mo, remote_ui, report_md, run_anomaly, run_btn):
    if not run_btn.value:
        _out = mo.md("Enter the settings and press **Run anomaly**.")
    else:
        _lim = int(limit_ui.value) or None
        try:
            _s = run_anomaly(final_dir_ui.value.strip(), _lim,
                             fetch=fetch_ui.value, remote_root=remote_ui.value.strip())
            _out = mo.md(report_md(_s))
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Google Drive (manual)

    When the run finishes `runs_anomaly/<run>/` is pushed automatically if credentials exist. To upload
    it again by hand, open the cell below.
    """)
    return


@app.cell
def _(OUT_ROOT, push_gdrive):
    # for _p in sorted(OUT_ROOT.glob("*/summary.json")):
    #     push_gdrive(_p.parent, phase=2)
    return


if __name__ == "__main__":
    app.run()
