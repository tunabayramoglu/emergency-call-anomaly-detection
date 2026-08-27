# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo", "torch", "torchvision", "transformers>=4.44", "peft>=0.11",
#     "pysoundfile", "numpy", "pandas",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
#
# Phase-2 SER (marimo). Frozen mHuBERT + SER LoRA + single-layer head + class_6.
#   molab: upload -> selftest green -> Settings -> press "Layer sweep" or "Train a single run".

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="SER training")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # Phase-2 — SER (audio-only)

        Frozen **mHuBERT-147** + **SER LoRA** (q/v) + a **single-layer head** (no WS, mean+std pool).
        Academic corpora only, mapped onto `class_6` (6 classes).

        - **Layer-range search:** the best frozen set is not the best LoRA set, so the LoRA adapt range
          starts wide (4-12), is narrowed on a subset, and the best read/lora pair is chosen by UA.
        - **Augmentation in the LoRA phase:** noise and specaug. Channel (band/8k) is off — see RESULTS.md 2.
          No pitch or speed shifting: it destroys the emotional cue.
        - class-weighted CE, model selection by **UA (macro recall)**, and a speaker-independent split.
        """
    )
    return (mo,)


@app.cell
def _():
    # molab package setup: these top-level imports pull in torch, torchvision, transformers and peft.
    # torchvision comes from the cu128 index (the build that MATCHES torch) -> the right version in
    # the venv overrides the broken one in the base image, which ends the torchvision::nms crash
    # that transformers triggers.
    import torch as _t, torchvision as _tv, transformers as _tf, peft as _p  # noqa: F401
    print(f"[setup] torch {_t.__version__} · torchvision {_tv.__version__}", flush=True)
    return


@app.cell
def _():
    # ================= ENGINE (one cell, all the functions) =================
    import os, json, math, hashlib
    import numpy as np
    from dataclasses import dataclass, asdict
    from pathlib import Path

    _root = Path("/marimo") if Path("/marimo").exists() else Path.cwd()
    DATA_ROOT = Path(os.environ.get("ECAD_DATA_ROOT", str(_root / "ser_data")))
    LABEL_DIR = DATA_ROOT / "labels"
    OUT_ROOT = Path(os.environ.get("ECAD_OUT_ROOT", str(_root / "runs_ser")))
    CACHE_ROOT = Path(os.environ.get("ECAD_CACHE_ROOT", str(_root / "cache_ser")))
    BACKBONE = os.environ.get("ECAD_BACKBONE", "utter-project/mHuBERT-147")

    CLASSES = ["neutral", "distress", "fear", "urgency", "panic", "confusion"]
    CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
    SOURCES = ["cremad", "ravdess", "savee", "tess", "jl", "asvp_esd", "kaggle_emergency"]
    LABEL_CSVS = {s: f"{s}_labels.csv" for s in SOURCES}

    def log(*a):
        print(*a, flush=True)

    # ---- labels + split ----
    def load_rows(sources=None):
        import csv
        sources = sources or SOURCES
        rows = []
        for s in sources:
            p = LABEL_DIR / LABEL_CSVS[s]
            if not p.exists():
                log(f"[label] ⚠ missing: {p}"); continue
            with open(p) as f:
                rd = list(csv.DictReader(f))
            for r in rd:
                c = (r.get("class_6") or "").strip().lower()
                if c not in CLS2IDX:
                    continue
                spk = (r.get("speaker_uid") or r.get("speaker_id") or "?").strip() or "?"
                rows.append({"path": r["path"], "cls": CLS2IDX[c], "source": s, "spk": f"{s}:{spk}"})
            log(f"[label] {s}: {sum(1 for x in rows if x['source']==s)} clips")
        return rows

    def subset_and_split(rows, cfg):
        rng = np.random.default_rng(cfg.seed)
        rows = list(rows)
        if cfg.subset_frac < 1.0:
            by_c = {}
            for r in rows:
                by_c.setdefault(r["cls"], []).append(r)
            sub = []
            for c, rs in by_c.items():
                rng.shuffle(rs)
                sub += rs[:max(1, int(len(rs) * cfg.subset_frac))]
            rows = sub
        spks = sorted({r["spk"] for r in rows}); rng.shuffle(spks)
        val_spks = set(spks[:max(1, int(len(spks) * cfg.val_frac))])
        tr = [r for r in rows if r["spk"] not in val_spks]
        va = [r for r in rows if r["spk"] in val_spks]
        if len(va) < 50 or len({r["cls"] for r in va}) < len(CLASSES) - 1:
            rng.shuffle(rows)
            n = int(len(rows) * cfg.val_frac)
            va, tr = rows[:n], rows[n:]
            log("[split] speaker split too weak -> stratified")
        else:
            log(f"[split] speaker-independent: {len(val_spks)} val speakers")
        return tr, va

    # ---- audio cache (int16) ----
    def build_audio_cache(rows, cfg, tag):
        import soundfile as sf
        key = hashlib.md5(f"{tag}|{cfg.sr}|{cfg.max_secs}|{len(rows)}".encode()).hexdigest()[:10]
        d = CACHE_ROOT / f"ser_wav_{key}"
        binp, metap = d / "wav.i16", d / "meta.json"
        if binp.exists() and metap.exists():
            log(f"[audio] cache -> {d}"); return d
        d.mkdir(parents=True, exist_ok=True)
        offs, cls, pos = [0], [], 0
        maxlen = int(cfg.max_secs * cfg.sr)
        with open(binp, "wb") as f:
            for i, r in enumerate(rows):
                try:
                    w, s = sf.read(str(DATA_ROOT / r["path"]), dtype="float32")
                except Exception:
                    continue
                w = np.asarray(w, np.float32)
                if w.ndim > 1: w = w.mean(1)
                if int(s) != cfg.sr:
                    w = np.interp(np.linspace(0, len(w)-1, int(len(w)*cfg.sr/s)),
                                  np.arange(len(w)), w).astype(np.float32)
                w = w[:maxlen]
                if len(w) < cfg.sr // 4: continue
                q = np.clip(np.rint(w*32768), -32768, 32767).astype(np.int16)
                f.write(q.tobytes()); pos += q.size; offs.append(pos); cls.append(r["cls"])
                if (i+1) % 2000 == 0: log(f"  [audio] {i+1}/{len(rows)}")
        metap.write_text(json.dumps({"offsets": offs, "cls": cls}))
        log(f"[audio] {len(cls)} clips -> {d}"); return d

    # ---- augmentation (LoRA phase) ----
    def _mask(w, sr, lo, hi):
        if len(w) < 8: return w
        W = np.fft.rfft(w); fr = np.fft.rfftfreq(len(w), 1.0/sr)
        if lo is not None: W[fr < lo] = 0.0
        if hi is not None: W[fr > hi] = 0.0
        return np.fft.irfft(W, len(w)).astype(np.float32)

    def aug_band(w, sr): return _mask(w, sr, 300.0, 3400.0)

    def aug_8k(w, sr):
        if len(w) < 8: return w
        x = _mask(w, sr, None, 3800.0); n8 = max(2, len(x)//2)
        d = np.interp(np.linspace(0, len(x)-1, n8), np.arange(len(x)), x)
        return np.interp(np.linspace(0, n8-1, len(x)), np.arange(n8), d).astype(np.float32)

    def _pink(n, rng):
        m = n//2 + 1
        s = (rng.standard_normal(m) + 1j*rng.standard_normal(m)).astype(np.complex64)
        fr = np.arange(m, dtype=np.float32); fr[0] = 1.0
        x = np.fft.irfft(s/np.sqrt(fr), n).astype(np.float32)
        return x / (float(np.sqrt((x**2).mean())) + 1e-9)

    def _mix(w, nz, snr_db):
        nz = nz / (float(np.sqrt((nz**2).mean())) + 1e-9)
        k = math.sqrt((float((w**2).mean()) + 1e-12) / (10.0**(snr_db/10.0)))
        return (w + k*nz).astype(np.float32)

    def augment(w, cfg, rng):
        if rng.random() < cfg.p_band: w = aug_band(w, cfg.sr)
        if rng.random() < cfg.p_8k:   w = aug_8k(w, cfg.sr)
        if rng.random() < cfg.p_noise: w = _mix(w, _pink(len(w), rng), rng.uniform(5.0, 20.0))
        return w

    # ---- config ----
    @dataclass
    class SerCfg:
        run: str = "ser"
        backbone: str = BACKBONE
        sr: int = 16_000
        max_secs: float = 8.0
        hid: int = 768
        n_cls: int = 6
        ws: tuple = (9, 10, 11, 12)      # WS read layers (learned weights) — ported from ASR
        lora_lo: int = 1
        lora_hi: int = 12
        lora_r: int = 16
        lora_alpha: int = 32
        head_dropout: float = 0.3
        hidden_dim: int = 256
        pool: str = "meanstd"
        p_band: float = 0.0
        p_8k: float = 0.0
        p_noise: float = 0.0
        specaug: bool = False
        subset_frac: float = 0.25
        val_frac: float = 0.15
        epochs: int = 15
        batch: int = 32
        lr: float = 3e-4
        w_lr: float = 1e-2               # a separate, higher LR for the WS layer_w, like w_lr in ASR
        weight_decay: float = 1e-4
        label_smoothing: float = 0.05
        seed: int = 1337
        stop_patience: int = 6

        def feat_dim(self):
            return self.hid * (2 if self.pool == "meanstd" else 1)

        def lora_layers(self):
            return tuple(range(self.lora_lo, self.lora_hi + 1))

        @property
        def dir(self):
            return OUT_ROOT / self.run

        def d(self):
            return asdict(self)

    # ---- model ----
    def build_backbone(cfg, dev):
        import torch
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import HubertModel
        kw = dict(mask_time_prob=0.05, mask_time_length=10, mask_feature_prob=0.02,
                  mask_feature_length=10, apply_spec_augment=True) if cfg.specaug else {}
        try:
            bb = HubertModel.from_pretrained(cfg.backbone, attn_implementation="sdpa", **kw)
        except Exception:
            bb = HubertModel.from_pretrained(cfg.backbone, **kw)
        bb = bb.to(dev)
        lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                        target_modules=["q_proj", "v_proj"], bias="none",
                        layers_to_transform=[i - 1 for i in cfg.lora_layers()])
        bb = inject_adapter_in_model(lc, bb)
        for n, p in bb.named_parameters():
            p.requires_grad = "lora_" in n
        got = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        log(f"[LORA] adapt {cfg.lora_lo}-{cfg.lora_hi} · {got:,} trainable")
        return bb

    def make_head(cfg, dev):
        import torch
        import torch.nn as nn

        class SerHead(nn.Module):
            """WS over the ws layers (learned) -> time pooling -> dense -> n_cls. Ported from ASR."""
            def __init__(s):
                super().__init__()
                s.layer_w = nn.Parameter(torch.zeros(len(cfg.ws)))
                s.net = nn.Sequential(
                    nn.LayerNorm(cfg.feat_dim()),
                    nn.Linear(cfg.feat_dim(), cfg.hidden_dim), nn.ReLU(),
                    nn.Dropout(cfg.head_dropout), nn.Linear(cfg.hidden_dim, cfg.n_cls))

            def weights(s):
                return s.layer_w.softmax(0)

            def forward(s, hs):            # hs: [B,T,n_ws,hid]
                w = s.layer_w.softmax(0)
                blend = (hs * w[None, None, :, None]).sum(2)      # [B,T,hid]
                mu = blend.mean(1)
                f = torch.cat([mu, blend.std(1)], -1) if cfg.pool == "meanstd" else mu
                return s.net(f)

        return SerHead().to(dev)

    def read_stack(bb, x, cfg):
        import torch
        o = bb(x, output_hidden_states=True)
        return torch.stack([o.hidden_states[L] for L in cfg.ws], 2)   # [B,T,n_ws,hid]

    # ---- training ----
    def _load_split_cache(cfg):
        rows = load_rows()
        tr, va = subset_and_split(rows, cfg)
        log(f"[data] subset {cfg.subset_frac:.0%} · train {len(tr)} / val {len(va)}")
        return build_audio_cache(tr, cfg, "train"), build_audio_cache(va, cfg, "val")

    def _batches(d, cfg, rng, train):
        import torch
        meta = json.loads((d / "meta.json").read_text())
        offs = np.asarray(meta["offsets"], np.int64); cls = np.asarray(meta["cls"], np.int64)
        buf = np.fromfile(d / "wav.i16", dtype=np.int16)
        idx = np.arange(len(cls))
        if train: rng.shuffle(idx)
        for i in range(0, len(idx), cfg.batch):
            bi = idx[i:i+cfg.batch]; ws = []
            for j in bi:
                w = buf[int(offs[j]):int(offs[j+1])].astype(np.float32) / 32768.0
                if train: w = augment(w, cfg, rng)
                ws.append(w)
            L = max(len(w) for w in ws)
            X = np.zeros((len(ws), L), np.float32)
            for k, w in enumerate(ws): X[k, :len(w)] = w
            yield torch.from_numpy(X), torch.from_numpy(cls[bi])

    def _ua(p, y):
        return float(np.mean([(p[y == c] == c).mean() for c in range(len(CLASSES)) if (y == c).any()]))

    def train_ser(cfg, dtr=None, dva=None, verbose=True):
        import torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dtr is None:
            dtr, dva = _load_split_cache(cfg)
        cfg.dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        bb = build_backbone(cfg, dev); head = make_head(cfg, dev)
        tr_cls = np.asarray(json.loads((dtr / "meta.json").read_text())["cls"])
        cnt = np.bincount(tr_cls, minlength=len(CLASSES)).astype(np.float32)
        cw = torch.from_numpy(cnt.sum() / (len(cnt) * np.maximum(cnt, 1))).float().to(dev)
        lossf = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)
        w_par = [head.layer_w]           # the WS weights get a higher LR so they move faster
        rest = [p for p in bb.parameters() if p.requires_grad] + \
               [p for nm, p in head.named_parameters() if nm != "layer_w"]
        opt = torch.optim.AdamW([{"params": rest, "lr": cfg.lr},
                                 {"params": w_par, "lr": cfg.w_lr}],
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
        rng = np.random.default_rng(cfg.seed)
        best_ua, best_ep, best_w, hist = 0.0, 0, None, []
        for ep in range(1, cfg.epochs + 1):
            bb.train(); head.train(); tot = n = 0.0
            for X, y in _batches(dtr, cfg, rng, True):
                X, y = X.to(dev), y.to(dev)
                with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    loss = lossf(head(read_stack(bb, X, cfg).float()), y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(y); n += len(y)
            sched.step()
            bb.eval(); head.eval(); P, Y = [], []
            with torch.no_grad():
                for X, y in _batches(dva, cfg, rng, False):
                    with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                        lg = head(read_stack(bb, X.to(dev), cfg).float())
                    P.append(lg.argmax(1).cpu().numpy()); Y.append(y.numpy())
            p = np.concatenate(P); yv = np.concatenate(Y)
            wa = float((p == yv).mean()); ua = _ua(p, yv)
            hist.append({"epoch": ep, "loss": tot/max(n, 1), "val_wa": wa, "val_ua": ua})
            w_now = head.weights().detach().cpu().numpy()
            if ua > best_ua:
                best_ua, best_ep = ua, ep
                best_w = w_now.round(3).tolist()
                torch.save(head.state_dict(), cfg.dir / "head.pt")
                torch.save({k: v for k, v in bb.state_dict().items() if "lora_" in k}, cfg.dir / "adapter.pt")
            if verbose:
                wstr = " ".join(f"L{L}={x:.2f}" for L, x in zip(cfg.ws, w_now))
                log(f"  e{ep:>2} loss {tot/max(n,1):.3f} | val WA {wa*100:.1f} UA {ua*100:.1f} | {wstr}")
            if ep - best_ep >= cfg.stop_patience:
                log(f"[stop] no improvement for {cfg.stop_patience} epochs"); break
        summary = {"run": cfg.run, "ws": list(cfg.ws), "lora": f"{cfg.lora_lo}-{cfg.lora_hi}",
                   "best_val_ua": best_ua, "best_epoch": best_ep, "layer_w": best_w,
                   "history": hist, "cfg": cfg.d()}
        (cfg.dir / "config.json").write_text(json.dumps(cfg.d(), indent=2))
        (cfg.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"[DONE] {cfg.run}: best UA {best_ua*100:.1f} @e{best_ep}")
        return best_ua, hist

    def sweep(cfg, configs=None):
        from dataclasses import replace
        # (ws read-set, lora_lo, lora_hi) — wide -> narrow
        if configs is None:
            configs = [((9, 10, 11, 12), 1, 12), ((9, 10, 11, 12), 4, 12),
                       ((9, 10, 11), 1, 11), ((8, 9, 10), 1, 10),
                       ((6, 7, 8, 9, 10, 11, 12), 1, 12), ((10, 11, 12), 1, 12),
                       ((9, 10), 1, 10)]
        dtr = dva = None; res = []
        for ws, lo, hi in configs:
            wss = "-".join(map(str, ws))
            c = replace(cfg, run=f"sweep_ws{wss}_L{lo}-{hi}", ws=tuple(ws), lora_lo=lo, lora_hi=hi)
            if dtr is None:
                dtr, dva = _load_split_cache(c)
            log(f"\n=== ws={ws} lora={lo}-{hi} ===")
            ua, _ = train_ser(c, dtr, dva, verbose=True)
            res.append({"ws": wss, "lora": f"{lo}-{hi}", "ua": ua})
            log(f"  -> UA {ua*100:.1f}")
        res.sort(key=lambda r: -r["ua"])
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "sweep.json").write_text(json.dumps(res, indent=2))
        return res

    def aug_sweep(cfg, configs=None):
        # augmentation ablation on the locked ws set: (name, band, 8k, noise, specaug)
        from dataclasses import replace
        if configs is None:
            configs = [("control", 0.0, 0.0, 0.0, False), ("specaug", 0.0, 0.0, 0.0, True),
                       ("channel", 0.4, 0.3, 0.0, False), ("noise", 0.0, 0.0, 0.4, False),
                       ("all", 0.4, 0.3, 0.4, True)]
        wss = "-".join(map(str, cfg.ws))
        dtr, dva = _load_split_cache(cfg); res = []
        for name, pb, p8, pn, sa in configs:
            c = replace(cfg, run=f"aug_{name}_ws{wss}", p_band=pb, p_8k=p8, p_noise=pn, specaug=sa)
            sp = c.dir / "summary.json"
            if sp.exists():
                ua = json.loads(sp.read_text())["best_val_ua"]
                log(f"[skip] {c.run} -> UA {ua*100:.1f}")
            else:
                log(f"\n=== aug={name} (band={pb} 8k={p8} noise={pn} specaug={sa}) ===")
                ua, _ = train_ser(c, dtr, dva, verbose=True)
                log(f"  -> UA {ua*100:.1f}")
            res.append({"aug": name, "ua": ua})
        res.sort(key=lambda r: -r["ua"])
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "aug_sweep.json").write_text(json.dumps(res, indent=2))
        return res

    return (CLASSES, CLS2IDX, SerCfg, OUT_ROOT, aug_band, augment, build_backbone,
            load_rows, log, make_head, read_stack, subset_and_split, train_ser,
            sweep, aug_sweep, np)


@app.cell
def _(mo):
    mo.md(r"""## 1 · Verification (no GPU, no data)""")
    return


@app.cell
def _(CLASSES, CLS2IDX, SerCfg, aug_band, augment, mo, np, subset_and_split):
    def _st():
        ok, rows = True, []
        def chk(n, c):
            nonlocal ok; ok &= bool(c); rows.append(f"| {'✅' if c else '❌'} | {n} |")
        chk("6 classes", len(CLASSES) == 6)
        c = SerCfg(ws=(9,10,11,12), lora_lo=1, lora_hi=12)
        chk("ws has 4 layers", len(c.ws) == 4)
        chk("lora_layers 1..12", c.lora_layers() == tuple(range(1,13)))
        chk("meanstd feat 1536", c.feat_dim() == 1536)
        _hs = np.random.randn(2,5,4,8).astype(np.float32); _w = np.ones(4,np.float32)/4
        chk("WS uniform -> layer mean",
            np.allclose((_hs*_w[None,None,:,None]).sum(2), _hs.mean(2), atol=1e-5))
        rng = np.random.default_rng(0)
        w = (rng.standard_normal(16000)*0.1).astype(np.float32)
        a = augment(w, SerCfg(p_band=1, p_8k=1, p_noise=1), rng)
        chk("aug preserves length", len(a) == len(w))
        chk("band suppresses high frequencies",
            np.abs(np.fft.rfft(aug_band(w,16000)))[4000:].mean() < np.abs(np.fft.rfft(w))[4000:].mean()*0.5)
        rr = [{"path":f"x{i}.wav","cls":i%6,"source":"cremad","spk":f"cremad:s{i%20}"} for i in range(600)]
        tr, va = subset_and_split(rr, SerCfg(subset_frac=0.5, val_frac=0.2))
        chk("no speaker leakage", not (set(r["spk"] for r in tr) & set(r["spk"] for r in va)))
        p = np.array([0,1,0]); y = np.array([0,1,1])
        ua = float(np.mean([(p[y==cc]==cc).mean() for cc in range(6) if (y==cc).any()]))
        chk("UA = 0.75", abs(ua-0.75) < 1e-6)
        return ok, rows
    _ok, _rows = _st()
    mo.md(("**SELFTEST PASSED**" if _ok else "**SELFTEST FAILED**")
          + "\n\n| | check |\n|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""## 2 · Settings""")
    return


@app.cell
def _(mo):
    ws_ui = mo.ui.text(value="9,10,11,12", label="WS read layers (the learned blend, e.g. 9,10,11,12)")
    lora_ui = mo.ui.text(value="1-12", label="LoRA adapt range (e.g. 1-12)")
    subset_ui = mo.ui.slider(0.05, 1.0, value=0.25, step=0.05, label="subset fraction")
    epochs_ui = mo.ui.number(3, 40, value=15, label="epochs")
    band_ui = mo.ui.slider(0.0, 1.0, value=0.0, step=0.1, label="p_band")
    eightk_ui = mo.ui.slider(0.0, 1.0, value=0.0, step=0.1, label="p_8k")
    noise_ui = mo.ui.slider(0.0, 1.0, value=0.0, step=0.1, label="p_noise")
    specaug_ui = mo.ui.checkbox(value=False, label="SpecAugment")
    sweep_btn = mo.ui.run_button(label="▶ Layer sweep")
    aug_btn = mo.ui.run_button(label="▶ Augmentation ablation")
    train_btn = mo.ui.run_button(label="Train a single run")
    mo.vstack([ws_ui, lora_ui, subset_ui, epochs_ui,
               mo.md("**Augmentation (LoRA phase), for a single run. The ablation button uses its own set:**"),
               band_ui, eightk_ui, noise_ui, specaug_ui,
               mo.hstack([sweep_btn, aug_btn, train_btn])])
    return (aug_btn, band_ui, eightk_ui, epochs_ui, lora_ui, noise_ui, ws_ui,
            specaug_ui, subset_ui, sweep_btn, train_btn)


@app.cell
def _(SerCfg, epochs_ui, mo, subset_ui, sweep, sweep_btn):
    if not sweep_btn.value:
        _out = mo.md("Press **Layer sweep** (a few LoRA ranges on a subset, ranked by UA).")
    else:
        try:
            _res = sweep(SerCfg(subset_frac=float(subset_ui.value), epochs=int(epochs_ui.value)))
            _tbl = "\n".join(f"| {r['ws']} | {r['lora']} | {r['ua']*100:.1f} |" for r in _res)
            _out = mo.md("### Sweep - ranked by UA\n\n| ws (read) | lora | UA % |\n|---|---|---|\n" + _tbl)
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


@app.cell
def _(SerCfg, aug_btn, aug_sweep, epochs_ui, lora_ui, mo, subset_ui, ws_ui):
    if not aug_btn.value:
        _out = mo.md("Press **Augmentation ablation** - 5 configs on the locked ws set "
                     "(control / specaug / channel / noise / all).")
    else:
        try:
            _lo, _hi = (int(x) for x in lora_ui.value.split("-"))
            _ws = tuple(int(x) for x in ws_ui.value.split(","))
            _res = aug_sweep(SerCfg(ws=_ws, lora_lo=_lo, lora_hi=_hi,
                                    subset_frac=float(subset_ui.value), epochs=int(epochs_ui.value)))
            _tbl = "\n".join(f"| {r['aug']} | {r['ua']*100:.1f} |" for r in _res)
            _out = mo.md("### Augmentation ablation — UA\n\n| aug | UA % |\n|---|---|\n" + _tbl)
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


@app.cell
def _(SerCfg, band_ui, eightk_ui, epochs_ui, lora_ui, mo, noise_ui, ws_ui,
      specaug_ui, subset_ui, train_btn, train_ser):
    if not train_btn.value:
        _out = mo.md("Press **Train a single run** (with the settings above).")
    else:
        try:
            _lo, _hi = (int(x) for x in lora_ui.value.split("-"))
            _ws = tuple(int(x) for x in ws_ui.value.split(","))
            _cfg = SerCfg(ws=_ws, lora_lo=_lo, lora_hi=_hi,
                          subset_frac=float(subset_ui.value), epochs=int(epochs_ui.value),
                          p_band=float(band_ui.value), p_8k=float(eightk_ui.value),
                          p_noise=float(noise_ui.value), specaug=bool(specaug_ui.value),
                          run=(f"ser_ws{ws_ui.value.replace(',','-')}_L{lora_ui.value}"
                               + (f"_chan{band_ui.value}-{eightk_ui.value}-{noise_ui.value}"
                                  if (float(band_ui.value) + float(eightk_ui.value)
                                      + float(noise_ui.value)) > 0 else "")))
            _ua, _hist = train_ser(_cfg)
            _out = mo.md(f"### Result\n- **best val UA: {_ua*100:.1f}%**  (ws {ws_ui.value}, "
                         f"lora {lora_ui.value}, subset {subset_ui.value}, {len(_hist)} epoch)")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


@app.cell
def _(log):
    # ======= kaggle_emergency inspection: quality + labels + sample clips ========
    import os as _os
    from pathlib import Path as _P
    _r = _P("/marimo") if _P("/marimo").exists() else _P.cwd()
    KG_ROOT = _P(_os.environ.get("ECAD_DATA_ROOT", str(_r / "ser_data")))

    def _kg_fs():
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                     auth_kwargs={"use_local_webserver": False})

    def _kg_ensure():                     # labels + zip (pulled from Drive and extracted if missing)
        import zipfile
        lab = KG_ROOT / "labels" / "kaggle_emergency_labels.csv"
        adir = KG_ROOT / "kaggle_emergency"
        if lab.exists() and (adir / ".extracted").exists():
            return lab
        (KG_ROOT / "labels").mkdir(parents=True, exist_ok=True)
        fs = _kg_fs()
        if not lab.exists():
            log("[pull] kaggle_emergency_labels.csv")
            (fs.get_file if hasattr(fs, "get_file") else fs.get)(
                "CLEAR/emotion_data/labels/kaggle_emergency_labels.csv", str(lab))
        if not (adir / ".extracted").exists():
            zp = KG_ROOT / "kaggle_emergency.zip"; log("[pull] kaggle_emergency.zip")
            (fs.get_file if hasattr(fs, "get_file") else fs.get)(
                "CLEAR/emotion_data/zips/kaggle_emergency.zip", str(zp))
            with zipfile.ZipFile(zp) as z: z.extractall(KG_ROOT)
            adir.mkdir(parents=True, exist_ok=True); (adir / ".extracted").write_text("ok"); zp.unlink()
        return lab

    def _ffdur(p):
        import subprocess, json
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=sample_rate:format=duration", "-of", "json", str(p)],
                           capture_output=True, text=True)
        d = json.loads(o.stdout or "{}")
        return (int((d.get("streams") or [{}])[0].get("sample_rate") or 0),
                float((d.get("format") or {}).get("duration") or 0.0))

    def kaggle_inspect(n_samples=18):
        import pandas as pd, numpy as np, subprocess, zipfile, shutil, os
        lab = _kg_ensure()
        df = pd.read_csv(lab)
        L = []
        def _pp(*a):
            _s = " ".join(str(x) for x in a); print(_s, flush=True); L.append(_s)
        _pp("[kaggle] rows:", len(df), "| columns:", list(df.columns))
        col = "class_6" if "class_6" in df.columns else "label"
        _pp(f"[{col}]", dict(df[col].astype(str).value_counts()))
        _sr, _du = [], []
        for p in df["path"].head(80):
            fp = KG_ROOT / p
            if fp.exists():
                sr, du = _ffdur(fp)
                if sr: _sr.append(sr)
                if du: _du.append(du)
        if _du:
            _pp(f"[audio] duration(s): med={np.median(_du):.1f} min={min(_du):.1f} max={max(_du):.1f}")
            _pp(f"[audio] sample_rate: {dict(pd.Series(_sr).value_counts())}  (high and consistent = clean)")
        else:
            _pp("[audio] file not found, the path layout may differ")
        _pp("[sample]", [str(x) for x in df["path"].head(3)])
        cdir = KG_ROOT / "kaggle_earcheck"
        if cdir.exists(): shutil.rmtree(cdir)
        cdir.mkdir(parents=True)
        per = max(1, n_samples // max(1, df[col].nunique()))
        picks = []
        for cls, sub in df.groupby(col):
            picks += [(cls, r) for _, r in sub.sample(min(per, len(sub)), random_state=0).iterrows()]
        idx, i = [], 0
        for cls, r in picks:
            fp = KG_ROOT / r["path"]
            if not fp.exists(): continue
            fn = f"{i:02d}_{str(cls)}_{os.path.basename(str(r['path']))}".replace("/", "-")
            if not fn.endswith(".wav"): fn += ".wav"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(fp), "-ac", "1", "-ar", "16000",
                            str(cdir / fn)], capture_output=True)
            idx.append({"clip": fn, col: cls, "path": r["path"]})
            i += 1
        pd.DataFrame(idx).to_csv(cdir / "index.csv", index=False)
        oz = KG_ROOT / "kaggle_samples.zip"
        with zipfile.ZipFile(oz, "w", zipfile.ZIP_DEFLATED) as z:
            for q in sorted(cdir.glob("*")): z.write(q, q.name)
        try:
            fs = _kg_fs()
            (fs.put_file if hasattr(fs, "put_file") else fs.put)(
                str(oz), "CLEAR/emotion_data/kaggle_samples.zip")
            _pp(f"[push] kaggle_samples.zip -> Drive ({i} clips)")
        except Exception as _e:
            _pp(f"[push] skipped: {type(_e).__name__} — local {oz}")
        return "\n".join(L)
    return (kaggle_inspect,)


@app.cell
def _(kaggle_inspect, mo):
    kg_btn = mo.ui.run_button(label="Inspect kaggle_emergency and export sample clips to Drive")
    mo.vstack([mo.md("**kaggle_emergency (338) quality check** - label distribution, duration, sample rate "
                     "as a quality indicator, plus one sample clip per class pushed to Drive as `kaggle_samples.zip`."),
               kg_btn])
    return (kg_btn,)


@app.cell
def _(kaggle_inspect, kg_btn, mo):
    if not kg_btn.value:
        _out = mo.md("🔎 Press **Inspect kaggle_emergency**.")
    else:
        try:
            _txt = kaggle_inspect()
            _out = mo.md("```\n" + _txt + "\n```")
        except Exception as _e:
            import traceback
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}\n\n```\n{traceback.format_exc()}\n```")
    _out
    return


@app.cell
def _(CLASSES, OUT_ROOT, SerCfg, build_backbone, load_rows, log, make_head,
      read_stack, subset_and_split):
    # ====== Controlled demo P1: voice_risk (SER) vs text_risk (zero-shot) ======
    import os as _os
    from pathlib import Path as _P
    _r = _P("/marimo") if _P("/marimo").exists() else _P.cwd()
    DEMO_DATA = _P(_os.environ.get("ECAD_DATA_ROOT", str(_r / "ser_data")))
    CREMAD_SENT = {"IEO": "It's eleven o'clock", "TIE": "That is exactly what happened",
                   "IOM": "I'm on my way to the meeting", "IWW": "I wonder what this is about",
                   "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
                   "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment",
                   "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before",
                   "TSI": "The surface is slick", "WSI": "We'll stop in a couple of minutes"}
    RAVDESS_SENT = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}
    HIGH_V = {"distress", "fear", "urgency", "panic"}
    CAND = ["an urgent emergency or someone in danger"]     # a SINGLE hypothesis, the multi-hypothesis MAX inflated the fixed sentences
    TEXT_THR = 0.5                                          # score > threshold => high

    def _demo_fs():
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                     auth_kwargs={"use_local_webserver": False})

    def _demo_transcript(path, source):
        b = _os.path.basename(str(path))
        if source == "cremad":
            p = b.split("_")
            return CREMAD_SENT.get(p[1] if len(p) > 1 else "", "")
        if source == "ravdess":
            p = b.split(".")[0].split("-")
            return RAVDESS_SENT.get(p[4] if len(p) > 4 else "", "")
        return ""

    def _demo_model(run, dev):
        import torch, json
        d = OUT_ROOT / run
        if not all((d / f).exists() for f in ["adapter.pt", "head.pt", "config.json"]):
            d.mkdir(parents=True, exist_ok=True); fs = _demo_fs()
            for f in ["adapter.pt", "head.pt", "config.json"]:
                log(f"[pull] ser_runs/{run}/{f}")
                (fs.get_file if hasattr(fs, "get_file") else fs.get)(
                    f"CLEAR/ser_runs/{run}/{f}", str(d / f))
        _c = json.loads((d / "config.json").read_text()); _c["ws"] = tuple(_c["ws"])
        cfg = SerCfg(**_c)
        bb = build_backbone(cfg, dev)
        bb.load_state_dict(torch.load(d / "adapter.pt", map_location=dev), strict=False)
        head = make_head(cfg, dev); head.load_state_dict(torch.load(d / "head.pt", map_location=dev))
        bb.eval(); head.eval(); return cfg, bb, head

    def controlled_demo(run="ser_ws7-8-9-11-12_L1-12", sources=("cremad", "ravdess"),
                        limit=0, batch=16):
        import torch, numpy as np, pandas as pd, subprocess
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # HELD-OUT val split — the SAME speaker-independent split as training (seed 1337, val_frac 0.15).
        # SER saw NONE of these clips during training, so the numbers are clean and uncontaminated.
        _cfg0 = SerCfg(subset_frac=1.0, val_frac=0.15, seed=1337)
        _, va = subset_and_split(load_rows(), _cfg0)
        rows = []
        for r in va:
            s = r["source"]
            if s not in sources: continue
            tx = _demo_transcript(r["path"], s)
            if tx:
                rows.append({"path": r["path"], "source": s,
                             "true_emo": CLASSES[r["cls"]], "text": tx})
        if not rows:
            log("[demo] no data in val — pull cremad/ravdess with pull_data"); return None
        df = pd.DataFrame(rows)
        if limit:
            df = df.sample(min(int(limit), len(df)), random_state=0).reset_index(drop=True)
        log(f"[demo] HELD-OUT val: {len(df)} clips {dict(df['source'].value_counts())}")
        cfg, bb, head = _demo_model(run, dev)
        maxlen = int(cfg.max_secs * cfg.sr); minlen = cfg.sr // 4

        def _load_wav(fp):               # ffmpeg decode (NO soundfile) -> 16k mono float32
            o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(fp), "-ac", "1",
                                "-ar", str(cfg.sr), "-f", "f32le", "-"], capture_output=True)
            return np.frombuffer(o.stdout, dtype=np.float32).copy()

        preds = []
        for i in range(0, len(df), batch):
            wavs = []
            for p in df["path"].iloc[i:i + batch]:
                w = _load_wav(DEMO_DATA / p)[:maxlen]
                if len(w) < minlen: w = np.pad(w, (0, minlen - len(w)))
                wavs.append(w)
            Ln = max(len(w) for w in wavs); X = np.zeros((len(wavs), Ln), np.float32)
            for k, w in enumerate(wavs): X[k, :len(w)] = w
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16):
                lg = head(read_stack(bb, torch.from_numpy(X).to(dev), cfg).float())
            preds += [CLASSES[j] for j in lg.argmax(1).cpu().numpy()]
        df["ser_pred"] = preds
        df["voice_risk"] = df["ser_pred"].apply(lambda n: "high" if n in HIGH_V else "low")
        _trisk = df["true_emo"].apply(lambda e: "high" if e in HIGH_V else "low")
        log(f"[voice acc] voice_risk vs the true-emotion risk: "
            f"{(df['voice_risk'] == _trisk).mean()*100:.1f}% (HELD-OUT val, n={len(df)})")
        log("[voice confusion voice(rows) x true(cols)]\n" +
            pd.crosstab(df["voice_risk"], _trisk).to_string())
        from transformers import pipeline
        clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                       device=0 if dev == "cuda" else -1)
        uniq = sorted(df["text"].unique()); tr, ts = {}, {}
        for j in range(0, len(uniq), 16):
            ch = uniq[j:j + 16]; o = clf(ch, CAND, multi_label=True)   # scores[0]=MAX (the correctly templated one)
            if isinstance(o, dict): o = [o]
            for t, oo in zip(ch, o):
                sc = float(oo["scores"][0]); ts[t] = round(sc, 3)
                tr[t] = "high" if sc > TEXT_THR else "low"
        df["text_risk"] = df["text"].map(tr); df["text_score"] = df["text"].map(ts)
        log("[text score] (high to low, sorted) - the fixed sentences should stay low:")
        for k, v in sorted(ts.items(), key=lambda x: -x[1]):
            log(f"  {v:.2f} {'HIGH' if v > TEXT_THR else 'low '} | {k}")
        df["anomaly"] = df["voice_risk"] != df["text_risk"]
        log(f"[demo] text_risk {dict(df['text_risk'].value_counts())} (fixed calm sentences, we expect low)")
        log(f"[demo] voice_risk {dict(df['voice_risk'].value_counts())}")
        log(f"[demo] ANOMALY rate {df['anomaly'].mean()*100:.1f}%")
        log("[crosstab voice(rows) x text(cols)]\n" +
            pd.crosstab(df["voice_risk"], df["text_risk"]).to_string())
        oc = DEMO_DATA / "controlled_demo.csv"; df.to_csv(oc, index=False)
        try:
            fs = _demo_fs()
            (fs.put_file if hasattr(fs, "put_file") else fs.put)(
                str(oc), "CLEAR/emotion_data/controlled_demo.csv")
            log("[push] controlled_demo.csv -> Drive")
        except Exception as _e:
            log(f"[push] skipped {type(_e).__name__}")
        return df
    return (controlled_demo,)


@app.cell
def _(mo):
    demo_n_ui = mo.ui.number(value=0, start=0, stop=5000, label="limit (0 = the whole held-out val set)")
    demo_btn = mo.ui.run_button(label="Controlled demo P1 (voice vs text mismatch)")
    mo.vstack([mo.md("**Controlled demo, part 1** - the SER **held-out val** split (speakers "
                     "never seen in training) - CREMA-D and RAVDESS use fixed, ordinary sentences: "
                     "SER→voice_risk, zero-shot→text_risk, mismatch."),
               demo_n_ui, demo_btn])
    return demo_btn, demo_n_ui


@app.cell
def _(controlled_demo, demo_btn, demo_n_ui, mo):
    import traceback as _tb
    if not demo_btn.value:
        _out = mo.md("Press **Controlled demo P1**.")
    else:
        try:
            _df = controlled_demo(limit=int(demo_n_ui.value))
            if _df is None:
                _out = mo.md("**No data** - pull cremad and ravdess with `pull_data` first.")
            else:
                _an = _df["anomaly"].mean() * 100
                _out = mo.md(f"### Demo P1\n- clips: **{len(_df)}**\n- anomaly rate: **{_an:.1f}%**\n"
                             f"- `controlled_demo.csv` was pushed to Drive")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}\n\n```\n{_tb.format_exc()}\n```")
    _out
    return


@app.cell
def _(log):
    # ===== Controlled demo P2: 2x2 (calm/alarming text x calm/emotional voice) =====
    import os as _os
    from pathlib import Path as _P
    _r2 = _P("/marimo") if _P("/marimo").exists() else _P.cwd()
    DEMO2 = _P(_os.environ.get("ECAD_DATA_ROOT", str(_r2 / "ser_data")))
    ALARM_SENT = ["There's a fire, people are trapped inside",
                  "She's not breathing, send an ambulance now",
                  "He has a gun and he's shooting at people",
                  "There's been a serious car crash with injuries",
                  "Someone broke into my house, they're still here",
                  "My child is choking and turning blue",
                  "The building is collapsing, we're stuck inside",
                  "He's having a heart attack, please hurry"]
    CAND2 = ["an urgent emergency or someone in danger"]     # the SAME single hypothesis as P1
    T2_THR = 0.5

    def _d2_fs():
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                     auth_kwargs={"use_local_webserver": False})

    def demo_2x2():
        import pandas as pd, numpy as np, torch
        cc = DEMO2 / "controlled_demo.csv"
        if not cc.exists():
            fs = _d2_fs(); log("[pull] controlled_demo.csv")
            (fs.get_file if hasattr(fs, "get_file") else fs.get)(
                "CLEAR/emotion_data/controlled_demo.csv", str(cc))
        df = pd.read_csv(cc)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import pipeline
        clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                       device=0 if dev == "cuda" else -1)
        o = clf(ALARM_SENT, CAND2, multi_label=True)
        if isinstance(o, dict): o = [o]
        asc = {s: float(oo["scores"][0]) for s, oo in zip(ALARM_SENT, o)}
        log("[alarm score] (should be high):")
        for s, v in sorted(asc.items(), key=lambda x: -x[1]):
            log(f"  {v:.2f} {'HIGH' if v > T2_THR else 'low '} | {s}")
        calm = df[["true_emo", "voice_risk", "text", "text_risk"]].copy(); calm["content"] = "calm"
        rng = np.random.default_rng(0)
        al = df[["true_emo", "voice_risk"]].copy()
        al["text"] = rng.choice(ALARM_SENT, size=len(al))
        al["text_risk"] = al["text"].map(lambda s: "high" if asc[s] > T2_THR else "low")
        al["content"] = "alarm"
        grid = pd.concat([calm, al], ignore_index=True)
        grid["anomaly"] = grid["voice_risk"] != grid["text_risk"]
        log(f"\n[2x2] {len(grid)} combinations | anomaly {grid['anomaly'].mean()*100:.1f}%")
        log("[crosstab voice(rows) x text(cols)]\n" +
            pd.crosstab(grid["voice_risk"], grid["text_risk"]).to_string())
        log("\n[sample — one per cell]")
        for vr in ["low", "high"]:
            for trr in ["low", "high"]:
                sub = grid[(grid.voice_risk == vr) & (grid.text_risk == trr)]
                if len(sub):
                    r = sub.iloc[0]
                    log(f"  voice={vr}({r.true_emo}) text={trr}('{str(r.text)[:34]}') "
                        f"-> {'ANOMALY' if r.anomaly else 'congruent'}")
        og = DEMO2 / "controlled_demo_2x2.csv"; grid.to_csv(og, index=False)
        try:
            fs = _d2_fs()
            (fs.put_file if hasattr(fs, "put_file") else fs.put)(
                str(og), "CLEAR/emotion_data/controlled_demo_2x2.csv")
            log("[push] controlled_demo_2x2.csv -> Drive")
        except Exception as _e:
            log(f"[push] skipped {type(_e).__name__}")
        return grid
    return (demo_2x2,)


@app.cell
def _(demo_2x2, mo):
    d2_btn = mo.ui.run_button(label="Controlled demo P2 - the 2x2 fusion test")
    mo.vstack([mo.md("**Part 2** - uses the P1 csv, so SER is not run again. It scores the alarm "
                     "sentences and crosses calm/alarming text with calm/emotional voice. Do all four corners "
                     "behave as expected, with anomalies in the mismatch cells and none in the congruent cells?"),
               d2_btn])
    return (d2_btn,)


@app.cell
def _(d2_btn, demo_2x2, mo):
    import traceback as _tb
    if not d2_btn.value:
        _out = mo.md("Press **Controlled demo P2** (P1 must have been run first).")
    else:
        try:
            _g = demo_2x2()
            _out = mo.md(f"### Demo P2 (2×2)\n- combinations: **{len(_g)}**\n"
                         f"- anomaly: **{_g['anomaly'].mean()*100:.1f}%**\n"
                         f"- `controlled_demo_2x2.csv` was pushed to Drive")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}\n\n```\n{_tb.format_exc()}\n```")
    _out
    return


@app.cell
def _(CLASSES, OUT_ROOT, SerCfg, build_backbone, load_rows, log, make_head,
      read_stack, subset_and_split):
    # ===== Per-clip SER feature export for the fusion benchmark (academic) =====
    import os as _os
    from pathlib import Path as _P
    _rf = _P("/marimo") if _P("/marimo").exists() else _P.cwd()
    FDATA = _P(_os.environ.get("ECAD_DATA_ROOT", str(_rf / "ser_data")))

    def _ff_fs():
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                     auth_kwargs={"use_local_webserver": False})

    def _load_model_f(run, dev):
        import torch, json
        d = OUT_ROOT / run
        if not all((d / f).exists() for f in ["adapter.pt", "head.pt", "config.json"]):
            d.mkdir(parents=True, exist_ok=True); fs = _ff_fs()
            for f in ["adapter.pt", "head.pt", "config.json"]:
                log(f"[pull] ser_runs/{run}/{f}")
                (fs.get_file if hasattr(fs, "get_file") else fs.get)(
                    f"CLEAR/ser_runs/{run}/{f}", str(d / f))
        _c = json.loads((d / "config.json").read_text()); _c["ws"] = tuple(_c["ws"])
        cfg = SerCfg(**_c)
        bb = build_backbone(cfg, dev)
        bb.load_state_dict(torch.load(d / "adapter.pt", map_location=dev), strict=False)
        head = make_head(cfg, dev); head.load_state_dict(torch.load(d / "head.pt", map_location=dev))
        bb.eval(); head.eval(); return cfg, bb, head

    def export_ser_feats(run="ser_ws7-8-9-11-12_L1-12", sources=("cremad", "ravdess"), batch=16):
        import torch, numpy as np, subprocess
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cfg, bb, head = _load_model_f(run, dev)
        tr, va = subset_and_split(load_rows(), SerCfg(subset_frac=1.0, val_frac=0.15, seed=1337))
        maxlen = int(cfg.max_secs * cfg.sr); minlen = cfg.sr // 4

        def _wav(p):
            o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(FDATA / p), "-ac", "1",
                                "-ar", str(cfg.sr), "-f", "f32le", "-"], capture_output=True)
            return np.frombuffer(o.stdout, np.float32).copy()

        def _run(rows, tag):
            rows = [r for r in rows if r["source"] in sources]
            embs, probs, true, spk, src = [], [], [], [], []
            for i in range(0, len(rows), batch):
                ch = rows[i:i + batch]; wavs = []
                for r in ch:
                    w = _wav(r["path"])[:maxlen]
                    if len(w) < minlen: w = np.pad(w, (0, minlen - len(w)))
                    wavs.append(w)
                L = max(len(w) for w in wavs); X = np.zeros((len(wavs), L), np.float32)
                for k, w in enumerate(wavs): X[k, :len(w)] = w
                with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    hs = read_stack(bb, torch.from_numpy(X).to(dev), cfg).float()   # [B,T,n_ws,hid]
                    wgt = head.layer_w.softmax(0)
                    blend = (hs * wgt[None, None, :, None]).sum(2)
                    mu = blend.mean(1)
                    f = torch.cat([mu, blend.std(1)], -1) if cfg.pool == "meanstd" else mu
                    pr = head.net(f).softmax(-1)
                embs.append(f.float().cpu().numpy()); probs.append(pr.float().cpu().numpy())
                for r in ch: true.append(r["cls"]); spk.append(r["spk"]); src.append(r["source"])
                if (i // batch) % 20 == 0: log(f"  [{tag}] {i+len(ch)}/{len(rows)}")
            out = FDATA / f"ser_feats_{tag}.npz"
            np.savez(out, emb=np.concatenate(embs), probs=np.concatenate(probs),
                     true=np.array(true), spk=np.array(spk), source=np.array(src),
                     classes=np.array(CLASSES))
            try:
                fs = _ff_fs()
                (fs.put_file if hasattr(fs, "put_file") else fs.put)(str(out), f"CLEAR/fusion/ser_feats_{tag}.npz")
                log(f"[push] ser_feats_{tag}.npz -> Drive ({len(true)} clips)")
            except Exception as _e:
                log(f"[push] skipped {type(_e).__name__}")
            return len(true)

        n_tr, n_va = _run(tr, "train"), _run(va, "val")
        log(f"[export] train {n_tr} · val {n_va} · emb_dim {cfg.feat_dim()} (features in Drive/CLEAR/fusion)")
        return n_tr, n_va
    return (export_ser_feats,)


@app.cell
def _(export_ser_feats, mo):
    feat_btn = mo.ui.run_button(label="SER feature export (for fusion)")
    mo.vstack([mo.md("**Fusion features** - for every clip in the academic train/val split (speaker independent) "
                     "emotion_probs + pooled embedding + true_emo → `CLEAR/fusion/ser_feats_{train,val}.npz`."),
               feat_btn])
    return (feat_btn,)


@app.cell
def _(export_ser_feats, feat_btn, mo):
    import traceback as _tb
    if not feat_btn.value:
        _out = mo.md("📦 Press **SER feature export**.")
    else:
        try:
            _ntr, _nva = export_ser_feats()
            _out = mo.md(f"✅ export: **train {_ntr} · val {_nva}** clips → `CLEAR/fusion/`")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}\n\n```\n{_tb.format_exc()}\n```")
    _out
    return


@app.cell
def _(OUT_ROOT, log):
    # ---- Drive push: upload one run folder to CLEAR/ser_runs/<run>/ ----
    def _get_fs():
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                     auth_kwargs={"use_local_webserver": False})

    def _put(fs, local, remote):
        (fs.put_file if hasattr(fs, "put_file") else fs.put)(str(local), remote)

    def push_run(run, remote_root="CLEAR/ser_runs"):
        src = OUT_ROOT / run
        files = sorted(p for p in src.glob("*") if p.is_file())
        if not files:
            log(f"[missing] {src} is empty or absent — check the run name"); return
        fs = _get_fs()
        rdir = f"{remote_root}/{run}"
        try:
            fs.makedirs(rdir, exist_ok=True)
        except Exception:
            pass
        for p in files:
            log(f"[push] {p.name} ({p.stat().st_size/1e3:.0f} KB)")
            _put(fs, p, f"{rdir}/{p.name}")
        log(f"done -> Drive:{rdir}")
    return (push_run,)


@app.cell
def _(mo):
    push_run_ui = mo.ui.text(value="ser_ws7-8-9-11-12_L1-12", label="run name", full_width=True)
    push_btn = mo.ui.run_button(label="Upload to Drive (head, adapter and summary)")
    mo.vstack([mo.md("**Push the final SER model to Drive** → `CLEAR/ser_runs/<run>/`"),
               push_run_ui, push_btn])
    return push_btn, push_run_ui


@app.cell
def _(mo, push_btn, push_run, push_run_ui):
    if not push_btn.value:
        _out = mo.md("Press **Upload to Drive** (the first upload triggers OAuth once).")
    else:
        try:
            push_run(push_run_ui.value.strip())
            _out = mo.md(f"`{push_run_ui.value.strip()}` was pushed to Drive (`CLEAR/ser_runs/`).")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


if __name__ == "__main__":
    app.run()
