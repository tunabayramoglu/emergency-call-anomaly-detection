# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers>=4.44",
#     "datasets==5.0.0",
#     "peft>=0.11",
#     "jiwer",
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
# Phase-1b - overnight ablation. SINGLE FILE, no external modules.
#   molab : upload, run the cells in order
#   local : marimo edit ablation_engine.py

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    app_title="Overnight ablation",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Phase-1b — augmentation & regularization ablation

        **Base** (pull from HF or train from scratch) -> a short run for **each axis separately**
        -> **combo** (the winners merged).

        Two things are critical.

        - **`X_control`** continues the base with the same short budget and nothing added.
          Without it every axis gain is confounded with the *"a few more epochs"* effect.
          The `d-clean` column measures against **control**, not against BASE.
        - **The target is the `tel8k/clean` ratio**, not absolute WER. Augmentation does not move
          the W/C ratio (WER divided by CER), only the language model does. That was your own analysis.

        The cache uses the same key as `sweep_v2.py`, so nothing is decoded again.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Environment
    """)
    return


@app.cell
def _():
    import contextlib, gc, glob, hashlib, io, json, math, os, random, time, traceback
    from dataclasses import dataclass, field, asdict, replace
    from itertools import groupby
    from pathlib import Path

    import numpy as np

    # Must be set BEFORE CUDA. Bucketing produces variable-length batches, and
    # every new shape asks the allocator for a new block. Once fragmentation grows
    # it triggers cudaFree/cudaMalloc, which is SYNCHRONOUS, and the card drops to 0%.
    # expandable_segments removes almost all of this.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    _root = Path("/marimo") if Path("/marimo").exists() else Path.cwd()
    os.environ.setdefault("ECAD_DATA_ROOT", str(_root / "data"))
    os.environ.setdefault("ECAD_OUT_ROOT", str(_root / "runs"))
    os.environ.setdefault("ECAD_CACHE_ROOT", str(_root / "cache"))

    DATA_ROOT = Path(os.environ["ECAD_DATA_ROOT"])
    OUT_ROOT = Path(os.environ["ECAD_OUT_ROOT"])
    CACHE_ROOT = Path(os.environ["ECAD_CACHE_ROOT"])
    NOISE_ROOT = Path(os.environ.setdefault("ECAD_NOISE_ROOT", str(_root / "noise")))
    BACKBONE = "utter-project/mHuBERT-147"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    return (
        BACKBONE,
        CACHE_ROOT,
        DATA_ROOT,
        Dataset,
        NOISE_ROOT,
        OUT_ROOT,
        Path,
        asdict,
        contextlib,
        dataclass,
        field,
        gc,
        glob,
        groupby,
        hashlib,
        io,
        json,
        log,
        math,
        nn,
        np,
        os,
        random,
        replace,
        time,
        torch,
        traceback,
    )


@app.cell
def _(BACKBONE, CACHE_ROOT, DATA_ROOT, OUT_ROOT, mo, torch):
    _has = torch.cuda.is_available()
    mo.md(
        f"""
    | | |
    |---|---|
    | GPU | {torch.cuda.get_device_name(0) if _has else "NONE - falls back to CPU"} |
    | VRAM | {f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB" if _has else "-"} |
    | torch | {torch.__version__} · bf16 {torch.cuda.is_bf16_supported() if _has else "-"} |
    | data / runs / cache | `{DATA_ROOT}`<br>`{OUT_ROOT}`<br>`{CACHE_ROOT}` |
    | backbone | `{BACKBONE}` |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Augmentation

    All of it is pure numpy FFT. torchaudio, scipy and librosa are not needed.
    The chain order is physical: **source -> room -> noise -> channel -> packet loss**.
    Changing the order changes the result (reverb after the codec is wrong).
    """)
    return


@app.cell
def _(glob, log, math, np, os):
    # Ko 2015 (speed) · Ko 2017 (RIR) · Snyder 2015 (MUSAN) · Park 2019 (SpecAugment)
    # ITU-T G.711 (mu-law) - G.712 (300-3400 Hz) - G.1050 (packet loss)

    def aug_speed(w, rate):
        if rate == 1.0 or len(w) < 2:
            return w
        n = max(2, int(round(len(w) / rate)))
        return np.interp(np.linspace(0.0, len(w) - 1.0, n, dtype=np.float32),
                         np.arange(len(w), dtype=np.float32), w).astype(np.float32)

    def _mask(w, sr, lo, hi):
        if len(w) < 8:
            return w
        W = np.fft.rfft(w)
        f = np.fft.rfftfreq(len(w), 1.0 / sr)
        if lo is not None:
            W[f < lo] = 0.0
        if hi is not None:
            W[f > hi] = 0.0
        return np.fft.irfft(W, len(w)).astype(np.float32)

    def aug_band(w, sr):
        """Telephone band, 300-3400 Hz."""
        return _mask(w, sr, 300.0, 3400.0)

    def aug_8k(w, sr=16000):
        """16k -> 8k -> 16k round-trip."""
        if len(w) < 8:
            return w
        x = _mask(w, sr, None, 3800.0)
        n8 = max(2, len(x) // 2)
        d = np.interp(np.linspace(0, len(x) - 1, n8), np.arange(len(x)), x)
        return np.interp(np.linspace(0, n8 - 1, len(x)), np.arange(n8), d).astype(np.float32)

    def aug_mulaw(w, mu=255.0):
        a = np.clip(w, -1.0, 1.0)
        y = np.sign(a) * np.log1p(mu * np.abs(a)) / math.log1p(mu)
        q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
        return (np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu).astype(np.float32)

    def aug_packet(w, sr, rate, rng):
        """VoIP packet loss. Blocks of 20-60 ms are zeroed out."""
        y = w.copy()
        i = 0
        while i < len(y):
            L = max(1, int(rng.uniform(20.0, 60.0) / 1000.0 * sr))
            if rng.random() < rate:
                y[i : i + L] = 0.0
            i += L
        return y

    def _pink(n, rng):
        m = n // 2 + 1
        s = (rng.standard_normal(m) + 1j * rng.standard_normal(m)).astype(np.complex64)
        f = np.arange(m, dtype=np.float32)
        f[0] = 1.0
        x = np.fft.irfft(s / np.sqrt(f), n).astype(np.float32)
        return x / (float(np.sqrt((x**2).mean())) + 1e-9)

    def _mix(w, nz, snr_db):
        nz = nz / (float(np.sqrt((nz**2).mean())) + 1e-9)
        k = math.sqrt((float((w**2).mean()) + 1e-12) / (10.0 ** (snr_db / 10.0)))
        return (w + k * nz).astype(np.float32)

    def _fit(src, n, rng):
        src = np.asarray(src, dtype=np.float32)
        if len(src) < n:
            src = np.tile(src, int(np.ceil(n / max(1, len(src)))))
        o = int(rng.integers(0, max(1, len(src) - n + 1)))
        return src[o : o + n]

    def aug_noise(w, snr_db, rng, bank=None):
        nz = _fit(bank[rng.integers(len(bank))], len(w), rng) if bank else _pink(len(w), rng)
        return _mix(w, nz, snr_db)

    def aug_babble(w, snr_db, rng, other_fn, k=3):
        """Speech on top of speech, taken from the train set, no extra data needed.
        More realistic than MUSAN for a call centre or 911 setting."""
        if other_fn is None:
            return w
        acc, got = np.zeros(len(w), dtype=np.float32), 0
        for _ in range(k):
            o = other_fn()
            if o is not None and len(o) >= 16:
                acc += _fit(o, len(w), rng)
                got += 1
        return _mix(w, acc, snr_db) if got else w

    def aug_rir(w, sr, t60, rng):
        """Synthetic room response. Direct path plus an exponentially decaying tail."""
        n = max(8, int(t60 * sr))
        h = rng.standard_normal(n).astype(np.float32)
        h *= np.exp(-6.9078 * np.arange(n, dtype=np.float32) / n)
        h[0] += 1.0
        h /= float(np.abs(h).sum()) + 1e-9
        L = len(w) + n - 1
        nfft = 1 << (L - 1).bit_length()
        y = np.fft.irfft(np.fft.rfft(w, nfft) * np.fft.rfft(h, nfft), nfft)
        return y[: len(w)].astype(np.float32)

    def input_norm(w, mode):
        if mode == "zscore":  # same as fairseq normalize=True
            return ((w - w.mean()) / (w.std() + 1e-7)).astype(np.float32)
        return w

    BANKS = {}

    def load_bank(d, limit=200):
        """Load the wav folder into memory. Each worker loads its own copy, hence the limit."""
        if not d:
            return None
        if d in BANKS:                    # <- BANKS, not _BANKS
            return BANKS[d] or None       # <- BANKS, not _BANKS
        files = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))[:limit]
        if not files:
            log(f"[BANK] warning: {d} is empty or missing, running without a noise bank")
            BANKS[d] = []                 # <- BANKS, not _BANKS
            return None
        import soundfile as sf
        bank = []
        for f in files:
            try:
                x, _ = sf.read(f, dtype="float32")
                bank.append(x.mean(1) if x.ndim > 1 else x)
            except Exception:
                pass
        BANKS[d] = bank                   # <- BANKS, not _BANKS
        log(f"[BANK] {len(bank)} files <- {d}")
        return bank or None

    def degrade_eval(w, mode, sr=16000):
        """Evaluation conditions. Deterministic."""
        if mode in (None, "clean"):
            return w
        if mode == "tel":
            return aug_band(w, sr)
        if mode == "tel8k":
            return aug_8k(aug_band(w, sr), sr)
        if mode == "noisy":  # fixed seed -> deterministic, comparable across runs
            return aug_noise(w, 10.0, np.random.default_rng(12345), None)
        raise ValueError(mode)

    return (
        aug_8k,
        aug_babble,
        aug_band,
        aug_mulaw,
        aug_noise,
        aug_packet,
        aug_rir,
        aug_speed,
        degrade_eval,
        input_norm,
        load_bank,
    )


@app.cell
def _(
    aug_8k,
    aug_babble,
    aug_band,
    aug_mulaw,
    aug_noise,
    aug_packet,
    aug_rir,
    aug_speed,
    dataclass,
    load_bank,
    np,
):
    @dataclass
    class Aug:
        p_clean: float = 0.40  # leave this fraction untouched, protects clean WER
        speed: tuple = ()
        p_speed: float = 0.0
        p_rir: float = 0.0
        p_noise: float = 0.0
        snr: tuple = (5.0, 20.0)
        p_babble: float = 0.0
        babble_snr: tuple = (10.0, 25.0)
        p_band: float = 0.0
        p_8k: float = 0.0
        p_mulaw: float = 0.0
        p_packet: float = 0.0
        noise_dir: str | None = None

        _K = ("p_speed", "p_rir", "p_noise", "p_babble", "p_band", "p_8k",
              "p_mulaw", "p_packet")

        def on(self):
            return any(getattr(self, k) > 0 for k in self._K)

        def names(self):
            return [k[2:] for k in self._K if getattr(self, k) > 0]

    def apply_aug(w, a: Aug, sr, rng, other_fn=None, noise_bank=None):
        if not a.on() or rng.random() < a.p_clean:
            return w
        if a.speed and rng.random() < a.p_speed:
            w = aug_speed(w, float(rng.choice(np.asarray(a.speed))))
        if rng.random() < a.p_rir:
            w = aug_rir(w, sr, float(rng.uniform(0.15, 0.50)), rng)
        if rng.random() < a.p_babble:
            w = aug_babble(w, float(rng.uniform(*a.babble_snr)), rng, other_fn)
        if rng.random() < a.p_noise:
            w = aug_noise(w, float(rng.uniform(*a.snr)), rng, load_bank(a.noise_dir) if a.noise_dir else noise_bank)
        if rng.random() < a.p_band:
            w = aug_band(w, sr)
        if rng.random() < a.p_8k:
            w = aug_8k(w, sr)
        if rng.random() < a.p_mulaw:
            w = aug_mulaw(w)
        if rng.random() < a.p_packet:
            w = aug_packet(w, sr, float(rng.uniform(0.01, 0.08)), rng)
        pk = float(np.abs(w).max()) + 1e-9
        return (w / pk).astype(np.float32) if pk > 1.0 else w

    return Aug, apply_aug


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Config
    """)
    return


@app.cell
def _(
    Aug,
    BACKBONE,
    CACHE_ROOT,
    DATA_ROOT,
    OUT_ROOT,
    asdict,
    dataclass,
    field,
):
    @dataclass
    class Cfg:
        run: str = "run"
        # architecture, same as sweep_v2
        ws: tuple = (9, 10, 11, 12)
        lora_layers: tuple | None = None
        lora_r: int = 16
        lora_alpha: int = 32
        hid: int = 768
        sr: int = 16_000
        # None means NO truncation. Truncating cut the audio but kept the transcript -> corrupted labels.
        max_secs: float | None = None
        # Memory ceiling. A batch holds at most batch x batch_secs seconds of audio.
        # On long utterances the batch shrinks automatically and VRAM stays flat.
        batch_secs: float = 20.0
        # regularization (all off in v2)
        weight_decay: float = 0.0
        lora_dropout: float = 0.0
        head_dropout: float = 0.0
        mask_time_prob: float = 0.0
        mask_feature_prob: float = 0.0
        bb_dropout: float = 0.0  # the backbone's own dropouts, enabled as a separate axis
        ws_layer_drop: float = 0.0
        hs_noise: float = 0.0
        aux_ctc_weight: float = 0.0  # InterCTC (Lee & Watanabe 2021)
        ema_decay: float = 0.0
        input_norm: str = "none"
        # augmentation
        aug: Aug = field(default_factory=Aug)
        # training
        epochs: int = 30
        batch: int = 64
        accum: int = 4
        head_lr: float = 1e-3
        lora_lr: float = 2e-4
        w_lr: float = 1e-3
        clip: float = 5.0
        patience: int = 4
        stop_patience: int = 12
        workers: int = 8
        ram: bool = False     # True: load the audio fully into RAM (instead of memmap)
        bucket: bool = True   # batch by length, cuts the padding waste
        grad_ckpt: bool = False
        seed: int = 1337
        init_from: str | None = None
        deadline: float | None = None
        conds: tuple = ("clean", "tel", "tel8k", "noisy")

        def __post_init__(self):
            self.ws = tuple(sorted(int(x) for x in self.ws))
            if self.lora_layers is None:
                self.lora_layers = tuple(range(1, max(self.ws) + 1))
            else:
                self.lora_layers = tuple(sorted(int(x) for x in self.lora_layers))

        @property
        def dir(self):
            return OUT_ROOT / self.run

        @property
        def spec_on(self):
            return self.mask_time_prob > 0 or self.mask_feature_prob > 0

        @property
        def bb_train(self):
            """SpecAugment and backbone dropout only run in train() mode."""
            return self.spec_on or self.bb_dropout > 0

        def n_adapter(self):
            return 2 * self.hid * self.lora_r * 2 * len(self.lora_layers)

        def regs(self):
            r = [f"{n}={v:g}" for n, v in (
                ("wd", self.weight_decay), ("lora_do", self.lora_dropout),
                ("head_do", self.head_dropout), ("mask_t", self.mask_time_prob),
                ("mask_f", self.mask_feature_prob), ("bb_do", self.bb_dropout),
                ("interctc", self.aux_ctc_weight), ("ws_drop", self.ws_layer_drop),
                ("hs_noise", self.hs_noise),
                ("ema", self.ema_decay)) if v > 0]
            if self.input_norm != "none":
                r.append(f"norm={self.input_norm}")
            return r

        def d(self):
            x = asdict(self)
            x["data_root"], x["backbone"] = str(DATA_ROOT), BACKBONE
            x["cache_root"] = str(CACHE_ROOT)
            return x

        def line(self):
            return (f"[CFG] {self.run} | read={list(self.ws)} "
                    f"| adapt={list(self.lora_layers)} | {self.epochs}ep "
                    f"bs{self.batch}x{self.accum} | aug=[{','.join(self.aug.names()) or '-'}] "
                    f"| reg=[{','.join(self.regs()) or '-'}]"
                    + (f" | init<-{self.init_from}" if self.init_from else ""))

    return (Cfg,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Data

    The cache key is **byte-for-byte the same** as `sweep_v2.py`, so the existing
    `audio.i16` is used as is and nothing is decoded again.
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    DATA_ROOT,
    Dataset,
    apply_aug,
    degrade_eval,
    glob,
    hashlib,
    input_norm,
    io,
    json,
    log,
    np,
    time,
    torch,
):
    CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
    HF_DS = "openslr/librispeech_asr"
    PREFIX = {"train": "clean/train.100/", "dev": "clean/validation/"}

    def build_vocab():
        v = {c: i for i, c in enumerate(CHARS)}
        v["|"], v["[UNK]"], v["[PAD]"] = len(v), len(v) + 1, len(v) + 2
        return v

    def _decode(cell, sr_t):
        import soundfile as sf

        if isinstance(cell, dict) and cell.get("array") is not None:
            w, sr = np.asarray(cell["array"], np.float32), cell.get("sampling_rate", sr_t)
        elif isinstance(cell, dict) and cell.get("bytes"):
            w, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif isinstance(cell, dict) and cell.get("path"):
            w, sr = sf.read(cell["path"], dtype="float32")
        else:
            w, sr = sf.read(cell if isinstance(cell, str) else io.BytesIO(cell),
                            dtype="float32")
        w = np.asarray(w, np.float32)
        if w.ndim > 1:
            w = w.mean(1)
        assert int(sr) == int(sr_t), f"sample rate {sr} != {sr_t}"
        return w

    def _hf(split, sr):
        from datasets import load_dataset
        from huggingface_hub import list_repo_files

        pat = str(DATA_ROOT / ("librispeech_train100/*.parquet" if split == "train"
                               else "librispeech_val/*.parquet"))
        files = sorted(glob.glob(pat))
        if files:
            ds = load_dataset("parquet", data_files=files, split="train",
                              verification_mode="no_checks")
        else:
            # WARNING. load_dataset(REPO,"clean",split=...) downloads the whole config (30 GB).
            urls = [f"hf://datasets/{HF_DS}/{f}" for f in
                    sorted(x for x in list_repo_files(HF_DS, repo_type="dataset")
                           if x.startswith(PREFIX[split]) and x.endswith(".parquet"))]
            log(f"[HF] {split}: {len(urls)} parquet")
            ds = load_dataset("parquet", data_files={"d": urls}, split="d",
                              verification_mode="no_checks")
        try:
            from datasets import Audio

            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
        return ds

    def prepare(split, sr=16000, max_secs=None, ram=False):
        """max_secs=None means NO truncation.

        Truncation cut the audio but left the transcript intact, so on every utterance
        longer than 20 s CTC was targeting words it never heard. Corrupted labels."""
        key = hashlib.md5(
            f"{DATA_ROOT}|{split}|{sr}|None|{max_secs}|int16".encode()
        ).hexdigest()[:12]
        d = CACHE_ROOT / f"{split}_{key}"
        bin_p, meta_p = d / "audio.i16", d / "meta.json"
        if not (bin_p.exists() and meta_p.exists()):
            d.mkdir(parents=True, exist_ok=True)
            ds = _hf(split, sr)
            cut = int(max_secs * sr) if max_secs else None
            t0, offs, txts, pos, n_cut = time.perf_counter(), [0], [], 0, 0
            with open(bin_p, "wb") as f:
                for i in range(len(ds)):
                    row = ds[i]
                    w = _decode(row["audio"], sr)
                    if cut and len(w) > cut:
                        w, n_cut = w[:cut], n_cut + 1
                    q = np.clip(np.rint(w * 32768.0), -32768, 32767).astype(np.int16)
                    f.write(q.tobytes())
                    pos += q.size
                    offs.append(pos)
                    txts.append(row["text"].upper().strip())
                    if (i + 1) % 2000 == 0:
                        log(f"  [cache:{split}] {i + 1}/{len(ds)} "
                            f"({pos * 2 / 1e9:.1f} GB, {time.perf_counter() - t0:.0f}s)")
            meta_p.write_text(json.dumps({"offsets": offs, "texts": txts}))
            log(f"[cache:{split}] {len(txts)} samples, {(time.perf_counter() - t0) / 60:.1f} min"
                + (f" · ⚠ {n_cut} utterance TRUNCATED (labels corrupted)" if n_cut else
                   " - no truncation"))
        else:
            log(f"[DATA:{split}] cache found -> {d}")
        meta = json.loads(meta_p.read_text())
        n = int(meta["offsets"][-1])
        _L = np.diff(np.asarray(meta["offsets"], np.int64))
        log(f"[DATA:{split}] {len(_L)} utt · {_L.sum() / sr / 3600:.2f} h · "
            f"avg {_L.mean() / sr:.1f} s · max {_L.max() / sr:.1f} s")
        if ram:
            # A single flat array, so fork workers SHARE it through copy-on-write.
            # We only read it, so no copy is made and we avoid 8 workers x 11.6 GB.
            buf = np.fromfile(bin_p, dtype=np.int16)
            log(f"[DATA:{split}] RAM - loaded {buf.nbytes / 1e9:.2f} GB")
        else:
            buf = np.memmap(bin_p, dtype=np.int16, mode="r", shape=(n,))
        return buf, np.asarray(meta["offsets"], np.int64), meta["texts"]

    class SpeechDS(Dataset):
        def __init__(self, raw, vocab, cfg, aug=None, degrade=None, noise_bank=None):
            self.buf, self.offs, self.texts = raw
            self.vocab, self.cfg = vocab, cfg
            self.aug, self.degrade, self.bank = aug, degrade, noise_bank
            self._rng = None

        def __len__(self):
            return len(self.texts)

        def rng(self):
            # a separate seed per worker, otherwise every worker produces the same sequence
            if self._rng is None:
                import torch.utils.data as tud

                w = tud.get_worker_info()
                self._rng = np.random.default_rng(self.cfg.seed * 100003 + (w.id if w else 0))
            return self._rng

        def _raw(self, i):
            a, b = int(self.offs[i]), int(self.offs[i + 1])
            return np.asarray(self.buf[a:b], np.float32) / 32768.0

        def __getitem__(self, i):
            w = self._raw(i)
            if self.aug is not None:
                w = apply_aug(w, self.aug, self.cfg.sr, self.rng(),
                              lambda: self._raw(int(self.rng().integers(len(self.texts)))),
                              self.bank)
            if self.degrade:
                w = degrade_eval(w, self.degrade, self.cfg.sr)
            w = input_norm(w, self.cfg.input_norm)
            if self.cfg.max_secs:  # no truncation when None
                w = w[: int(self.cfg.max_secs * self.cfg.sr)]
            w = np.ascontiguousarray(w, np.float32)
            ids = [self.vocab.get(c, self.vocab["[UNK]"])
                   for c in self.texts[i].replace(" ", "|")]
            return torch.from_numpy(w), torch.tensor(ids, dtype=torch.long), i

    class LengthBucket(torch.utils.data.Sampler):
        """Groups by length and forms batches under a FRAME BUDGET.

        Two jobs at once.
        1. Similar lengths land in the same batch, cutting padding waste from 36% to about 1%.
        2. `budget` = batch x batch_secs x sr. When long utterances arrive the batch
           shrinks automatically (40 instead of 64), so the VRAM ceiling stays flat
           without any max_secs truncation. That is what lets us remove
           the truncation.

        The pool is chosen at random, so shuffling is preserved."""

        def __init__(self, lengths, batch, budget, shuffle=True, seed=0, pool_mult=50):
            self.L = np.asarray(lengths, np.int64)
            self.b, self.budget = batch, int(budget)
            self.shuffle, self.seed = shuffle, seed
            self.pool, self.epoch = batch * pool_mult, 0
            self._cache = self._build(0)  # so __len__ is correct even before the first iter

        def _build(self, epoch):
            g = np.random.default_rng(self.seed + epoch)
            idx = g.permutation(len(self.L)) if self.shuffle else np.arange(len(self.L))
            out, cur = [], []
            for i in range(0, len(idx), self.pool):
                ch = idx[i : i + self.pool]
                ch = ch[np.argsort(self.L[ch], kind="stable")]  # ascending within the pool
                for j in ch:
                    Lj = int(self.L[j])  # sorted, so the longest is always the last one added
                    if cur and (len(cur) + 1 > self.b
                                or Lj * (len(cur) + 1) > self.budget):
                        out.append(cur)
                        cur = [int(j)]
                    else:
                        cur.append(int(j))
                if cur:  # close at the pool boundary so different lengths do not mix
                    out.append(cur)
                    cur = []
            if cur:
                out.append(cur)
            if self.shuffle:
                g.shuffle(out)
            return out

        def __iter__(self):
            out = self._cache if self._cache is not None else self._build(self.epoch)
            self._cache = None
            self.epoch += 1
            self._n = len(out)
            return iter(out)

        def __len__(self):
            return len(self._cache) if self._cache is not None else getattr(
                self, "_n", (len(self.L) + self.b - 1) // self.b)

    def _fork_ctx(workers):
        """Classes defined in a notebook cell CANNOT BE IMPORTED inside a spawn worker,
        (AttributeError: Can't get attribute 'SpeechDS' on __mp_main__).
        fork inherits memory instead of pickling it, which makes the problem disappear."""
        if workers <= 0:
            return None
        import multiprocessing as _mp

        return _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else None

    def make_loader(ds, raw, cfg, collate, shuffle, workers, persist=None):
        from torch.utils.data import DataLoader as _DL

        # eval loaders are used once, leaving persistent workers piles up processes
        persist = (workers > 0) if persist is None else (persist and workers > 0)
        kw = dict(collate_fn=collate, pin_memory=True, num_workers=workers,
                  persistent_workers=persist)
        if workers > 0:
            kw["prefetch_factor"] = 4
            ctx = _fork_ctx(workers)
            if ctx is not None:
                kw["multiprocessing_context"] = ctx
        if cfg.bucket:
            kw["batch_sampler"] = LengthBucket(
                np.diff(raw[1]), cfg.batch,
                budget=cfg.batch * cfg.batch_secs * cfg.sr,
                shuffle=shuffle, seed=cfg.seed)
        else:
            kw["batch_size"], kw["shuffle"] = cfg.batch, shuffle
        return _DL(ds, **kw)

    def probe_workers(ds, raw, cfg, collate, want):
        """Rehearse the workers BEFORE TRAINING STARTS. Normally the error surfaces on the first batch,
        which is after the model is built and minutes have already gone by."""
        if want <= 0:
            return 0
        try:
            import torch as _t

            _t.multiprocessing.set_sharing_strategy("file_system")  # fd limit
        except Exception:
            pass
        from dataclasses import replace as _rep

        for n in dict.fromkeys((want, 2)):
            try:
                dl = make_loader(ds, raw, _rep(cfg, batch=2), collate, False, n)
                it = iter(dl)
                next(it)
                del it, dl
                if n != want:
                    log(f"[WORKER] {want} did not work, continuing with {n}")
                return n
            except Exception as e:
                log(f"[WORKER] num_workers={n} failed ({type(e).__name__}: "
                    f"{str(e)[:120]})")
        log("[WORKER] dropping to 0, the GPU will idle a little but the run is safe")
        return 0

    class Collate:
        """A class, not a closure, because spawn workers pickle the argument."""

        def __init__(self, pad):
            self.pad = pad

        def __call__(self, b):
            ws, ls, ix = zip(*b)
            wl = torch.tensor([len(w) for w in ws])
            ll = torch.tensor([len(l) for l in ls])
            X = torch.zeros(len(ws), int(wl.max()))
            Y = torch.zeros(len(ls), int(ll.max()), dtype=torch.long)
            for i, (w, l) in enumerate(zip(ws, ls)):
                X[i, : len(w)] = w
                Y[i, : len(l)] = l
            return X, Y, wl, ll, torch.tensor(ix)

    return Collate, SpeechDS, build_vocab, make_loader, prepare, probe_workers


@app.cell
def _(mo):
    banks_btn = mo.ui.run_button(label="Download the ESC-50 and UrbanSound8K noise banks")
    mo.vstack([mo.md("**Real noise banks** — about 800 wav files streamed from "
                     "HuggingFace. Only the `noise_real` axis needs them; every other "
                     "axis uses synthetic pink noise."),
               banks_btn])
    return (banks_btn,)


@app.cell
def _(NOISE_ROOT, Path, banks_btn, io, mo, np):
    import soundfile as sf
    from datasets import load_dataset, Audio

    # The ESC-50 classes that make sense in a 911 context (cow/frog/rooster dropped)
    ESC50_911 = {"siren", "car_horn", "engine", "train", "helicopter", "airplane",
                 "fireworks", "glass_breaking", "clock_alarm", "crying_baby",
                 "coughing", "sneezing", "footsteps", "clapping", "laughing",
                 "thunderstorm", "wind", "rain", "crackling_fire", "chainsaw"}

    def _dec(cell, sr=16000):
        if isinstance(cell, dict) and cell.get("array") is not None:
            w, s = np.asarray(cell["array"], np.float32), cell["sampling_rate"]
        elif isinstance(cell, dict) and cell.get("bytes"):
            w, s = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif isinstance(cell, dict) and cell.get("path"):
            w, s = sf.read(cell["path"], dtype="float32")
        else:
            return None
        w = np.asarray(w, np.float32)
        if w.ndim > 1:
            w = w.mean(1)
        if int(s) != sr:  # simple resampling
            w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr / s)),
                          np.arange(len(w)), w).astype(np.float32)
        return w

    def dump_noise(repo, out, keep=None, col="category", n=400, sr=16000):
        d = Path(out); d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob("*.wav")))
        if have >= 50:
            print(f"{out}: already has {have} files, skipped"); return
        ds = load_dataset(repo, split="train", streaming=True)
        try:
            ds = ds.cast_column("audio", Audio(decode=False))   # so it will not want torchcodec
        except Exception:
            pass
        i = 0
        for r in ds:
            if i >= n: break
            if keep and str(r.get(col, "")).lower() not in keep: continue
            w = _dec(r.get("audio"), sr)
            if w is None or len(w) < sr // 2: continue
            pk = float(np.abs(w).max())
            if pk < 1e-4: continue                    # drop the silent clips
            sf.write(d / f"{i:05d}.wav", w / pk * 0.9, sr); i += 1
        print(f"{repo} -> {out} ({i} files)")

    # Guarded: marimo runs every cell when the file is opened, and this streams
    # two HuggingFace datasets and writes ~800 wav files. It must be a decision,
    # not a side effect of opening the notebook.
    mo.stop(not banks_btn.value,
            mo.md("*Noise banks not downloaded. Press the button above if an "
                  "axis needs a real bank; synthetic pink noise is used otherwise.*"))
    dump_noise("ashraq/esc50", str(NOISE_ROOT / "esc50"), ESC50_911)   # siren, scream, glass
    dump_noise("danavery/urbansound8K", str(NOISE_ROOT / "urban"))     # 10 classes, all urban
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Model and training
    """)
    return


@app.cell
def _(BACKBONE, log, nn, torch):
    def build_backbone(cfg, device):
        from transformers import HubertModel
        from peft import LoraConfig, inject_adapter_in_model

        _kw = dict(
            mask_time_prob=cfg.mask_time_prob, mask_time_length=10,
            mask_feature_prob=cfg.mask_feature_prob, mask_feature_length=10,
            apply_spec_augment=cfg.spec_on,
            # CRITICAL. bb.train() turns on SpecAugment but also turns on the backbone's OWN
            # dropouts (HuBERT defaults to 0.1). Without zeroing them the
            # X_specaug axis measures "SpecAugment + dropout" and is no longer isolated.
            hidden_dropout=cfg.bb_dropout,
            attention_dropout=cfg.bb_dropout,
            activation_dropout=cfg.bb_dropout,
            feat_proj_dropout=cfg.bb_dropout,
            final_dropout=cfg.bb_dropout,
            layerdrop=0.0,  # >0 can make the very layer weighted-sum reads disappear
        )
        # SDPA: it does NOT materialize the [B,12,T,T] attention matrix. In a 20 s batch
        # T=1000 means about 1.5 GB per layer, roughly 18 GB for 12 layers. The single largest VRAM item.
        try:
            bb = HubertModel.from_pretrained(BACKBONE, attn_implementation="sdpa", **_kw)
            log("[BB] attention: sdpa")
        except Exception as _e:
            bb = HubertModel.from_pretrained(BACKBONE, **_kw)
            log(f"[BB] attention: eager (no sdpa: {type(_e).__name__}), VRAM will be high")
        bb = bb.to(device)
        if cfg.grad_ckpt:
            bb.gradient_checkpointing_enable()
            log("[BB] gradient checkpointing ON (about 30% slower, far less VRAM)")
        cfgl = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "v_proj"], bias="none",
            layers_to_transform=[i - 1 for i in cfg.lora_layers],
        )
        bb = inject_adapter_in_model(cfgl, bb)
        for n, p in bb.named_parameters():
            p.requires_grad = "lora_" in n
        got, exp = sum(p.numel() for p in bb.parameters() if p.requires_grad), cfg.n_adapter()
        log(f"[LORA] {got:,} param (expected {exp:,}) · dropout={cfg.lora_dropout}")
        assert got == exp, f"LoRA has the wrong scope: {got:,} != {exp:,}"
        bb.eval()
        return bb, bb._get_feat_extract_output_lengths

    class Head(nn.Module):
            """x: [B,T,N,D] -> (logits, aux)"""

            def __init__(self, n, dim, V, dropout=0.0, aux_idx=None,
                         ws_drop=0.0, hs_noise=0.0):
                super().__init__()
                self.n, self.aux_idx = n, aux_idx
                self.ws_drop, self.hs_noise = ws_drop, hs_noise
                self.layer_w = nn.Parameter(torch.zeros(n))
                # Dropout is ALWAYS in the stack (a no-op at p=0). Adding it conditionally
                # shifted the Sequential indices and the output layer started from random weights.
                self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(),
                                         nn.Dropout(dropout), nn.Linear(dim, V))
                self.aux = nn.Linear(dim, V) if aux_idx is not None else None

            def weights(self):
                return self.layer_w.softmax(0)

            def forward(self, x):
                if self.training and self.hs_noise > 0:
                    x = x + torch.randn_like(x) * (self.hs_noise * x.std())
                w = self.layer_w.softmax(0)
                if self.training and self.ws_drop > 0 and self.n > 1:
                    m = (torch.rand(self.n, device=x.device) > self.ws_drop).float()
                    if m.sum() == 0:
                        m = torch.ones_like(m)
                    w = w * m
                    w = w / w.sum()          # redistribute over the remaining layers
                f = (x * w[None, None, :, None]).sum(2)
                a = self.aux(x[:, :, self.aux_idx, :]) if self.aux is not None else None
                return self.net(f), a

    class EMA:
        """Cheap, and the literature reports a small but consistent gain."""

        def __init__(self, ps, decay):
            self.d, self.ps = decay, list(ps)
            self.sh = [p.detach().clone() for p in self.ps]
            self.bk, self.t = None, 0

        def update(self):
            if self.d <= 0:
                return
            self.t += 1
            d = min(self.d, (1.0 + self.t) / (10.0 + self.t))
            for s, p in zip(self.sh, self.ps):
                s.mul_(d).add_(p.detach(), alpha=1 - d)

        def apply(self):
            if self.d > 0:
                self.bk = [p.detach().clone() for p in self.ps]
                for s, p in zip(self.sh, self.ps):
                    p.data.copy_(s)

        def restore(self):
            if self.d > 0 and self.bk:
                for b, p in zip(self.bk, self.ps):
                    p.data.copy_(b)
                self.bk = None

    return EMA, Head, build_backbone


@app.cell
def _(
    Collate,
    EMA,
    Head,
    OUT_ROOT,
    SpeechDS,
    build_backbone,
    build_vocab,
    contextlib,
    gc,
    groupby,
    json,
    log,
    make_loader,
    nn,
    np,
    prepare,
    probe_workers,
    random,
    time,
    torch,
):
    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    def decode(ids, i2c, blank, unk):
        return "".join(i2c.get(k, "") for k, _ in groupby(ids)
                       if k not in (blank, unk)).replace("|", " ").strip()

    @torch.no_grad()
    def evaluate(head, bb, dl, texts, flen, cfg, dev, i2c, blank, unk):
        import jiwer

        head.eval()
        bb.eval()  # ALWAYS off during eval, no dropout and no SpecAugment
        H, R = [], []
        for X, _, wl, _, ix in dl:
            X = X.to(dev, non_blocking=True)
            am = (torch.arange(X.shape[1], device=dev)[None, :] < wl.to(dev)[:, None]).long()
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                o = bb(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([o.hidden_states[L] for L in cfg.ws], 2)
            xl = flen(wl.to(dev))
            pr = head(hs.float())[0].argmax(-1).cpu().numpy()
            for b, i in enumerate(ix.tolist()):
                H.append(decode(pr[b, : int(xl[b])].tolist(), i2c, blank, unk))
                R.append(texts[i])
        return jiwer.wer(R, H), jiwer.cer(R, H)

    def train_one(cfg, noise_bank=None):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(cfg.seed)
        if dev == "cuda":
            # Release the blocks left over from the previous run and reset the peak counter.
            # nvidia-smi reports RESERVED memory, so without clearing it the previous run leaves a trace.
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg.dir.mkdir(parents=True, exist_ok=True)
        (cfg.dir / "config.json").write_text(json.dumps(cfg.d(), indent=2, default=str))
        log(cfg.line())

        vocab = build_vocab()
        blank, unk = vocab["[PAD]"], vocab["[UNK]"]
        i2c = {v: k for k, v in vocab.items()}
        tr_raw = prepare("train", cfg.sr, cfg.max_secs, cfg.ram)
        dv_raw = prepare("dev", cfg.sr, cfg.max_secs, cfg.ram)
        aug = cfg.aug if cfg.aug.on() else None
        tr = SpeechDS(tr_raw, vocab, cfg, aug=aug, noise_bank=noise_bank)
        dv = SpeechDS(dv_raw, vocab, cfg)  # dev is always clean
        col = Collate(blank)
        # Tying the workers to augmentation was a MISTAKE. Even without aug, collate fills a
        # [64, 320k] tensor on a single thread while the GPU sits idle.
        # Rehearse BEFORE training. If fork does not work we want to know now, not minutes later.
        nw = probe_workers(tr, tr_raw, cfg, col, cfg.workers)
        tdl = make_loader(tr, tr_raw, cfg, col, True, nw)
        ddl = make_loader(dv, dv_raw, cfg, col, False, min(nw, 4))
        _bt = np.diff(tr_raw[1])
        log(f"[DATA] train={len(tr)} dev={len(dv)} · ~{len(tdl)} batch/ep · nw={nw} "
            f"- bucket={cfg.bucket} - truncation={cfg.max_secs or 'NONE'} "
            f"- max {_bt.max() / cfg.sr:.1f} s - budget "
            f"{cfg.batch}×{cfg.batch_secs:.0f}s")

        bb, flen = build_backbone(cfg, dev)
        aux_idx = list(cfg.ws).index(min(cfg.ws)) if cfg.aux_ctc_weight > 0 else None
        head = Head(len(cfg.ws), cfg.hid, len(vocab), cfg.head_dropout, aux_idx, cfg.ws_layer_drop, cfg.hs_noise).to(dev)

        if cfg.init_from:
            src = OUT_ROOT / cfg.init_from
            if (src / "head.pt").exists():
                sd = torch.load(src / "head.pt", map_location=dev)
                if ("net.2.weight" in sd
                        and sd["net.2.weight"].shape[0] != sd["net.0.weight"].shape[0]):
                    sd["net.3.weight"] = sd.pop("net.2.weight")
                    sd["net.3.bias"] = sd.pop("net.2.bias")
                r = head.load_state_dict(sd, strict=False)
                crit = [k for k in r.missing_keys if not k.startswith("aux.")]
                if crit:
                    log(f"[INIT] WARNING, COULD NOT LOAD: {crit}. Starting from random, the comparison is invalid.")
                if (src / "adapter.pt").exists():
                    bb.load_state_dict(torch.load(src / "adapter.pt", map_location=dev), strict=False)
                log(f"[INIT] <- {src}")
            else:
                log(f"[WARN] '{cfg.init_from}' is missing, starting FROM SCRATCH, the comparison breaks")

        groups = [
            {"params": [p for n, p in head.named_parameters() if n != "layer_w"],
             "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
            {"params": [head.layer_w], "lr": cfg.w_lr, "weight_decay": 0.0},
            {"params": [p for p in bb.parameters() if p.requires_grad],
             "lr": cfg.lora_lr, "weight_decay": cfg.weight_decay},
        ]
        opt = torch.optim.AdamW(groups, fused=(dev == "cuda"))
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.5, patience=cfg.patience, threshold=0.005,
            threshold_mode="rel")
        ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        trainable = [p for g in opt.param_groups for p in g["params"]]
        ema = EMA(trainable, cfg.ema_decay)

        hp, lp = cfg.dir / "history.jsonl", cfg.dir / "last.pt"
        ep0, best, best_ep, hist = 1, float("inf"), 0, []
        if lp.exists():
            ck = torch.load(lp, map_location=dev, weights_only=False)
            head.load_state_dict(ck["head"])
            if ck.get("adapter"):
                bb.load_state_dict(ck["adapter"], strict=False)
            opt.load_state_dict(ck["opt"])
            with contextlib.suppress(Exception):
                sch.load_state_dict(ck["sch"])
            ep0, best, best_ep = ck["epoch"] + 1, ck["best"], ck["best_ep"]
            # do not crash when last.pt exists but history.jsonl does not (partial copy, manual delete)
            hist = ([json.loads(l) for l in hp.read_text().splitlines() if l.strip()]
                    if hp.exists() else [])
            log(f"[RESUME] e{ck['epoch']} · best {best * 100:.2f}%")
        if not hp.exists():
            hp.write_text("")

        stopped = None
        for ep in range(ep0, cfg.epochs + 1):
            if cfg.deadline and time.time() > cfg.deadline:
                stopped = "deadline"
                log("[DEADLINE] time is up, moving to evaluation")
                break
            head.train()
            # CRITICAL. v2 called bb.eval() unconditionally here, and HF applies masking
            # only while self.training is set, so SpecAugment had never actually run.
            bb.train() if cfg.bb_train else bb.eval()
            t0, tot, nb = time.perf_counter(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for X, Y, wl, ll, _ in tdl:
                X, Y = X.to(dev, non_blocking=True), Y.to(dev, non_blocking=True)
                am = (torch.arange(X.shape[1], device=dev)[None, :] < wl.to(dev)[:, None]).long()
                with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    o = bb(X, attention_mask=am, output_hidden_states=True)
                    hs = torch.stack([o.hidden_states[L] for L in cfg.ws], 2)
                xl = flen(wl.to(dev))
                lg, aux = head(hs.float())
                loss = ctc(lg.log_softmax(-1).transpose(0, 1), Y, xl, ll.to(dev))
                if aux is not None:
                    la = ctc(aux.log_softmax(-1).transpose(0, 1), Y, xl, ll.to(dev))
                    loss = (1 - cfg.aux_ctc_weight) * loss + cfg.aux_ctc_weight * la
                (loss / cfg.accum).backward()
                nb += 1
                # do NOT TRUST len(tdl). Under a frame budget the batch count changes from
                # epoch to epoch. We now collect the remainder after the loop.
                if nb % cfg.accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    ema.update()
                tot += loss.item()
            if nb % cfg.accum:  # the last accumulated gradients
                torch.nn.utils.clip_grad_norm_(trainable, cfg.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update()

            ema.apply()
            wer, cer = evaluate(head, bb, ddl, dv.texts, flen, cfg, dev, i2c, blank, unk)
            rec = {"epoch": ep, "loss": tot / max(1, nb), "wer": wer, "cer": cer,
                   "secs": time.perf_counter() - t0,
                   "w": head.weights().detach().cpu().numpy().round(4).tolist()}
            hist.append(rec)
            with hp.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if dev == "cuda":
                rec["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log(f"  e{ep:>3} | loss {rec['loss']:.3f} | {rec['secs']:.0f}s "
                f"| VAL wer {wer * 100:.2f} cer {cer * 100:.2f}"
                + (f" | peak {rec['vram_gb']:.1f} GB" if "vram_gb" in rec else ""))
            sch.step(cer)
            if cer < best * 0.995:
                best, best_ep = cer, ep
                torch.save(head.state_dict(), cfg.dir / "head.pt")
                torch.save({k: v.detach().cpu().clone() for k, v in bb.state_dict().items()
                            if "lora_" in k}, cfg.dir / "adapter.pt")
                log(f"     [SAVE] {cer * 100:.2f}%")
            ema.restore()
            torch.save({"head": head.state_dict(),
                        "adapter": {k: v.detach().cpu().clone()
                                    for k, v in bb.state_dict().items() if "lora_" in k},
                        "opt": opt.state_dict(), "sch": sch.state_dict(),
                        "epoch": ep, "best": best, "best_ep": best_ep}, lp)
            if ep - best_ep >= cfg.stop_patience:
                stopped = "early_stop"
                log("[STOP] no improvement")
                break

        # final: the best checkpoint, clean / tel / tel8k
        if (cfg.dir / "head.pt").exists():
            head.load_state_dict(torch.load(cfg.dir / "head.pt", map_location=dev), strict=False)
            if (cfg.dir / "adapter.pt").exists():  # a crash can land between the two saves
                bb.load_state_dict(torch.load(cfg.dir / "adapter.pt", map_location=dev),
                                   strict=False)
        conds = {}
        for m in cfg.conds:
            ds = SpeechDS(dv_raw, vocab, cfg, degrade=(None if m == "clean" else m))
            dl = make_loader(ds, dv_raw, cfg, col, False, min(nw, 4), persist=False)
            w_, c_ = evaluate(head, bb, dl, ds.texts, flen, cfg, dev, i2c, blank, unk)
            conds[m] = {"wer": w_, "cer": c_}
            log(f"  [{m:6s}] CER {c_ * 100:5.2f}  WER {w_ * 100:6.2f}")

        ratio = (conds["tel8k"]["wer"] / conds["clean"]["wer"]
                 if conds.get("clean", {}).get("wer") and "tel8k" in conds else None)
        s = {"run": cfg.run, "best_cer": best, "best_epoch": best_ep,
             "epochs_done": hist[-1]["epoch"] if hist else 0, "stopped": stopped,
             "final": conds, "ratio": ratio,
             "vram_peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                              if dev == "cuda" else None),
             "sec_per_epoch": float(np.median([h["secs"] for h in hist])) if hist else None,
             "aug": cfg.aug.names(), "reg": cfg.regs(), "history": hist}
        (cfg.dir / "summary.json").write_text(json.dumps(s, indent=2))
        log(f"[DONE] {cfg.run}: CER {best * 100:.2f}% @e{best_ep}"
            + (f" · tel8k/clean ×{ratio:.2f}" if ratio else ""))
        return s

    return evaluate, train_one


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Axes

    **Independent, not cumulative.** All of them warm start from the same base.
    Each axis is measured on its own, there is no ordering effect, and if the deadline
    cuts the run short whatever was measured stays valid. **combo** picks up the interactions.
    """)
    return


@app.cell
def _(Aug, Cfg, NOISE_ROOT, replace):
    SPEC = dict(mask_time_prob=0.05, mask_feature_prob=0.004)

    AXES = {
        "control":   {},                                              # reference
        "specaug":   SPEC,                                            # Park 2019
        "channel":   dict(aug=Aug(p_band=0.4, p_8k=0.3, p_mulaw=0.25)),  # G.711/712
        "noise":     dict(aug=Aug(p_noise=0.5)),                      # Snyder 2015
        "babble":    dict(aug=Aug(p_babble=0.5)),                     # call centre
        "speed":     dict(aug=Aug(speed=(0.9, 1.0, 1.1), p_speed=0.6)),  # Ko 2015
        "packet":    dict(aug=Aug(p_packet=0.4)),                     # VoIP loss
        "rir":       dict(aug=Aug(p_rir=0.4)),                        # Ko 2017
        "wd":        dict(weight_decay=0.01),
        "dropout":   dict(lora_dropout=0.05, head_dropout=0.1),
        "bbdrop":    dict(bb_dropout=0.05),   # now isolated from specaug
        "inputnorm": dict(input_norm="zscore"),
        "ema":       dict(ema_decay=0.999),
        "interctc":  dict(aux_ctc_weight=0.3),                        # Lee & Watanabe 2021

        # --- round 2: representation level (this family won round 1) ---
        "specaug_hi": dict(mask_time_prob=0.30, mask_feature_prob=0.05),
        "bbdrop_hi":  dict(bb_dropout=0.10),
        "wsdrop":     dict(ws_layer_drop=0.25),
        "hsnoise":    dict(hs_noise=0.10),
        "droplite":   dict(head_dropout=0.05),   # NO lora_dropout
        # --- round 2: real noise (do not run before the bank is downloaded) ---
        "noise_real": dict(aug=Aug(p_noise=0.5, noise_dir=str(NOISE_ROOT / "esc50"))),
        "noise_911":  dict(aug=Aug(p_noise=0.5, snr=(0.0, 15.0), noise_dir=str(NOISE_ROOT / "urban"))),
        # --- narrowed combination: the best from each family ---
        "combo2":     dict(mask_time_prob=0.05, mask_feature_prob=0.004, bb_dropout=0.05, aug=Aug(p_clean=0.5, p_band=0.4, p_8k=0.3, p_mulaw=0.25, p_rir=0.3, p_babble=0.3)),    
        "combo3":  dict(mask_time_prob=0.0, mask_feature_prob=0.0, bb_dropout=0.05,
                            aug=Aug(p_clean=0.5, p_band=0.4, p_8k=0.3, p_mulaw=0.25, p_noise=0.4)),
    }

    ORDER = ["combo3", "combo2", "specaug_hi", "bbdrop_hi", "wsdrop", "hsnoise", "droplite",
                 "noise_real", "noise_911",
                 "control", "specaug", "channel", "noise", "speed", "wd", "babble",
                 "dropout", "packet", "ema", "rir", "interctc", "bbdrop", "inputnorm"]
    # inputnorm goes last. BASE was trained on UNNORMALISED input, and a warm start
    # cannot absorb the distribution shift in 10 epochs, so it looks unfairly bad.
    # The real test is a run from scratch.
    BASE = "BASE"

    def make_abl(base: Cfg, name, epochs, deadline=None):
        c = replace(base, run=f"X_{name}", init_from=base.run, epochs=epochs,
                    head_lr=5e-4, lora_lr=1e-4, w_lr=5e-4,
                    stop_patience=max(4, epochs // 2), patience=3,
                    deadline=deadline, aug=Aug())
        return replace(c, **AXES[name])

    def merge(base: Cfg, names, epochs, deadline=None):
        """Merge the winners. Probabilities take the max, scalars take the last writer."""
        over, ak = {}, {}
        for n in names:
            for k, v in AXES[n].items():
                if k == "aug":
                    for f in v._K:
                        ak[f] = max(ak.get(f, 0.0), getattr(v, f))
                    if v.speed:
                        ak["speed"] = v.speed
                else:
                    over[k] = v
        c = make_abl(base, "control", epochs, deadline)
        if ak:
            ak["p_clean"] = 0.6
            over["aug"] = Aug(**ak)
        return replace(c, run="COMBO", **over)

    def winners(out_root, tol=1.05):
        """Relative to control. Improves the ratio without hurting clean by more than 5%, OR improves clean."""
        import json as _j

        def L(n):
            p = out_root / n / "summary.json"
            return _j.loads(p.read_text()) if p.exists() else None

        c = L("X_control")
        if not c or not c["final"].get("clean", {}).get("wer"):
            return [], None
        cc, cr = c["final"]["clean"]["wer"], c.get("ratio")
        out = []
        for n in ORDER:
            if n == "control":
                continue
            s = L(f"X_{n}")
            if not s or not s["final"].get("clean", {}).get("wer"):
                continue
            cl, ra = s["final"]["clean"]["wer"], s.get("ratio")
            if cl <= cc * tol and (cl < cc or (ra and cr and ra < cr)):
                out.append(n)
        return out, c

    def plan(hours, base_ep, abl_ep, spe, have_base, n_sel):
        pe = spe / 60.0
        ov = 3 * pe * 0.35  # final eval over 3 conditions
        b = 0.0 if have_base else base_ep * pe + ov
        a = abl_ep * pe + ov
        fits = max(0, int((hours * 60 - b - a) // a))
        n = min(fits, n_sel)
        return {"pe": pe, "base": b, "abl": a, "n": n, "total": b + n * a + a}

    return BASE, ORDER, make_abl, merge, plan, winners


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Hugging Face  *(optional)*

    The base checkpoint is pulled from here and the results are uploaded here.
    Leave it empty and everything stays local, which is fine as long as the session survives.

    Token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Write**
    """)
    return


@app.cell
def _(mo, os):
    tok_ui = mo.ui.text(label="HF token", kind="password", full_width=True,
                        placeholder="hf_...  (if empty, HF_TOKEN is tried)")
    repo_ui = mo.ui.text(label="Repo", value=os.environ.get("ECAD_HF_REPO", ""),
                         placeholder="tuna/clear-phase1-runs  (empty means local)",
                         full_width=True)
    check_btn = mo.ui.run_button(label="Verify and list the runs")
    mo.vstack([tok_ui, repo_ui, check_btn])
    return check_btn, repo_ui, tok_ui


@app.cell
def _(OUT_ROOT, check_btn, json, log, mo, os, repo_ui, tok_ui):
    def _tok():
        t = tok_ui.value.strip() or os.environ.get("HF_TOKEN", "").strip()
        if t:
            return t
        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception:
            return None

    def hf_runs(repo):
        from huggingface_hub import list_repo_files

        return sorted({f.split("/")[0] for f in
                       list_repo_files(repo, repo_type="dataset", token=_tok())
                       if f.endswith("/summary.json")})

    def hf_pull(repo, run=None):
        """Download the run with the best CER. The architecture is derived from config.json,
        If ws_layers or lora_layers disagree, the head and adapter will not load."""
        from huggingface_hub import hf_hub_download

        t = _tok()
        best, bc = run, float("inf")
        if not run:
            for r in hf_runs(repo):
                try:
                    p = hf_hub_download(repo, f"{r}/summary.json", repo_type="dataset", token=t)
                    c = json.loads(open(p).read()).get("best_cer", float("inf"))
                    log(f"[HF] {r}: {c * 100:.2f}%")
                    if c < bc:
                        best, bc = r, c
                except Exception:
                    pass
        if not best:
            raise FileNotFoundError("no suitable run")
        d = OUT_ROOT / best
        d.mkdir(parents=True, exist_ok=True)
        arch = {}
        for fn in ("head.pt", "adapter.pt", "config.json"):
            try:
                p = hf_hub_download(repo, f"{best}/{fn}", repo_type="dataset", token=t)
                (d / fn).write_bytes(open(p, "rb").read())
                if fn == "config.json":
                    c = json.loads((d / fn).read_text())
                    for k in ("ws", "ws_layers"):
                        if k in c:
                            arch["ws"] = tuple(c[k])
                    if "lora_layers" in c and c["lora_layers"]:
                        arch["lora_layers"] = tuple(c["lora_layers"])
                    for k in ("lora_r", "lora_alpha"):
                        if k in c:
                            arch[k] = c[k]
            except Exception:
                pass
        if not (d / "head.pt").exists():
            raise FileNotFoundError(f"{best}/head.pt could not be downloaded")
        log(f"[HF] base '{best}' ready - architecture {arch}")
        return best, arch

    def hf_push(run_name, repo):
        t = _tok()
        if not (repo and t):
            return
        try:
            from huggingface_hub import HfApi, create_repo

            create_repo(repo, repo_type="dataset", private=True, exist_ok=True, token=t)
            HfApi(token=t).upload_folder(
                folder_path=str(OUT_ROOT / run_name), path_in_repo=run_name,
                repo_id=repo, repo_type="dataset", ignore_patterns=["last.pt"])
            log(f"[PUSH] {run_name} -> {repo}")
        except Exception as e:
            log(f"[PUSH] failed: {type(e).__name__}: {e}")

    if not check_btn.value:
        RUNS, hf_md = [], mo.md("*Waiting for verification (optional).*")
    else:
        _r = repo_ui.value.strip()
        with mo.redirect_stdout():
            if not _r:
                RUNS, hf_md = [], mo.md("No repo, everything stays in the local `runs/` folder.")
            elif not _tok():
                RUNS, hf_md = [], mo.md("No token, the base cannot be pulled and nothing can be pushed.")
            else:
                try:
                    from huggingface_hub import HfApi

                    who = HfApi(token=_tok()).whoami()["name"]
                    RUNS = hf_runs(_r)
                    hf_md = mo.md(f"`{who}` - found **{len(RUNS)}** runs"
                                  + (f": `{'`, `'.join(RUNS)}`" if RUNS else " (the repo is empty)"))
                except Exception as _e:
                    RUNS, hf_md = [], mo.md(f"❌ {type(_e).__name__}: {_e}")
    hf_md
    return RUNS, hf_pull, hf_push


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Budget
    """)
    return


@app.cell
def _(ORDER, RUNS, mo):
    hours_ui = mo.ui.slider(1, 12, value=6, step=0.5, label="Budget (hours)", show_value=True)
    base_ep_ui = mo.ui.slider(10, 60, value=30, step=5, label="Base epoch", show_value=True)
    # 10 epochs: 13 axes plus combo fit into 6 hours at 130 s/epoch. Going to 12 drops 2 axes.
    abl_ep_ui = mo.ui.slider(4, 30, value=10, step=1, label="Ablation epoch", show_value=True)
    spe_ui = mo.ui.number(30, 600, value=130, step=10, label="Epoch time estimate (s)")
    base_ui = mo.ui.dropdown(["(train from scratch)", "(HF: best CER)"] + list(RUNS),
                             value="(HF: best CER)" if RUNS else "(train from scratch)",
                             label="Base")
    axes_ui = mo.ui.multiselect(options=list(ORDER), value=list(ORDER),
                                label="Axes", full_width=True)
    combo_ui = mo.ui.switch(label="Run COMBO at the end", value=True)
    ram_ui = mo.ui.switch(label="Load the audio into RAM (about 13 GB, instead of memmap)", value=False)
    bs_ui = mo.ui.dropdown(["64", "32", "16"], value="64",
                           label="Batch (lower it if VRAM is tight, accumulation compensates)")
    mo.vstack([hours_ui, base_ep_ui, abl_ep_ui, spe_ui, base_ui, axes_ui,
               mo.hstack([combo_ui, ram_ui], justify="start", gap=2), bs_ui])
    return (
        abl_ep_ui,
        axes_ui,
        base_ep_ui,
        base_ui,
        bs_ui,
        combo_ui,
        hours_ui,
        ram_ui,
        spe_ui,
    )


@app.cell
def _(
    BASE,
    ORDER,
    OUT_ROOT,
    abl_ep_ui,
    axes_ui,
    base_ep_ui,
    base_ui,
    hours_ui,
    mo,
    plan,
    spe_ui,
):
    _have = base_ui.value != "(train from scratch)" or (OUT_ROOT / BASE / "head.pt").exists()
    _sel = [a for a in ORDER if a in axes_ui.value]
    _p = plan(hours_ui.value, base_ep_ui.value, abl_ep_ui.value, spe_ui.value,
              _have, len(_sel))
    _fit, _skip = _sel[: _p["n"]], _sel[_p["n"]:]
    mo.md(
        f"""
    | stage | time |
    |---|---|
    | base | {"**0 min** (ready)" if _have else f"**{_p['base']:.0f} min** ({base_ep_ui.value} ep)"} |
    | ablation | **{_p['abl']:.0f} min** × {_p['n']} |
    | combo | **{_p['abl']:.0f} min** |
    | **total** | **{_p['total'] / 60:.1f} h** / {hours_ui.value:.1f} h budget |

    **Will run ({len(_fit)}):** `{'`, `'.join(_fit) or '-'}`

    {("**Does not fit:** `" + "`, `".join(_skip) + "`") if _skip else "All of them fit."}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Run
    """)
    return


@app.cell
def _(mo):
    go_btn = mo.ui.run_button(label="Start the night")
    go_btn
    return (go_btn,)


@app.cell
def _(
    BASE,
    Cfg,
    ORDER,
    OUT_ROOT,
    abl_ep_ui,
    axes_ui,
    base_ep_ui,
    base_ui,
    bs_ui,
    combo_ui,
    gc,
    go_btn,
    hf_pull,
    hf_push,
    hours_ui,
    log,
    make_abl,
    merge,
    mo,
    ram_ui,
    repo_ui,
    spe_ui,
    time,
    torch,
    traceback,
    train_one,
    winners,
):
    mo.stop(not go_btn.value, mo.md("*Waiting to run.*"))

    with mo.redirect_stdout():
        _t0 = time.time()
        _dl = _t0 + hours_ui.value * 3600
        _repo = repo_ui.value.strip() or None
        _spe = spe_ui.value

        # --- base ---
        _name, _arch, _have = BASE, {}, False
        if base_ui.value != "(train from scratch)" and _repo:
            try:
                _name, _arch = hf_pull(
                    _repo, None if base_ui.value == "(HF: best CER)" else base_ui.value)
                _have = True
            except Exception as _e:
                log(f"[HF] could not pull the base ({type(_e).__name__}: {_e}), from scratch")
        if not _have and (OUT_ROOT / BASE / "head.pt").exists():
            log("[BASE] ready locally")
            _have = True

        _bs = int(bs_ui.value)
        base_cfg = Cfg(run=_name, epochs=base_ep_ui.value, deadline=_dl,
                       ram=ram_ui.value, batch=_bs,
                       accum=max(1, 256 // _bs),  # effective batch fixed at 256
                       **_arch)
        if not _have:
            log(f"\n### BASE - {base_ep_ui.value} epochs from scratch")
            _s = train_one(base_cfg)
            _spe = _s.get("sec_per_epoch") or _spe
            hf_push(base_cfg.run, _repo)
        else:
            log(f"### BASE ready: {_name}")

        # --- axes ---
        for _ax in [a for a in ORDER if a in axes_ui.value]:
            _c = make_abl(base_cfg, _ax, abl_ep_ui.value, _dl)
            if (_c.dir / "summary.json").exists():
                log(f"[SKIP] {_c.run}")
                continue
            _left = (_dl - time.time()) / 60
            _need = abl_ep_ui.value * _spe / 60 * 1.15
            if _left < _need:
                log(f"[BUDGET] {_ax} skipped, {_left:.0f} min left < {_need:.0f} min")
                continue
            log(f"\n### {_ax}  ({_left:.0f} min left)")
            try:
                _s = train_one(_c)
                _spe = 0.5 * _spe + 0.5 * (_s.get("sec_per_epoch") or _spe)
                hf_push(_c.run, _repo)
            except Exception:  # one failing axis must not end the whole night
                log(f"[ERROR] {_ax}:\n{traceback.format_exc()}")
            finally:
                gc.collect()
                torch.cuda.is_available() and torch.cuda.empty_cache()

        # --- combo ---
        _w, _ = winners(OUT_ROOT)
        log(f"\n### WINNERS: {_w or 'none'}")
        if combo_ui.value and _w and (_dl - time.time()) / 60 > abl_ep_ui.value * _spe / 60:
            try:
                _cc = merge(base_cfg, _w, abl_ep_ui.value, _dl)
                log(f"\n### COMBO — {_cc.line()}")
                train_one(_cc)
                hf_push("COMBO", _repo)
            except Exception:
                log(f"[ERROR] combo:\n{traceback.format_exc()}")

        log(f"\nElapsed: {(time.time() - _t0) / 60:.0f} min")

    night_done = time.time()
    return (base_cfg,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Results
    """)
    return


@app.cell
def _(mo):
    rep_btn = mo.ui.run_button(label="Refresh (works while a run is going)")
    rep_btn
    return (rep_btn,)


@app.cell
def _(BASE, OUT_ROOT, json, mo, rep_btn):
    rep_btn
    _rows = []
    for _p in sorted(OUT_ROOT.glob("*/summary.json")):
        try:
            _s = json.loads(_p.read_text())
        except Exception:
            continue
        _f = _s.get("final", {})
        _g = lambda k: _f.get(k, {}).get("wer", float("nan")) * 100
        _rows.append((_s.get("run", _p.parent.name), (_s.get("best_cer") or float("nan")) * 100,
                          _g("clean"), _g("tel"), _g("tel8k"), _g("noisy"),
                          _s.get("ratio") or float("nan"), _s.get("epochs_done", 0)))
    if not _rows:
        table_md = mo.md("*No results yet.*")
    else:
        _c = next((r for r in _rows if r[0] == "X_control"), None)
        _b = []
        for r in sorted(_rows, key=lambda x: (x[0] != BASE, x[0])):
            _d = f"{r[2] - _c[2]:+.2f}" if _c and r[2] == r[2] else "—"
            _b.append(f"| `{r[0]}` | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | "
                      f"{r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {_d} | {r[7]} |")
        table_md = mo.md(
            "| run | valCER | clean W | tel W | tel8k W | noisy W | tel8k/clean | d-clean | ep |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(_b)
            + "\n\n**Rule:** take the axis that lowers the `tel8k/clean` ratio while keeping "
              "`d-clean` within +5% relative. Compare against `X_control`, not against `BASE`.\n\n"
              "*KenLM is not in the ablation. It is a fixed offset, it does not change the ranking, "
              "and it costs decode time. Enable it once, after the winner is chosen.*"
        )
    table_md
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 · Final

    Run the winning setup long and **from scratch**, not from a warm start. If your
    observation that it plateaus at 50 epochs holds, the final number should come from a clean run.
    """)
    return


@app.cell
def _(mo):
    final_ep_ui = mo.ui.slider(20, 100, value=50, step=5, label="Final epoch", show_value=True)
    final_btn = mo.ui.run_button(label="Run FINAL")
    mo.hstack([final_ep_ui, final_btn], justify="start", gap=1)
    return final_btn, final_ep_ui


@app.cell
def _(
    OUT_ROOT,
    base_cfg,
    final_btn,
    final_ep_ui,
    hf_push,
    log,
    merge,
    mo,
    replace,
    repo_ui,
    train_one,
    winners,
):
    mo.stop(not final_btn.value, mo.md("*Waiting for FINAL.*"))
    with mo.redirect_stdout():
        _w, _ = winners(OUT_ROOT)
        if not _w:
            log("No winner. No axis beat control without hurting clean. "
                "That is a result too, augmentation does not help at this budget.")
            final_res = None
        else:
            _c = merge(base_cfg, _w, final_ep_ui.value)
            _c = replace(_c, run="FINAL", init_from=None, epochs=final_ep_ui.value,
                         head_lr=1e-3, lora_lr=2e-4, w_lr=1e-3, stop_patience=12,
                         patience=4, deadline=None)
            log(f"FINAL — winners {_w}\n{_c.line()}")
            final_res = train_one(_c)
            hf_push("FINAL", repo_ui.value.strip() or None)
    final_res
    return


@app.cell
def _(mo):
    reeval_btn = mo.ui.run_button(label="Backfill the missing conditions (rewrites summary.json)")
    mo.vstack([mo.md("**Backfill.** Re-evaluates every finished run whose "
                     "`summary.json` was written before the `noisy` condition "
                     "existed, and **rewrites those files in place**."),
               reeval_btn])
    return (reeval_btn,)


@app.cell
def _(
    Aug,
    Cfg,
    Collate,
    Head,
    OUT_ROOT,
    SpeechDS,
    build_backbone,
    build_vocab,
    evaluate,
    gc,
    json,
    make_loader,
    mo,
    reeval_btn,
    prepare,
    replace,
    torch,
):
    def reeval(run, conds=("clean", "tel", "tel8k", "noisy")):
        d = OUT_ROOT / run
        if not (d / "head.pt").exists():
            print(f"{run}: no head.pt, skipped"); return
        raw = json.loads((d / "config.json").read_text())
        keep = {k: v for k, v in raw.items() if k in Cfg.__dataclass_fields__}
        if isinstance(keep.get("aug"), dict):
            keep["aug"] = Aug(**keep["aug"])
        cfg = replace(Cfg(**keep), run=run, conds=tuple(conds), deadline=None)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = build_vocab(); blank, unk = vocab["[PAD]"], vocab["[UNK]"]
        i2c = {v: k for k, v in vocab.items()}
        dv_raw = prepare("dev", cfg.sr, cfg.max_secs, cfg.ram)
        col = Collate(blank)
        bb, flen = build_backbone(cfg, dev)
        aux_idx = list(cfg.ws).index(min(cfg.ws)) if cfg.aux_ctc_weight > 0 else None
        head = Head(len(cfg.ws), cfg.hid, len(vocab), cfg.head_dropout, aux_idx,
                    cfg.ws_layer_drop, cfg.hs_noise).to(dev)
        sd = torch.load(d / "head.pt", map_location=dev)
        if ("net.2.weight" in sd
                and sd["net.2.weight"].shape[0] != sd["net.0.weight"].shape[0]):
            sd["net.3.weight"] = sd.pop("net.2.weight")
            sd["net.3.bias"] = sd.pop("net.2.bias")
        head.load_state_dict(sd, strict=False)
        if (d / "adapter.pt").exists():
            bb.load_state_dict(torch.load(d / "adapter.pt", map_location=dev), strict=False)

        out = {}
        for m in conds:
            ds = SpeechDS(dv_raw, vocab, cfg, degrade=(None if m == "clean" else m))
            dl = make_loader(ds, dv_raw, cfg, col, False, 4, persist=False)
            w_, c_ = evaluate(head, bb, dl, ds.texts, flen, cfg, dev, i2c, blank, unk)
            out[m] = {"wer": w_, "cer": c_}
        s = json.loads((d / "summary.json").read_text())
        s["final"] = out
        s["ratio"] = out["tel8k"]["wer"] / out["clean"]["wer"]
        (d / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"{run:14s} clean {out['clean']['wer']*100:5.2f} · "
              f"tel8k {out['tel8k']['wer']*100:5.2f} · noisy {out['noisy']['wer']*100:5.2f} · "
              f"ratio {s['ratio']:.2f}")
        del bb, head; gc.collect(); torch.cuda.empty_cache()

    # Guarded: this rewrites summary.json in place for every finished run, so on
    # load it would silently overwrite recorded results.
    mo.stop(not reeval_btn.value,
            mo.md("*Backfill not run. Press the button above to re-evaluate the "
                  "runs whose `summary.json` predates the `noisy` condition — it "
                  "rewrites those files in place.*"))
    for _r in sorted(p.parent.name for p in OUT_ROOT.glob("*/summary.json")):
        if "noisy" not in json.loads((OUT_ROOT / _r / "summary.json").read_text()).get("final", {}):
            reeval(_r)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## KenLM - Stage 1, dev logit dump

    `dump_dev_logits(run)` writes the winning run's dev-clean **log_softmax** logits and
    references to `runs/<run>/dev_logits.npz`. Then, standalone:
    `python kenlm_grid.py --grid --npz runs/<run>/dev_logits.npz`.
    Model loading and decoding are identical to `reeval`, except the logits are stored instead of the argmax.
    """)
    return


@app.cell
def _(Aug, Cfg, Collate, Head, OUT_ROOT, SpeechDS, build_backbone, build_vocab,
      gc, json, make_loader, np, prepare, replace, torch):
    def dump_dev_logits(run, max_utt=None):
        """Produce runs/<run>/dev_logits.npz: for every dev-clean utterance the
        log_softmax logits [T,V] plus the reference text. `kenlm_grid --grid` reads it."""
        d = OUT_ROOT / run
        if not (d / "head.pt").exists():
            print(f"{run}: no head.pt, skipped"); return None
        # architecture from config.json (identical to reeval)
        raw = json.loads((d / "config.json").read_text())
        keep = {k: v for k, v in raw.items() if k in Cfg.__dataclass_fields__}
        if isinstance(keep.get("aug"), dict):
            keep["aug"] = Aug(**keep["aug"])
        cfg = replace(Cfg(**keep), run=run, conds=("clean",), deadline=None)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = build_vocab(); blank = vocab["[PAD]"]
        dv_raw = prepare("dev", cfg.sr, cfg.max_secs, cfg.ram)
        col = Collate(blank)
        bb, flen = build_backbone(cfg, dev)
        aux_idx = list(cfg.ws).index(min(cfg.ws)) if cfg.aux_ctc_weight > 0 else None
        head = Head(len(cfg.ws), cfg.hid, len(vocab), cfg.head_dropout, aux_idx,
                    cfg.ws_layer_drop, cfg.hs_noise).to(dev)
        sd = torch.load(d / "head.pt", map_location=dev)
        if ("net.2.weight" in sd
                and sd["net.2.weight"].shape[0] != sd["net.0.weight"].shape[0]):
            sd["net.3.weight"] = sd.pop("net.2.weight")
            sd["net.3.bias"] = sd.pop("net.2.bias")
        head.load_state_dict(sd, strict=False)
        if (d / "adapter.pt").exists():
            bb.load_state_dict(torch.load(d / "adapter.pt", map_location=dev), strict=False)
        head.eval(); bb.eval()

        ds = SpeechDS(dv_raw, vocab, cfg, degrade=None)     # clean
        dl = make_loader(ds, dv_raw, cfg, col, False, 4, persist=False)
        logits_list, refs = [], []
        with torch.no_grad():
            for X, _, wl, _, ix in dl:
                X = X.to(dev, non_blocking=True)
                am = (torch.arange(X.shape[1], device=dev)[None, :]
                      < wl.to(dev)[:, None]).long()
                with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    o = bb(X, attention_mask=am, output_hidden_states=True)
                    hs = torch.stack([o.hidden_states[L] for L in cfg.ws], 2)
                xl = flen(wl.to(dev))
                lg = head(hs.float())[0].log_softmax(-1)    # [B,T,V]
                for b, i in enumerate(ix.tolist()):
                    T = int(xl[b])
                    logits_list.append(lg[b, :T].detach().cpu().numpy().astype("float32"))
                    refs.append(ds.texts[i])
                if max_utt and len(refs) >= max_utt:
                    break

        arr = np.empty(len(logits_list), dtype=object)      # variable T, so an object array
        for k, a in enumerate(logits_list):
            arr[k] = a
        out = d / "dev_logits.npz"
        np.savez(out, logits=arr, refs=np.array(refs))
        print(f"[DUMP] {run}: {len(refs)} utterance -> {out} "
              f"(~{out.stat().st_size/1e6:.0f} MB)")
        del bb, head; gc.collect(); torch.cuda.empty_cache()
        return out

    return (dump_dev_logits,)


@app.cell
def _():
    # KenLM stage 1, run it on the winner (we will test both).
    # dump_dev_logits("FINAL")
    # dump_dev_logits("FINAL_hdo")
    # then standalone:
    #   python kenlm_grid.py --grid --npz runs/FINAL/dev_logits.npz
    #   python kenlm_grid.py --grid --npz runs/FINAL_hdo/dev_logits.npz
    return


@app.cell
def _(mo):
    final_pair_btn = mo.ui.run_button(label="Train FINAL and FINAL_hdo (2 x 75 epochs)")
    mo.vstack([mo.md("**The two long runs.** `FINAL` is the winning configuration; "
                     "`FINAL_hdo` is the same with head dropout at 0.05."),
               final_pair_btn])
    return (final_pair_btn,)


@app.cell
def _(Aug, Cfg, final_pair_btn, mo, replace, train_one):
    base_final = dict(
        epochs=75, init_from=None,
        head_lr=1e-3, lora_lr=2e-4, w_lr=1e-3,
        stop_patience=12, patience=4, deadline=None,
        conds=("clean", "tel", "tel8k", "noisy"),
        bb_dropout=0.05,
        aug=Aug(p_clean=0.5, p_band=0.4, p_8k=0.3, p_mulaw=0.25, p_noise=0.4),
    )

    # Guarded: 2 x 75 epochs. Opening the notebook must not start training.
    mo.stop(not final_pair_btn.value,
            mo.md("*Not started. This trains `FINAL` and `FINAL_hdo` back to back, "
                  "75 epochs each.*"))
    train_one(replace(Cfg(run="FINAL"),      head_dropout=0.0,  **base_final))
    train_one(replace(Cfg(run="FINAL_hdo"),  head_dropout=0.05, **base_final))
    return


@app.cell
def _():
    # from gdrive_fsspec import GoogleDriveFileSystem

    # fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
    #                            auth_kwargs={"use_local_webserver": False})

    # REMOTE = "CLEAR/Phase 1/runs"     # ASR = Phase 1

    # # --- does the nested folder path really get created, and is there duplication ---
    # try:
    #     fs.makedirs(REMOTE, exist_ok=True)
    # except Exception as e:
    #     print("[makedirs] note:", type(e).__name__, e)

    # open("/tmp/_ovtest.txt", "w").write("1")
    # fs.put("/tmp/_ovtest.txt", f"{REMOTE}/_ovtest.txt")
    # fs.put("/tmp/_ovtest.txt", f"{REMOTE}/_ovtest.txt")   # same name, second time

    # _matches = [x for x in fs.ls(REMOTE) if "_ovtest.txt" in str(x)]
    # print(f"_ovtest.txt appears {len(_matches)} times on Drive")
    # if len(_matches) == 1:
    #     print("put OVERWRITES, no fs.rm needed during transfer")
    #     NEEDS_RM = False
    # else:
    #     print("put CREATES A COPY, delete first during transfer")
    #     NEEDS_RM = True

    # # clean up the test file (all copies if any)
    # while any("_ovtest.txt" in str(x) for x in fs.ls(REMOTE)):
    #     try:
    #         fs.rm(f"{REMOTE}/_ovtest.txt")
    #     except Exception:
    #         break
    # print("cleaned -", "runs folder:", REMOTE)
    return


@app.cell
def _():
    # def push_gdrive(only=None, skip=("last.pt",)):
    #     """Upload finished runs (the ones with a summary.json) under CLEAR/Phase 1/runs.
    #     Every file except last.pt. Overwrite safety follows the result of Cell 1."""
    #     rm_first = globals().get("NEEDS_RM", True)   # the safe side when the test was not run
    #     up, skipped = 0, 0
    #     for p in sorted(OUT_ROOT.glob("*/summary.json")):
    #         run = p.parent.name
    #         if only and run not in only:
    #             continue
    #         rdst = f"{REMOTE}/{run}"
    #         try:
    #             fs.makedirs(rdst, exist_ok=True)
    #         except Exception:
    #             pass
    #         for f in sorted(p.parent.iterdir()):
    #             if not f.is_file() or f.name in skip:
    #                 continue
    #             dst = f"{rdst}/{f.name}"
    #             try:
    #                 if rm_first and fs.exists(dst):
    #                     fs.rm(dst)
    #                 fs.put(str(f), dst)
    #                 up += 1
    #             except Exception as e:
    #                 print(f"  [ERROR] {run}/{f.name}: {type(e).__name__}: {e}")
    #                 skipped += 1
    #         print(f"[GDRIVE] {run}")
    #     print(f"[GDRIVE] done - {up} files uploaded"
    #           + (f" · {skipped} errors" if skipped else "") + f" -> {REMOTE}")

    # push_gdrive()                                  # every finished run
    # # push_gdrive(only={"FINAL", "FINAL_hdo"})     # only specific runs
    # # push_gdrive(skip=("last.pt", "dev_logits.npz"))   # also skip the large npz
    return


if __name__ == "__main__":
    app.run()
