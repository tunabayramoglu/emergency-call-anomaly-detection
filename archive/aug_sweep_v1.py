# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.44",
#     "datasets>=2.20",
#     "peft>=0.11",
#     "jiwer",
#     "soundfile",
#     "huggingface_hub",
#     "numpy",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""
Phase-1b — augmentation + regularization ablation.

The architecture of sweep_v2.py is preserved exactly (weighted-sum head,
LoRA, int16 memmap cache, CTC). The only difference is that augmentation and
regularization arms were added, and A0..A5 run in sequence over a single night.

CACHE: _cache_key() is byte-for-byte the same as v2, so the existing cache is reused
and NOTHING is decoded again.

Usage:
    python aug_sweep_v1.py --stage all          # A0 (50ep) + A1..A5 (15ep, warm start)
    python aug_sweep_v1.py --stage a0           # the reference only
    python aug_sweep_v1.py --stage a3 --epochs 20
    python aug_sweep_v1.py --report             # print the table only
    python aug_sweep_v1.py --selftest           # augmentation test, no GPU
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict, replace
from itertools import groupby
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- paths ----
_DEFAULT_ROOT = Path("/marimo") if Path("/marimo").exists() else Path.cwd()
os.environ.setdefault("ECAD_DATA_ROOT", str(_DEFAULT_ROOT / "data"))
os.environ.setdefault("ECAD_OUT_ROOT", str(_DEFAULT_ROOT / "runs"))
os.environ.setdefault("ECAD_CACHE_ROOT", str(_DEFAULT_ROOT / "cache"))
os.environ.setdefault("ECAD_BACKBONE", "utter-project/mHuBERT-147")


def _envp(key, default):
    return Path(os.environ.get(key) or default).expanduser()


# ============================================================================
# 1 - AUGMENTATION. Pure numpy, no extra dependencies
# ============================================================================
# The chain order is physical. speed -> RIR -> noise -> band/codec -> gain.
# Audio degrades in this order in real life, and changing the order changes the result.


def aug_speed(w, rate):
    """Speed change, pitch shifts with it (classic 'speed perturb')."""
    if rate == 1.0 or len(w) < 2:
        return w
    n = max(2, int(round(len(w) / rate)))
    src = np.arange(len(w), dtype=np.float32)
    dst = np.linspace(0.0, len(w) - 1.0, n, dtype=np.float32)
    return np.interp(dst, src, w).astype(np.float32)


def aug_gain(w, db):
    return (w * (10.0 ** (db / 20.0))).astype(np.float32)


def aug_clip(w, thresh):
    return np.clip(w, -thresh, thresh).astype(np.float32)


def _rfft_mask(w, sr, lo, hi):
    n = len(w)
    if n < 8:
        return w
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    if lo is not None:
        W[f < lo] = 0.0
    if hi is not None:
        W[f > hi] = 0.0
    return np.fft.irfft(W, n).astype(np.float32)


def aug_bandpass(w, sr, lo=300.0, hi=3400.0):
    """Telephone band (G.712). Identical to the 'tel' condition in the benchmark."""
    return _rfft_mask(w, sr, lo, hi)


def aug_8k_roundtrip(w, sr=16000):
    """16k -> 8k -> 16k. Anti-alias lowpass + decimate + interpolate."""
    if len(w) < 8:
        return w
    x = _rfft_mask(w, sr, None, 3800.0)
    n8 = max(2, int(len(x) / 2))
    down = np.interp(
        np.linspace(0, len(x) - 1, n8), np.arange(len(x)), x
    ).astype(np.float32)
    up = np.interp(
        np.linspace(0, n8 - 1, len(x)), np.arange(n8), down
    ).astype(np.float32)
    return up


def aug_mulaw(w, mu=255.0):
    """G.711 mu-law encode/decode, including 8-bit quantisation loss."""
    a = np.clip(w, -1.0, 1.0)
    y = np.sign(a) * np.log1p(mu * np.abs(a)) / math.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0  # 8 bit
    return (np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu).astype(np.float32)


def _noise(kind, n, rng):
    if kind == "white":
        x = rng.standard_normal(n).astype(np.float32)
    else:  # pink 1/f, far closer to speech noise than white
        m = n // 2 + 1
        spec = (rng.standard_normal(m) + 1j * rng.standard_normal(m)).astype(
            np.complex64
        )
        f = np.arange(m, dtype=np.float32)
        f[0] = 1.0
        spec /= np.sqrt(f)
        x = np.fft.irfft(spec, n).astype(np.float32)
    p = float(np.sqrt((x**2).mean()) + 1e-9)
    return x / p


def aug_noise(w, snr_db, rng, kind="pink", bank=None):
    """Add noise at the given SNR. If a bank is supplied the noise is real (MUSAN etc.)."""
    if bank:
        src = bank[rng.integers(len(bank))]
        if len(src) < len(w):
            src = np.tile(src, int(np.ceil(len(w) / len(src))))
        off = int(rng.integers(0, max(1, len(src) - len(w) + 1)))
        nz = np.asarray(src[off : off + len(w)], dtype=np.float32)
        nz = nz / (float(np.sqrt((nz**2).mean())) + 1e-9)
    else:
        nz = _noise(kind, len(w), rng)
    ps = float((w**2).mean()) + 1e-12
    k = math.sqrt(ps / (10.0 ** (snr_db / 10.0)))
    return (w + k * nz).astype(np.float32)


def _fftconvolve(w, h):
    n = len(w) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(w, nfft) * np.fft.rfft(h, nfft), nfft)
    return y[: len(w)].astype(np.float32)


def aug_rir(w, sr, t60, rng):
    """Synthetic room response. Direct path plus an exponentially decaying noise tail.

    Not as good as a real RIR (OpenSLR-28) but it needs no extra data and
    captures the main effect of reverb, temporal smearing."""
    n = max(8, int(t60 * sr))
    tail = rng.standard_normal(n).astype(np.float32)
    tail *= np.exp(-6.9078 * np.arange(n, dtype=np.float32) / n)  # -60 dB @ n
    h = tail
    h[0] += 1.0
    h /= float(np.abs(h).sum()) + 1e-9
    return _fftconvolve(w, h)


@dataclass
class AugConfig:
    """All off means A0, the reference. Each arm is enabled independently."""

    p_clean: float = 0.40  # nothing is applied to this fraction (prevents forgetting)
    # speed perturb
    speed_rates: tuple = ()
    p_speed: float = 0.0
    # reverb
    p_rir: float = 0.0
    rir_t60: tuple = (0.15, 0.50)
    # additive noise
    p_noise: float = 0.0
    snr_db: tuple = (5.0, 20.0)
    noise_kind: str = "pink"
    noise_dir: str | None = None  # MUSAN/DEMAND wav folder, if available
    # channel / codec
    p_band: float = 0.0  # 300-3400 Hz
    p_8k: float = 0.0  # 8 kHz round-trip
    p_mulaw: float = 0.0  # G.711
    # level
    p_gain: float = 0.0
    gain_db: tuple = (-6.0, 6.0)
    p_clip: float = 0.0
    clip_thresh: tuple = (0.3, 0.9)

    def active(self):
        return any(
            getattr(self, k) > 0
            for k in ("p_speed", "p_rir", "p_noise", "p_band", "p_8k",
                      "p_mulaw", "p_gain", "p_clip")
        )


_NOISE_BANK: list | None = None


def _load_noise_bank(d):
    global _NOISE_BANK
    if _NOISE_BANK is not None or not d:
        return _NOISE_BANK
    import soundfile as sf

    files = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))[:400]
    bank = []
    for p in files:
        try:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            bank.append(x)
        except Exception:
            pass
    _NOISE_BANK = bank or None
    print(f"[NOISE] loaded {len(bank)} files <- {d}")
    return _NOISE_BANK


def apply_aug(w, a: AugConfig, sr: int, rng: np.random.Generator):
    if not a.active() or rng.random() < a.p_clean:
        return w
    bank = _load_noise_bank(a.noise_dir) if a.noise_dir else None

    if a.speed_rates and rng.random() < a.p_speed:
        w = aug_speed(w, float(rng.choice(np.asarray(a.speed_rates))))
    if rng.random() < a.p_rir:
        w = aug_rir(w, sr, float(rng.uniform(*a.rir_t60)), rng)
    if rng.random() < a.p_noise:
        w = aug_noise(w, float(rng.uniform(*a.snr_db)), rng, a.noise_kind, bank)
    if rng.random() < a.p_band:
        w = aug_bandpass(w, sr)
    if rng.random() < a.p_8k:
        w = aug_8k_roundtrip(w, sr)
    if rng.random() < a.p_mulaw:
        w = aug_mulaw(w)
    if rng.random() < a.p_gain:
        w = aug_gain(w, float(rng.uniform(*a.gain_db)))
    if rng.random() < a.p_clip:
        w = aug_clip(w, float(rng.uniform(*a.clip_thresh)))

    peak = float(np.abs(w).max()) + 1e-9
    if peak > 1.0:
        w = (w / peak).astype(np.float32)
    return w


def degrade_eval(w, mode, sr=16000):
    """Evaluation conditions. Deterministic, identical to the benchmark."""
    if mode == "clean":
        return w
    if mode == "tel":
        return aug_bandpass(w, sr)
    if mode == "tel8k":
        return aug_8k_roundtrip(aug_bandpass(w, sr), sr)
    raise ValueError(mode)


# ============================================================================
# 2 - CONFIGURATION
# ============================================================================


@dataclass
class Config:
    # ---- paths ----
    data_root: Path = field(default_factory=lambda: _envp("ECAD_DATA_ROOT", "./data"))
    out_root: Path = field(default_factory=lambda: _envp("ECAD_OUT_ROOT", "./runs"))
    cache_root: Path = field(default_factory=lambda: _envp("ECAD_CACHE_ROOT", "./cache"))
    backbone: str = field(
        default_factory=lambda: os.environ.get("ECAD_BACKBONE")
        or "utter-project/mHuBERT-147"
    )
    train_glob: str = "librispeech_train100/*.parquet"
    dev_glob: str = "librispeech_val/*.parquet"
    hf_fallback: bool = True
    run_name: str = "run"

    # ---- audio ----
    sr: int = 16_000
    hid: int = 768
    max_secs: float = 20.0

    # ---- architecture (same as v2) ----
    ws_layers: tuple = (9, 10, 11, 12)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple = ("q_proj", "v_proj")
    lora_layers: tuple | None = None

    # ---- REGULARIZATION (all zero / off in v2) ----
    weight_decay: float = 0.0
    lora_dropout: float = 0.0
    head_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    layerdrop: float = 0.0  # >0 SKIPS LAYERS and breaks weighted-sum, keep it at 0
    # SpecAugment, masking inside the backbone, only active in backbone.train() mode
    mask_time_prob: float = 0.0
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10

    # ---- AUGMENTATION ----
    aug: AugConfig = field(default_factory=AugConfig)

    # ---- training ----
    train_batch: int = 64
    accumulation_steps: int = 4
    num_epochs: int = 50
    head_lr: float = 1e-3
    lora_lr: float = 2e-4
    layer_w_lr: float = 1e-3
    grad_clip: float = 5.0
    lr_patience: int = 4
    lr_factor: float = 0.5
    stop_patience: int = 12
    delta_rel: float = 0.005
    num_workers: int = 8
    seed: int = 1337

    # ---- warm start ----
    init_from: str | None = None  # another run_name, loads its head.pt and adapter.pt

    # ---- resources ----
    audio_mode: str = "memmap"
    amp_dtype: str = "bfloat16"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    resume: bool = True
    final_conditions: tuple = ("clean", "tel", "tel8k")

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.out_root = Path(self.out_root)
        self.cache_root = Path(self.cache_root)
        self.ws_layers = tuple(sorted(int(x) for x in self.ws_layers))
        if self.use_lora and self.lora_layers is None:
            self.lora_layers = tuple(range(1, max(self.ws_layers) + 1))
        elif self.lora_layers is not None:
            self.lora_layers = tuple(sorted(int(x) for x in self.lora_layers))

    # backbone.train() is only needed when masking or backbone dropout is wanted
    @property
    def backbone_train_mode(self):
        return (
            self.mask_time_prob > 0
            or self.mask_feature_prob > 0
            or self.hidden_dropout > 0
            or self.attention_dropout > 0
            or self.activation_dropout > 0
            or self.feat_proj_dropout > 0
            or self.layerdrop > 0
        )

    @property
    def out_dir(self):
        return self.out_root / self.run_name

    @property
    def train_files(self):
        return str(self.data_root / self.train_glob)

    @property
    def dev_files(self):
        return str(self.data_root / self.dev_glob)

    @property
    def torch_dtype(self):
        import torch as _t

        return {"bfloat16": _t.bfloat16, "float16": _t.float16}[self.amp_dtype]

    def expected_adapter_params(self):
        if not self.use_lora:
            return 0
        return 2 * self.hid * self.lora_r * len(self.lora_targets) * len(self.lora_layers)

    def to_dict(self):
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    def summary(self):
        a = self.aug
        on = [
            n
            for n, v in (
                ("speed", a.p_speed), ("rir", a.p_rir), ("noise", a.p_noise),
                ("band", a.p_band), ("8k", a.p_8k), ("mulaw", a.p_mulaw),
                ("gain", a.p_gain), ("clip", a.p_clip),
            )
            if v > 0
        ]
        reg = [
            f"{n}={v:g}"
            for n, v in (
                ("wd", self.weight_decay), ("lora_do", self.lora_dropout),
                ("head_do", self.head_dropout), ("mask_t", self.mask_time_prob),
                ("mask_f", self.mask_feature_prob), ("hid_do", self.hidden_dropout),
            )
            if v > 0
        ]
        return (
            f"[CFG] {self.run_name} | read={list(self.ws_layers)} "
            f"| adapt={list(self.lora_layers) if self.use_lora else 'NONE'} "
            f"| {self.num_epochs}ep bs{self.train_batch}x{self.accumulation_steps} "
            f"| aug=[{','.join(on) or 'none'}] | reg=[{','.join(reg) or 'none'}] "
            f"| bb_train={self.backbone_train_mode}"
            + (f" | init<-{self.init_from}" if self.init_from else "")
        )


# ============================================================================
# 3 - DATA. Byte-for-byte the same cache key as v2
# ============================================================================

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
HF_REPO = "openslr/librispeech_asr"
HF_PREFIX = {"train": "clean/train.100/", "dev": "clean/validation/"}


def build_vocab():
    v = {c: i for i, c in enumerate(CHARS)}
    v["|"] = len(v)
    v["[UNK]"] = len(v)
    v["[PAD]"] = len(v)
    return v


def hf_parquet_urls(split):
    from huggingface_hub import list_repo_files

    pre = HF_PREFIX[split]
    files = sorted(
        f
        for f in list_repo_files(HF_REPO, repo_type="dataset")
        if f.startswith(pre) and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"no '{pre}*.parquet' inside {HF_REPO}")
    print(f"[HF] {HF_REPO} :: {pre} — {len(files)} parquet")
    return [f"hf://datasets/{HF_REPO}/{f}" for f in files]


def decode_audio(cell, target_sr):
    import soundfile as sf

    if isinstance(cell, dict) and cell.get("array") is not None:
        w = np.asarray(cell["array"], dtype=np.float32)
        sr = cell.get("sampling_rate", target_sr)
    elif isinstance(cell, dict) and cell.get("bytes"):
        w, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=False)
    elif isinstance(cell, dict) and cell.get("path"):
        w, sr = sf.read(cell["path"], dtype="float32", always_2d=False)
    elif isinstance(cell, (str, bytes)):
        src = cell if isinstance(cell, str) else io.BytesIO(cell)
        w, sr = sf.read(src, dtype="float32", always_2d=False)
    else:
        raise TypeError(f"could not decode the audio cell: {type(cell)}")
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if int(sr) != int(target_sr):
        raise ValueError(f"sample rate {sr} != {target_sr}")
    return w


def _load_hf(cfg, split):
    from datasets import load_dataset

    pattern = cfg.train_files if split == "train" else cfg.dev_files
    files = sorted(glob.glob(pattern))
    if files:
        hf = load_dataset(
            "parquet", data_files=files, split="train", verification_mode="no_checks"
        )
    elif cfg.hf_fallback:
        hf = load_dataset(
            "parquet",
            data_files={"d": hf_parquet_urls(split)},
            split="d",
            verification_mode="no_checks",
        )
    else:
        raise FileNotFoundError(f"[{split}] '{pattern}' is missing and hf_fallback is off.")
    try:
        from datasets import Audio

        hf = hf.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    cap = cfg.max_train_samples if split == "train" else cfg.max_eval_samples
    if cap:
        hf = hf.select(range(min(cap, len(hf))))
    return hf


def _cache_key(cfg, split):
    """Byte-for-byte identical to v2 so the existing cache is reused."""
    cap = cfg.max_train_samples if split == "train" else cfg.max_eval_samples
    raw = f"{cfg.data_root}|{split}|{cfg.sr}|{cap}|{cfg.max_secs}|int16"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _build_cache(cfg, split, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_p, meta_p = cache_dir / "audio.i16", cache_dir / "meta.json"
    hf = _load_hf(cfg, split)
    max_len = int(cfg.max_secs * cfg.sr)
    t0, offsets, texts, pos = time.perf_counter(), [0], [], 0
    with open(bin_p, "wb") as f:
        for i in range(len(hf)):
            row = hf[i]
            w = decode_audio(row["audio"], cfg.sr)[:max_len]
            q = np.clip(np.rint(w * 32768.0), -32768, 32767).astype(np.int16)
            f.write(q.tobytes())
            pos += q.size
            offsets.append(pos)
            texts.append(row["text"].upper().strip())
            if (i + 1) % 2000 == 0:
                print(
                    f"  [cache:{split}] {i + 1}/{len(hf)} "
                    f"({pos * 2 / 1e9:.1f} GB, {time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
    meta_p.write_text(json.dumps({"offsets": offsets, "texts": texts}))
    print(
        f"[cache:{split}] ready - {len(texts)} samples, {pos * 2 / 1e9:.2f} GB, "
        f"{(time.perf_counter() - t0) / 60:.1f} min"
    )


def prepare_split(cfg, split):
    cache_dir = cfg.cache_root / f"{split}_{_cache_key(cfg, split)}"
    bin_p, meta_p = cache_dir / "audio.i16", cache_dir / "meta.json"
    if not (bin_p.exists() and meta_p.exists()):
        _build_cache(cfg, split, cache_dir)
    else:
        print(f"[DATA:{split}] cache found -> {cache_dir}")
    meta = json.loads(meta_p.read_text())
    offsets = np.asarray(meta["offsets"], dtype=np.int64)
    texts, n = meta["texts"], int(meta["offsets"][-1])
    if cfg.audio_mode == "memmap":
        buf = np.memmap(bin_p, dtype=np.int16, mode="r", shape=(n,))
        print(f"[DATA:{split}] memmap - {len(texts)} samples, {n * 2 / 1e9:.2f} GB")
    else:
        buf = np.fromfile(bin_p, dtype=np.int16)
        print(f"[DATA:{split}] ram - {len(texts)} samples, {buf.nbytes / 1e9:.2f} GB")
    return buf, offsets, texts


def _make_ds_class():
    from torch.utils.data import Dataset
    import torch

    class SpeechDS(Dataset):
        """train: random augmentation · dev: deterministic degradation (or clean)"""

        def __init__(self, buf, offsets, texts, vocab, cfg, aug=None, degrade=None):
            self.buf, self.offsets, self.texts = buf, offsets, texts
            self.vocab, self.cfg = vocab, cfg
            self.aug, self.degrade = aug, degrade
            self.max_len = int(cfg.max_secs * cfg.sr)
            self._rng = None

        def __len__(self):
            return len(self.texts)

        def rng(self):
            # each worker gets its own seed, otherwise EVERY worker produces the same
            # augmentation and the diversity is fake.
            if self._rng is None:
                import torch.utils.data as tud

                info = tud.get_worker_info()
                wid = info.id if info else 0
                self._rng = np.random.default_rng(self.cfg.seed * 1000 + wid)
            return self._rng

        def __getitem__(self, i):
            a, b = int(self.offsets[i]), int(self.offsets[i + 1])
            w = np.asarray(self.buf[a:b], dtype=np.float32) / 32768.0
            if self.aug is not None:
                w = apply_aug(w, self.aug, self.cfg.sr, self.rng())
            if self.degrade:
                w = degrade_eval(w, self.degrade, self.cfg.sr)
            w = np.ascontiguousarray(w[: self.max_len], dtype=np.float32)
            ids = [
                self.vocab.get(c, self.vocab["[UNK]"])
                for c in self.texts[i].replace(" ", "|")
            ]
            return torch.from_numpy(w), torch.tensor(ids, dtype=torch.long), i

    class Collate:
        def __init__(self, pad_id):
            self.pad_id = pad_id

        def __call__(self, batch):
            waves, labels, idxs = zip(*batch)
            wl = torch.tensor([len(w) for w in waves], dtype=torch.long)
            ll = torch.tensor([len(l) for l in labels], dtype=torch.long)
            X = torch.zeros(len(waves), int(wl.max()), dtype=torch.float32)
            Y = torch.zeros(len(labels), int(ll.max()), dtype=torch.long)
            for i, (w, l) in enumerate(zip(waves, labels)):
                X[i, : len(w)] = w
                Y[i, : len(l)] = l
            return X, Y, wl, ll, torch.tensor(idxs)

    return SpeechDS, Collate


# ============================================================================
# 4 · MODEL
# ============================================================================


def hs_to_module_idx(hs_layers):
    return [int(i) - 1 for i in hs_layers]


def build_backbone(cfg, device):
    from transformers import HubertModel
    from peft import LoraConfig, inject_adapter_in_model

    over = dict(
        hidden_dropout=cfg.hidden_dropout,
        attention_dropout=cfg.attention_dropout,
        activation_dropout=cfg.activation_dropout,
        feat_proj_dropout=cfg.feat_proj_dropout,
        layerdrop=cfg.layerdrop,
        mask_time_prob=cfg.mask_time_prob,
        mask_time_length=cfg.mask_time_length,
        mask_feature_prob=cfg.mask_feature_prob,
        mask_feature_length=cfg.mask_feature_length,
        apply_spec_augment=(cfg.mask_time_prob > 0 or cfg.mask_feature_prob > 0),
    )
    bb = HubertModel.from_pretrained(cfg.backbone, **over).to(device)
    n_layers = bb.config.num_hidden_layers
    feat_len_fn = bb._get_feat_extract_output_lengths

    bad = [l for l in cfg.ws_layers if not (1 <= l <= n_layers)]
    if bad:
        raise ValueError(f"ws_layers={list(cfg.ws_layers)} is invalid ({n_layers} layers)")
    if cfg.layerdrop > 0:
        print("[WARN] layerdrop>0, if a layer is skipped weighted-sum loses the "
              "hidden_state it reads. 0 is recommended.")
    print(f"[BB] {cfg.backbone} | layers={n_layers} | hid={bb.config.hidden_size} "
          f"| spec_aug={over['apply_spec_augment']}")

    if cfg.use_lora:
        dead = [l for l in cfg.lora_layers if l > max(cfg.ws_layers)]
        if dead:
            print(f"[WARN] {dead} sits above the deepest layer read -> IT RECEIVES NO GRADIENT")
        lcfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets),
            bias="none",
            layers_to_transform=hs_to_module_idx(cfg.lora_layers),
        )
        bb = inject_adapter_in_model(lcfg, bb)
        for n, p in bb.named_parameters():
            p.requires_grad = "lora_" in n
        n_tr = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        exp = cfg.expected_adapter_params()
        print(f"[LORA] adapter {n_tr:,} (expected {exp:,}) | dropout={cfg.lora_dropout}")
        assert n_tr == exp, f"LoRA has the wrong scope: {n_tr:,} != {exp:,}"
    else:
        for p in bb.parameters():
            p.requires_grad = False
        print("[LORA] off, frozen backbone")

    bb.eval()
    return bb, feat_len_fn, n_layers


def _make_head_class():
    import torch
    import torch.nn as nn

    class WeightedSumHead(nn.Module):
        """x: [B, T, N_WS, D] -> logits: [B, T, V]"""

        def __init__(self, n_ws, dim, vocab_size, dropout=0.0):
            super().__init__()
            self.n_ws = n_ws
            self.layer_w = nn.Parameter(torch.zeros(n_ws))
            layers = [nn.Linear(dim, dim), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim, vocab_size))
            self.net = nn.Sequential(*layers)

        def weights(self):
            return self.layer_w.softmax(0)

        def forward(self, x):
            if self.n_ws == 1:
                feat = x[:, :, 0, :]
            else:
                w = self.layer_w.softmax(0)
                feat = (x * w[None, None, :, None]).sum(2)
            return self.net(feat)

    return WeightedSumHead


def param_groups(head, backbone, cfg):
    rest = [p for n, p in head.named_parameters() if n != "layer_w"]
    groups = [
        {"params": rest, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": [head.layer_w], "lr": cfg.layer_w_lr, "weight_decay": 0.0},
    ]
    if cfg.use_lora:
        groups.append(
            {
                "params": [p for p in backbone.parameters() if p.requires_grad],
                "lr": cfg.lora_lr,
                "weight_decay": cfg.weight_decay,
            }
        )
    return groups


def adapter_state_dict(backbone):
    return {
        k: v.detach().cpu().clone()
        for k, v in backbone.state_dict().items()
        if "lora_" in k
    }


GROUP_NAMES = ["head", "layer_w", "lora"]


# ============================================================================
# 5 - TRAINING
# ============================================================================


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def greedy_decode(ids, id2ch, blank_id, unk_id):
    dec = [id2ch.get(k, "") for k, _ in groupby(ids) if k not in (blank_id, unk_id)]
    return "".join(dec).replace("|", " ").strip()


def evaluate(head, backbone, dl, texts, feat_len_fn, cfg, device, id2ch,
             blank_id, unk_id):
    import jiwer
    import torch

    head.eval()
    backbone.eval()  # ALWAYS off during eval, no dropout and no SpecAugment
    hyps, refs = [], []
    with torch.no_grad():
        for X, Y, wl, ll, idxs in dl:
            X = X.to(device, non_blocking=True)
            am = (
                torch.arange(X.shape[1], device=device)[None, :] < wl.to(device)[:, None]
            ).long()
            with torch.autocast(device_type=device, dtype=cfg.torch_dtype):
                out = backbone(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([out.hidden_states[L] for L in cfg.ws_layers], dim=2)
            xlen = feat_len_fn(wl.to(device))
            pred = head(hs.float()).argmax(-1).cpu().numpy()
            for b, i in enumerate(idxs.tolist()):
                T = int(xlen[b])
                hyps.append(greedy_decode(pred[b, :T].tolist(), id2ch, blank_id, unk_id))
                refs.append(texts[i])
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def train_one(cfg: Config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    SpeechDS, Collate = _make_ds_class()
    WeightedSumHead = _make_head_class()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(cfg.summary(), flush=True)

    vocab = build_vocab()
    blank_id, unk_id = vocab["[PAD]"], vocab["[UNK]"]
    id2ch = {v: k for k, v in vocab.items()}

    tr_raw = prepare_split(cfg, "train")
    dv_raw = prepare_split(cfg, "dev")
    aug = cfg.aug if cfg.aug.active() else None
    tr = SpeechDS(*tr_raw, vocab, cfg, aug=aug)
    dv = SpeechDS(*dv_raw, vocab, cfg)  # dev is ALWAYS clean
    collate = Collate(blank_id)

    nw = cfg.num_workers if aug is not None else 0  # workers are pointless without aug
    train_dl = DataLoader(
        tr, batch_size=cfg.train_batch, shuffle=True, num_workers=nw,
        collate_fn=collate, pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None, drop_last=False,
    )
    dev_dl = DataLoader(
        dv, batch_size=cfg.train_batch, shuffle=False, num_workers=0,
        collate_fn=collate, pin_memory=True,
    )
    print(f"[DATA] train={len(tr)} dev={len(dv)} | {len(train_dl)} batch/ep | nw={nw}")

    backbone, feat_len_fn, _ = build_backbone(cfg, device)
    head = WeightedSumHead(
        len(cfg.ws_layers), cfg.hid, len(vocab), cfg.head_dropout
    ).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    n_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[MODEL] trainable: head={n_head:,} + lora={n_lora:,} = {n_head + n_lora:,}")

    # ---- warm start ----
    if cfg.init_from:
        src = cfg.out_root / cfg.init_from
        hp, ap = src / "head.pt", src / "adapter.pt"
        if hp.exists():
            head.load_state_dict(torch.load(hp, map_location=device), strict=False)
            print(f"[INIT] head <- {hp}")
        if ap.exists() and cfg.use_lora:
            backbone.load_state_dict(torch.load(ap, map_location=device), strict=False)
            print(f"[INIT] adapter <- {ap}")
        if not hp.exists():
            print(f"[WARN] init_from='{cfg.init_from}' not found, starting from scratch")

    opt = torch.optim.AdamW(param_groups(head, backbone, cfg), fused=(device == "cuda"))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
        threshold=0.005, threshold_mode="rel",
    )
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    trainable = [p for g in opt.param_groups for p in g["params"]]

    hist_p, last_p = cfg.out_dir / "history.jsonl", cfg.out_dir / "last.pt"
    start_ep, best_cer, best_ep, history = 1, float("inf"), 0, []

    if cfg.resume and last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        head.load_state_dict(ck["head"])
        if ck.get("adapter"):
            backbone.load_state_dict(ck["adapter"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best_cer, best_ep = ck["epoch"] + 1, ck["best_cer"], ck["best_ep"]
        history = [json.loads(l) for l in hist_p.read_text().splitlines() if l.strip()]
        print(f"[RESUME] resuming from e{ck['epoch']} · best {best_cer * 100:.2f}%")
    elif not hist_p.exists():
        hist_p.write_text("")

    for epoch in range(start_ep, cfg.num_epochs + 1):
        head.train()
        # CRITICAL. SpecAugment and backbone dropout only run in train() mode.
        # v2 called eval() unconditionally here, so masking never actually ran.
        backbone.train() if cfg.backbone_train_mode else backbone.eval()
        t0, tot, nb = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for X, Y, wl, ll, _ in train_dl:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            am = (
                torch.arange(X.shape[1], device=device)[None, :] < wl.to(device)[:, None]
            ).long()
            with torch.autocast(device_type=device, dtype=cfg.torch_dtype):
                out = backbone(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([out.hidden_states[L] for L in cfg.ws_layers], dim=2)
            xlen = feat_len_fn(wl.to(device))
            logits = head(hs.float())
            logp = logits.log_softmax(-1).transpose(0, 1)
            loss = ctc(logp, Y, xlen, ll.to(device)) / cfg.accumulation_steps
            loss.backward()
            if ((nb + 1) % cfg.accumulation_steps == 0) or ((nb + 1) == len(train_dl)):
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item() * cfg.accumulation_steps
            nb += 1

        wer, cer = evaluate(head, backbone, dev_dl, dv.texts, feat_len_fn, cfg,
                            device, id2ch, blank_id, unk_id)
        w = head.weights().detach().cpu().numpy().round(4).tolist()
        rec = {
            "epoch": epoch, "loss": tot / nb, "wer": wer, "cer": cer,
            "secs": time.perf_counter() - t0, "w": w,
            "layers": list(cfg.ws_layers), "run": cfg.run_name,
            "lr": {n: g["lr"] for n, g in zip(GROUP_NAMES, opt.param_groups)},
        }
        history.append(rec)
        with hist_p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"epoch {epoch:>3} | loss {rec['loss']:.3f} | {rec['secs']:.0f}s "
            f"| VAL wer {wer * 100:.2f}% cer {cer * 100:.2f}%",
            flush=True,
        )

        sched.step(cer)
        if cer < best_cer * (1.0 - cfg.delta_rel):
            best_cer, best_ep = cer, epoch
            torch.save(head.state_dict(), cfg.out_dir / "head.pt")
            if cfg.use_lora:
                torch.save(adapter_state_dict(backbone), cfg.out_dir / "adapter.pt")
            print(f"   [SAVE] new best val-CER {cer * 100:.2f}%")
        torch.save(
            {
                "head": head.state_dict(),
                "adapter": adapter_state_dict(backbone) if cfg.use_lora else None,
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best_cer": best_cer, "best_ep": best_ep,
            },
            last_p,
        )
        if epoch - best_ep >= cfg.stop_patience:
            print(f"[STOP] no improvement for {cfg.stop_patience} epochs")
            break

    # ---- final: clean / tel / tel8k, using the BEST checkpoint ----
    if (cfg.out_dir / "head.pt").exists():
        head.load_state_dict(torch.load(cfg.out_dir / "head.pt", map_location=device))
        if cfg.use_lora and (cfg.out_dir / "adapter.pt").exists():
            backbone.load_state_dict(
                torch.load(cfg.out_dir / "adapter.pt", map_location=device), strict=False
            )
    conds = {}
    for mode in cfg.final_conditions:
        ds = SpeechDS(*dv_raw, vocab, cfg, degrade=(None if mode == "clean" else mode))
        dl = DataLoader(ds, batch_size=cfg.train_batch, shuffle=False,
                        num_workers=0, collate_fn=collate, pin_memory=True)
        wer, cer = evaluate(head, backbone, dl, ds.texts, feat_len_fn, cfg,
                            device, id2ch, blank_id, unk_id)
        conds[mode] = {"wer": wer, "cer": cer}
        print(f"[FINAL:{mode:6s}] CER {cer * 100:5.2f}  WER {wer * 100:6.2f}")

    ratio = (
        conds["tel8k"]["wer"] / conds["clean"]["wer"]
        if "tel8k" in conds and "clean" in conds and conds["clean"]["wer"] > 0
        else None
    )
    summary = {
        "run": cfg.run_name, "best_cer": best_cer, "best_epoch": best_ep,
        "epochs_done": history[-1]["epoch"] if history else 0,
        "final": conds, "tel8k_over_clean": ratio,
        "aug": asdict(cfg.aug),
        "reg": {
            "weight_decay": cfg.weight_decay, "lora_dropout": cfg.lora_dropout,
            "head_dropout": cfg.head_dropout, "mask_time_prob": cfg.mask_time_prob,
            "mask_feature_prob": cfg.mask_feature_prob,
            "hidden_dropout": cfg.hidden_dropout,
        },
        "history": history,
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] {cfg.run_name}: val-CER {best_cer * 100:.2f}% @ e{best_ep}"
          + (f" · tel8k/clean ×{ratio:.2f}" if ratio else ""), flush=True)
    return summary


# ============================================================================
# 6 · ABLATION PLAN
# ============================================================================
# A0 is a full run from scratch, the reference. A1..A5 warm start from A0 at a lower LR.
# Each step adds exactly ONE axis on top of the previous one, so contributions stay isolated.

BASE = "A0_ref"
WARM = dict(init_from=BASE, num_epochs=15, head_lr=5e-4, lora_lr=1e-4,
            layer_w_lr=5e-4, stop_patience=8, lr_patience=3)


def plan() -> dict[str, Config]:
    p: dict[str, Config] = {}

    p["a0"] = Config(run_name=BASE, num_epochs=50)

    # A1 - SpecAugment. Does not touch the waveform, zero CPU cost.
    p["a1"] = Config(
        run_name="A1_specaug", mask_time_prob=0.05, mask_time_length=10,
        mask_feature_prob=0.004, mask_feature_length=10, **WARM,
    )

    a1 = dict(mask_time_prob=0.05, mask_time_length=10,
              mask_feature_prob=0.004, mask_feature_length=10)

    # A2 — + speed perturb
    p["a2"] = Config(
        run_name="A2_speed", **a1, **WARM,
        aug=AugConfig(speed_rates=(0.9, 1.0, 1.1), p_speed=0.6),
    )

    # A3 — + additive noise
    p["a3"] = Config(
        run_name="A3_noise", **a1, **WARM,
        aug=AugConfig(speed_rates=(0.9, 1.0, 1.1), p_speed=0.6,
                      p_noise=0.5, snr_db=(5.0, 20.0), noise_kind="pink",
                      noise_dir=os.environ.get("ECAD_NOISE_DIR")),
    )

    # A4 - plus channel (band-limit + 8k + mu-law). The axis that targets the tel/clean ratio.
    p["a4"] = Config(
        run_name="A4_channel", **a1, **WARM,
        aug=AugConfig(speed_rates=(0.9, 1.0, 1.1), p_speed=0.6,
                      p_noise=0.5, snr_db=(5.0, 20.0), noise_kind="pink",
                      noise_dir=os.environ.get("ECAD_NOISE_DIR"),
                      p_band=0.4, p_8k=0.3, p_mulaw=0.3),
    )

    # A5 — + reverb + level
    p["a5"] = Config(
        run_name="A5_rir", **a1, **WARM,
        aug=AugConfig(speed_rates=(0.9, 1.0, 1.1), p_speed=0.6,
                      p_noise=0.5, snr_db=(5.0, 20.0), noise_kind="pink",
                      noise_dir=os.environ.get("ECAD_NOISE_DIR"),
                      p_band=0.4, p_8k=0.3, p_mulaw=0.3,
                      p_rir=0.3, p_gain=0.3, p_clip=0.1),
    )

    # R1 - regularization ONLY, no augmentation. A test for whether overfitting exists.
    p["r1"] = Config(
        run_name="R1_reg", weight_decay=0.01, lora_dropout=0.05,
        head_dropout=0.1, **WARM,
    )
    return p


ORDER = ["a0", "a1", "a2", "a3", "a4", "a5", "r1"]


def report(out_root: Path):
    rows = []
    for d in sorted(out_root.glob("*/summary.json")):
        try:
            s = json.loads(d.read_text())
        except Exception:
            continue
        f = s.get("final", {})
        rows.append(
            (
                s["run"],
                s.get("best_cer", float("nan")) * 100,
                f.get("clean", {}).get("wer", float("nan")) * 100,
                f.get("tel", {}).get("wer", float("nan")) * 100,
                f.get("tel8k", {}).get("wer", float("nan")) * 100,
                s.get("tel8k_over_clean") or float("nan"),
                s.get("epochs_done", 0),
            )
        )
    if not rows:
        print("No summary.json yet.")
        return
    print("\n" + "=" * 84)
    print("ABLATION — LibriSpeech dev-clean · greedy CTC (no KenLM)")
    print("=" * 84)
    print(f"{'run':<16}{'valCER':>8}{'clean W':>9}{'tel W':>8}{'tel8k W':>9}"
          f"{'tel8k/cl':>10}{'ep':>5}")
    print("-" * 84)
    for r in rows:
        print(f"{r[0]:<16}{r[1]:>8.2f}{r[2]:>9.2f}{r[3]:>8.2f}{r[4]:>9.2f}"
              f"{r[5]:>10.2f}{r[6]:>5}")
    print("=" * 84)
    print("DECISION RULE: drop any axis that hurts clean WER by more than 5% relative")
    print("without lowering the tel8k/clean ratio. The ratio is the real target, and")
    print("augmentation does not move W/C at all, only the LM does.")


# ============================================================================
# 7 - SELFTEST. No GPU, checks augmentation correctness
# ============================================================================


def selftest():
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr, dtype=np.float32) / sr
    w = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)).astype(
        np.float32
    )
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {name} {extra}")

    def energy(x, lo, hi):
        X = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return float(X[(f >= lo) & (f < hi)].sum())

    print("[selftest] augmentation")
    s = aug_speed(w, 1.1)
    chk("speed 1.1 shortens", abs(len(s) - len(w) / 1.1) <= 2, f"({len(s)} samples)")
    chk("speed 1.0 no-op", np.array_equal(aug_speed(w, 1.0), w))

    b = aug_bandpass(w, sr)
    chk("bandpass removes 5 kHz", energy(b, 4000, 8000) < 0.01 * energy(w, 4000, 8000))
    chk("bandpass preserves 440 Hz",
        energy(b, 300, 1000) > 0.5 * energy(w, 300, 1000))

    k = aug_8k_roundtrip(w, sr)
    chk("the 8k round trip preserves length", len(k) == len(w))
    chk("8k removes >4 kHz", energy(k, 4500, 8000) < 0.02 * energy(w, 4500, 8000))

    m = aug_mulaw(w)
    chk("mulaw length and bound", len(m) == len(w) and np.abs(m).max() <= 1.01)

    for snr in (0.0, 10.0, 20.0):
        n = aug_noise(w, snr, rng)
        d = n - w
        got = 10 * np.log10(((w**2).mean() + 1e-12) / ((d**2).mean() + 1e-12))
        chk(f"noise SNR {snr:.0f} dB", abs(got - snr) < 1.0, f"(measured {got:.2f})")

    r = aug_rir(w, sr, 0.3, rng)
    chk("rir preserves length", len(r) == len(w))
    chk("rir does not blow up the energy", np.abs(r).max() < 3 * np.abs(w).max())

    g = aug_gain(w, 6.0)
    chk("gain +6 dB ≈ ×2", abs(np.abs(g).max() / np.abs(w).max() - 2.0) < 0.05)

    print("[selftest] chain")
    a = AugConfig(p_clean=0.0, speed_rates=(0.9, 1.0, 1.1), p_speed=1.0,
                  p_noise=1.0, p_band=1.0, p_8k=1.0, p_mulaw=1.0,
                  p_gain=1.0, p_rir=1.0, p_clip=1.0)
    y = apply_aug(w.copy(), a, sr, rng)
    chk("the full chain runs", np.isfinite(y).all() and len(y) > 0, f"({len(y)} samples)")
    chk("the full chain is clipped", np.abs(y).max() <= 1.001)

    a_off = AugConfig()
    chk("an off config is a no-op", np.array_equal(apply_aug(w.copy(), a_off, sr, rng), w))
    a_clean = AugConfig(p_clean=1.0, p_band=1.0)
    chk("p_clean=1 no-op", np.array_equal(apply_aug(w.copy(), a_clean, sr, rng), w))

    print("[selftest] degrade_eval")
    chk("clean no-op", np.array_equal(degrade_eval(w, "clean"), w))
    chk("tel8k is deterministic",
        np.array_equal(degrade_eval(w, "tel8k"), degrade_eval(w, "tel8k")))

    print("[selftest] plan")
    P = plan()
    chk("7 runs are defined", len(P) == 7, f"({list(P)})")
    chk("A0 has aug off", not P["a0"].aug.active())
    chk("A0 backbone is in eval mode", not P["a0"].backbone_train_mode)
    chk("A1 backbone is in train mode", P["a1"].backbone_train_mode)
    chk("A1 has no waveform aug", not P["a1"].aug.active())
    chk("A4 has channel on", P["a4"].aug.p_band > 0 and P["a4"].aug.p_8k > 0)
    chk("warm start is from A0", all(P[k].init_from == BASE for k in ORDER[1:]))
    chk("layerdrop is 0 in every run", all(P[k].layerdrop == 0 for k in ORDER))
    chk("the adapter param count is consistent",
        P["a0"].expected_adapter_params() == 2 * 768 * 16 * 2 * 12)

    print("\n" + ("selftest PASSED" if ok else "selftest FAILED"))
    return 0 if ok else 1


# ============================================================================
# 8 · CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="all | " + " | ".join(ORDER) + " | comma-separated list")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--noise-dir", default=None, help="MUSAN/DEMAND wav folder")
    ap.add_argument("--smoke", action="store_true",
                    help="64 train / 32 dev, 1 epoch, verifies the path")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    out_root = _envp("ECAD_OUT_ROOT", "./runs")
    if args.report:
        report(out_root)
        return

    if args.noise_dir:
        os.environ["ECAD_NOISE_DIR"] = args.noise_dir

    P = plan()
    stages = ORDER if args.stage == "all" else [
        s.strip() for s in args.stage.split(",")
    ]
    bad = [s for s in stages if s not in P]
    if bad:
        raise SystemExit(f"unknown stage: {bad} - options are {ORDER}")

    import torch

    results = {}
    for s in stages:
        cfg = P[s]
        over = {}
        if args.epochs:
            over["num_epochs"] = args.epochs
        if args.batch:
            over["train_batch"] = args.batch
            over["accumulation_steps"] = max(1, 256 // args.batch)
        if args.workers is not None:
            over["num_workers"] = args.workers
        if args.smoke:
            over.update(run_name=f"smoke_{s}", num_epochs=1, max_train_samples=64,
                        max_eval_samples=32, train_batch=8, accumulation_steps=2,
                        num_workers=0, resume=False, init_from=None,
                        final_conditions=("clean",))
        if over:
            cfg = replace(cfg, **over)

        sp = cfg.out_dir / "summary.json"
        if sp.exists() and not args.smoke:
            results[s] = json.loads(sp.read_text())
            print(f"[SKIP] {cfg.run_name} already finished - "
                  f"CER {results[s]['best_cer'] * 100:.2f}%")
            continue

        if cfg.init_from and not (cfg.out_root / cfg.init_from / "head.pt").exists():
            print(f"[WARN] {s}: warm start source '{cfg.init_from}' is missing. "
                  f"Run 'a0' first, otherwise this run starts from scratch and the comparison breaks.")

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        results[s] = train_one(cfg)
        print(f"[TIME] {s}: {(time.perf_counter() - t0) / 60:.1f} min")
        if torch.cuda.is_available():
            print(f"[VRAM] peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            gc.collect()
            torch.cuda.empty_cache()

    report(out_root)


if __name__ == "__main__":
    main()
