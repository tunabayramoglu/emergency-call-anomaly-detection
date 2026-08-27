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
Phase-1b - overnight ablation run.

Single file. Augmentation and regularization are written in pure numpy/torch,
with NO torchaudio, scipy, audiomentations or librosa needed.

The architecture is identical to sweep_v2.py (weighted-sum head + LoRA + CTC),
and the data cache key matches too, so the existing cache is reused.

FLOW
    1. BASE: pull the best checkpoint from HF, or train from scratch if there is none.
    2. ABLATION: every axis warm starts from BASE, short run, INDEPENDENT.
    3. COMBO: the winning axes are merged and run together.
    4. The winning setup is then run long (--stage final).

Usage
    python aug_night_v2.py --selftest                  # no GPU, ~5 s
    python aug_night_v2.py --smoke                     # end-to-end path test
    python aug_night_v2.py --plan --hours 6            # show what it will run
    python aug_night_v2.py --night --hours 6 \
        --hf-repo username/clear-phase1-runs           # overnight run
    python aug_night_v2.py --report
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import glob
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import traceback
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


def log(*a):
    print(*a, flush=True)


# ============================================================================
# 1 · AUGMENTATION — pure numpy
# ============================================================================
# References:
#   speed perturb .............. Ko et al., Interspeech 2015
#   RIR / noise (MUSAN) ........ Ko et al., ICASSP 2017 · Snyder et al. 2015
#   SpecAugment ................ Park et al., Interspeech 2019
#   babble (speech-on-speech) .. Hu & Wang 2010, critical for call centres
#   packet loss ................ VoIP/GSM loss simulation (ITU-T G.1050)
#   μ-law / A-law .............. ITU-T G.711
#   band-limit 300-3400 Hz ..... ITU-T G.712


def aug_speed(w, rate):
    """Speed and pitch shift together (classic speed perturb)."""
    if rate == 1.0 or len(w) < 2:
        return w
    n = max(2, int(round(len(w) / rate)))
    return np.interp(
        np.linspace(0.0, len(w) - 1.0, n, dtype=np.float32),
        np.arange(len(w), dtype=np.float32),
        w,
    ).astype(np.float32)


def aug_tempo(w, rate, sr=16000, win_ms=30.0):
    """Changes duration, PRESERVES pitch (OLA). A different axis from speed.
    speed shifts the formants, tempo does not."""
    if rate == 1.0 or len(w) < 4:
        return w
    win = max(64, int(win_ms / 1000 * sr))
    hop_a = win // 2
    hop_s = max(1, int(hop_a * rate))
    win_fn = np.hanning(win).astype(np.float32)
    n_out = int(len(w) / rate) + win
    out = np.zeros(n_out, dtype=np.float32)
    norm = np.zeros(n_out, dtype=np.float32)
    i_s, i_a = 0, 0
    while i_s + win < len(w) and i_a + win < n_out:
        out[i_a : i_a + win] += w[i_s : i_s + win] * win_fn
        norm[i_a : i_a + win] += win_fn
        i_s += hop_s
        i_a += hop_a
    out = out[: max(1, int(len(w) / rate))]
    norm = norm[: len(out)]
    return (out / np.maximum(norm, 1e-3)).astype(np.float32)


def aug_pitch(w, semitones, sr=16000):
    """Shift pitch, keep duration. Resample plus tempo compensation."""
    if semitones == 0:
        return w
    r = 2.0 ** (semitones / 12.0)
    return aug_tempo(aug_speed(w, r), 1.0 / r, sr)


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
    return _rfft_mask(w, sr, lo, hi)


def aug_8k_roundtrip(w, sr=16000):
    if len(w) < 8:
        return w
    x = _rfft_mask(w, sr, None, 3800.0)
    n8 = max(2, len(x) // 2)
    down = np.interp(np.linspace(0, len(x) - 1, n8), np.arange(len(x)), x)
    return np.interp(
        np.linspace(0, n8 - 1, len(x)), np.arange(n8), down
    ).astype(np.float32)


def aug_mulaw(w, mu=255.0):
    """G.711 mu-law, including 8-bit quantisation loss."""
    a = np.clip(w, -1.0, 1.0)
    y = np.sign(a) * np.log1p(mu * np.abs(a)) / math.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu).astype(np.float32)


def aug_alaw(w, A=87.6):
    """G.711 A-law (European telephone network)."""
    a = np.clip(w, -1.0, 1.0)
    ab = np.abs(a)
    lo = ab < 1.0 / A
    y = np.where(
        lo,
        A * ab / (1 + math.log(A)),
        (1 + np.log(np.maximum(A * ab, 1e-9))) / (1 + math.log(A)),
    )
    y = np.sign(a) * y
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    qb = np.abs(q)
    thr = 1.0 / (1 + math.log(A))
    inv = np.where(
        qb < thr,
        qb * (1 + math.log(A)) / A,
        np.exp(qb * (1 + math.log(A)) - 1) / A,
    )
    return (np.sign(q) * inv).astype(np.float32)


def aug_packet_loss(w, sr, rate, chunk_ms=(20.0, 60.0), rng=None):
    """VoIP packet loss. Random blocks are zeroed out."""
    rng = rng or np.random.default_rng()
    y = w.copy()
    i = 0
    while i < len(y):
        L = max(1, int(rng.uniform(*chunk_ms) / 1000.0 * sr))
        if rng.random() < rate:
            y[i : i + L] = 0.0
        i += L
    return y


def _noise(kind, n, rng):
    if kind == "white":
        x = rng.standard_normal(n).astype(np.float32)
    else:  # pink 1/f, far closer to speech background than white
        m = n // 2 + 1
        spec = (rng.standard_normal(m) + 1j * rng.standard_normal(m)).astype(np.complex64)
        f = np.arange(m, dtype=np.float32)
        f[0] = 1.0
        spec /= np.sqrt(f)
        x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (float(np.sqrt((x**2).mean())) + 1e-9)


def _mix_at_snr(w, nz, snr_db):
    nz = nz / (float(np.sqrt((nz**2).mean())) + 1e-9)
    ps = float((w**2).mean()) + 1e-12
    k = math.sqrt(ps / (10.0 ** (snr_db / 10.0)))
    return (w + k * nz).astype(np.float32)


def _fit_len(src, n, rng):
    src = np.asarray(src, dtype=np.float32)
    if len(src) < n:
        src = np.tile(src, int(np.ceil(n / max(1, len(src)))))
    off = int(rng.integers(0, max(1, len(src) - n + 1)))
    return src[off : off + n]


def aug_noise(w, snr_db, rng, kind="pink", bank=None):
    nz = _fit_len(bank[rng.integers(len(bank))], len(w), rng) if bank else _noise(
        kind, len(w), rng
    )
    return _mix_at_snr(w, nz, snr_db)


def aug_babble(w, snr_db, rng, other_fn, n_speakers=3):
    """Speech on top of speech. No extra data needed, drawn from the train set.
    The most realistic degradation for a call centre or 911 setting."""
    if other_fn is None:
        return w
    acc = np.zeros(len(w), dtype=np.float32)
    got = 0
    for _ in range(n_speakers):
        o = other_fn()
        if o is None or len(o) < 16:
            continue
        acc += _fit_len(o, len(w), rng)
        got += 1
    if got == 0:
        return w
    return _mix_at_snr(w, acc, snr_db)


def _fftconvolve(w, h):
    n = len(w) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(w, nfft) * np.fft.rfft(h, nfft), nfft)
    return y[: len(w)].astype(np.float32)


def aug_rir(w, sr, t60, rng, bank=None):
    """Use a real RIR bank if present, otherwise a synthetic exponentially decaying tail."""
    if bank:
        h = np.asarray(bank[rng.integers(len(bank))], dtype=np.float32)
    else:
        n = max(8, int(t60 * sr))
        h = rng.standard_normal(n).astype(np.float32)
        h *= np.exp(-6.9078 * np.arange(n, dtype=np.float32) / n)  # -60 dB @ t60
        h[0] += 1.0
    h = h / (float(np.abs(h).sum()) + 1e-9)
    return _fftconvolve(w, h)


@dataclass
class AugConfig:
    """All zero means augmentation is off."""

    p_clean: float = 0.40  # leave this fraction untouched (protects clean WER)
    # speed / tempo / pitch
    speed_rates: tuple = ()
    p_speed: float = 0.0
    tempo_rates: tuple = ()
    p_tempo: float = 0.0
    pitch_semitones: tuple = ()
    p_pitch: float = 0.0
    # room
    p_rir: float = 0.0
    rir_t60: tuple = (0.15, 0.50)
    rir_dir: str | None = None
    # noise
    p_noise: float = 0.0
    snr_db: tuple = (5.0, 20.0)
    noise_kind: str = "pink"
    noise_dir: str | None = None
    p_babble: float = 0.0
    babble_snr_db: tuple = (10.0, 25.0)
    babble_speakers: int = 3
    # channel / codec
    p_band: float = 0.0
    p_8k: float = 0.0
    p_mulaw: float = 0.0
    p_alaw: float = 0.0
    p_packet: float = 0.0
    packet_rate: tuple = (0.01, 0.08)
    # level
    p_gain: float = 0.0
    gain_db: tuple = (-6.0, 6.0)
    p_clip: float = 0.0
    clip_thresh: tuple = (0.3, 0.9)

    _KEYS = (
        "p_speed", "p_tempo", "p_pitch", "p_rir", "p_noise", "p_babble",
        "p_band", "p_8k", "p_mulaw", "p_alaw", "p_packet", "p_gain", "p_clip",
    )

    def active(self):
        return any(getattr(self, k) > 0 for k in self._KEYS)

    def names(self):
        return [k[2:] for k in self._KEYS if getattr(self, k) > 0]


_BANKS: dict[str, list] = {}


def _load_bank(d, limit=200):
    """Load the wav folder into memory. Every dataloader worker loads its own
    copy, which is why the limit exists."""
    if not d:
        return None
    if d in _BANKS:
        return _BANKS[d] or None
    import soundfile as sf

    files = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))[:limit]
    bank = []
    for p in files:
        try:
            x, _ = sf.read(p, dtype="float32")
            bank.append(x.mean(1) if x.ndim > 1 else x)
        except Exception:
            pass
    _BANKS[d] = bank
    log(f"[BANK] {len(bank)} files <- {d}")
    return bank or None


def apply_aug(w, a: AugConfig, sr, rng, other_fn=None):
    """The chain order is physical. source -> room -> noise -> channel -> level.
    Changing the order changes the result (reverb AFTER the codec is wrong)."""
    if not a.active() or rng.random() < a.p_clean:
        return w

    if a.speed_rates and rng.random() < a.p_speed:
        w = aug_speed(w, float(rng.choice(np.asarray(a.speed_rates))))
    if a.tempo_rates and rng.random() < a.p_tempo:
        w = aug_tempo(w, float(rng.choice(np.asarray(a.tempo_rates))), sr)
    if a.pitch_semitones and rng.random() < a.p_pitch:
        w = aug_pitch(w, float(rng.choice(np.asarray(a.pitch_semitones))), sr)

    if rng.random() < a.p_rir:
        w = aug_rir(w, sr, float(rng.uniform(*a.rir_t60)), rng, _load_bank(a.rir_dir))

    if rng.random() < a.p_babble:
        w = aug_babble(w, float(rng.uniform(*a.babble_snr_db)), rng,
                       other_fn, a.babble_speakers)
    if rng.random() < a.p_noise:
        w = aug_noise(w, float(rng.uniform(*a.snr_db)), rng,
                      a.noise_kind, _load_bank(a.noise_dir))

    if rng.random() < a.p_band:
        w = aug_bandpass(w, sr)
    if rng.random() < a.p_8k:
        w = aug_8k_roundtrip(w, sr)
    if rng.random() < a.p_mulaw:
        w = aug_mulaw(w)
    elif rng.random() < a.p_alaw:  # applying both at once makes no sense
        w = aug_alaw(w)
    if rng.random() < a.p_packet:
        w = aug_packet_loss(w, sr, float(rng.uniform(*a.packet_rate)), rng=rng)

    if rng.random() < a.p_gain:
        w = aug_gain(w, float(rng.uniform(*a.gain_db)))
    if rng.random() < a.p_clip:
        w = aug_clip(w, float(rng.uniform(*a.clip_thresh)))

    peak = float(np.abs(w).max()) + 1e-9
    return (w / peak).astype(np.float32) if peak > 1.0 else w


def degrade_eval(w, mode, sr=16000):
    """Evaluation conditions. Deterministic, identical to the benchmark."""
    if mode in (None, "clean"):
        return w
    if mode == "tel":
        return aug_bandpass(w, sr)
    if mode == "tel8k":
        return aug_8k_roundtrip(aug_bandpass(w, sr), sr)
    if mode == "noisy10":  # extra diagnostic condition, fixed seed -> deterministic
        return aug_noise(w, 10.0, np.random.default_rng(12345), "pink")
    raise ValueError(mode)


def input_norm(w, mode):
    if mode == "zscore":  # same as fairseq 'normalize=True'
        return ((w - w.mean()) / (w.std() + 1e-7)).astype(np.float32)
    if mode == "peak":
        return (w / (np.abs(w).max() + 1e-7)).astype(np.float32)
    return w


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

    # ---- audio / architecture (same as v2) ----
    sr: int = 16_000
    hid: int = 768
    max_secs: float = 20.0
    ws_layers: tuple = (9, 10, 11, 12)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple = ("q_proj", "v_proj")
    lora_layers: tuple | None = None

    # ---- REGULARIZATION ----
    weight_decay: float = 0.0
    lora_dropout: float = 0.0
    head_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    layerdrop: float = 0.0  # >0 can drop the very layer weighted-sum reads
    mask_time_prob: float = 0.0
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    entropy_reg: float = 0.0  # EnCTC (Liu 2018), RAISE entropy to break early overconfidence
    aux_ctc_layer: int | None = None  # InterCTC (Lee & Watanabe 2021)
    aux_ctc_weight: float = 0.0
    ema_decay: float = 0.0  # 0 means off, 0.999 is typical
    input_norm: str = "none"  # none | zscore | peak
    sched: str = "plateau"  # plateau | cosine
    warmup_steps: int = 0

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

    # ---- warm start / persistence ----
    init_from: str | None = None
    resume: bool = True
    final_conditions: tuple = ("clean", "tel", "tel8k")

    # ---- resources ----
    audio_mode: str = "memmap"
    amp_dtype: str = "bfloat16"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    deadline_ts: float | None = None  # the epoch loop breaks once this time passes

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.out_root = Path(self.out_root)
        self.cache_root = Path(self.cache_root)
        self.ws_layers = tuple(sorted(int(x) for x in self.ws_layers))
        if self.use_lora and self.lora_layers is None:
            self.lora_layers = tuple(range(1, max(self.ws_layers) + 1))
        elif self.lora_layers is not None:
            self.lora_layers = tuple(sorted(int(x) for x in self.lora_layers))
        if self.aux_ctc_layer is not None and self.aux_ctc_layer not in self.ws_layers:
            raise ValueError(
                f"aux_ctc_layer={self.aux_ctc_layer} is not in ws_layers={self.ws_layers}"
            )

    @property
    def backbone_train_mode(self):
        """SpecAugment and backbone dropout only run in train() mode."""
        return (
            self.mask_time_prob > 0 or self.mask_feature_prob > 0
            or self.hidden_dropout > 0 or self.attention_dropout > 0
            or self.activation_dropout > 0 or self.feat_proj_dropout > 0
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

    def reg_names(self):
        out = []
        for n, v in (
            ("wd", self.weight_decay), ("lora_do", self.lora_dropout),
            ("head_do", self.head_dropout), ("hid_do", self.hidden_dropout),
            ("attn_do", self.attention_dropout), ("mask_t", self.mask_time_prob),
            ("mask_f", self.mask_feature_prob), ("ent", self.entropy_reg),
            ("interctc", self.aux_ctc_weight), ("ema", self.ema_decay),
        ):
            if v and v > 0:
                out.append(f"{n}={v:g}")
        if self.input_norm != "none":
            out.append(f"norm={self.input_norm}")
        if self.sched != "plateau":
            out.append(f"sched={self.sched}")
        return out

    def to_dict(self):
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    def summary(self):
        return (
            f"[CFG] {self.run_name} | read={list(self.ws_layers)} "
            f"| adapt={list(self.lora_layers) if self.use_lora else 'NONE'} "
            f"| {self.num_epochs}ep bs{self.train_batch}x{self.accumulation_steps} "
            f"| aug=[{','.join(self.aug.names()) or '-'}] "
            f"| reg=[{','.join(self.reg_names()) or '-'}] "
            f"| bb_train={self.backbone_train_mode}"
            + (f" | init<-{self.init_from}" if self.init_from else "")
        )


# ============================================================================
# 3 - DATA. The cache key is byte-for-byte the same as sweep_v2
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
        f for f in list_repo_files(HF_REPO, repo_type="dataset")
        if f.startswith(pre) and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"no '{pre}*.parquet' inside {HF_REPO}")
    log(f"[HF] {HF_REPO} :: {pre} — {len(files)} parquet")
    return [f"hf://datasets/{HF_REPO}/{f}" for f in files]


def decode_audio(cell, target_sr):
    import soundfile as sf

    if isinstance(cell, dict) and cell.get("array") is not None:
        w, sr = np.asarray(cell["array"], np.float32), cell.get("sampling_rate", target_sr)
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
        hf = load_dataset("parquet", data_files=files, split="train",
                          verification_mode="no_checks")
    elif cfg.hf_fallback:
        hf = load_dataset("parquet", data_files={"d": hf_parquet_urls(split)},
                          split="d", verification_mode="no_checks")
    else:
        raise FileNotFoundError(f"[{split}] '{pattern}' is missing and hf_fallback is off.")
    with contextlib.suppress(Exception):
        from datasets import Audio

        hf = hf.cast_column("audio", Audio(decode=False))
    cap = cfg.max_train_samples if split == "train" else cfg.max_eval_samples
    return hf.select(range(min(cap, len(hf)))) if cap else hf


def _cache_key(cfg, split):
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
                log(f"  [cache:{split}] {i + 1}/{len(hf)} "
                    f"({pos * 2 / 1e9:.1f} GB, {time.perf_counter() - t0:.0f}s)")
    meta_p.write_text(json.dumps({"offsets": offsets, "texts": texts}))
    log(f"[cache:{split}] ready - {len(texts)} samples, {pos * 2 / 1e9:.2f} GB, "
        f"{(time.perf_counter() - t0) / 60:.1f} min")


def prepare_split(cfg, split):
    cache_dir = cfg.cache_root / f"{split}_{_cache_key(cfg, split)}"
    bin_p, meta_p = cache_dir / "audio.i16", cache_dir / "meta.json"
    if not (bin_p.exists() and meta_p.exists()):
        _build_cache(cfg, split, cache_dir)
    else:
        log(f"[DATA:{split}] cache found -> {cache_dir}")
    meta = json.loads(meta_p.read_text())
    offsets = np.asarray(meta["offsets"], dtype=np.int64)
    texts, n = meta["texts"], int(meta["offsets"][-1])
    if cfg.audio_mode == "memmap":
        buf = np.memmap(bin_p, dtype=np.int16, mode="r", shape=(n,))
    else:
        buf = np.fromfile(bin_p, dtype=np.int16)
    log(f"[DATA:{split}] {cfg.audio_mode} - {len(texts)} samples, {n * 2 / 1e9:.2f} GB")
    return buf, offsets, texts


def _make_data_classes():
    import torch
    from torch.utils.data import Dataset
    import torch.utils.data as tud

    class SpeechDS(Dataset):
        def __init__(self, buf, offsets, texts, vocab, cfg, aug=None, degrade=None):
            self.buf, self.offsets, self.texts = buf, offsets, texts
            self.vocab, self.cfg = vocab, cfg
            self.aug, self.degrade = aug, degrade
            self.max_len = int(cfg.max_secs * cfg.sr)
            self._rng = None

        def __len__(self):
            return len(self.texts)

        def rng(self):
            # A separate seed per worker. Without it EVERY worker produces the same
            # augmentation sequence and the diversity is fake.
            if self._rng is None:
                info = tud.get_worker_info()
                self._rng = np.random.default_rng(
                    self.cfg.seed * 100003 + (info.id if info else 0)
                )
            return self._rng

        def _raw(self, i):
            a, b = int(self.offsets[i]), int(self.offsets[i + 1])
            return np.asarray(self.buf[a:b], dtype=np.float32) / 32768.0

        def _other(self):
            return self._raw(int(self.rng().integers(len(self.texts))))

        def __getitem__(self, i):
            w = self._raw(i)
            if self.aug is not None:
                w = apply_aug(w, self.aug, self.cfg.sr, self.rng(), self._other)
            if self.degrade:
                w = degrade_eval(w, self.degrade, self.cfg.sr)
            w = input_norm(w, self.cfg.input_norm)
            w = np.ascontiguousarray(w[: self.max_len], dtype=np.float32)
            ids = [
                self.vocab.get(c, self.vocab["[UNK]"])
                for c in self.texts[i].replace(" ", "|")
            ]
            return torch.from_numpy(w), torch.tensor(ids, dtype=torch.long), i

    class Collate:
        """A class, not a closure, because spawn workers pickle the argument."""

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

    if any(not (1 <= l <= n_layers) for l in cfg.ws_layers):
        raise ValueError(f"ws_layers={list(cfg.ws_layers)} is invalid ({n_layers} layers)")
    if cfg.layerdrop > 0:
        log("[WARN] layerdrop>0, the layer it skips may be the very "
            "hidden_state that weighted-sum reads. 0 is recommended.")
    log(f"[BB] {cfg.backbone} | layers={n_layers} | hid={bb.config.hidden_size} "
        f"| spec_aug={over['apply_spec_augment']}")

    if cfg.use_lora:
        dead = [l for l in cfg.lora_layers if l > max(cfg.ws_layers)]
        if dead:
            log(f"[WARN] {dead} sits above the deepest layer read -> NO GRADIENT")
        lcfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets), bias="none",
            layers_to_transform=hs_to_module_idx(cfg.lora_layers),
        )
        bb = inject_adapter_in_model(lcfg, bb)
        for n, p in bb.named_parameters():
            p.requires_grad = "lora_" in n
        n_tr = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        exp = cfg.expected_adapter_params()
        log(f"[LORA] adapter {n_tr:,} (expected {exp:,}) | dropout={cfg.lora_dropout}")
        assert n_tr == exp, f"LoRA has the wrong scope: {n_tr:,} != {exp:,}"
    else:
        for p in bb.parameters():
            p.requires_grad = False
        log("[LORA] off, frozen backbone")

    bb.eval()
    return bb, feat_len_fn, n_layers


def _make_head_class():
    import torch
    import torch.nn as nn

    class WeightedSumHead(nn.Module):
        """x: [B,T,N_WS,D] -> (logits, aux_logits|None)"""

        def __init__(self, n_ws, dim, vocab_size, dropout=0.0, aux_idx=None):
            super().__init__()
            self.n_ws, self.aux_idx = n_ws, aux_idx
            self.layer_w = nn.Parameter(torch.zeros(n_ws))
            layers = [nn.Linear(dim, dim), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim, vocab_size))
            self.net = nn.Sequential(*layers)
            # InterCTC. CTC applied directly to an intermediate layer, a deep supervision effect
            self.aux = nn.Linear(dim, vocab_size) if aux_idx is not None else None

        def weights(self):
            return self.layer_w.softmax(0)

        def forward(self, x):
            if self.n_ws == 1:
                feat = x[:, :, 0, :]
            else:
                feat = (x * self.layer_w.softmax(0)[None, None, :, None]).sum(2)
            aux = self.aux(x[:, :, self.aux_idx, :]) if self.aux is not None else None
            return self.net(feat), aux

    return WeightedSumHead


def param_groups(head, backbone, cfg):
    rest = [p for n, p in head.named_parameters() if n != "layer_w"]
    groups = [
        {"params": rest, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": [head.layer_w], "lr": cfg.layer_w_lr, "weight_decay": 0.0},
    ]
    if cfg.use_lora:
        groups.append({
            "params": [p for p in backbone.parameters() if p.requires_grad],
            "lr": cfg.lora_lr, "weight_decay": cfg.weight_decay,
        })
    return groups


def adapter_state_dict(backbone):
    return {k: v.detach().cpu().clone()
            for k, v in backbone.state_dict().items() if "lora_" in k}


GROUP_NAMES = ["head", "layer_w", "lora"]


class EMA:
    """Exponential moving average of the trainable parameters.
    Cheap, and the literature almost always reports a small but consistent gain."""

    def __init__(self, params, decay):
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @property
    def on(self):
        return self.decay > 0

    def update(self):
        if not self.on:
            return
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    def apply(self):
        if not self.on:
            return
        self.backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s)

    def restore(self):
        if not self.on or self.backup is None:
            return
        for b, p in zip(self.backup, self.params):
            p.data.copy_(b)
        self.backup = None


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
    return "".join(
        id2ch.get(k, "") for k, _ in groupby(ids) if k not in (blank_id, unk_id)
    ).replace("|", " ").strip()


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
            am = (torch.arange(X.shape[1], device=device)[None, :]
                  < wl.to(device)[:, None]).long()
            with torch.autocast(device_type=device, dtype=cfg.torch_dtype):
                out = backbone(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([out.hidden_states[L] for L in cfg.ws_layers], dim=2)
            xlen = feat_len_fn(wl.to(device))
            pred = head(hs.float())[0].argmax(-1).cpu().numpy()
            for b, i in enumerate(idxs.tolist()):
                T = int(xlen[b])
                hyps.append(greedy_decode(pred[b, :T].tolist(), id2ch, blank_id, unk_id))
                refs.append(texts[i])
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def _make_sched(opt, cfg, steps_per_epoch):
    import torch

    if cfg.sched == "cosine":
        total = max(1, cfg.num_epochs * steps_per_epoch)
        warm = max(0, cfg.warmup_steps)

        def fn(step):
            if warm and step < warm:
                return step / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

        return torch.optim.lr_scheduler.LambdaLR(opt, fn), "step"
    return (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
            threshold=0.005, threshold_mode="rel",
        ),
        "epoch",
    )


def train_one(cfg: Config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    SpeechDS, Collate = _make_data_classes()
    WeightedSumHead = _make_head_class()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log(cfg.summary())

    vocab = build_vocab()
    blank_id, unk_id = vocab["[PAD]"], vocab["[UNK]"]
    id2ch = {v: k for k, v in vocab.items()}

    tr_raw, dv_raw = prepare_split(cfg, "train"), prepare_split(cfg, "dev")
    aug = cfg.aug if cfg.aug.active() else None
    tr = SpeechDS(*tr_raw, vocab, cfg, aug=aug)
    dv = SpeechDS(*dv_raw, vocab, cfg)  # dev is always clean
    collate = Collate(blank_id)

    nw = cfg.num_workers if aug is not None else 0
    train_dl = DataLoader(
        tr, batch_size=cfg.train_batch, shuffle=True, num_workers=nw,
        collate_fn=collate, pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    dev_dl = DataLoader(dv, batch_size=cfg.train_batch, shuffle=False,
                        num_workers=0, collate_fn=collate, pin_memory=True)
    log(f"[DATA] train={len(tr)} dev={len(dv)} | {len(train_dl)} batch/ep | nw={nw}")

    backbone, feat_len_fn, _ = build_backbone(cfg, device)
    aux_idx = (list(cfg.ws_layers).index(cfg.aux_ctc_layer)
               if cfg.aux_ctc_layer is not None else None)
    head = WeightedSumHead(len(cfg.ws_layers), cfg.hid, len(vocab),
                           cfg.head_dropout, aux_idx).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    n_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    log(f"[MODEL] trainable: head={n_head:,} + lora={n_lora:,} = {n_head + n_lora:,}")

    if cfg.init_from:
        src = cfg.out_root / cfg.init_from
        hp, ap = src / "head.pt", src / "adapter.pt"
        if hp.exists():
            miss = head.load_state_dict(torch.load(hp, map_location=device), strict=False)
            log(f"[INIT] head <- {hp} (missing: {list(miss.missing_keys)})")
        else:
            log(f"[WARN] init_from='{cfg.init_from}' has no head.pt, starting FROM SCRATCH, "
                f"this run is not comparable with the others")
        if ap.exists() and cfg.use_lora:
            backbone.load_state_dict(torch.load(ap, map_location=device), strict=False)
            log(f"[INIT] adapter <- {ap}")

    opt = torch.optim.AdamW(param_groups(head, backbone, cfg), fused=(device == "cuda"))
    sched, sched_when = _make_sched(opt, cfg, len(train_dl))
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    trainable = [p for g in opt.param_groups for p in g["params"]]
    ema = EMA(trainable, cfg.ema_decay)

    hist_p, last_p = cfg.out_dir / "history.jsonl", cfg.out_dir / "last.pt"
    start_ep, best_cer, best_ep, history = 1, float("inf"), 0, []

    if cfg.resume and last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        head.load_state_dict(ck["head"])
        if ck.get("adapter"):
            backbone.load_state_dict(ck["adapter"], strict=False)
        opt.load_state_dict(ck["opt"])
        with contextlib.suppress(Exception):
            sched.load_state_dict(ck["sched"])
        start_ep, best_cer, best_ep = ck["epoch"] + 1, ck["best_cer"], ck["best_ep"]
        history = [json.loads(l) for l in hist_p.read_text().splitlines() if l.strip()]
        log(f"[RESUME] resuming from e{ck['epoch']} · best {best_cer * 100:.2f}%")
    elif not hist_p.exists():
        hist_p.write_text("")

    stopped = None
    for epoch in range(start_ep, cfg.num_epochs + 1):
        if cfg.deadline_ts and time.time() > cfg.deadline_ts:
            stopped = "deadline"
            log("[DEADLINE] time is up, cutting training short and moving to evaluation")
            break

        head.train()
        # CRITICAL. v2 called backbone.eval() unconditionally here, and HF applies masking
        # only while self.training is set, so SpecAugment never actually ran.
        backbone.train() if cfg.backbone_train_mode else backbone.eval()
        t0, tot, nb = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for X, Y, wl, ll, _ in train_dl:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            am = (torch.arange(X.shape[1], device=device)[None, :]
                  < wl.to(device)[:, None]).long()
            with torch.autocast(device_type=device, dtype=cfg.torch_dtype):
                out = backbone(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([out.hidden_states[L] for L in cfg.ws_layers], dim=2)
            xlen = feat_len_fn(wl.to(device))
            logits, aux = head(hs.float())
            logp = logits.log_softmax(-1)
            loss = ctc(logp.transpose(0, 1), Y, xlen, ll.to(device))

            if aux is not None and cfg.aux_ctc_weight > 0:  # InterCTC
                la = ctc(aux.log_softmax(-1).transpose(0, 1), Y, xlen, ll.to(device))
                loss = (1 - cfg.aux_ctc_weight) * loss + cfg.aux_ctc_weight * la
            if cfg.entropy_reg > 0:  # EnCTC: RAISE the entropy
                H = -(logp.exp() * logp).sum(-1).mean()
                loss = loss - cfg.entropy_reg * H

            (loss / cfg.accumulation_steps).backward()
            if ((nb + 1) % cfg.accumulation_steps == 0) or ((nb + 1) == len(train_dl)):
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update()
                if sched_when == "step":
                    sched.step()
            tot += loss.item()
            nb += 1

        ema.apply()
        wer, cer = evaluate(head, backbone, dev_dl, dv.texts, feat_len_fn, cfg,
                            device, id2ch, blank_id, unk_id)
        rec = {
            "epoch": epoch, "loss": tot / max(1, nb), "wer": wer, "cer": cer,
            "secs": time.perf_counter() - t0,
            "w": head.weights().detach().cpu().numpy().round(4).tolist(),
            "layers": list(cfg.ws_layers), "run": cfg.run_name,
            "lr": {n: g["lr"] for n, g in zip(GROUP_NAMES, opt.param_groups)},
        }
        history.append(rec)
        with hist_p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"epoch {epoch:>3} | loss {rec['loss']:.3f} | {rec['secs']:.0f}s "
            f"| VAL wer {wer * 100:.2f}% cer {cer * 100:.2f}%")

        if sched_when == "epoch":
            sched.step(cer)
        if cer < best_cer * (1.0 - cfg.delta_rel):
            best_cer, best_ep = cer, epoch
            torch.save(head.state_dict(), cfg.out_dir / "head.pt")
            if cfg.use_lora:
                torch.save(adapter_state_dict(backbone), cfg.out_dir / "adapter.pt")
            log(f"   [SAVE] new best val-CER {cer * 100:.2f}%")
        ema.restore()

        torch.save({
            "head": head.state_dict(),
            "adapter": adapter_state_dict(backbone) if cfg.use_lora else None,
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "epoch": epoch, "best_cer": best_cer, "best_ep": best_ep,
        }, last_p)
        if epoch - best_ep >= cfg.stop_patience:
            stopped = "early_stop"
            log(f"[STOP] no improvement for {cfg.stop_patience} epochs")
            break

    # ---- final: best checkpoint, clean / tel / tel8k ----
    if (cfg.out_dir / "head.pt").exists():
        head.load_state_dict(torch.load(cfg.out_dir / "head.pt", map_location=device),
                             strict=False)
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
        log(f"[FINAL:{mode:8s}] CER {cer * 100:5.2f}  WER {wer * 100:6.2f}")

    ratio = None
    if conds.get("clean", {}).get("wer"):
        ratio = conds.get("tel8k", {}).get("wer", float("nan")) / conds["clean"]["wer"]
    summary = {
        "run": cfg.run_name, "best_cer": best_cer, "best_epoch": best_ep,
        "epochs_done": history[-1]["epoch"] if history else 0,
        "stopped": stopped, "final": conds, "tel8k_over_clean": ratio,
        "sec_per_epoch": float(np.median([h["secs"] for h in history])) if history else None,
        "aug": asdict(cfg.aug), "reg": cfg.reg_names(),
        "cfg": cfg.to_dict(), "history": history,
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[DONE] {cfg.run_name}: val-CER {best_cer * 100:.2f}% @ e{best_ep}"
        + (f" · tel8k/clean ×{ratio:.2f}" if ratio and np.isfinite(ratio) else ""))
    return summary


# ============================================================================
# 6 - HUGGING FACE. Pull and push checkpoints
# ============================================================================


# Credential sources, in order of precedence.
#   1. the --hf-token flag          (written here via set_token)
#   2. the HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable
#   3. ~/.cache/huggingface/token   (output of `huggingface-cli login`)
# Repo:
#   1. the --hf-repo flag
#   2. the ECAD_HF_REPO environment variable
_TOKEN_OVERRIDE: str | None = None
REPO_TYPE = "dataset"  # push_run in sweep_v2 also used a dataset repo


def set_repo_type(t):
    global REPO_TYPE
    REPO_TYPE = t or "dataset"


def set_token(tok: str | None):
    """Store the token from the CLI and also export it so subprocesses can see it."""
    global _TOKEN_OVERRIDE
    tok = (tok or "").strip() or None
    if tok:
        _TOKEN_OVERRIDE = tok
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok


def _tok():
    if _TOKEN_OVERRIDE:
        return _TOKEN_OVERRIDE
    env = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
           or "").strip()
    if env:
        return env
    # token stored by `huggingface-cli login`
    with contextlib.suppress(Exception):
        from huggingface_hub import get_token

        return get_token() or None
    with contextlib.suppress(Exception):
        from huggingface_hub import HfFolder  # older versions

        return HfFolder.get_token() or None
    return None


def token_source():
    """Report where the token came from, without leaking the secret."""
    if _TOKEN_OVERRIDE:
        return "--hf-token"
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        return "environment variable"
    return "huggingface-cli login" if _tok() else None


def hf_preflight(repo, repo_type=None, require_write=True):
    """VERIFY identity and write permission before the overnight run starts.

    Finding out after six hours that there was no token is expensive. This runs
    whoami, create_repo and a small test upload up front."""
    repo_type = repo_type or REPO_TYPE
    if not repo:
        log("[HF] no repo given, the results will stay LOCAL ONLY.")
        log("     In a molab/container environment they vanish when the session ends.")
        return False
    tok = _tok()
    if not tok:
        log("[HF] x no token found. Results cannot be pushed.")
        log("     Fix: --hf-token hf_xxx  |  export HF_TOKEN=hf_xxx  |  "
            "huggingface-cli login")
        return False
    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=tok)
        who = api.whoami()
        log(f"[HF] ✓ identity: {who.get('name')} (source: {token_source()})")
        if not require_write:
            return True
        create_repo(repo, repo_type=repo_type, private=True, exist_ok=True, token=tok)
        api.upload_file(
            path_or_fileobj=json.dumps(
                {"ok": True, "ts": time.time()}
            ).encode(),
            path_in_repo=".preflight.json",
            repo_id=repo, repo_type=repo_type,
        )
        log(f"[HF] ok write access verified: {repo} ({repo_type}, private)")
        return True
    except Exception as e:
        log(f"[HF] x could not verify {repo} - {type(e).__name__}: {e}")
        log("     Does the token have WRITE scope? A read token cannot upload.")
        return False


def hf_list_runs(repo, repo_type=None):
    repo_type = repo_type or REPO_TYPE
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo, repo_type=repo_type, token=_tok())
    return sorted({f.split("/")[0] for f in files if f.endswith("/summary.json")})


def hf_pull_best(repo, out_root: Path, run=None, repo_type=None):
    """Scan the runs in the repo and download the one with the best val CER (or a named run).

    Returns (local_run_name, cfg_dict), where cfg_dict is the downloaded config.json.
    The architecture (ws_layers / lora_layers / r / alpha) is derived from it, otherwise
    the head and adapter shapes will not match."""
    from huggingface_hub import hf_hub_download

    repo_type = repo_type or REPO_TYPE
    tok = _tok()
    runs = [run] if run else hf_list_runs(repo, repo_type)
    if not runs:
        raise FileNotFoundError(f"no run with a summary.json inside {repo}")

    best, best_cer = None, float("inf")
    for r in runs:
        try:
            p = hf_hub_download(repo, f"{r}/summary.json", repo_type=repo_type, token=tok)
            c = json.loads(Path(p).read_text()).get("best_cer", float("inf"))
            log(f"[HF] {r}: val-CER {c * 100:.2f}%")
            if c < best_cer:
                best, best_cer = r, c
        except Exception as e:
            log(f"[HF] could not read {r}: {type(e).__name__}")
    if best is None:
        raise FileNotFoundError("no suitable run")

    dest = out_root / best
    dest.mkdir(parents=True, exist_ok=True)
    got = {}
    for fn in ("head.pt", "adapter.pt", "config.json", "summary.json"):
        try:
            p = hf_hub_download(repo, f"{best}/{fn}", repo_type=repo_type, token=tok)
            (dest / fn).write_bytes(Path(p).read_bytes())
            got[fn] = True
        except Exception:
            got[fn] = False
    if not got.get("head.pt"):
        raise FileNotFoundError(f"{best}/head.pt could not be downloaded")
    log(f"[HF] base '{best}' downloaded (val-CER {best_cer * 100:.2f}%) -> {dest}")
    cfgd = json.loads((dest / "config.json").read_text()) if got.get("config.json") else {}
    return best, cfgd


_PUSH_WARNED = False


def hf_push(cfg: Config, repo, repo_type=None):
    global _PUSH_WARNED
    repo_type = repo_type or REPO_TYPE
    tok = _tok()  # BEFORE the import: if there is nothing to upload the library is not needed
    if not repo or not tok:
        if not _PUSH_WARNED:  # do not swallow silently, say it out loud once
            why = "no repo given" if not repo else "no token"
            log(f"[PUSH] x {why}, runs stay local only: {cfg.out_root}")
            _PUSH_WARNED = True
        return None
    try:
        from huggingface_hub import HfApi, create_repo

        create_repo(repo, repo_type=repo_type, private=True, exist_ok=True, token=tok)
        HfApi(token=tok).upload_folder(
            folder_path=str(cfg.out_dir), path_in_repo=cfg.run_name,
            repo_id=repo, repo_type=repo_type,
            ignore_patterns=["last.pt", "*.npz"],  # last.pt is large and not needed
        )
        log(f"[PUSH] {cfg.run_name} -> {repo}")
        return repo
    except Exception as e:
        log(f"[PUSH] failed ({type(e).__name__}: {e}), the run stays local")
        return None


def cfg_from_downloaded(cfgd: dict, **over) -> dict:
    """Read the architecture from the downloaded config.json. If these disagree the state_dict will not load."""
    keys = ("ws_layers", "lora_layers", "lora_r", "lora_alpha", "lora_targets",
            "hid", "use_lora", "sr", "max_secs", "backbone")
    out = {k: cfgd[k] for k in keys if k in cfgd}
    for k in ("ws_layers", "lora_layers", "lora_targets"):
        if k in out and out[k] is not None:
            out[k] = tuple(out[k])
    out.update(over)
    return out


# ============================================================================
# 7 · ABLATION PLAN
# ============================================================================
# DESIGN. The axes are INDEPENDENT, not cumulative. All warm start from the same base.
#   + Each axis contributes in isolation, no ordering effect.
#   + If the deadline cuts it short the remaining runs are lost but the measured ones stay valid.
#   - It misses interactions, which the COMBO stage compensates for.
#
# C0_control is CRITICAL. It continues the base with the SAME short budget and
# nothing added. Without it every ablation gain is confounded with "12 more epochs".

SPECAUG = dict(mask_time_prob=0.05, mask_time_length=10,
               mask_feature_prob=0.004, mask_feature_length=10)


def _A(**kw):
    return AugConfig(**kw)


def axes() -> dict[str, dict]:
    """name -> Config overrides. Precedence is given by ORDER."""
    nd = os.environ.get("ECAD_NOISE_DIR") or None
    rd = os.environ.get("ECAD_RIR_DIR") or None
    return {
        # -- control --
        "control": {},
        # -- regularization --
        "specaug": SPECAUG,
        "wd": dict(weight_decay=0.01),
        "dropout": dict(lora_dropout=0.05, head_dropout=0.1),
        "bbdrop": dict(hidden_dropout=0.05, attention_dropout=0.05,
                       activation_dropout=0.05),
        "inputnorm": dict(input_norm="zscore"),
        "ema": dict(ema_decay=0.999),
        "interctc": dict(aux_ctc_weight=0.3),  # aux_ctc_layer is assigned from the base
        "entropy": dict(entropy_reg=0.01),
        "cosine": dict(sched="cosine", warmup_steps=200),
        # -- augmentation --
        "speed": dict(aug=_A(speed_rates=(0.9, 1.0, 1.1), p_speed=0.6)),
        "tempo": dict(aug=_A(tempo_rates=(0.9, 1.0, 1.1), p_tempo=0.6)),
        "pitch": dict(aug=_A(pitch_semitones=(-2, -1, 1, 2), p_pitch=0.5)),
        "noise": dict(aug=_A(p_noise=0.5, snr_db=(5.0, 20.0), noise_dir=nd)),
        "babble": dict(aug=_A(p_babble=0.5, babble_snr_db=(10.0, 25.0))),
        "channel": dict(aug=_A(p_band=0.4, p_8k=0.3, p_mulaw=0.25, p_alaw=0.15)),
        "packet": dict(aug=_A(p_packet=0.4, packet_rate=(0.01, 0.08))),
        "rir": dict(aug=_A(p_rir=0.4, rir_dir=rd)),
        "level": dict(aug=_A(p_gain=0.5, p_clip=0.2)),
    }


# Most valuable first, most speculative last. The deadline cuts from the end.
ORDER = [
    "control", "specaug", "channel", "noise", "speed", "wd", "inputnorm",
    "babble", "dropout", "ema", "packet", "interctc", "rir", "cosine",
    "tempo", "level", "entropy", "bbdrop", "pitch",
]

BASE_RUN = "BASE"
ABL_PREFIX = "X_"


def make_ablation(base_cfg: Config, name: str, over: dict, epochs: int,
                  deadline=None) -> Config:
    kw = dict(
        run_name=f"{ABL_PREFIX}{name}", init_from=base_cfg.run_name,
        num_epochs=epochs, head_lr=5e-4, lora_lr=1e-4, layer_w_lr=5e-4,
        stop_patience=max(4, epochs // 2), lr_patience=3, deadline_ts=deadline,
        aug=AugConfig(),  # each axis brings its own aug
    )
    kw.update(over)
    if kw.get("aux_ctc_weight", 0) > 0:
        kw.setdefault("aux_ctc_layer", min(base_cfg.ws_layers))
    keep = {k: getattr(base_cfg, k) for k in
            ("data_root", "out_root", "cache_root", "backbone", "train_glob",
             "dev_glob", "hid", "sr", "max_secs", "ws_layers", "lora_layers",
             "lora_r", "lora_alpha", "lora_targets", "use_lora", "train_batch",
             "accumulation_steps", "num_workers", "seed", "audio_mode", "amp_dtype",
             "max_train_samples", "max_eval_samples", "final_conditions")}
    keep.update(kw)
    return Config(**keep)


# ---- winner selection -----------------------------------------------------

CLEAN_TOL = 1.05  # clean WER is allowed to degrade by this factor


def pick_winners(out_root: Path, verbose=True):
    """Relative to control. Either the tel8k/clean ratio without hurting clean too much, OR
    axes that improve clean WER win."""
    def load(name):
        p = out_root / name / "summary.json"
        return json.loads(p.read_text()) if p.exists() else None

    ctrl = load(f"{ABL_PREFIX}control")
    if not ctrl:
        return [], None
    c_clean = ctrl["final"].get("clean", {}).get("wer")
    c_ratio = ctrl.get("tel8k_over_clean")
    if not c_clean:
        return [], ctrl

    wins = []
    for name in ORDER:
        if name == "control":
            continue
        s = load(f"{ABL_PREFIX}{name}")
        if not s or not s.get("final", {}).get("clean", {}).get("wer"):
            continue
        cl, ra = s["final"]["clean"]["wer"], s.get("tel8k_over_clean")
        ok_clean = cl <= c_clean * CLEAN_TOL
        better = (cl < c_clean) or (ra is not None and c_ratio is not None
                                    and ra < c_ratio)
        if verbose:
            log(f"  {name:10s} clean {cl * 100:5.2f} (ctrl {c_clean * 100:5.2f}) "
                f"ratio {ra if ra else float('nan'):.2f} -> "
                f"{'KEEP' if ok_clean and better else 'DROP'}")
        if ok_clean and better:
            wins.append((name, cl, ra))
    return wins, ctrl


def merge_axes(base_cfg: Config, names, epochs, deadline=None) -> Config:
    """Combine the winning axes into a single configuration.
    Probabilities take the max, lists take the union, scalars take the last writer."""
    A = axes()
    over: dict = {}
    aug_kw: dict = {}
    for n in names:
        for k, v in A[n].items():
            if k == "aug":
                for f, val in asdict(v).items():
                    if f.startswith("_"):
                        continue
                    if f.startswith("p_") and f != "p_clean":
                        aug_kw[f] = max(aug_kw.get(f, 0.0), val)
                    elif isinstance(val, (list, tuple)) and val:
                        aug_kw[f] = tuple(dict.fromkeys(tuple(aug_kw.get(f, ())) + tuple(val))) \
                            if f.endswith(("rates", "semitones")) else tuple(val)
                    elif val not in (None, 0, 0.0, (), ""):
                        aug_kw.setdefault(f, val)
            else:
                over[k] = v
    if aug_kw:
        aug_kw.setdefault("p_clean", 0.40)
        over["aug"] = AugConfig(**aug_kw)
    cfg = make_ablation(base_cfg, "combo", over, epochs, deadline)
    return replace(cfg, run_name="COMBO")


# ============================================================================
# 8 - OVERNIGHT RUNNER, deadline aware
# ============================================================================


def _fmt(mins):
    return f"{int(mins) // 60}h {int(mins) % 60}m"


def plan_night(hours, base_epochs, abl_epochs, sec_per_epoch, have_base,
               n_axes=None):
    """Split the budget across base + N axes + combo. Returns how many axes fit."""
    per_ep = sec_per_epoch / 60.0
    eval_overhead = 3 * per_ep * 0.35  # final eval on 3 conditions, about 35% of an epoch
    base_min = 0.0 if have_base else base_epochs * per_ep + eval_overhead
    abl_min = abl_epochs * per_ep + eval_overhead
    combo_min = abl_min
    budget = hours * 60 - base_min - combo_min
    fits = max(0, int(budget // abl_min))
    total = len(ORDER) if n_axes is None else n_axes
    return {
        "per_epoch_min": per_ep, "base_min": base_min, "abl_min": abl_min,
        "combo_min": combo_min, "n_fit": min(fits, total), "n_total": total,
        "est_total_min": base_min + min(fits, total) * abl_min + combo_min,
    }


def run_night(hours=6.0, base_epochs=30, abl_epochs=12, final_epochs=50,
              hf_repo=None, hf_run=None, only=None, base_cfg_over=None,
              do_combo=True, dry=False, sec_per_epoch=130.0):
    out_root = _envp("ECAD_OUT_ROOT", "./runs")
    out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    deadline = t_start + hours * 3600
    base_over = dict(base_cfg_over or {})

    # ---- 0) CREDENTIAL PRE-CHECK, before burning 6 hours ----
    can_push = hf_preflight(hf_repo) if not dry else bool(_tok() and hf_repo)
    if hf_repo and not can_push:
        log("[HF] warning: the base can be downloaded but results CANNOT be pushed.")

    # ---- 1) BASE ----
    base_name, pulled = BASE_RUN, False
    if hf_repo:
        try:
            base_name, cfgd = hf_pull_best(hf_repo, out_root, hf_run)
            base_over = cfg_from_downloaded(cfgd, **base_over)
            pulled = True
        except Exception as e:
            log(f"[HF] could not pull the base ({type(e).__name__}: {e}), training from scratch")
    if not pulled and (out_root / BASE_RUN / "head.pt").exists():
        log(f"[BASE] '{BASE_RUN}' exists locally, it will not be retrained")
        pulled = True

    base_cfg = Config(run_name=base_name, num_epochs=base_epochs,
                      deadline_ts=deadline, **base_over)

    est = plan_night(hours, base_epochs, abl_epochs, sec_per_epoch, pulled,
                     len(only) if only else None)
    order = [a for a in ORDER if (only is None or a in only)]
    log("=" * 78)
    log(f"OVERNIGHT PLAN - budget {hours:.1f} h - epoch approx {est['per_epoch_min']:.1f} min")
    log(f"  base       : {'HF/local (0 min)' if pulled else _fmt(est['base_min'])}"
        f"{'' if pulled else f' — {base_epochs} epoch'}")
    log(f"  ablation   : {_fmt(est['abl_min'])} × {est['n_fit']}/{len(order)} axes"
        f" ({abl_epochs} epoch)")
    log(f"  combo      : {_fmt(est['combo_min'])}")
    log(f"  TOTAL est. : {_fmt(est['est_total_min'])}")
    log(f"  order      : {', '.join(order[: est['n_fit']])}")
    if est["n_fit"] < len(order):
        log(f"  will not fit: {', '.join(order[est['n_fit']:])}")
    log("=" * 78)
    if dry:
        return est

    if not pulled:
        log(f"\n### BASE - {base_epochs} epochs from scratch")
        try:
            s = train_one(base_cfg)
            if s.get("sec_per_epoch"):
                sec_per_epoch = s["sec_per_epoch"]
            hf_push(base_cfg, hf_repo)
        except Exception:
            log("[ERROR] base training crashed:\n" + traceback.format_exc())
            return None
    else:
        log(f"\n### BASE ready: {base_name}")

    # ---- 2) ABLATION ----
    A = axes()
    done = []
    for name in order:
        cfg = make_ablation(base_cfg, name, A[name], abl_epochs, deadline)
        sp = cfg.out_dir / "summary.json"
        if sp.exists():
            log(f"[SKIP] {cfg.run_name} already finished")
            done.append(name)
            continue
        left = (deadline - time.time()) / 60.0
        need = abl_epochs * sec_per_epoch / 60.0 * 1.15
        if left < need:
            log(f"[BUDGET] {name} skipped, {left:.0f} min left < {need:.0f} min needed")
            continue
        log(f"\n### AXIS {name}  ({left:.0f} min left)")
        try:
            s = train_one(cfg)
            if s.get("sec_per_epoch"):
                sec_per_epoch = 0.5 * sec_per_epoch + 0.5 * s["sec_per_epoch"]
            done.append(name)
            hf_push(cfg, hf_repo)
        except Exception:
            # One failing axis must not end the whole night.
            log(f"[ERROR] {name} crashed, skipping:\n" + traceback.format_exc())
        finally:
            _free()

    # ---- 3) COMBO ----
    log("\n### WINNER SELECTION")
    wins, ctrl = pick_winners(out_root)
    log(f"winners: {[w[0] for w in wins] or 'none'}")
    if do_combo and wins and (deadline - time.time()) / 60 > abl_epochs * sec_per_epoch / 60:
        cfg = merge_axes(base_cfg, [w[0] for w in wins], abl_epochs, deadline)
        log(f"\n### COMBO — {cfg.summary()}")
        try:
            train_one(cfg)
            hf_push(cfg, hf_repo)
        except Exception:
            log("[ERROR] combo crashed:\n" + traceback.format_exc())
        finally:
            _free()
    elif do_combo:
        log("[BUDGET] no time left for combo, run 'python aug_night_v2.py --stage combo' in the morning")

    report(out_root)
    log(f"\nElapsed: {_fmt((time.time() - t_start) / 60)}")
    log(f"Next step: run the best setup for {final_epochs} epochs -> --stage final")
    return {"done": done, "winners": [w[0] for w in wins]}


def _free():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 9 · REPORT
# ============================================================================


def report(out_root: Path):
    rows = []
    for p in sorted(out_root.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        f = s.get("final", {})
        rows.append((
            s.get("run", p.parent.name),
            (s.get("best_cer") or float("nan")) * 100,
            f.get("clean", {}).get("wer", float("nan")) * 100,
            f.get("tel", {}).get("wer", float("nan")) * 100,
            f.get("tel8k", {}).get("wer", float("nan")) * 100,
            s.get("tel8k_over_clean") or float("nan"),
            s.get("epochs_done", 0),
            s.get("stopped") or "",
        ))
    if not rows:
        log("No summary.json yet.")
        return
    ctrl = next((r for r in rows if r[0] == f"{ABL_PREFIX}control"), None)
    log("\n" + "=" * 96)
    log("ABLATION — LibriSpeech dev-clean · greedy CTC (no KenLM)")
    log("=" * 96)
    log(f"{'run':<18}{'valCER':>8}{'clean W':>9}{'tel W':>8}{'tel8k W':>9}"
        f"{'tel8k/cl':>10}{'Δclean':>9}{'ep':>4}  not")
    log("-" * 96)
    for r in sorted(rows, key=lambda x: (x[0] != BASE_RUN, x[0])):
        d = f"{r[2] - ctrl[2]:+.2f}" if ctrl and np.isfinite(r[2]) else "—"
        log(f"{r[0]:<18}{r[1]:>8.2f}{r[2]:>9.2f}{r[3]:>8.2f}{r[4]:>9.2f}"
            f"{r[5]:>10.2f}{d:>9}{r[6]:>4}  {r[7]}")
    log("=" * 96)
    log("RULE: take the axis that lowers the tel8k/clean ratio while keeping d-clean within +5% rel.")
    log("Always compare against X_control, never against BASE. The gap between them")
    log("is the 'extra epochs' effect and must not be credited to augmentation.")
    log("Augmentation does not change the W/C ratio, only the LM does.")


# ============================================================================
# 10 · SELFTEST
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
        log(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")

    def energy(x, lo, hi):
        X = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return float(X[(f >= lo) & (f < hi)].sum())

    log("[1] speed / tempo / pitch")
    chk("speed 1.1 shortens", abs(len(aug_speed(w, 1.1)) - sr / 1.1) <= 2)
    chk("speed 1.0 no-op", np.array_equal(aug_speed(w, 1.0), w))
    tp = aug_tempo(w, 1.2, sr)
    chk("tempo 1.2 shortens", abs(len(tp) - sr / 1.2) <= sr * 0.05, f"({len(tp)})")
    chk("tempo preserves pitch",
        energy(tp, 400, 480) > 0.3 * energy(tp, 0, 8000) * 0.1, "(440 Hz in place)")
    pc = aug_pitch(w, 2, sr)
    chk("pitch roughly preserves duration", abs(len(pc) - len(w)) < sr * 0.1, f"({len(pc)})")

    log("[2] channel / codec")
    b = aug_bandpass(w, sr)
    chk("bandpass removes 5 kHz", energy(b, 4000, 8000) < 0.01 * energy(w, 4000, 8000))
    chk("bandpass preserves 440 Hz", energy(b, 300, 1000) > 0.5 * energy(w, 300, 1000))
    k = aug_8k_roundtrip(w, sr)
    chk("8k preserves length", len(k) == len(w))
    chk("8k removes >4 kHz", energy(k, 4500, 8000) < 0.02 * energy(w, 4500, 8000))
    for fn, nm in ((aug_mulaw, "mulaw"), (aug_alaw, "alaw")):
        y = fn(w)
        err = float(np.abs(y - w).mean())
        chk(f"{nm} close but lossy", len(y) == len(w) and 1e-5 < err < 0.1,
            f"(mean error {err:.4f})")
    pl = aug_packet_loss(w, sr, 0.5, rng=np.random.default_rng(1))
    zf = float((pl == 0).mean())
    chk("packet loss zeroes about 50%", 0.25 < zf < 0.75, f"({zf * 100:.0f}%)")

    log("[3] noise / room")
    for snr in (0.0, 10.0, 20.0):
        d = aug_noise(w, snr, rng) - w
        got = 10 * np.log10(((w**2).mean() + 1e-12) / ((d**2).mean() + 1e-12))
        chk(f"noise SNR {snr:.0f} dB", abs(got - snr) < 1.0, f"(measured {got:.2f})")
    other = lambda: 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    bb = aug_babble(w, 10.0, rng, other, 3)
    d = bb - w
    got = 10 * np.log10(((w**2).mean() + 1e-12) / ((d**2).mean() + 1e-12))
    chk("babble SNR 10 dB", abs(got - 10.0) < 1.0, f"(measured {got:.2f})")
    chk("babble other_fn=None -> no-op", np.array_equal(aug_babble(w, 10, rng, None), w))
    r = aug_rir(w, sr, 0.3, rng)
    chk("rir preserves length", len(r) == len(w))
    chk("rir does not blow up the energy", np.abs(r).max() < 3 * np.abs(w).max())

    log("[4] level / norm")
    chk("gain +6 dB ~ ×2",
        abs(np.abs(aug_gain(w, 6.0)).max() / np.abs(w).max() - 2.0) < 0.05)
    z = input_norm(w, "zscore")
    chk("zscore mean~0 std~1", abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3)
    chk("peak norm peak=1", abs(np.abs(input_norm(w, "peak")).max() - 1) < 1e-5)
    chk("norm none no-op", np.array_equal(input_norm(w, "none"), w))

    log("[5] chain")
    full = AugConfig(
        p_clean=0.0, speed_rates=(0.9, 1.1), p_speed=1.0,
        tempo_rates=(0.95,), p_tempo=1.0, pitch_semitones=(1,), p_pitch=1.0,
        p_rir=1.0, p_noise=1.0, p_babble=1.0, p_band=1.0, p_8k=1.0,
        p_mulaw=1.0, p_alaw=1.0, p_packet=1.0, p_gain=1.0, p_clip=1.0,
    )
    y = apply_aug(w.copy(), full, sr, rng, other)
    chk("full chain is finite", np.isfinite(y).all() and len(y) > 0, f"({len(y)} samples)")
    chk("full chain is clipped", np.abs(y).max() <= 1.001)
    chk("an off config is a no-op", np.array_equal(apply_aug(w.copy(), AugConfig(), sr, rng), w))
    chk("p_clean=1 no-op",
        np.array_equal(apply_aug(w.copy(), AugConfig(p_clean=1.0, p_band=1.0), sr, rng), w))
    chk("aug names", set(AugConfig(p_band=0.5, p_noise=0.1).names()) == {"band", "noise"})

    log("[6] eval conditions")
    for m in ("clean", "tel", "tel8k", "noisy10"):
        chk(f"{m} is deterministic",
            np.array_equal(degrade_eval(w, m), degrade_eval(w, m)))

    log("[7] config / plan")
    base = Config(run_name=BASE_RUN, ws_layers=(9, 10, 11, 12))
    chk("adapter param", base.expected_adapter_params() == 2 * 768 * 16 * 2 * 12)
    chk("base bb is in eval mode", not base.backbone_train_mode)
    A = axes()
    chk("every axis is in ORDER", set(A) == set(ORDER), f"({len(A)} axes)")
    for nm in ORDER:
        c = make_ablation(base, nm, A[nm], 12)
        chk(f"  {nm:10s} builds", c.run_name == f"{ABL_PREFIX}{nm}"
            and c.init_from == BASE_RUN and c.layerdrop == 0)
    chk("control is completely empty",
        not make_ablation(base, "control", A["control"], 12).aug.active()
        and not make_ablation(base, "control", A["control"], 12).reg_names())
    chk("specaug bb is in train mode", make_ablation(base, "specaug", A["specaug"], 12)
        .backbone_train_mode)
    chk("specaug has no waveform aug",
        not make_ablation(base, "specaug", A["specaug"], 12).aug.active())
    ic = make_ablation(base, "interctc", A["interctc"], 12)
    chk("the interctc aux layer is inside ws", ic.aux_ctc_layer in ic.ws_layers,
        f"(L{ic.aux_ctc_layer})")
    try:
        Config(ws_layers=(9, 10), aux_ctc_layer=5)
        chk("an invalid aux layer is caught", False)
    except ValueError:
        chk("an invalid aux layer is caught", True)

    log("[8] combo merging")
    cb = merge_axes(base, ["channel", "noise", "speed", "wd", "specaug"], 12)
    chk("combo aug merged",
        cb.aug.p_band > 0 and cb.aug.p_noise > 0 and cb.aug.p_speed > 0,
        f"({cb.aug.names()})")
    chk("combo reg merged", cb.weight_decay == 0.01 and cb.mask_time_prob > 0)
    chk("combo preserves p_clean", cb.aug.p_clean == 0.40)
    chk("combo name", cb.run_name == "COMBO")

    log("[9] budget planner")
    p6 = plan_night(6.0, 30, 12, 130.0, have_base=True)
    p6n = plan_night(6.0, 30, 12, 130.0, have_base=False)
    chk("more axes fit when the base is ready", p6["n_fit"] > p6n["n_fit"],
        f"({p6['n_fit']} vs {p6n['n_fit']})")
    chk("the 6 h estimate stays within budget", p6["est_total_min"] <= 6 * 60 + 1,
        f"({p6['est_total_min']:.0f} min)")
    chk("0 hours -> 0 axes", plan_night(0.2, 30, 12, 130.0, True)["n_fit"] == 0)

    log("[10] HF credentials")
    _saved = (os.environ.pop("HF_TOKEN", None),
              os.environ.pop("HUGGING_FACE_HUB_TOKEN", None))
    global _TOKEN_OVERRIDE, REPO_TYPE, _PUSH_WARNED
    _TOKEN_OVERRIDE = None
    try:
        set_token("hf_TEST123")
        chk("--hf-token is read", _tok() == "hf_TEST123")
        chk("the token is exported", os.environ.get("HF_TOKEN") == "hf_TEST123")
        chk("the source is reported correctly", token_source() == "--hf-token")
        _TOKEN_OVERRIDE = None
        os.environ["HF_TOKEN"] = "hf_ENV456"
        chk("the environment variable is read", _tok() == "hf_ENV456")
        chk("the environment source is reported", token_source() == "environment variable")
        set_token("hf_CLI789")
        chk("the CLI overrides the environment", _tok() == "hf_CLI789")
        chk("an empty token is ignored", (set_token("  ") or _tok()) == "hf_CLI789")
        _TOKEN_OVERRIDE = None
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        chk("preflight without a repo is False", hf_preflight(None) is False)
        chk("the repo type is set",
            (set_repo_type("model") or REPO_TYPE) == "model")
        set_repo_type("dataset")
        _PUSH_WARNED = False
        chk("a push without a repo returns None",
            hf_push(Config(run_name="t"), None) is None)
        chk("the push warning is printed once", _PUSH_WARNED is True)
    finally:
        _TOKEN_OVERRIDE = None
        for k, v in zip(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"), _saved):
            os.environ.pop(k, None)
            if v:
                os.environ[k] = v
        set_repo_type("dataset")
        _PUSH_WARNED = False

    log("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


# ============================================================================
# 11 · CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="Overnight ablation run")
    ap.add_argument("--night", action="store_true", help="the full overnight flow")
    ap.add_argument("--plan", action="store_true", help="show the plan only")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--base-epochs", type=int, default=30)
    ap.add_argument("--abl-epochs", type=int, default=12)
    ap.add_argument("--final-epochs", type=int, default=50)
    ap.add_argument("--sec-per-epoch", type=float, default=130.0,
                    help="initial estimate, refined as it is measured")
    ap.add_argument("--hf-repo", default=os.environ.get("ECAD_HF_REPO"),
                    help="e.g. username/clear-phase1-runs (pulls the base and pushes results). "
                         "Environment: ECAD_HF_REPO")
    ap.add_argument("--hf-token", default=None,
                    help="a token with write scope. If omitted, HF_TOKEN / "
                         "HUGGING_FACE_HUB_TOKEN / huggingface-cli login are tried")
    ap.add_argument("--hf-repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--hf-check", action="store_true",
                    help="verify identity and write access only, then exit")
    ap.add_argument("--hf-run", default=None, help="a specific run name, otherwise the best CER")
    ap.add_argument("--only", default=None, help="comma-separated list of axes")
    ap.add_argument("--no-combo", action="store_true")
    ap.add_argument("--stage", default=None,
                    help="base | combo | final | <axis name> | comma-separated list")
    ap.add_argument("--noise-dir", default=None, help="MUSAN/DEMAND wav folder")
    ap.add_argument("--rir-dir", default=None, help="OpenSLR-28 RIR wav folder")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="64/32 samples, 1 epoch")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list-hf", action="store_true", help="list the runs in the HF repo")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    set_token(args.hf_token)
    set_repo_type(args.hf_repo_type)
    out_root = _envp("ECAD_OUT_ROOT", "./runs")
    if args.noise_dir:
        os.environ["ECAD_NOISE_DIR"] = args.noise_dir
    if args.rir_dir:
        os.environ["ECAD_RIR_DIR"] = args.rir_dir

    if args.hf_check:
        ok = hf_preflight(args.hf_repo, args.hf_repo_type)
        if ok and args.hf_repo:
            runs = hf_list_runs(args.hf_repo, args.hf_repo_type)
            log(f"[HF] runs in the repo ({len(runs)}): {', '.join(runs) or '-'}")
        raise SystemExit(0 if ok else 1)

    if args.list_hf:
        if not args.hf_repo:
            raise SystemExit("--hf-repo is required (or ECAD_HF_REPO)")
        for r in hf_list_runs(args.hf_repo, args.hf_repo_type):
            log(r)
        return
    if args.report:
        report(out_root)
        return

    over = {}
    if args.batch:
        over["train_batch"] = args.batch
        over["accumulation_steps"] = max(1, 256 // args.batch)
    if args.workers is not None:
        over["num_workers"] = args.workers

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    if args.smoke:
        log("### SMOKE — 64 train / 32 dev, 1 epoch, 3 axes")
        base = Config(run_name="smoke_BASE", num_epochs=1, max_train_samples=64,
                      max_eval_samples=32, train_batch=8, accumulation_steps=2,
                      num_workers=0, resume=False, final_conditions=("clean",), **over)
        train_one(base)
        A = axes()
        for nm in ("control", "specaug", "channel"):
            c = make_ablation(base, nm, A[nm], 1)
            c = replace(c, run_name=f"smoke_{nm}", max_train_samples=64,
                        max_eval_samples=32, train_batch=8, accumulation_steps=2,
                        num_workers=0, resume=False, final_conditions=("clean", "tel8k"))
            train_one(c)
        log("SMOKE ok, the path works end to end.")
        return

    if args.night or args.plan:
        run_night(hours=args.hours, base_epochs=args.base_epochs,
                  abl_epochs=args.abl_epochs, final_epochs=args.final_epochs,
                  hf_repo=args.hf_repo, hf_run=args.hf_run, only=only,
                  base_cfg_over=over, do_combo=not args.no_combo,
                  dry=args.plan, sec_per_epoch=args.sec_per_epoch)
        return

    if not args.stage:
        ap.print_help()
        return

    # ---- individual stages ----
    base_over = dict(over)
    base_name = BASE_RUN
    if args.hf_repo:
        with contextlib.suppress(Exception):
            base_name, cfgd = hf_pull_best(args.hf_repo, out_root, args.hf_run)
            base_over = cfg_from_downloaded(cfgd, **base_over)
    base_cfg = Config(run_name=base_name, num_epochs=args.base_epochs, **base_over)
    A = axes()

    for st in [s.strip() for s in args.stage.split(",")]:
        if st == "base":
            train_one(base_cfg)
        elif st == "combo":
            wins, _ = pick_winners(out_root)
            if not wins:
                log("No winner, run the ablation first.")
                continue
            train_one(merge_axes(base_cfg, [w[0] for w in wins], args.abl_epochs))
        elif st == "final":
            wins, _ = pick_winners(out_root)
            cfg = merge_axes(base_cfg, [w[0] for w in wins], args.final_epochs)
            cfg = replace(cfg, run_name="FINAL", init_from=None,
                          num_epochs=args.final_epochs, head_lr=1e-3, lora_lr=2e-4,
                          layer_w_lr=1e-3, stop_patience=12, lr_patience=4,
                          final_conditions=("clean", "tel", "tel8k", "noisy10"))
            log("FINAL - from scratch, with the winning setup: " + cfg.summary())
            train_one(cfg)
        elif st in A:
            train_one(make_ablation(base_cfg, st, A[st], args.abl_epochs))
        else:
            raise SystemExit(f"unknown stage '{st}' - {['base','combo','final'] + ORDER}")
        _free()
    report(out_root)


if __name__ == "__main__":
    main()
