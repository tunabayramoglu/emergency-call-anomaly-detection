# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchaudio", "soundfile==0.14.0"]
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
ASR -- 300h retrain, GPU-side augmentation.

Runs on the BATCHED waveform tensor, on GPU, inside the training loop --
not per-sample on CPU in the DataLoader. The backbone is frozen (only LoRA +
weighted-sum + CTC head are trainable), so there is plenty of spare GPU
compute; doing this per-sample on CPU workers would starve the GPU instead.

Chain order matches the physical story used in aug_night_v2.py / ablation_engine.py
(source -> room -> noise -> channel -> spec masking): reversing it gives
wrong results (e.g. reverb AFTER the telephone round-trip is physically
backwards -- a room impulse response never happens inside a phone network).

What's reused from the existing CPU augmentation work (aug_night_v2.py,
aug_sweep_v1.py, ablation_engine.py):
  - the AugConfig-style dataclass of independent per-effect probabilities
    (p_clean escape hatch, p_speed, p_rir, p_noise, p_band/p_8k) and the
    SNR / T60 ranges those files already tuned (5-20 dB noise, 0.15-0.50 s
    T60 for reverb, 300-3400 Hz telephone band).
  - the physical ordering of the augmentation chain.
  - the "never always-on" philosophy (moderate probabilities, p_clean floor).
This module reimplements the *mechanics* with torch/torchaudio ops so the
whole chain runs batched on GPU; the numpy FFT versions in aug_night_v2.py /
ablation_engine.py were CPU, per-sample and are not reused verbatim for that reason.

Additions the SER work deliberately avoided but which apply here:
  - speed perturbation (0.9/1.0/1.1) -- SER avoided it because it corrupts
    emotion labels; that concern doesn't exist for ASR, where it's a
    standard, cheap accuracy win.
  - reverb is NOT down-weighted -- the live demo is a laptop mic in a room,
    not a phone line, so room acoustics matter more than channel/codec here.

Channel randomisation deliberately does NOT use torchaudio codec APIs
(io.AudioEffector / functional.apply_codec): torchaudio has been in
maintenance mode since 2.8 and encode/decode moved to TorchCodec, so those
APIs may simply be gone on molab's stack. A plain 16k->8k->16k
functional.resample round-trip gives the same "telephone bandwidth" effect
without any extra dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


# ============================================================================
# 1 . Noise / RIR bank -- loaded into RAM ONCE as fp32, shared across steps
# ============================================================================


AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".sph")


def _read_audio(fp: str):
    """Decode one file, trying soundfile before torchaudio.

    torchaudio 2.8+ is in maintenance mode and has been moving decoding out to
    TorchCodec; `torchaudio.load` can raise on an environment where soundfile
    reads the very same file without complaint. soundfile handles every format
    in the noise/RIR banks (OpenSLR-28 and MUSAN are plain PCM wav), so it goes
    first and torchaudio is the fallback rather than the other way round.
    """
    try:
        import soundfile as sf
        import numpy as np

        w, s = sf.read(fp, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(w.T)), int(s)
    except Exception as sf_exc:
        try:
            return torchaudio.load(fp)
        except Exception as ta_exc:
            raise RuntimeError(
                f"soundfile: {type(sf_exc).__name__}: {sf_exc} | "
                f"torchaudio: {type(ta_exc).__name__}: {ta_exc}") from ta_exc


class AudioBank:
    """Loads a folder of audio files into RAM once (~1.4 GB as fp32 for a few
    hundred MUSAN/DEMAND/OpenSLR-28 files at 16 kHz), and serves batched GPU
    tensors on demand. Kept as a plain python list of 1-D CPU tensors -- we
    only move the slice we need to GPU per batch, not the whole bank.

    WHY THE REPORTING IS THIS VERBOSE
    ---------------------------------
    The previous version globbed for `**/*.wav`, wrapped the decode in
    `except Exception: continue`, and printed only the number of clips it ended
    up with. That makes THREE completely different situations print the same
    `0 clips, 0.00 h` line:

      1. the directory is genuinely empty (fetch_noise_banks.py never ran)
      2. the files are there but under an extension the glob missed
      3. the files are there and EVERY decode failed

    Case 3 actually happened -- the banks were downloaded and verified, and the
    trainer still announced "noise bank is EMPTY", sending the search to the
    download step which was never the problem. So this version counts the files
    it found separately from the clips it decoded, and surfaces the most common
    decode error instead of swallowing all of them.
    """

    def __init__(self, root: str | None, sr: int = 16000, limit: int = 4000,
                 device: str | None = None, max_resident_gb: float = 6.0):
        from collections import Counter

        self.sr = sr
        self.clips: list[torch.Tensor] = []
        self.device = "cpu"
        self.root = root
        self.n_found = 0
        self.errors: "Counter[str]" = Counter()
        if not root:
            return

        if not Path(root).is_dir():
            print(f"[BANK] {root}: DIRECTORY DOES NOT EXIST -- nothing to load")
            return

        found = sorted(str(p) for p in Path(root).rglob("*")
                       if p.suffix.lower() in AUDIO_EXTS and p.is_file())
        self.n_found = len(found)
        # Deterministic RANDOM sample, not the alphabetical head. OpenSLR-28
        # unpacks to RIRS_NOISES/simulated_rirs/{largeroom,mediumroom,smallroom}/...
        # and a run reported "capped at 4000 of 60218" -- sorted, that cap is 4000
        # largeroom impulses and zero small rooms. The bank would then teach the
        # model exactly one acoustic environment while claiming to teach reverb.
        if len(found) > limit:
            files = [found[i] for i in
                     sorted(random.Random(1337).sample(range(len(found)), limit))]
        else:
            files = found

        total_s = 0.0
        for fp in files:
            try:
                w, s = _read_audio(fp)
            except Exception as exc:
                self.errors[str(exc)[:200]] += 1
                continue
            w = w.mean(0) if w.dim() > 1 else w
            if s != sr:
                w = torchaudio.functional.resample(w, s, sr)
            self.clips.append(w.float())
            total_s += w.numel() / sr

        # Park the whole bank on the GPU once instead of copying crops across the
        # PCIe bus every batch. `sample_batch` used to do one `.to(device)` PER ROW,
        # so a 64-utterance batch meant 128 tiny host->device transfers (noise +
        # RIR) per step, each a potential stall -- and the cost grew with the batch
        # size, which is exactly backwards when the goal is to feed the GPU more.
        if device and device != "cpu" and self.clips:
            gb = sum(c.numel() for c in self.clips) * 4 / 2**30
            if gb <= max_resident_gb:
                self.clips = [c.to(device, non_blocking=True) for c in self.clips]
                self.device = device
                print(f"[BANK] {root}: resident on {device} ({gb:.2f} GB) -- no "
                      f"per-batch host->device copies")
            else:
                print(f"[BANK] {root}: {gb:.2f} GB exceeds max_resident_gb="
                      f"{max_resident_gb}, staying on CPU (per-batch copies remain)")

        capped = f" (capped at {limit} of {self.n_found})" if self.n_found > limit else ""
        print(f"[BANK] {root}: {len(self.clips)} clips from {self.n_found} audio files"
              f"{capped}, {total_s / 3600:.2f} h loaded")

        if self.n_found == 0:
            others = Counter(p.suffix.lower() for p in Path(root).rglob("*") if p.is_file())
            present = dict(others.most_common(8)) if others else "none, the directory is empty"
            print(f"[BANK]   no {'/'.join(AUDIO_EXTS)} files under this root. "
                  f"Extensions actually present: {present}")
        elif not self.clips:
            top = self.errors.most_common(1)[0]
            print(f"[BANK]   *** {self.n_found} files found but NONE could be decoded. "
                  "This is a decoder problem, NOT a missing-download problem -- "
                  "re-running fetch_noise_banks.py will not help. ***")
            print(f"[BANK]   most common error ({top[1]}x): {top[0]}")
        elif self.errors:
            top = self.errors.most_common(1)[0]
            print(f"[BANK]   {sum(self.errors.values())} of {len(files)} files failed to "
                  f"decode; most common ({top[1]}x): {top[0]}")

    def empty(self) -> bool:
        return len(self.clips) == 0

    def sample_batch(self, n: int, length: int, device, rng: random.Random) -> torch.Tensor:
        """Returns [n, length] float32 tensor on `device`, each row a random
        crop (looped if shorter than `length`) from a random bank clip."""
        out = torch.zeros(n, length, device=device)
        resident = self.device == device
        for i in range(n):
            c = self.clips[rng.randrange(len(self.clips))]
            if c.numel() < length:
                reps = math.ceil(length / max(1, c.numel()))
                c = c.repeat(reps)
            off = rng.randrange(0, max(1, c.numel() - length + 1))
            crop = c[off : off + length]
            # Device-to-device when the bank is resident; only fall back to a
            # host->device copy when it is not.
            out[i] = crop if resident else crop.to(device, non_blocking=True)
        return out


# ============================================================================
# 2 . Config -- mirrors the AugConfig shape from aug_night_v2.py / ablation_engine.py
# ============================================================================


@dataclass
class GPUAugConfig:
    p_clean: float = 0.35          # never-touch floor -- protects clean WER

    # speed perturbation (Ko et al. 2015) -- ENABLED here (unlike SER, which
    # avoided it to protect emotion labels; that reasoning is ASR-irrelevant)
    speed_rates: tuple = (0.9, 1.0, 1.1)
    p_speed: float = 0.6           # applied often; 1.0 is a no-op 1/3 of the time

    # reverb -- weighted UP, not down: the demo is a laptop mic in a ROOM
    p_rir: float = 0.35
    rir_dir: str | None = None     # OpenSLR-28 RIR wavs

    # additive noise -- MUSAN + DEMAND
    p_noise: float = 0.5
    snr_db: tuple = (5.0, 20.0)
    noise_dir: str | None = None

    # SpecAugment (Park et al. 2019) via torchaudio transforms, applied on a
    # complex STFT and inverted back to waveform (see apply_specaugment_gpu)
    p_specaug: float = 0.4
    freq_mask_param: int = 15
    time_mask_param: int = 35
    n_freq_masks: int = 2
    n_time_masks: int = 2

    # channel: 16k->8k->16k round trip via plain resample, NOT codec APIs
    # (io.AudioEffector / apply_codec may not exist on molab's torchaudio,
    # which has been in maintenance mode since 2.8 -- TorchCodec owns
    # encode/decode now). Also cheaper and dependency-free.
    p_channel_8k: float = 0.25

    _KEYS = ("p_speed", "p_rir", "p_noise", "p_specaug", "p_channel_8k")

    def any_on(self) -> bool:
        return any(getattr(self, k) > 0 for k in self._KEYS)


# ============================================================================
# 3 . Batched GPU ops
# ============================================================================


def _mix_at_snr(wave: torch.Tensor, noise: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """torchaudio.functional.add_noise wrapper -- both args [B, T], snr_db [B]."""
    return torchaudio.functional.add_noise(wave, noise, snr_db)


def apply_speed_gpu(wave: torch.Tensor, lengths: torch.Tensor, sr: int,
                     rates: tuple, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """One speed factor per UTTERANCE (not per batch): resample each row at
    its own factor, then re-pad the batch to the new max length. A no-op for
    rows that draw rate==1.0."""
    B, T = wave.shape
    out_rows, new_lens = [], []
    for i in range(B):
        rate = rng.choice(rates)
        w = wave[i, : lengths[i]]
        if rate != 1.0:
            # resample-based speed change: change the "declared" sample rate
            # by `rate`, then resample back to sr -> shortens/lengthens the
            # signal exactly like classic sox speed perturbation
            w = torchaudio.functional.resample(w.unsqueeze(0), int(sr * rate), sr).squeeze(0)
        out_rows.append(w)
        new_lens.append(w.numel())
    max_len = max(new_lens)
    out = torch.zeros(B, max_len, device=wave.device, dtype=wave.dtype)
    for i, w in enumerate(out_rows):
        out[i, : w.numel()] = w
    return out, torch.tensor(new_lens, device=wave.device, dtype=lengths.dtype)


def apply_rir_gpu(wave: torch.Tensor, bank: AudioBank, rng: random.Random) -> torch.Tensor:
    """Batched FFT convolution with a random RIR per row, using
    torchaudio.functional.fftconvolve (falls back to a manual torch.fft
    implementation if the installed torchaudio predates that function)."""
    B, T = wave.shape
    rirs = bank.sample_batch(B, min(T, 16000), wave.device, rng)  # cap RIR length ~1s
    # normalise each RIR to unit L1 energy so the reverberated signal doesn't
    # blow up in level (same convention as aug_rir in ablation_engine.py)
    rirs = rirs / (rirs.abs().sum(-1, keepdim=True) + 1e-9)
    try:
        wet = torchaudio.functional.fftconvolve(wave, rirs, mode="full")[:, :T]
    except AttributeError:
        n = T + rirs.shape[-1] - 1
        nfft = 1 << (n - 1).bit_length()
        W = torch.fft.rfft(wave, nfft)
        H = torch.fft.rfft(rirs, nfft)
        wet = torch.fft.irfft(W * H, nfft)[:, :T]
    return wet


def apply_noise_gpu(wave: torch.Tensor, lengths: torch.Tensor, bank: AudioBank,
                     snr_range: tuple, rng: random.Random) -> torch.Tensor:
    B, T = wave.shape
    noise = bank.sample_batch(B, T, wave.device, rng)
    snr = torch.empty(B, device=wave.device).uniform_(*snr_range)
    return _mix_at_snr(wave, noise, snr)


def apply_channel_8k_gpu(wave: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """16k -> 8k -> 16k round trip. Plain resample, no codec API (see module
    docstring for why codec APIs are avoided)."""
    down = torchaudio.functional.resample(wave, sr, sr // 2)
    back = torchaudio.functional.resample(down, sr // 2, sr)
    T = wave.shape[-1]
    if back.shape[-1] < T:
        back = F.pad(back, (0, T - back.shape[-1]))
    return back[..., :T]


def _mask_along_axis(spec: torch.Tensor, mask_param: int, axis: int) -> torch.Tensor:
    """Zero out one random band along `axis`, independently per batch item.

    WHY THIS IS HAND-WRITTEN INSTEAD OF torchaudio.transforms.FrequencyMasking
    -------------------------------------------------------------------------
    torchaudio's masking builds its index ramp with

        torch.arange(..., dtype=specgram.dtype)

    i.e. it inherits the dtype of the tensor being masked. We deliberately mask a
    COMPLEX spectrogram (power=None) so that InverseSpectrogram can reconstruct
    the waveform with its original phase, and `arange` has no complex CUDA
    kernel, so that call dies with:

        NotImplementedError: "arange_cuda" not implemented for 'ComplexFloat'

    The masking itself has nothing to do with complex numbers -- it is a 0/1
    band. So we build the ramp in an integer dtype and multiply, which works for
    real and complex spectrograms alike. The alternative fixes are both worse:
    masking the magnitude only throws the phase away, and masking real and
    imaginary parts separately with two torchaudio calls would draw two
    DIFFERENT random bands and corrupt the signal instead of masking it.

    Semantics match SpecAugment: width ~ U[0, mask_param], start ~ U[0, n-width].
    `spec` is [B, F, T] or [F, T]; `axis` must be -2 (frequency) or -1 (time).
    The draw is independent per BATCH ITEM only -- a separate draw per frequency
    bin would scatter noise rather than mask a contiguous band.
    """
    assert axis in (-2, -1), f"axis must be -2 (freq) or -1 (time), got {axis}"
    n = spec.shape[axis]
    dev = spec.device
    batched = spec.dim() == 3
    nb = spec.shape[0] if batched else 1

    width = torch.randint(0, int(mask_param) + 1, (nb,), device=dev).clamp(max=n)
    start = (torch.rand(nb, device=dev) * (n - width + 1).to(torch.float32)).long()
    ramp = torch.arange(n, device=dev)            # integer dtype -- the fix
    keep = (ramp[None, :] < start[:, None]) | (ramp[None, :] >= (start + width)[:, None])

    # [B, n] -> broadcastable against [B, F, T]
    shape = ([nb] if batched else []) + ([n, 1] if axis == -2 else [1, n])
    real_dtype = spec.real.dtype if spec.is_complex() else spec.dtype
    return spec * keep.reshape(shape).to(real_dtype)


def apply_specaugment_gpu(wave: torch.Tensor, cfg: GPUAugConfig, sr: int = 16000) -> torch.Tensor:
    """SpecAugment (Park et al. 2019) applied on a complex STFT
    (torchaudio.transforms.Spectrogram with power=None) and inverted back to
    waveform with InverseSpectrogram. Because encode/decode is a matched
    STFT/ISTFT pair, this round-trip is lossless except in the masked
    bins/frames -- so the backbone still sees a raw waveform (as ablation_engine.py's
    CTC pipeline expects), not a spectrogram.

    Masking uses `_mask_along_axis` rather than torchaudio's FrequencyMasking /
    TimeMasking; see that docstring for why the torchaudio version cannot touch a
    complex spectrogram on CUDA."""
    n_fft, hop = 400, 160  # 25ms / 10ms @ 16kHz, standard ASR STFT config
    spec_fn = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop,
                                                 power=None).to(wave.device)
    ispec_fn = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop
                                                        ).to(wave.device)

    spec = spec_fn(wave)  # [B, F, T] complex
    for _ in range(cfg.n_freq_masks):
        spec = _mask_along_axis(spec, cfg.freq_mask_param, axis=-2)
    for _ in range(cfg.n_time_masks):
        spec = _mask_along_axis(spec, cfg.time_mask_param, axis=-1)
    out = ispec_fn(spec, length=wave.shape[-1])
    return out.real if out.is_complex() else out


# ============================================================================
# 4 . Top-level pipeline
# ============================================================================


class GPUAugmentPipeline:
    """Owns the noise/RIR banks and applies the full chain to a batch.

    Usage inside the training loop (batch already on GPU):
        aug = GPUAugmentPipeline(cfg, noise_dir=..., rir_dir=..., device=dev)
        X, wl = aug(X, wl)   # X: [B, T] float32 waveform, wl: [B] lengths
    """

    def __init__(self, cfg: GPUAugConfig, device: str, seed: int = 1337):
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(seed)
        self.noise_bank = AudioBank(cfg.noise_dir, device=device)
        self.rir_bank = AudioBank(cfg.rir_dir, device=device)

        # An empty bank makes the corresponding effect a no-op further down
        # (`if not bank.empty() and rng.random() < p`). That is the right
        # behaviour, but it must not be SILENT: a run that trains happily while
        # the two demo-critical augmentations do nothing is the worst outcome,
        # because nothing in the log looks wrong. Say it loudly instead.
        # The advice depends on WHY the bank is empty. Telling someone to re-run
        # the download when the files are already on disk and merely failed to
        # decode sends them to fix a step that was never broken -- which is
        # exactly what happened: the banks were fetched and verified, and this
        # warning still said "run fetch_noise_banks.py".
        def _why(bank) -> str:
            if bank.root is None:
                return ("no directory was passed -- the trainer was launched without "
                        "--noise-dir/--rir-dir")
            if bank.n_found == 0:
                return f"no audio files under {bank.root} -- run fetch_noise_banks.py"
            return (f"{bank.n_found} files ARE present under {bank.root} but none could "
                    "be decoded -- see the [BANK] error above; re-downloading will not "
                    "help, the decoder is the problem")

        if self.noise_bank.empty() and cfg.p_noise > 0:
            print(f"[AUG] *** noise bank is EMPTY -- additive-noise augmentation will "
                  f"NOT be applied. Reason: {_why(self.noise_bank)}. Set p_noise=0 to "
                  "make this intentional. ***", flush=True)
        if self.rir_bank.empty() and cfg.p_rir > 0:
            print(f"[AUG] *** RIR bank is EMPTY -- reverb augmentation will NOT be "
                  f"applied. This is the demo-critical one: a laptop mic in a room is "
                  f"reverberant and the model will never have seen that. Reason: "
                  f"{_why(self.rir_bank)}. Set p_rir=0 to make this intentional. ***",
                  flush=True)

    def __call__(self, wave: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if not cfg.any_on() or self.rng.random() < cfg.p_clean:
            return wave, lengths

        # 1. source: speed perturbation (changes length -> do first, before
        #    any effect that assumes a fixed T)
        if self.rng.random() < cfg.p_speed:
            wave, lengths = apply_speed_gpu(wave, lengths, 16000, cfg.speed_rates, self.rng)

        # 2. room: reverb
        if not self.rir_bank.empty() and self.rng.random() < cfg.p_rir:
            wave = apply_rir_gpu(wave, self.rir_bank, self.rng)

        # 3. noise: MUSAN/DEMAND additive noise at random SNR
        if not self.noise_bank.empty() and self.rng.random() < cfg.p_noise:
            wave = apply_noise_gpu(wave, lengths, self.noise_bank, cfg.snr_db, self.rng)

        # 4. channel: telephone-band round trip (plain resample, no codec API)
        if self.rng.random() < cfg.p_channel_8k:
            wave = apply_channel_8k_gpu(wave)

        # 5. SpecAugment -- time/frequency masking, applied last (regularises
        #    the representation the backbone actually consumes)
        if self.rng.random() < cfg.p_specaug:
            wave = apply_specaugment_gpu(wave, cfg)

        peak = wave.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        over = peak > 1.0
        wave = torch.where(over, wave / peak, wave)
        return wave, lengths


# ============================================================================
# 5 . Deterministic eval-time degradations (mirrors ablation_engine.py degrade_eval,
#     kept torch-native so the same eval harness can run on GPU batches)
# ============================================================================


def degrade_eval_gpu(wave: torch.Tensor, mode: str | None, sr: int = 16000) -> torch.Tensor:
    if mode in (None, "clean"):
        return wave
    if mode == "tel8k":
        return apply_channel_8k_gpu(wave, sr)
    if mode == "noisy":
        g = torch.Generator(device="cpu").manual_seed(12345)
        noise = torch.randn(wave.shape, generator=g).to(wave.device)
        snr = torch.full((wave.shape[0],), 10.0, device=wave.device)
        return _mix_at_snr(wave, noise, snr)
    raise ValueError(mode)


def check_banks(noise_dir: str | None, rir_dir: str | None) -> int:
    """Load the banks and say exactly what happened. Returns a shell exit code.

    Exists because `fetch_noise_banks.py --verify-only` counts wav FILES and the
    trainer needs DECODED clips, and those two are not the same thing: a run was
    seen reporting verified banks on disk and "0 clips" in the same session. This
    closes that gap without paying for a training start-up.
    """
    ok = True
    for label, root, why_it_matters in (
        ("noise", noise_dir, "additive-noise augmentation"),
        ("rir", rir_dir, "reverb -- the demo-critical one, a laptop mic in a room"),
    ):
        print(f"\n=== {label} bank ===")
        bank = AudioBank(root, limit=50)   # 50 is enough to prove decodability
        if bank.clips:
            secs = sum(c.numel() for c in bank.clips) / bank.sr
            print(f"  OK: {len(bank.clips)} of {min(50, bank.n_found)} sampled files "
                  f"decoded ({secs:.1f}s of audio). {why_it_matters} will be applied.")
        else:
            ok = False
            print(f"  FAIL: nothing decoded -- {why_it_matters} would be a silent no-op.")
    print("\n" + ("READY: both banks decode." if ok else
                  "NOT READY: fix the bank(s) above, or set p_noise/p_rir to 0 so the "
                  "run states on the record that augmentation is off ON PURPOSE."))
    return 0 if ok else 2


if __name__ == "__main__":
    import argparse
    import sys as _sys

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--check-banks", action="store_true",
                     help="load the noise/RIR banks and report decode results, then exit")
    _ap.add_argument("--noise-dir", default=None)
    _ap.add_argument("--rir-dir", default=None)
    _args = _ap.parse_args()
    if _args.check_banks:
        _sys.exit(check_banks(_args.noise_dir, _args.rir_dir))

    # Minimal self-test -- runs on CPU, no bank files needed. Confirms the
    # pipeline is at least shape-correct end to end.
    torch.manual_seed(0)
    cfg = GPUAugConfig(p_clean=0.0, p_speed=1.0, p_rir=0.0, p_noise=0.0,
                       p_specaug=1.0, p_channel_8k=1.0)
    pipe = GPUAugmentPipeline(cfg, device="cpu")
    X = torch.randn(4, 16000)
    L = torch.tensor([16000, 12000, 8000, 4000])
    Y, YL = pipe(X, L)
    assert Y.shape[0] == X.shape[0]
    assert YL.max().item() <= Y.shape[1]
    print("[selftest] OK", Y.shape, YL.tolist())
