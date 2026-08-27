# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch", "torchaudio", "transformers>=4.44", "peft>=0.11", "jiwer", "numpy",
#     "soundfile==0.14.0",   # augment.AudioBank decodes the noise/RIR banks with it
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
ASR -- 300h retrain, training script.

Reuses the architecture, LoRA config, weighted-sum head and CTC training
loop from ablation_engine.py VERBATIM (frozen mHuBERT-147 + LoRA on q_proj/v_proj
layers 1-12 + weighted-sum over configurable `ws` layers + 2-layer MLP CTC
head, AdamW with three param groups at different LRs, ReduceLROnPlateau on
CER, gradient accumulation, length-bucketed batching). What's NEW here:

  - data comes from the packed cache built by build_cache.py out of the
    300h combined manifest (prepare_data.py), instead of ablation_engine.py's
    LibriSpeech-only parquet cache.
  - augmentation is GPUAugmentPipeline (augment.py) applied to the batched
    GPU waveform tensor, instead of ablation_engine.py's per-sample CPU numpy aug.
  - the model is fully parameterised (--ws, --lora-layers, --lr-scale,
    --hours-subset) so this ONE script serves both the 50h probe (three WS
    arms, high LR) and the full 300h run -- avoiding a second, drifting copy
    of the training loop.
  - per-epoch checkpoints are kept as immutable ep{N:03d}.pt snapshots IN
    ADDITION to the resumable last.pt ablation_engine.py already writes: an ~8h
    unattended cloud run must survive a disconnect, and a single overwritten
    last.pt is one bad write away from losing everything.

Usage:
    python train_asr.py --run FINAL_300h --cache-dir /marimo/cache/combined_XXXX \
        --ws 9,10,11,12 --lora-layers 1-12 --epochs 30 \
        --micro-secs 200 --micro-batch 16 --effective-secs 800 \
        --noise-dir /marimo/noise --rir-dir /marimo/rir --out /marimo/runs

    # 50h probe, control arm:
    python train_asr.py --run probe_control --cache-dir ... --hours-subset 50 \
        --ws 9,10,11,12 --lora-layers 1-12 --epochs 6 --lr-scale 3.0

    # 50h probe, lower-layer arm:
    python train_asr.py --run probe_lowerA --cache-dir ... --hours-subset 50 \
        --ws 5,6,7,8 --lora-layers 1-12 --epochs 6 --lr-scale 3.0
"""

from __future__ import annotations

import argparse
import contextlib
import os
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from itertools import groupby
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import build_vocab  # same vocab as ablation_engine.py / kenlm_grid.py
from augment import GPUAugConfig, GPUAugmentPipeline


# Set BEFORE torch initialises CUDA (torch is imported lazily inside functions
# here, so module scope is early enough). Length-bucketed batching gives the
# allocator a new tensor shape almost every step; without expandable segments the
# pool fragments and the process holds several times its live-tensor peak -- 4.9 GB
# of tensors sat inside a 42 GB pool. That surplus is invisible to PyTorch's own
# `max_memory_allocated`, but it is very visible to nvidia-smi and to anything
# else trying to use the card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(*a):
    print(*a, flush=True)


def _sync_to_drive(paths, run_name: str) -> None:
    """Mirror the given files to Google Drive, once per epoch.

    NEVER raises. A Drive hiccup, an expired token or an unmounted volume must
    not take down an 8-hour unattended training run — losing the mirror is
    recoverable, losing the run is not. Every failure is logged and swallowed.

    `gdrive_sync.sync_checkpoint` prefers a mounted Drive (a plain file copy)
    and falls back to the Google API upload path.
    """
    try:
        # train_asr.py and gdrive_sync.py are written side by side, but the
        # process may be launched from a different cwd, so make the sibling
        # importable explicitly rather than relying on it.
        _here = str(Path(__file__).resolve().parent)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import gdrive_sync
    except Exception as exc:
        log(f"     [drive] gdrive_sync unavailable ({exc}) — checkpoints stay local only")
        return

    ok, failed = [], []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            # sync_checkpoint returns (ok, reason). It used to return None
            # whether it uploaded or did nothing, so this loop reported "mirrored"
            # for a silent no-op — a false confirmation, which is the worst
            # possible outcome for an unattended overnight run.
            done, reason = gdrive_sync.sync_checkpoint(p, run_name)
            (ok if done else failed).append(f"{p.name}" if done else f"{p.name}: {reason}")
        except Exception as exc:
            failed.append(f"{p.name}: {type(exc).__name__}: {exc}")
    if ok:
        log(f"     [drive] mirrored: {', '.join(ok)}")
    if failed:
        log(f"     [drive] NOT mirrored ({len(failed)}): {failed[0]}")
        if len(failed) > 1:
            log(f"     [drive] ...and {len(failed) - 1} more with the same problem")


# ============================================================================
# Config
# ============================================================================


@dataclass
class Cfg:
    run: str = "run"
    ws: tuple = (9, 10, 11, 12)
    lora_layers: tuple = tuple(range(1, 13))
    lora_r: int = 16
    lora_alpha: int = 32
    hid: int = 768
    sr: int = 16000

    # ---- batching: MEMORY and OPTIMISATION are two separate knobs ------------
    # The old `batch` / `batch_secs` pair conflated them. The sampler's budget was
    # `batch * batch_secs * sr`, so raising `batch` to 64 asked for 64*20 = 1280
    # SECONDS of padded audio in a single forward pass (20.5 M samples). The conv
    # frontend then tried to allocate 1.64 GiB in one go and the run died -- while
    # `accum` sat at 4, silently making the optimisation batch 4x bigger too.
    #
    #   micro_secs      -> padded audio seconds per GPU forward. THE ONLY knob
    #                      that determines peak memory.
    #   micro_batch     -> utterance cap per forward. Secondary guard so a bucket
    #                      of 0.3 s clips does not become a 700-item batch whose
    #                      per-item overhead dominates.
    #   effective_secs  -> audio seconds per OPTIMISER STEP. accum is derived from
    #                      it, so halving micro_secs to fit a smaller card leaves
    #                      the optimisation batch -- and therefore the learning
    #                      rate and the ablation comparison -- unchanged.
    micro_secs: float = 200.0
    micro_batch: int = 16
    effective_secs: float = 800.0
    accum: int = 1                 # DERIVED in __post_init__, do not set by hand
    # Recorded as a field so it lands in config.json: the effective batch is the
    # number that must match across ablation arms, and provenance beats memory.
    effective_secs_actual: float = 0.0
    epochs: int = 30
    head_lr: float = 1e-3
    lora_lr: float = 2e-4
    w_lr: float = 1e-3
    lr_scale: float = 1.0          # multiplies all three LRs -- probe uses >1
    weight_decay: float = 0.0
    clip: float = 5.0
    patience: int = 4
    stop_patience: int = 12
    workers: int = 8
    seed: int = 1337
    hours_subset: float | None = None  # None = full cache; 50.0 for the probe
    aug_on: bool = True
    noise_dir: str | None = None
    rir_dir: str | None = None

    # The 100h FINAL baseline was trained with bb_dropout=0.05, and ablation_engine.py's
    # own ablation found it the single strongest regulariser it tested (-0.86 WER):
    # it acts on the REPRESENTATION rather than the parameters, so it behaves like
    # augmentation on a frozen backbone. Defaulting to 0.0 here silently dropped it
    # from the 300h recipe.
    #
    # It only has any effect in train() mode -- torch dropout is a no-op under
    # eval(). So a non-zero value also flips the backbone into train() for the
    # training pass (never for evaluation). `apply_spec_augment` stays False and
    # the mask probabilities stay 0, so train() enables dropout and nothing else.
    bb_dropout: float = 0.0

    # Start from another run's head.pt/adapter.pt with a FRESH optimiser, rather
    # than resuming that run. This is what "fine-tune the finished 300h model with
    # bb_dropout on" needs: same weights, different regularisation, clean schedule.
    init_from: str | None = None

    def __post_init__(self):
        self.ws = tuple(sorted(int(x) for x in self.ws))
        self.lora_layers = tuple(sorted(int(x) for x in self.lora_layers))

        # accum is derived, never configured. If effective < micro the user asked
        # for an optimisation batch smaller than one forward pass, which cannot be
        # honoured by accumulation -- so report the batch we will ACTUALLY use
        # rather than pretending.
        if self.micro_secs <= 0:
            raise ValueError("micro_secs must be > 0")
        # `output_hidden_states=True` on a 12-layer backbone returns THIRTEEN
        # tensors: index 0 is the feature-projection output (before any transformer
        # layer) and 1..12 are the layer outputs, so 12 is the final layer. An out-of
        # -range entry would only surface as an IndexError deep inside the training
        # step, and a 0 would silently mix in the pre-transformer embedding.
        bad = [L for L in self.ws if not 1 <= L <= 12]
        if bad:
            raise ValueError(
                f"--ws {bad} out of range: valid layers are 1..12 (index 12 IS the "
                "final layer; index 0 would be the pre-transformer feature projection, "
                "which is not a hidden layer and is excluded on purpose)")
        self.accum = max(1, round(self.effective_secs / self.micro_secs))
        self.effective_secs_actual = self.accum * self.micro_secs


# ============================================================================
# Data: reads the packed int16 cache built by build_cache.py -- SAME format
# ablation_engine.py's prepare() uses (audio.i16 memmap + offsets + texts).
# ============================================================================


class SpeechDS:
    def __init__(self, cache_dir: Path, vocab: dict, sr: int, subset_hours: float | None = None,
                 seed: int = 1337):
        meta = json.loads((cache_dir / "meta.json").read_text())
        offs = np.asarray(meta["offsets"], dtype=np.int64)
        texts = meta["texts"]
        # build_cache.py has always written this and nothing ever read it. Without
        # it the dev metric is one number over a LibriSpeech+CommonVoice+AMI+VCTK
        # mixture, which cannot say whether a plateau is the model's ceiling or
        # just AMI's -- a question that has now come up three times.
        corpora = meta.get("corpora") or ["?"] * len(texts)
        n = int(offs[-1])
        buf = np.memmap(cache_dir / "audio.i16", dtype=np.int16, mode="r", shape=(n,))

        # `0` means the same thing as `None` here: use the whole cache. Without
        # this guard `subset_hours=0.0` passes the `is not None` test and the
        # budget loop keeps ZERO rows -- an empty dataset, which is the worst
        # possible reading of 'no subset'.
        if subset_hours:
            lens = np.diff(offs)
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(texts))
            budget = subset_hours * 3600.0 * sr
            keep, acc = [], 0.0
            for i in order:
                if acc >= budget:
                    break
                keep.append(int(i))
                acc += lens[i]
            keep = sorted(keep)
            self._idx = keep
        else:
            self._idx = list(range(len(texts)))

        self.buf, self.offs, self.texts, self.vocab, self.sr = buf, offs, texts, vocab, sr
        self.corpora = corpora
        self._idx = self._drop_infeasible(self._idx, offs, texts)

    @staticmethod
    def _feat_len(n_samples: int) -> int:
        """mHuBERT conv frontend output length. Mirrors
        `_get_feat_extract_output_lengths`, but computable without a model."""
        L = n_samples
        for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
            L = (L - k) // s + 1
        return L

    def _drop_infeasible(self, idx, offs, texts) -> list:
        """Remove rows the CTC loss cannot accept, and say how many and why.

        A run died with `Expected input_lengths to have value at least 0, but got
        value -1`. The conv stack maps 0 samples to exactly -1, so ONE zero-length
        row in a 186,789-row cache is enough. The 50 h probe never hit it because
        its stratified subset happened not to draw one -- which is precisely how a
        data defect survives a smaller pilot run.

        `prepare_data.combine()` already gates the MANIFEST on
        `0.2 <= duration_s <= 30` and `duration_s * 50 >= 2 * len(text)`. That gate
        uses the duration recorded in the manifest. This gate uses the number of
        samples ACTUALLY IN THE CACHE, which is the only quantity the trainer
        consumes. When a decoder returns fewer samples than the metadata promised,
        those two disagree and only this check notices.
        """
        lens = np.diff(offs)
        empty, short, infeasible = [], [], []
        for i in idx:
            n = int(lens[i])
            f = self._feat_len(n)
            t = len(texts[i].replace(" ", "|"))
            if n <= 0:
                empty.append(i)
            elif f < 1:
                short.append(i)
            elif f < t:
                # CTC cannot align more labels than it has frames. Keeping these
                # gives inf loss, which poisons the running average and every
                # gradient in the accumulation window it lands in.
                infeasible.append(i)
        bad = set(empty) | set(short) | set(infeasible)
        if bad:
            log(f"[DATA] dropped {len(bad)} of {len(idx)} rows the CTC loss cannot take: "
                f"{len(empty)} zero-length, {len(short)} shorter than one frame, "
                f"{len(infeasible)} with more characters than frames")
            for label, group in (("zero-length", empty), ("sub-frame", short),
                                 ("infeasible", infeasible)):
                if group:
                    i = group[0]
                    log(f"       e.g. {label} idx={i} samples={int(lens[i])} "
                        f"frames={self._feat_len(int(lens[i]))} "
                        f"chars={len(texts[i])} text={texts[i][:40]!r}")
            if len(bad) > 0.01 * len(idx):
                raise RuntimeError(
                    f"{len(bad)} rows ({len(bad) / len(idx):.2%}) are unusable. Over 1% "
                    "means the cache is broken, not merely imperfect -- rebuild it "
                    "rather than training on whatever survived.")
        return [i for i in idx if i not in bad]

    def __len__(self):
        return len(self._idx)

    def _raw(self, j):
        i = self._idx[j]
        a, b = int(self.offs[i]), int(self.offs[i + 1])
        return np.asarray(self.buf[a:b], np.float32) / 32768.0

    def text(self, j):
        return self.texts[self._idx[j]]

    def corpus(self, j):
        return self.corpora[self._idx[j]]

    def __getitem__(self, j):
        import torch

        w = self._raw(j)
        ids = [self.vocab.get(c, self.vocab["[UNK]"]) for c in self.text(j).replace(" ", "|")]
        return torch.from_numpy(np.ascontiguousarray(w)), torch.tensor(ids, dtype=torch.long), j


class LengthBucket:
    """Same frame-budget bucketing as ablation_engine.py's LengthBucket."""

    def __init__(self, lengths, batch, budget, shuffle=True, seed=0, pool_mult=50):
        self.L = np.asarray(lengths, np.int64)
        self.b, self.budget = batch, int(budget)
        self.shuffle, self.seed = shuffle, seed
        self.pool, self.epoch = batch * pool_mult, 0
        self._cache = self._build(0)

    def _build(self, epoch):
        g = np.random.default_rng(self.seed + epoch)
        idx = g.permutation(len(self.L)) if self.shuffle else np.arange(len(self.L))
        out, cur = [], []
        for i in range(0, len(idx), self.pool):
            ch = idx[i:i + self.pool]
            ch = ch[np.argsort(self.L[ch], kind="stable")]
            for j in ch:
                Lj = int(self.L[j])
                if cur and (len(cur) + 1 > self.b or Lj * (len(cur) + 1) > self.budget):
                    out.append(cur)
                    cur = [int(j)]
                else:
                    cur.append(int(j))
            if cur:
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
        return len(self._cache) if self._cache is not None else getattr(self, "_n", 1)


def collate(batch, pad):
    import torch

    ws, ls, ix = zip(*batch)
    wl = torch.tensor([len(w) for w in ws])
    ll = torch.tensor([len(l) for l in ls])
    X = torch.zeros(len(ws), int(wl.max()))
    Y = torch.zeros(len(ls), int(ll.max()), dtype=torch.long)
    for i, (w, l) in enumerate(zip(ws, ls)):
        X[i, : len(w)] = w
        Y[i, : len(l)] = l
    return X, Y, wl, ll, torch.tensor(ix)


def make_loader(ds, cfg, shuffle):
    import torch
    from torch.utils.data import DataLoader

    lens = np.diff(ds.offs)[ds._idx]
    # Budget is micro_secs ALONE -- it must not be multiplied by the utterance cap.
    # `micro_batch * micro_secs * sr` was the old expression and it made the two
    # knobs multiply each other, so nudging the utterance cap from 16 to 64
    # quadrupled peak memory with no indication that it would.
    sampler = LengthBucket(lens, cfg.micro_batch, cfg.micro_secs * cfg.sr,
                            shuffle=shuffle, seed=cfg.seed)
    return DataLoader(ds, batch_sampler=sampler,
                       collate_fn=lambda b: collate(b, None),
                       num_workers=cfg.workers, pin_memory=True,
                       persistent_workers=cfg.workers > 0)


# ============================================================================
# Model -- verbatim from ablation_engine.py (backbone + LoRA + weighted-sum head)
# ============================================================================


def build_backbone(cfg: Cfg, device: str):
    import torch
    import torch.nn as nn
    from transformers import HubertModel
    from peft import LoraConfig, inject_adapter_in_model

    BACKBONE = "utter-project/mHuBERT-147"
    # mask_* and apply_spec_augment stay OFF regardless: waveform-level SpecAugment
    # is done by augment.py, and HF's masking would stack a second, unmeasured one
    # on top the moment the backbone goes into train() mode.
    _bd = cfg.bb_dropout
    kw = dict(mask_time_prob=0.0, mask_feature_prob=0.0, apply_spec_augment=False,
              hidden_dropout=_bd, attention_dropout=_bd, activation_dropout=_bd,
              feat_proj_dropout=_bd, final_dropout=_bd, layerdrop=0.0)
    try:
        bb = HubertModel.from_pretrained(BACKBONE, attn_implementation="sdpa", **kw)
        log("[BB] attention: sdpa")
    except Exception as e:
        bb = HubertModel.from_pretrained(BACKBONE, **kw)
        log(f"[BB] attention: eager (sdpa unavailable: {type(e).__name__})")
    bb = bb.to(device)

    lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                          target_modules=["q_proj", "v_proj"], bias="none",
                          layers_to_transform=[i - 1 for i in cfg.lora_layers])
    bb = inject_adapter_in_model(lora_cfg, bb)
    for n, p in bb.named_parameters():
        p.requires_grad = "lora_" in n
    got = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    exp = 2 * cfg.hid * cfg.lora_r * 2 * len(cfg.lora_layers)
    log(f"[LORA] {got:,} trainable params (expected {exp:,})")
    assert got == exp, f"LoRA out of scope: {got:,} != {exp:,}"
    bb.eval()  # backbone always in eval() -- SpecAugment/dropout are handled
               # by augment.py's GPUAugmentPipeline on the waveform instead
    return bb, bb._get_feat_extract_output_lengths


def make_head(cfg: Cfg, vocab_size: int, device: str):
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self, n, dim, V):
            super().__init__()
            self.n = n
            self.layer_w = nn.Parameter(torch.zeros(n))
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Dropout(0.0),
                                     nn.Linear(dim, V))

        def weights(self):
            return self.layer_w.softmax(0)

        def forward(self, x):
            w = self.layer_w.softmax(0)
            f = (x * w[None, None, :, None]).sum(2)
            return self.net(f)

    return Head(len(cfg.ws), cfg.hid, vocab_size).to(device)


def decode_greedy(ids, i2c, blank, unk):
    return "".join(i2c.get(k, "") for k, _ in groupby(ids) if k not in (blank, unk)
                   ).replace("|", " ").strip()


# ============================================================================
# Train / eval loop
# ============================================================================


def evaluate(head, bb, dl, ds, flen, dev, i2c, blank, unk):
    import torch
    import jiwer

    head.eval()
    bb.eval()
    H, R, C = [], [], []
    with torch.no_grad():
        for X, _, wl, _, ix in dl:
            X = X.to(dev, non_blocking=True)
            am = (torch.arange(X.shape[1], device=dev)[None, :] < wl.to(dev)[:, None]).long()
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                o = bb(X, attention_mask=am, output_hidden_states=True)
                cfg_ws = getattr(evaluate, "_ws", None)
            xl = flen(wl.to(dev))
            pr = head(torch.stack([o.hidden_states[L] for L in evaluate._ws], 2).float())
            pr = pr.argmax(-1).cpu().numpy()
            for b, j in enumerate(ix.tolist()):
                H.append(decode_greedy(pr[b, : int(xl[b])].tolist(), i2c, blank, unk))
                R.append(ds.text(j))
                C.append(ds.corpus(j))

    by_corpus = {}
    for c in sorted(set(C)):
        _r = [r for r, cc in zip(R, C) if cc == c]
        _h = [h for h, cc in zip(H, C) if cc == c]
        if _r:
            by_corpus[c] = {"n": len(_r), "wer": jiwer.wer(_r, _h), "cer": jiwer.cer(_r, _h)}
    return jiwer.wer(R, H), jiwer.cer(R, H), by_corpus


def _fmt_ws(ws: tuple, w: list) -> str:
    """One-line view of the weighted-sum distribution over hidden layers.

    These numbers were already written to history.jsonl every epoch, but never
    printed, so the one signal that says WHICH layer the model is leaning on was
    invisible during an 8-hour run.

    The normalised entropy matters as much as the argmax. `layer_w` is
    zero-initialised, so the softmax starts perfectly uniform (H/Hmax = 1.00). A
    run that ends near 1.00 has not selected anything -- reading its argmax as
    "the model prefers layer 10" would be reading noise. Only once H/Hmax drops
    meaningfully below 1 is the distribution actually informative.

    IMPORTANT -- what this CANNOT tell you: the softmax only ranks the layers in
    `ws`. If ws=(9,10,11,12) it can never reveal that layer 6 would have been
    better, because layer 6 was never on the menu. That is precisely why the
    three-arm ablation is still needed; this is a cheap hint, not a substitute.
    """
    import math

    w = [float(x) for x in w]
    top = max(range(len(w)), key=lambda i: w[i])
    cells = " ".join(f"L{L}{'*' if i == top else ' '}{w[i]:.3f}"
                     for i, L in enumerate(ws))
    h = -sum(x * math.log(x) for x in w if x > 0)
    hmax = math.log(len(w)) if len(w) > 1 else 1.0
    ratio = h / hmax if hmax else 1.0
    order = sorted(w, reverse=True)
    verdict = ("UNIFORM, argmax is not meaningful yet" if ratio > 0.99
               else "barely selective" if ratio > 0.97 else "selective")
    return (f"WS {cells} | top2 mass {sum(order[:2]):.2f} | "
            f"H/Hmax {ratio:.3f} ({verdict})")


def _report_occupancy(loader, ds, cfg) -> None:
    """Measure how full the micro-batches actually are, and name the binding cap.

    `effective_secs` is a CEILING. The sampler stops a batch when EITHER the
    utterance cap or the seconds budget is hit, so if the utterance cap binds
    first the seconds budget is never reached and the true optimisation batch is
    smaller than the configured one. Reporting the configured number as though it
    were achieved would be a quietly wrong provenance record -- and it also hides
    the reason the GPU is half idle.
    """
    lens = np.diff(ds.offs)
    # PEEK, do not consume. `list(sampler)` calls __iter__, which drops the
    # pre-built epoch-0 batching and increments the epoch counter -- so this
    # diagnostic was silently changing which permutation the first training epoch
    # got. A measurement that alters what it measures is the bug class this file
    # keeps closing; read the cache instead.
    _s = loader.batch_sampler
    batches = _s._cache if getattr(_s, "_cache", None) is not None else _s._build(_s.epoch)
    if not batches:
        return
    utts = np.array([len(b) for b in batches], float)
    # The sampler budgets PADDED samples (max_len * count), which is also what
    # determines memory, so that is the number to compare against micro_secs.
    padded = np.array([max(int(lens[ds._idx[j]]) for j in b) * len(b) for b in batches],
                      float) / cfg.sr
    real = np.array([sum(int(lens[ds._idx[j]]) for j in b) for b in batches],
                    float) / cfg.sr

    utt_bound = float((utts >= cfg.micro_batch).mean())
    fill = float(padded.mean() / cfg.micro_secs)
    binding = ("utterance cap" if utt_bound > 0.5 else "seconds budget")
    log(f"[BATCH] occupancy over {len(batches):,} batches: "
        f"{utts.mean():.1f} utts/forward (cap {cfg.micro_batch}), "
        f"{padded.mean():.0f}s padded / {real.mean():.0f}s real "
        f"(budget {cfg.micro_secs:.0f}s, {fill:.0%} used)")
    log(f"[BATCH] binding constraint: {binding} "
        f"({utt_bound:.0%} of batches hit the utterance cap)")
    log(f"[BATCH] ACHIEVED effective batch ~{cfg.accum * padded.mean():.0f}s "
        f"per optimiser step (configured ceiling {cfg.effective_secs_actual:.0f}s)")
    if fill < 0.7:
        log(f"[BATCH] !! only {fill:.0%} of the memory budget is used. Raising "
            f"--micro-secs will NOT help while the {binding} binds; raise "
            f"--micro-batch instead.")


MIN_FREE_GIB = 8.0


def _gpu_preflight(cfg: Cfg) -> None:
    """Fail fast if the card is already occupied by someone else.

    A run died with `OutOfMemoryError: Tried to allocate 1.64 GiB. GPU 0 has a
    total capacity of 94.97 GiB of which 186.38 MiB is free. Process 1 has
    94.77 GiB memory in use. Of the allocated memory 1.24 GiB is allocated by
    PyTorch`. Read those numbers together: OUR process held 1.24 GiB, and
    something else held 94.77 GiB of a 95 GiB card. Shrinking the batch could not
    have helped -- there was nothing to shrink into.

    The usual cause is a previous training subprocess that never exited. The
    notebook launches the trainer with subprocess.Popen and streams its stdout;
    interrupting the marimo cell stops the STREAMING, not the child, so the child
    keeps the model resident on the GPU. Checking here turns a confusing OOM
    thirty seconds into the run into an actionable message before any work starts.
    """
    import torch

    if not torch.cuda.is_available():
        log("[GPU] no CUDA device -- running on CPU, this will be far too slow for 300h")
        return

    # Release anything THIS process is still caching before measuring. In a fresh
    # subprocess that is nearly nothing, so this is not the fix for a card held by
    # someone else -- no process can free another process's memory. It matters on
    # a resume, where the model has already been built and torn down once.
    before = torch.cuda.mem_get_info()[0]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    freed = torch.cuda.mem_get_info()[0] - before
    if freed > 64 * 2**20:
        log(f"[GPU] released {freed / 2**30:.2f} GiB of our own cached blocks before start")

    free_b, total_b = torch.cuda.mem_get_info()
    free, total = free_b / 2**30, total_b / 2**30
    used_by_us = torch.cuda.memory_allocated() / 2**30
    foreign = total - free - used_by_us
    log(f"[GPU] {torch.cuda.get_device_name(0)} | {free:.1f} GiB free / {total:.1f} GiB total "
        f"| ours {used_by_us:.2f} GiB | other processes ~{foreign:.1f} GiB")

    if free >= MIN_FREE_GIB:
        return

    procs = ""
    try:
        import subprocess as _sp
        procs = _sp.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                         "--format=csv,noheader"], capture_output=True, text=True,
                        timeout=20).stdout.strip()
    except Exception:
        pass

    raise RuntimeError(
        f"Only {free:.2f} GiB free on a {total:.1f} GiB GPU; ~{foreign:.1f} GiB is held by "
        f"another process. Lowering micro_secs will NOT fix this.\n"
        + (f"Processes on the device:\n{procs}\n" if procs else
           "nvidia-smi gave no process list (containers often hide other tenants' PIDs).\n")
        + "Most likely a previous trainer is still alive: interrupting the notebook cell "
          "stops the log stream, not the child process. Kill it "
          "(`pkill -f train_asr.py`) and re-run. If the memory belongs to another tenant "
          "on a shared GPU, wait rather than shrinking the batch -- a smaller batch that "
          f"fits in {free:.2f} GiB would not train the same model."
    )


def train_one(cfg: Cfg, out_root: Path, cache_dir: Path):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = out_root / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log(f"[CFG] {cfg.run} | ws={list(cfg.ws)} | lora={list(cfg.lora_layers)} | "
        f"epochs={cfg.epochs} | lr_scale={cfg.lr_scale} | subset={cfg.hours_subset}h | "
        f"aug={'on' if cfg.aug_on else 'off'} | bb_dropout={cfg.bb_dropout}"
        + (f" | init_from={cfg.init_from}" if cfg.init_from else ""))
    if cfg.bb_dropout > 0:
        log(f"[BB] backbone in train() mode for dropout={cfg.bb_dropout} "
            "(still frozen; eval passes always use eval())")
    log(f"[BATCH] micro={cfg.micro_secs:.0f}s audio (<={cfg.micro_batch} utts) per forward "
        f"x accum {cfg.accum} -> effective {cfg.effective_secs_actual:.0f}s per optimiser step "
        f"(CEILING -- see [BATCH] occupancy below for what is actually reached)")
    if abs(cfg.effective_secs_actual - cfg.effective_secs) > 1e-6:
        log(f"[BATCH] note: requested {cfg.effective_secs:.0f}s is not an integer multiple "
            f"of micro_secs, so the ACTUAL effective batch is {cfg.effective_secs_actual:.0f}s. "
            "Keep this identical across ablation arms or the comparison is confounded.")
    _gpu_preflight(cfg)

    vocab = build_vocab()
    blank, unk = vocab["[PAD]"], vocab["[UNK]"]
    i2c = {v: k for k, v in vocab.items()}

    tr = SpeechDS(cache_dir, vocab, cfg.sr, subset_hours=cfg.hours_subset, seed=cfg.seed)
    # ------------------------------------------------------------------
    # dev split: 5% from EACH corpus, not the last 5% of the cache.
    #
    # The old comment claimed "a pre-shuffled manifest". It is not shuffled:
    # prepare_data.combine() concatenates librispeech, then common_voice, then
    # ami, then vctk, and build_cache.py writes rows in that order. So the last
    # 5% of the cache is the tail of VCTK and NOTHING else -- a run reported
    # `PER-CORPUS vctk: n=9830` for the entire dev set, which is what exposed it.
    #
    # Consequences of the old behaviour, all of which were invisible:
    #   * every VAL wer/cer for a full-cache run measured clean read speech from
    #     a handful of VCTK speakers, not the 300 h mixture it appeared to;
    #   * [BEST] selected checkpoints on that one corpus;
    #   * the 50 h probe was NOT affected -- its `--hours-subset` path shuffles --
    #     so probe and full-run numbers were never on the same test set, and any
    #     comparison between them was meaningless.
    #
    # Stratifying also guarantees AMI is represented, which is the corpus that
    # actually decides whether a plateau is the model's ceiling or the data's.
    # ------------------------------------------------------------------
    _rng_dev = np.random.default_rng(cfg.seed)
    _by_corpus: dict[str, list[int]] = {}
    for _j, _i in enumerate(tr._idx):
        _by_corpus.setdefault(tr.corpora[_i], []).append(_j)

    _dev_pos = []
    for _c, _pos in sorted(_by_corpus.items()):
        _pos = np.asarray(_pos)
        _rng_dev.shuffle(_pos)
        _k = max(1, int(round(0.05 * len(_pos))))
        _dev_pos.extend(_pos[:_k].tolist())
    _dev_pos = set(_dev_pos)

    dv_idx = [tr._idx[j] for j in sorted(_dev_pos)]
    tr._idx = [tr._idx[j] for j in range(len(tr._idx)) if j not in _dev_pos]
    n_dev = len(dv_idx)

    from collections import Counter as _Ctr
    _dev_mix = _Ctr(tr.corpora[i] for i in dv_idx)
    _tr_mix = _Ctr(tr.corpora[i] for i in tr._idx)
    log("[DEV] stratified 5% per corpus: "
        + "  ".join(f"{c}={n}" for c, n in sorted(_dev_mix.items())))
    _absent = sorted(set(_tr_mix) - set(_dev_mix))
    if _absent:
        raise RuntimeError(f"dev split has no rows from {_absent} -- a corpus present "
                           "in training must be present in dev, or VAL measures "
                           "something other than what is being trained")
    dv = SpeechDS.__new__(SpeechDS)
    # Hand-rolled shallow copy, so every attribute the class gained has to be
    # copied here too. `corpora` was added for the per-corpus dev breakdown and
    # would have raised AttributeError on the first evaluation -- exactly the kind
    # of bug a __new__-based copy invites.
    dv.buf, dv.offs, dv.texts, dv.vocab, dv.sr = tr.buf, tr.offs, tr.texts, tr.vocab, tr.sr
    dv.corpora = tr.corpora
    dv._idx = dv_idx
    _missing = [a for a in vars(tr) if a not in vars(dv)]
    assert not _missing, f"dev split copy is missing SpeechDS attributes: {_missing}"

    tdl = make_loader(tr, cfg, True)
    _report_occupancy(tdl, tr, cfg)
    ddl = make_loader(dv, cfg, False)
    log(f"[DATA] train={len(tr)} dev={len(dv)} utterances")

    bb, flen = build_backbone(cfg, dev)
    head = make_head(cfg, len(vocab), dev)
    evaluate._ws = cfg.ws

    aug_cfg = GPUAugConfig(noise_dir=cfg.noise_dir, rir_dir=cfg.rir_dir)
    if not cfg.aug_on:
        aug_cfg.p_clean = 1.0  # forces every batch through unmodified
    augmenter = GPUAugmentPipeline(aug_cfg, device=dev, seed=cfg.seed)

    groups = [
        {"params": [p for n, p in head.named_parameters() if n != "layer_w"],
         "lr": cfg.head_lr * cfg.lr_scale, "weight_decay": cfg.weight_decay},
        {"params": [head.layer_w], "lr": cfg.w_lr * cfg.lr_scale, "weight_decay": 0.0},
        {"params": [p for p in bb.parameters() if p.requires_grad],
         "lr": cfg.lora_lr * cfg.lr_scale, "weight_decay": cfg.weight_decay},
    ]
    opt = torch.optim.AdamW(groups, fused=(dev == "cuda"))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                      patience=cfg.patience, threshold=0.005)
    ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
    trainable = [p for g in opt.param_groups for p in g["params"]]

    hist_path, last_path = run_dir / "history.jsonl", run_dir / "last.pt"
    ep0, best, best_ep, hist = 1, float("inf"), 0, []
    if cfg.init_from and not last_path.exists():
        _src = Path(cfg.init_from)
        _h, _a = _src / "head.pt", _src / "adapter.pt"
        if not (_h.is_file() and _a.is_file()):
            raise SystemExit(f"--init-from {_src}: needs both head.pt and adapter.pt")
        head.load_state_dict(torch.load(_h, map_location=dev))
        bb.load_state_dict(torch.load(_a, map_location=dev), strict=False)
        log(f"[INIT] weights from {_src} (fresh optimiser and schedule -- this is a "
            f"fine-tune, NOT a resume: epoch counter starts at 1 and `best` is "
            f"unset, so the first epoch of this run always writes a checkpoint)")
        # Print what the source run actually was. A fine-tune is only interpretable
        # against its starting point, and "which checkpoint is this and how was it
        # batched" is exactly what gets misremembered an hour later.
        _ssum = _src / "summary.json"
        _scfg = _src / "config.json"
        if _ssum.is_file():
            try:
                _sd = json.loads(_ssum.read_text())
                log(f"[INIT] source run: best CER {100 * _sd.get('best_cer', float('nan')):.2f}% "
                    f"@ epoch {_sd.get('best_epoch')} of {_sd.get('epochs_done')}")
            except Exception:
                pass
        if _scfg.is_file():
            try:
                _sc = json.loads(_scfg.read_text())
                _sm, _sb = _sc.get("micro_secs"), _sc.get("micro_batch")
                _se = _sc.get("effective_secs_actual")
                log(f"[INIT] source batching: micro={_sm}s cap={_sb} effective={_se}s")
                if (_sm, _sb) != (cfg.micro_secs, cfg.micro_batch):
                    log(f"[INIT] !! this run uses micro={cfg.micro_secs}s "
                        f"cap={cfg.micro_batch}. Changing the batching mid-fine-tune "
                        "changes the gradient-noise regime, so any difference in the "
                        "result can no longer be attributed to what you meant to "
                        "change (here: bb_dropout).")
            except Exception:
                pass

        log("[INIT] !! VAL is a training MONITOR for this run, not a result. The dev "
            "rows are drawn fresh, so most of them were in the SOURCE run's training "
            "set -- the model has already seen them and VAL will read optimistic. "
            "Checkpoint selection inherits that bias. The reportable numbers are "
            "eval_asr.py's dev-clean and L2-ARCTIC, which no run has ever trained on.")
    elif cfg.init_from:
        log(f"[INIT] ignoring --init-from: {last_path} exists, so this is a RESUME "
            "of an interrupted run. Delete the run directory to start a fine-tune.")

    if last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        head.load_state_dict(ck["head"])
        bb.load_state_dict(ck["adapter"], strict=False)
        opt.load_state_dict(ck["opt"])
        with contextlib.suppress(Exception):
            sch.load_state_dict(ck["sch"])
        ep0, best, best_ep = ck["epoch"] + 1, ck["best"], ck["best_ep"]
        hist = ([json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
                if hist_path.exists() else [])
        log(f"[RESUME] from epoch {ck['epoch']}, best CER {best * 100:.2f}%")
    if not hist_path.exists():
        hist_path.write_text("")

    for ep in range(ep0, cfg.epochs + 1):
        head.train()
        # Frozen either way -- no backbone parameter has requires_grad. The mode
        # only decides whether its dropout layers fire, and dropout is a no-op
        # under eval(), so bb_dropout>0 would be silently ignored without this.
        bb.train() if cfg.bb_dropout > 0 else bb.eval()
        t0, tot, nb = time.perf_counter(), 0.0, 0
        n_skipped, _reported_skip, n_oom = 0, False, 0
        # Progress inside an epoch. With ~5,700 batches the old code printed
        # NOTHING between the epoch header and the epoch summary, so an overnight
        # run was indistinguishable from a hung one for ten minutes at a time, and
        # there was no way to tell whether a throughput change had helped.
        # `audio_s` accumulates ON THE GPU and is only `.item()`d when printing, so
        # the heartbeat adds no per-batch synchronisation.
        # Reset the peak counters EVERY epoch. They were reset once at startup, so
        # `max_memory_allocated` was a high-water mark since process start: it can
        # only ever go up, which makes a perfectly healthy run look like a leak in
        # the logs. Per-epoch peaks are what distinguish the two -- a leak makes the
        # ALLOCATED peak climb epoch over epoch; fragmentation leaves it flat while
        # only the reserved pool grows.
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        n_batches = len(tdl.batch_sampler)
        every = max(1, n_batches // 20)          # ~20 lines per epoch
        audio_s = torch.zeros((), device=dev, dtype=torch.float64)
        # How much of the epoch does augmentation actually cost? Throughput alone
        # cannot answer it -- batching and bank residency changed at the same time,
        # so the numbers only bound the cost, they do not measure it. CUDA events
        # are recorded on the stream and only READ once per epoch, so this does not
        # add a host synchronisation per batch the way time.perf_counter() would.
        aug_events, aug_ms = [], 0.0
        _ev_a = torch.cuda.Event(enable_timing=True) if dev == "cuda" else None
        _ev_b = torch.cuda.Event(enable_timing=True) if dev == "cuda" else None
        opt.zero_grad(set_to_none=True)
        for X, Y, wl, ll, _ in tdl:
          # An unattended overnight run must not die on one transient allocation
          # failure at hour six. The padded-seconds budget bounds the worst-case
          # batch, so an OOM here means allocator fragmentation or a co-tenant
          # taking the card -- both transient. Dropping the batch costs one
          # gradient step out of thousands; dying costs the night.
          try:
            X, Y = X.to(dev, non_blocking=True), Y.to(dev, non_blocking=True)
            wl = wl.to(dev)
            if _ev_a is not None and nb % 20 == 0:
                # Sample every 20th batch. Timing every batch would need two events
                # per step and a growing list; a 5% sample is plenty for a ratio.
                _a = torch.cuda.Event(enable_timing=True)
                _b = torch.cuda.Event(enable_timing=True)
                _a.record()
                X, wl = augmenter(X, wl)
                _b.record()
                aug_events.append((_a, _b))
            else:
                X, wl = augmenter(X, wl)
            audio_s += wl.sum()
            am = (torch.arange(X.shape[1], device=dev)[None, :] < wl[:, None]).long()
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                o = bb(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([o.hidden_states[L] for L in cfg.ws], 2)
            xl = flen(wl)
            _ll = ll.to(dev)
            # Second line of defence. SpeechDS filters the cache up front, but
            # augmentation also rewrites lengths (speed perturbation, the 8 kHz
            # round trip), so a batch can in principle still arrive infeasible.
            # Crashing here would kill an 8-hour unattended run over one bad batch;
            # skipping SILENTLY is the failure mode this project keeps closing. So:
            # skip, count, and report the count at the end of the epoch.
            _ok = (xl >= 1) & (xl >= _ll)
            if not bool(_ok.all()):
                n_skipped += int((~_ok).sum())
                if not _reported_skip:
                    _reported_skip = True
                    _b = int((~_ok).nonzero()[0, 0])
                    log(f"     [SKIP] batch has an infeasible row: frames={int(xl[_b])} "
                        f"labels={int(_ll[_b])} samples={int(wl[_b])} -- skipping it. "
                        "Further occurrences are counted, not printed.")
                if not bool(_ok.any()):
                    continue
                keep_ix = _ok.nonzero().flatten()
                hs, Y, xl, _ll = hs[keep_ix], Y[keep_ix], xl[keep_ix], _ll[keep_ix]
            lg = head(hs.float())
            loss = ctc(lg.log_softmax(-1).transpose(0, 1), Y, xl, _ll)
            (loss / cfg.accum).backward()
            nb += 1
            if nb % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item()

            if nb % every == 0 and nb:
                _el = time.perf_counter() - t0
                _done = nb / n_batches
                _eta = _el / max(_done, 1e-9) - _el
                _rt = (audio_s.item() / cfg.sr) / max(_el, 1e-9)
                log(f"       e{ep:>3} {nb:>5}/{n_batches} ({_done:>3.0%}) | "
                    f"loss {tot / nb:.3f} | {_el / 60:.1f}m elapsed, ~{_eta / 60:.1f}m left "
                    f"| {_rt:.0f}x realtime")
          except torch.OutOfMemoryError:
            n_oom += 1
            # Discard the whole accumulation window rather than stepping on
            # gradients from a partial one. A slightly smaller effective batch for
            # one step is harmless; an optimiser step built from an unknown
            # fraction of the intended batch is not.
            opt.zero_grad(set_to_none=True)
            nb -= nb % cfg.accum
            # Drop every reference the failed step may still hold. `del <name>` is
            # unavoidable here rather than a loop over a list of names: deleting a
            # loop variable would delete the STRING, not the tensor it names, which
            # is a no-op that looks like cleanup.
            X = Y = wl = am = o = hs = lg = loss = xl = _ll = None
            gc.collect()
            torch.cuda.empty_cache()
            if n_oom <= 3 or n_oom % 50 == 0:
                _free, _tot = torch.cuda.mem_get_info()
                log(f"     [OOM] batch {nb} dropped and accumulation window reset "
                    f"({n_oom} so far). {_free / 2**30:.1f} GiB free of "
                    f"{_tot / 2**30:.1f} GiB after emptying the cache.")
            if n_oom == 25:
                log("     [OOM] 25 OOMs in one epoch -- this is no longer transient. "
                    "Lower --micro-secs (memory scales with it) or free the card; "
                    "training continues but is losing real batches.")
        if nb % cfg.accum:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        wer, cer, by_corpus = evaluate(head, bb, ddl, dv, flen, dev, i2c, blank, unk)
        # The scheduler was stepping and nobody could see it. "Has the LR dropped
        # yet?" is the first question a plateau raises, and it was unanswerable
        # from the log.
        _lrs = [g["lr"] for g in opt.param_groups]
        rec = {"epoch": ep, "loss": tot / max(1, nb), "wer": wer, "cer": cer,
               "lr_head": _lrs[0], "lr_w": _lrs[1], "lr_lora": _lrs[2],
               "by_corpus": by_corpus,
               "rows_skipped": n_skipped, "oom_batches": n_oom,
               "secs": time.perf_counter() - t0,
               "w": head.weights().detach().cpu().numpy().round(4).tolist()}
        if dev == "cuda":
            # THREE different numbers, and only the last one resembles nvidia-smi:
            #   max_memory_allocated -> peak bytes in LIVE tensors. This is what the
            #     old single `vram_gb` field reported, and it is why the log said
            #     4.9 GB while nvidia-smi said 42 GB. Both were correct.
            #   max_memory_reserved  -> peak size of the caching allocator's POOL.
            #     Freed blocks are kept, not returned to the driver, so with
            #     length-bucketed batches (a new tensor shape almost every step) the
            #     pool fragments and grows far past the live-tensor peak.
            #   + the CUDA context, cuDNN/cuBLAS workspaces and kernels, a few
            #     hundred MB that PyTorch never counts at all.
            rec["vram_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rec["vram_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            # Live at the END of the epoch, after eval. If THIS climbs epoch over
            # epoch something is genuinely being retained; the two peaks above
            # cannot tell you that on their own.
            rec["vram_live_end_gb"] = torch.cuda.memory_allocated() / 1e9
            # Kept under the old key so existing history.jsonl files stay readable.
            rec["vram_gb"] = rec["vram_alloc_gb"]
        hist.append(rec)
        with hist_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        rec["realtime_factor"] = (audio_s.item() / cfg.sr) / max(rec["secs"], 1e-9)
        if aug_events:
            torch.cuda.synchronize()
            _per = [a.elapsed_time(b) for a, b in aug_events]
            # Scale the 1-in-20 sample up to the whole epoch.
            rec["aug_ms_per_batch"] = float(np.mean(_per))
            rec["aug_frac_of_epoch"] = (np.mean(_per) / 1000.0 * n_batches) / max(rec["secs"], 1e-9)
        log(f"  e{ep:>3} | loss {rec['loss']:.3f} | {rec['secs']:.0f}s | "
            f"{rec['realtime_factor']:.0f}x realtime | "
            f"VAL wer {wer * 100:.2f} cer {cer * 100:.2f}")
        log("       " + _fmt_ws(cfg.ws, rec["w"]))
        if by_corpus:
            log("       PER-CORPUS " + "  ".join(
                f"{c}: cer {100 * v['cer']:.2f} wer {100 * v['wer']:.2f} (n={v['n']})"
                for c, v in by_corpus.items()))
        log(f"       LR head {_lrs[0]:.2e} · w {_lrs[1]:.2e} · lora {_lrs[2]:.2e}")
        _prev_lr = [h for h in hist[:-1] if "lr_head" in h]
        # `sch.step(cer)` runs at the END of an epoch, so a reduction it triggers
        # first shows up in the NEXT epoch's learning rate. The message says that
        # rather than claiming the drop happened during this epoch.
        if _prev_lr and _lrs[0] < _prev_lr[-1]["lr_head"] * 0.99:
            log(f"       [SCHED] LR dropped after epoch {ep - 1} "
                f"({_prev_lr[-1]['lr_head']:.2e} -> {_lrs[0]:.2e}) -- "
                "ReduceLROnPlateau judged the plateau real")
        if "aug_frac_of_epoch" in rec:
            log(f"       AUG {rec['aug_ms_per_batch']:.1f} ms/batch = "
                f"{rec['aug_frac_of_epoch']:.1%} of the epoch "
                f"(p_clean={aug_cfg.p_clean:.2f} means {aug_cfg.p_clean:.0%} of batches "
                "skip the chain entirely)")
        if n_skipped:
            log(f"       [SKIP] {n_skipped} infeasible rows skipped this epoch "
                "-- recorded in history.jsonl as rows_skipped")
        if n_oom:
            log(f"       [OOM] {n_oom} batches dropped to out-of-memory this epoch "
                "-- recorded in history.jsonl as oom_batches")
        if dev == "cuda":
            _al, _rs = rec["vram_alloc_gb"], rec["vram_reserved_gb"]
            _lv = rec["vram_live_end_gb"]
            log(f"       VRAM this epoch: peak {_al:.1f} GB live / {_rs:.1f} GB pool "
                f"| {_lv:.2f} GB still live at epoch end")
            # Leak test on the TREND, not on one epoch-to-epoch difference.
            # The first version compared against the previous epoch only and cried
            # leak twice on a series that was simply oscillating:
            #   4.97  4.30  4.80  4.41  4.96   (slope +0.009 GB/epoch)
            # Live-at-epoch-end depends on which tensors the last eval batch still
            # holds, so +-0.6 GB of noise is normal. A leak is a SUSTAINED rise, so
            # require a positive slope over at least four epochs and a total climb
            # bigger than the observed swing.
            _series = [h["vram_live_end_gb"] for h in hist if "vram_live_end_gb" in h]
            if len(_series) >= 4:
                _n = len(_series)
                _slope = float(np.polyfit(range(_n), _series, 1)[0])
                _swing = max(_series) - min(_series)
                if _slope > 0.15 and _slope * _n > _swing:
                    log(f"       !! live memory trending up {_slope:+.2f} GB/epoch over "
                        f"{_n} epochs (swing {_swing:.2f} GB) -- that looks like a real "
                        "LEAK. A growing POOL with flat live memory would be normal; "
                        "a rising trend in live memory is not.")
        sch.step(cer)

        if cer < best * 0.995:
            best, best_ep = cer, ep
            torch.save(head.state_dict(), run_dir / "head.pt")
            torch.save({k: v.detach().cpu().clone() for k, v in bb.state_dict().items()
                        if "lora_" in k}, run_dir / "adapter.pt")
            log(f"     [BEST] {cer * 100:.2f}%")

        # per-epoch IMMUTABLE checkpoint -- an ~8h unattended run must survive
        # a disconnect; a single overwritten last.pt is one bad write from
        # losing everything, so every epoch also gets its own snapshot file.
        ckpt = {"head": head.state_dict(),
                "adapter": {k: v.detach().cpu().clone() for k, v in bb.state_dict().items()
                           if "lora_" in k},
                "opt": opt.state_dict(), "sch": sch.state_dict(),
                "epoch": ep, "best": best, "best_ep": best_ep}
        torch.save(ckpt, run_dir / f"ep{ep:03d}.pt")
        torch.save(ckpt, last_path)  # resumable pointer to "latest"

        # Mirror to Drive EVERY epoch, not once at the end. molab is a cloud
        # notebook: if the session dies at 03:00 the local disk goes with it, and
        # per-epoch snapshots that only ever existed locally are worth nothing.
        # This was the gap that made `gdrive_sync.py` dead code -- it was written
        # to disk and never called from anywhere.
        _sync_to_drive([run_dir / f"ep{ep:03d}.pt", last_path, hist_path,
                        run_dir / "head.pt", run_dir / "adapter.pt"], cfg.run)

        if ep - best_ep >= cfg.stop_patience:
            log("[STOP] no improvement, early stopping")
            break

    summary = {"run": cfg.run, "best_cer": best, "best_epoch": best_ep,
               "epochs_done": hist[-1]["epoch"] if hist else 0,
               "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None,
               "vram_reserved_peak_gb": (torch.cuda.max_memory_reserved() / 1e9
                                         if dev == "cuda" else None),
               "sec_per_epoch": float(np.median([h["secs"] for h in hist])) if hist else None,
               "final_layer_weights": hist[-1]["w"] if hist else None}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Final mirror: the best-checkpoint pair and the run metadata. The per-epoch
    # sync above already covered the resumable state, so this is about making the
    # DEPLOYABLE artefacts (head.pt + adapter.pt + config.json) plus the summary
    # available even if the session dies immediately after training finishes.
    _sync_to_drive([run_dir / "summary.json", run_dir / "config.json",
                    run_dir / "head.pt", run_dir / "adapter.pt",
                    run_dir / "history.jsonl"], cfg.run)
    log(f"[DONE] {cfg.run}: best CER {best * 100:.2f}% @ epoch {best_ep}")
    return summary


def parse_layers(s: str) -> tuple:
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in s.split(","))


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


def main():
    log(f"[src] {_selfstamp()}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", default="/marimo/runs")
    ap.add_argument("--ws", default="9,10,11,12")
    ap.add_argument("--lora-layers", default="1-12")
    ap.add_argument("--epochs", type=int, default=30)
    # Batching: memory knob and optimisation knob, deliberately separate.
    # `--accum` is NOT accepted -- it is derived from the two below, so it cannot
    # drift out of sync with the effective batch the run reports.
    ap.add_argument("--micro-secs", type=float, default=200.0,
                    help="padded audio seconds per GPU forward -- lower this to fit "
                         "a smaller card; it does NOT change the optimisation batch")
    ap.add_argument("--micro-batch", type=int, default=16,
                    help="utterance cap per forward (guard against buckets of very "
                         "short clips)")
    ap.add_argument("--effective-secs", type=float, default=800.0,
                    help="audio seconds per optimiser step; MUST be identical across "
                         "ablation arms for the comparison to mean anything")
    ap.add_argument("--batch", type=int, default=None,
                    help="DEPRECATED alias for --micro-batch. The old flag also scaled "
                         "the memory budget, which is exactly the bug this replaces.")
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--hours-subset", type=float, default=None,
                    help="train on a random subset of this many hours; "
                         "0 or omitted = use the whole cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--bb-dropout", type=float, default=0.0,
                    help="backbone dropout; 0.05 matches the 100h FINAL baseline. "
                         "Non-zero puts the frozen backbone in train() mode so the "
                         "dropout layers actually fire")
    ap.add_argument("--init-from", default=None,
                    help="run dir to take head.pt/adapter.pt from, with a fresh "
                         "optimiser (fine-tune, not resume)")
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--rir-dir", default=None)
    args = ap.parse_args()

    micro_batch = args.micro_batch
    if args.batch is not None:
        log(f"[BATCH] --batch {args.batch} is deprecated; treating it as "
            f"--micro-batch {args.batch}. It no longer scales the memory budget: "
            f"use --micro-secs (currently {args.micro_secs:.0f}s) for that.")
        micro_batch = args.batch

    cfg = Cfg(run=args.run, ws=parse_layers(args.ws), lora_layers=parse_layers(args.lora_layers),
              epochs=args.epochs, micro_secs=args.micro_secs, micro_batch=micro_batch,
              effective_secs=args.effective_secs, lr_scale=args.lr_scale,
              hours_subset=args.hours_subset, workers=args.workers, aug_on=not args.no_aug,
              noise_dir=args.noise_dir, rir_dir=args.rir_dir,
              bb_dropout=args.bb_dropout, init_from=args.init_from)
    train_one(cfg, Path(args.out), Path(args.cache_dir))


if __name__ == "__main__":
    main()
