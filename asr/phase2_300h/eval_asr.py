# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch", "torchaudio", "transformers>=4.44", "peft>=0.11", "jiwer",
#     "psutil", "numpy", "datasets==5.0.0", "soundfile==0.14.0",
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
ASR -- 300h retrain, headline evaluation.

TWO TEST COLUMNS (not two separate tables) for every system:
    dev-clean    -- comparability anchor; both the 100h baseline and the
                    300h model are scored here so the new number is directly
                    comparable to the published 10.1% / 5.1% WER.
    L2-ARCTIC    -- held-out OOD accent test. NEVER seen in training
                    (enforced upstream by prepare_data.assert_no_l2arctic).

Each column reports greedy AND +KenLM separately -- the KenLM gain is
expected to SHRINK out-of-domain (the LM was built on LibriSpeech text), and
that shrinkage is reported as a real finding, not hidden by only reporting
one decode mode. Whisper (base/small/medium) is re-run on BOTH columns as
the external reference, same protocol as whisper_bench.ipynb.

Efficiency stats follow the existing project's convention:
    CPU RTF   = wall_time_on_cpu / audio_duration_s     (psutil process)
    GPU RTF   = wall_time_on_gpu / audio_duration_s
    peak RAM  = psutil RSS delta (CPU) / torch.cuda.max_memory_allocated (GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from itertools import groupby
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import build_vocab, normalize_text

_NORM_RE = re.compile(r"[^A-Z' ]+")


def wer_normalize(s: str) -> str:
    """Common normalizer applied to BOTH ref and hyp before jiwer -- same
    convention as kenlm_grid.py's normalize(), so WER isn't inflated by a
    spurious ref/hyp mismatch in punctuation handling."""
    s = s.upper().replace("|", " ")
    s = _NORM_RE.sub(" ", s)
    return " ".join(s.split())


def log(*a):
    print(*a, flush=True)


def sync_to_drive(paths, run_name: str) -> None:
    """Mirror small result artefacts to Drive. NEVER raises.

    train_asr.py mirrors checkpoints every epoch, but `lm_params*.json` and
    `eval_results.json` were written to molab's local disk and nowhere else --
    and molab is ephemeral. Those two files ARE the results table; losing the
    session would mean re-running the eval, not just re-copying a file.
    """
    try:
        _here = str(Path(__file__).resolve().parent)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import gdrive_sync
    except Exception as exc:
        log(f"[drive] gdrive_sync unavailable ({exc}) -- results stay local only")
        return
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            ok, reason = gdrive_sync.sync_checkpoint(p, run_name)
            log(f"[drive] {p.name}: {'mirrored' if ok else 'NOT mirrored -- ' + reason}")
        except Exception as exc:
            log(f"[drive] {p.name}: NOT mirrored -- {type(exc).__name__}: {exc}")


# ============================================================================
# 1 . Test set loaders
# ============================================================================


def load_devclean(limit=None):
    from datasets import load_dataset

    ds = _no_decode(load_dataset("openslr/librispeech_asr", "clean", split="validation"))
    rows = []
    for i in range(len(ds) if limit is None else min(limit, len(ds))):
        r = ds[i]
        rows.append({"audio": r["audio"], "text": r["text"]})
    log(f"[devclean] {len(rows)} utterances")
    return rows


def load_l2arctic(limit=None):
    """L2-ARCTIC -- held-out OOD accent test set. Loaded HERE ONLY, at eval time,
    never in prepare_data.py / train_asr.py.

    The repo is `KoelLabs/L2Arctic`. Two things about it broke the old code:

      * there is NO `train` split. The splits are `scripted` (3,599 utterances,
        the ARCTIC prompts read by 24 non-native speakers) and `spontaneous`
        (22). `scripted` is the one that corresponds to L2-ARCTIC as normally
        reported; 22 utterances is not a test set.
      * the dataset is GATED ("gated: auto"), cc-by-nc-4.0. The HF token from
        §3 must have accepted the terms on the dataset page, otherwise the load
        fails with a 401/403 rather than a missing-repo error.

    It also carries `speaker_native_language`, which is kept so accuracy can be
    broken down by L1 (Arabic / Mandarin / Spanish / Hindi / Vietnamese /
    Korean). A single averaged accent number hides which accents the model
    actually handles -- the same lesson the per-corpus dev split taught.
    """
    from datasets import load_dataset

    ds = None
    tried = []
    for repo, split in (("KoelLabs/L2Arctic", "scripted"),
                        ("KoelLabs/L2Arctic", "train"),
                        ("babels/l2-arctic", "train")):
        try:
            ds = _no_decode(load_dataset(repo, split=split))
            log(f"[l2arctic] loaded {repo} split={split}")
            break
        except Exception as e:
            tried.append(f"{repo}/{split}: {type(e).__name__}: {str(e)[:120]}")
    if ds is None:
        raise RuntimeError(
            "Could not load L2-ARCTIC. Tried:\n  " + "\n  ".join(tried)
            + "\nIf the error is 401/403 rather than 'not found', the dataset is "
              "gated: open https://huggingface.co/datasets/KoelLabs/L2Arctic , "
              "accept the terms with the same account as the §3 token, and re-run.")

    rows = []
    n = len(ds) if limit is None else min(limit, len(ds))
    # Fixed stride, not the first n: the split is ordered by speaker, so rows[:500]
    # would be a handful of speakers -- and with only 24 speakers across 6 L1s,
    # that could silently reduce the accent test to two or three accents.
    step = max(1, len(ds) // n)
    for i in list(range(0, len(ds), step))[:n]:
        r = ds[i]
        text = r.get("text") or r.get("transcript") or r.get("sentence")
        rows.append({"audio": r["audio"], "text": text,
                     "l1": r.get("speaker_native_language") or "?",
                     "speaker": r.get("speaker_code") or "?"})
    from collections import Counter as _C
    log(f"[l2arctic] {len(rows)} utterances (every {step}th of {len(ds)}) | "
        f"L1: {dict(_C(r['l1'] for r in rows))} | "
        f"{len(set(r['speaker'] for r in rows))} speakers")
    return rows


# ============================================================================
# 2 . Our model: greedy + KenLM decode
# ============================================================================


def _decode_audio_array(cell, sr_target=16000):
    """Decode one HF audio cell, whichever shape it arrives in.

    `datasets` v5 routes Audio decoding through **torchcodec**, and on this stack
    torchcodec cannot load at all:

        RuntimeError: Could not load libtorchcodec
        OSError: libnvrtc.so.13: cannot open shared object file

    prepare_data.py already sidesteps this with `Audio(decode=False)` and decodes
    the raw bytes itself; the eval path had not been given the same treatment, so
    it died on the first `ds[i]["audio"]`. With decode=False the cell is
    `{"bytes": ..., "path": ...}` instead of `{"array": ..., "sampling_rate": ...}`,
    so this handles both -- an already-decoded cell still works if some other
    environment does have a functioning torchcodec.
    """
    import io

    import numpy as np

    if isinstance(cell, dict) and cell.get("array") is not None:
        w = np.asarray(cell["array"], dtype=np.float32)
        sr = cell["sampling_rate"]
    else:
        import soundfile as sf

        if isinstance(cell, dict) and cell.get("bytes"):
            w, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif isinstance(cell, dict) and cell.get("path"):
            w, sr = sf.read(cell["path"], dtype="float32")
        elif isinstance(cell, str):
            w, sr = sf.read(cell, dtype="float32")
        else:
            raise ValueError(f"unrecognised audio cell: {type(cell)} {list(cell) if isinstance(cell, dict) else ''}")
        w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if int(sr) != sr_target:
        w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr_target / sr)),
                      np.arange(len(w)), w).astype(np.float32)
    return w


def _no_decode(ds):
    """Turn OFF the Audio feature's decoding, so torchcodec is never imported."""
    from datasets import Audio

    try:
        return ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:      # column may already be raw, or named differently
        log(f"[audio] cast_column(decode=False) skipped: {type(exc).__name__}: {exc}")
        return ds


def greedy_decode(logits, vocab):
    blank, unk = vocab["[PAD]"], vocab["[UNK]"]
    i2c = {i: c for c, i in vocab.items()}
    ids = logits.argmax(-1)
    out = [i2c[k] for k, _ in groupby(ids.tolist()) if k not in (blank, unk)]
    return "".join(out).replace("|", " ").strip()


def load_our_model(run_dir: Path, device: str):
    import torch
    import torch.nn as nn
    from transformers import HubertModel
    from peft import LoraConfig, inject_adapter_in_model

    cfg = json.loads((run_dir / "config.json").read_text())
    ws, lora_layers = cfg["ws"], cfg["lora_layers"]
    hid = cfg.get("hid", 768)

    bb = HubertModel.from_pretrained("utter-project/mHuBERT-147")
    lora_cfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                          target_modules=["q_proj", "v_proj"], bias="none",
                          layers_to_transform=[i - 1 for i in lora_layers])
    bb = inject_adapter_in_model(lora_cfg, bb)
    bb.load_state_dict(torch.load(run_dir / "adapter.pt", map_location=device), strict=False)
    bb = bb.to(device).eval()

    vocab = build_vocab()

    class Head(nn.Module):
        def __init__(self, n, dim, V):
            super().__init__()
            self.layer_w = nn.Parameter(torch.zeros(n))
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Dropout(0.0),
                                     nn.Linear(dim, V))

        def forward(self, x):
            w = self.layer_w.softmax(0)
            return self.net((x * w[None, None, :, None]).sum(2))

    head = Head(len(ws), hid, len(vocab)).to(device)
    head.load_state_dict(torch.load(run_dir / "head.pt", map_location=device))
    head.eval()
    return bb, head, ws, vocab


def run_our_model(rows, run_dir: Path, lm_path: str | None, device: str,
                  alpha: float = 0.5, beta: float = 1.0,
                  beam_width: int = 100) -> dict:
    import torch
    import psutil

    bb, head, ws, vocab = load_our_model(run_dir, device)
    blank, unk = vocab["[PAD]"], vocab["[UNK]"]
    flen = bb._get_feat_extract_output_lengths

    decoder = None
    if lm_path:
        try:
            import kenlm  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                f"KenLM python bindings are not installed ({exc}). Either install "
                "them (pip install kenlm, or the github archive if there is no "
                "wheel) or drop --lm to score the greedy column only."
            ) from exc
        from pyctcdecode import build_ctcdecoder

        # inlined from kenlm_grid.py's vocab_to_labels -- kept local instead
        # of importing across the _Staj/ vs _Staj/asr/ directory boundary,
        # since these notebooks are meant to be standalone (kenlm_grid.py's
        # own header rule: no imports across scripts)
        def _vocab_to_labels(v):
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

        # alpha/beta were hardcoded 0.5/1.0 here with no evidence behind either
        # number. They now come from tune_lm.py, which fits them on dev-other --
        # a set that is NOT reported -- and records which checkpoint they belong
        # to. Passing them in also means the 100h baseline row can be re-decoded
        # under the identical protocol instead of inheriting a guess.
        log(f"[kenlm] alpha={alpha} beta={beta} beam_width={beam_width}")
        decoder = build_ctcdecoder(_vocab_to_labels(vocab), kenlm_model_path=lm_path,
                                   alpha=alpha, beta=beta)

    proc = psutil.Process()
    refs, hyps_greedy, hyps_lm = [], [], []
    total_audio_s, t_greedy, t_lm = 0.0, 0.0, 0.0
    rss0 = proc.memory_info().rss
    peak_rss = rss0

    with torch.no_grad():
        for r in rows:
            w = _decode_audio_array(r["audio"])
            total_audio_s += len(w) / 16000.0
            X = torch.from_numpy(w).unsqueeze(0).to(device)
            am = torch.ones_like(X, dtype=torch.long)

            t0 = time.perf_counter()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                o = bb(X, attention_mask=am, output_hidden_states=True)
                hs = torch.stack([o.hidden_states[L] for L in ws], 2)
                logits = head(hs.float())[0]
            if device == "cuda":
                torch.cuda.synchronize()
            _fwd = time.perf_counter() - t0
            t_greedy += _fwd

            # `.float()` is NOT optional. The forward runs under
            # autocast(bfloat16), so `logits` comes back bf16, and numpy has no
            # bfloat16 dtype -- `probs.cpu().numpy()` for pyctcdecode dies with
            #     TypeError: Got unsupported ScalarType BFloat16
            # tune_lm.py already cast here; this copy of the same computation did
            # not, which is what two hand-written copies of one forward pass do.
            probs = logits.log_softmax(-1).float()
            hyps_greedy.append(wer_normalize(greedy_decode(probs.cpu(), vocab)))
            refs.append(wer_normalize(r["text"]))

            if decoder is not None:
                t1 = time.perf_counter()
                hyp = decoder.decode(probs.cpu().numpy(), beam_width=beam_width)
                # BUG FIXED: this used to add the CUMULATIVE `t_greedy`, so by
                # utterance N it had charged the forward pass N times over and the
                # KenLM RTF grew quadratically with the size of the test set --
                # a reported efficiency number that was pure artefact. The LM
                # decode does include the forward pass, but only THIS one.
                t_lm += (time.perf_counter() - t1) + _fwd
                hyps_lm.append(wer_normalize(hyp))

            peak_rss = max(peak_rss, proc.memory_info().rss)

    import jiwer

    def _by_group(key):
        """WER/CER per group (L1 accent for L2-ARCTIC). Empty for sets without it."""
        groups = {}
        keys = [r.get(key) for r in rows]
        if not any(keys):
            return {}
        for g in sorted({k for k in keys if k}):
            _i = [i for i, k in enumerate(keys) if k == g]
            _r = [refs[i] for i in _i]
            _hg = [hyps_greedy[i] for i in _i]
            groups[g] = {"n": len(_i), "greedy_wer": jiwer.wer(_r, _hg),
                         "greedy_cer": jiwer.cer(_r, _hg)}
            if hyps_lm:
                _hl = [hyps_lm[i] for i in _i]
                groups[g]["kenlm_wer"] = jiwer.wer(_r, _hl)
                groups[g]["kenlm_cer"] = jiwer.cer(_r, _hl)
        return groups

    out = {
        "greedy": {"wer": jiwer.wer(refs, hyps_greedy), "cer": jiwer.cer(refs, hyps_greedy),
                   "rtf": t_greedy / max(total_audio_s, 1e-6)},
    }
    # One averaged accent number hides which accents the model can actually
    # handle -- exactly what the VCTK-only dev split taught. L2-ARCTIC carries the
    # speaker's L1, so break it down.
    _bl1 = _by_group("l1")
    if _bl1:
        out["by_l1"] = _bl1
        log("[eval] per-L1: " + "  ".join(
            f"{g}: greedy {100 * v['greedy_wer']:.1f}"
            + (f" / +LM {100 * v['kenlm_wer']:.1f}" if "kenlm_wer" in v else "")
            + f" (n={v['n']})" for g, v in _bl1.items()))
    if decoder is not None:
        out["kenlm"] = {"wer": jiwer.wer(refs, hyps_lm), "cer": jiwer.cer(refs, hyps_lm),
                        "rtf": t_lm / max(total_audio_s, 1e-6)}
    # Was `(peak - current + peak)`, i.e. 2*peak - current, which is not a
    # quantity. Report the peak, and the delta over the pre-run baseline
    # separately, since "how much did THIS add" is the useful figure.
    out["peak_ram_mb"] = peak_rss / 1e6
    out["ram_delta_mb"] = (peak_rss - rss0) / 1e6
    if device == "cuda":
        out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


# ============================================================================
# 3 . Whisper baseline -- same protocol, both columns
# ============================================================================


def run_whisper(rows, model_name: str, device: str) -> dict:
    import torch
    import psutil
    import jiwer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    processor = WhisperProcessor.from_pretrained(model_name)
    ps = psutil.Process()

    refs, hyps = [], []
    total_audio_s, t_total = 0.0, 0.0
    with torch.no_grad():
        for r in rows:
            w = _decode_audio_array(r["audio"])
            total_audio_s += len(w) / 16000.0
            inputs = processor(w, sampling_rate=16000, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            ids = proc_model.generate(inputs["input_features"], language="en", task="transcribe")
            if device == "cuda":
                torch.cuda.synchronize()
            t_total += time.perf_counter() - t0
            hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
            refs.append(wer_normalize(r["text"]))
            hyps.append(wer_normalize(hyp))

    out = {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps),
           "rtf": t_total / max(total_audio_s, 1e-6),
           "peak_ram_mb": ps.memory_info().rss / 1e6}
    if device == "cuda":
        out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


# ============================================================================
# 4 . Orchestration
# ============================================================================


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
    ap.add_argument("--run-dir", required=True, help="trained run dir (has config.json/head.pt/adapter.pt)")
    ap.add_argument("--lm", default=None, help="path to KenLM .arpa (omit for greedy-only)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per test set (debug)")
    ap.add_argument("--whisper", default="openai/whisper-base,openai/whisper-small,openai/whisper-medium",
                    help="comma-separated model ids; pass '' to skip Whisper entirely")
    ap.add_argument("--whisper-on", default="l2-arctic",
                    choices=["both", "dev-clean", "l2-arctic", "none"],
                    help="which test sets to run Whisper on. Default l2-arctic: "
                         "Whisper is a FIXED external reference and its dev-clean "
                         "numbers were already measured in whisper_bench.ipynb "
                         "for the 100h comparison -- re-running it there costs ~100 "
                         "minutes and returns the identical number. L2-ARCTIC is the "
                         "new set, so that is where a new Whisper run is needed.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lm-params", default=None,
                    help="lm_params.json from tune_lm.py; overrides --alpha/--beta/--beam")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beam", type=int, default=100)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_rows = load_devclean(args.limit)
    l2_rows = load_l2arctic(args.limit)

    # Decoder params: a tuned file wins over the CLI defaults. The defaults are
    # still 0.5/1.0/100 -- the OLD hardcoded guess -- kept only so an untuned run
    # reproduces the previous behaviour instead of silently changing under you.
    alpha, beta, beam = args.alpha, args.beta, args.beam
    if args.lm_params:
        _pp = json.loads(Path(args.lm_params).read_text())
        alpha, beta, beam = _pp["alpha"], _pp["beta"], _pp["beam_width"]
        log(f"[eval] decoder params from {args.lm_params}: "
            f"alpha={alpha} beta={beta} beam={beam}")
        # Params fitted to a DIFFERENT checkpoint are not transferable: alpha
        # trades off against how well-calibrated this particular acoustic model
        # is. Refuse quietly-wrong reuse.
        _want = str(Path(args.run_dir).resolve())
        _got = str(Path(_pp.get("run_dir", "")).resolve()) if _pp.get("run_dir") else ""
        if _got and _got != _want:
            raise SystemExit(
                f"{args.lm_params} was tuned on {_pp['run_dir']} but --run-dir is "
                f"{args.run_dir}. Run tune_lm.py for THIS checkpoint; alpha is not "
                "transferable between acoustic models of different quality.")
        if _pp.get("lm_path") and args.lm and Path(_pp["lm_path"]).name != Path(args.lm).name:
            raise SystemExit(
                f"{args.lm_params} was tuned against {_pp['lm_path']} but --lm is "
                f"{args.lm}. Re-tune, or the LM weight belongs to a different LM.")
        for _w in _pp.get("warnings", []):
            log(f"[eval] !! tuning warning carried over: {_w}")
    elif args.lm:
        log("[eval] !! decoding with UNTUNED alpha=0.5 beta=1.0 beam=100 (the old "
            "hardcoded guess). Run tune_lm.py and pass --lm-params for a fair number.")

    results = {"dev-clean": {}, "l2-arctic": {},
               "decoder_params": {"alpha": alpha, "beta": beta, "beam_width": beam,
                                  "source": args.lm_params or "CLI/default"}}

    log("[eval] scoring FINAL model on dev-clean...")
    results["dev-clean"]["FINAL_300h"] = run_our_model(
        dev_rows, Path(args.run_dir), args.lm, device, alpha, beta, beam)
    log("[eval] scoring FINAL model on l2-arctic...")
    results["l2-arctic"]["FINAL_300h"] = run_our_model(
        l2_rows, Path(args.run_dir), args.lm, device, alpha, beta, beam)

    _wsets = {"both": ("dev-clean", "l2-arctic"), "dev-clean": ("dev-clean",),
              "l2-arctic": ("l2-arctic",), "none": ()}[args.whisper_on]
    _wmodels = [w for w in args.whisper.split(",") if w.strip()]
    if not _wmodels or not _wsets:
        log("[eval] Whisper skipped. If the table needs a dev-clean Whisper row, "
            "take it from whisper_bench.ipynb / meeting/B1_WER.png -- it is "
            "the same model on the same set, so the number is unchanged.")
    results["whisper_run_on"] = list(_wsets)
    for wm in _wmodels:
        tag = wm.split("/")[-1]
        for _setname, _rows in (("dev-clean", dev_rows), ("l2-arctic", l2_rows)):
            if _setname not in _wsets:
                continue
            log(f"[eval] scoring {tag} on {_setname}...")
            results[_setname][tag] = run_whisper(_rows, wm, device)

    log(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"[eval] written to {args.out}")
        _rd = Path(args.run_dir)
        sync_to_drive([args.out, _rd / "lm_params.json",
                       _rd / "lm_params_clean.json", _rd / "lm_params_other.json"],
                      _rd.name)

    # Same kenlm teardown abort tune_lm.py hits:
    #   util/mmap.cc:138 SyncOrThrow ... Fatal Python error: Aborted
    # It fires during garbage collection, long after the results are written and
    # mirrored. Left alone it would give a non-zero exit code for a run that
    # actually succeeded -- and this one is meant to be left running unattended,
    # where "did it work?" is answered by the exit code and the Drive copy.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
