import marimo

__generated_by_class = "A"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import subprocess
    import sys
    import json
    import os
    import shutil
    from pathlib import Path
    return mo, subprocess, sys, json, os, shutil, Path


@app.cell
def _(mo):
    mo.md(
        r"""
        # ASR — 300h Retrain Interactive Dashboard

        Extends the finished 100h LibriSpeech-only model (dev-clean greedy WER 10.1% / CER 2.8%; +KenLM 5.1% / 1.8%) with accent, spontaneous-speech, and channel/noise robustness, for a **live demo**: laptop microphone, in a room, non-native English speaker.

        ### Architecture Highlights (Unchanged):
        - **Backbone**: Frozen `mHuBERT-147` + LoRA (layers 1–12, r=16, alpha=32 on `q_proj`, `v_proj`).
        - **Head**: Weighted-sum over hidden-state layers + 2-layer MLP + CTC Loss.
        - **Vocabulary**: `A-Z`, `'` (apostrophe), `|` (word separator), `[UNK]`, `[PAD]`.

        This interactive marimo notebook serves as an orchestration dashboard for:
        1. Setting up the Python 3.11 virtual environment.
        2. Automatically compiling pipeline modules to disk (`prepare_data.py`, `augment.py`, `gdrive_sync.py`, `build_cache.py`, `train_asr.py`, `eval_asr.py`).
        3. Fetching background noise/RIR banks (OpenSLR-28 RIRs + optional MUSAN noise) via `fetch_noise_banks.py` -- section 7. Verify the banks are non-empty before training: an empty bank makes reverb/noise a silent no-op.
        4. Building a 300-hour manifest and binary cache (100h LibriSpeech + 106h Common Voice + 50h AMI + 44h VCTK).
        5. **Probing WS Layers**: Conducting LoRA + Weighted-Sum Ablation studies with high LR to determine ideal phonetic layers.
        6. **Evaluating**: two-column (dev-clean / L2-ARCTIC) WER/CER scoring, greedy and +KenLM, against a real Whisper baseline -- no hardcoded numbers.
        7. **Google Drive Syncing**: Automatically mirrors all checkpoints, configs, history logs, and `summary.json` files to Google Drive root at `CLEAR/Phase 1/ASR-300` if mounted!
        """
    )


@app.cell
def _(mo):
    dir_input = mo.ui.text(value="./marimo_asr_work", label="ASR Work Directory")
    return dir_input,


@app.cell
def _(mo, dir_input):
    mo.md(
        f"""
        ## 1 · Directory configuration
        Configure the root working directory where datasets, runs, caches, and the isolated Python 3.11 environment will reside:
        
        {dir_input}
        """
    )


@app.cell
def _(Path, dir_input):
    base_dir = Path(dir_input.value).resolve()
    asr_dir = base_dir / "asr"
    data_dir = base_dir / "data"
    cache_dir = base_dir / "cache"
    runs_dir = base_dir / "runs"
    noise_dir = base_dir / "noise"
    rir_dir = base_dir / "rir"
    lm_dir = base_dir / "lm"
    py_bin = base_dir / "asr311" / "bin" / "python"
    
    for d in (asr_dir, data_dir, cache_dir, runs_dir, noise_dir, rir_dir, lm_dir):
        d.mkdir(parents=True, exist_ok=True)
        
    return base_dir, asr_dir, data_dir, cache_dir, runs_dir, noise_dir, rir_dir, lm_dir, py_bin


@app.cell
def _(mo, asr_dir):
    # 1. prepare_data.py (verbatim port of _Staj/asr/prepare_data.py)
    prepare_data_code = r'''# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets==5.0.0",
#     "huggingface-hub>=0.24",
#     "soundfile==0.14.0",
#     "torchcodec",       # datasets 5.x routes Audio decoding through this;
#                         # streaming paths use decode=False and skip it, but
#                         # VCTK/Common Voice still decode normally.
#     "numpy",
#     "pandas",
#     "tqdm",
# ]
# ///
"""
ASR -- 300h retrain, data pipeline.

Builds the training manifest from four sources:
    LibriSpeech train-clean-100   100 h   comparability anchor (unchanged)
    Common Voice 22 EN            106 h   accent-stratified sample
    AMI (ihm+sdm, disjoint mtgs)   50 h   accent/spontaneous/far-field
    VCTK                           44 h   accent, studio-clean
                                  -----
                                  300 h

L2-ARCTIC IS NEVER TOUCHED HERE. It is the held-out OOD test set used only
in evaluation (see asr_300h.ipynb). assert_no_l2arctic() below is a
hard gate -- every manifest-writing path runs through it before the combined
manifest is written.

Text is normalised to the EXACT character vocabulary used by ablation_engine.py /
kenlm_grid.py:
    A-Z, ' (apostrophe), | (word separator), [UNK], [PAD] (=CTC blank)
No new characters are ever added to that vocabulary -- the existing KenLM
3-gram was built on LibriSpeech-normalised text and does not know any other
symbol. Rows that still contain out-of-vocabulary characters after
normalisation are dropped, and the drop rate is logged per corpus.

Usage:
    python prepare_data.py --corpus librispeech --out /marimo/data
    python prepare_data.py --corpus common_voice --out /marimo/data
    python prepare_data.py --corpus ami --out /marimo/data
    python prepare_data.py --corpus vctk --out /marimo/data
    python prepare_data.py --combine --out /marimo/data
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import os
import random
import re
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ============================================================================
# 0 . Vocabulary (must match ablation_engine.py / kenlm_grid.py exactly)
# ============================================================================

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")


def build_vocab() -> dict:
    v = {c: i for i, c in enumerate(CHARS)}
    v["|"], v["[UNK]"], v["[PAD]"] = len(v), len(v) + 1, len(v) + 2
    return v


VOCAB = build_vocab()
_ALLOWED = set(CHARS) | {" "}  # space becomes "|" downstream; both are legal here

# digits -> words, so "911" survives normalisation instead of being dropped as OOV
_DIGIT_WORDS = {
    "0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR",
    "5": "FIVE", "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE",
}

_WS_RE = re.compile(r"\s+")


def normalize_text(raw: str) -> str | None:
    """Normalise to the CTC/KenLM character set. Returns None if the row is
    unrecoverable (still has OOV chars after normalisation, or is empty).

    Rules (per spec):
      - expand digits to words (911 -> NINE ONE ONE)
      - strip hyphens and periods (AMI spells letters as "S. S. H.")
      - uppercase
      - keep apostrophes
      - collapse whitespace
      - drop rows still containing OOV chars after all of the above

    IMPORTANT: only hyphens/periods are explicitly stripped. Any OTHER
    character outside A-Z/'/space causes the whole row to be DROPPED (return
    None), not silently deleted -- silently deleting stray punctuation would
    let rows with real OOV content (accented letters, stray symbols) sail
    through as if they were clean, which is exactly the failure mode the
    "drop rows still containing OOV chars" rule exists to prevent.
    """
    if not raw:
        return None
    s = raw.upper()
    # digit expansion BEFORE stripping punctuation, so "9-1-1" and "9.1.1."
    # both become "NINE ONE ONE" rather than "911" surviving as a bare token
    s = "".join(f" {_DIGIT_WORDS[ch]} " if ch in _DIGIT_WORDS else ch for ch in s)
    s = s.replace("-", " ").replace(".", " ")
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return None
    if any(c not in _ALLOWED for c in s):
        return None
    return s


# ============================================================================
# 1 . L2-ARCTIC leakage gate
# ============================================================================

_L2ARCTIC_MARKERS = ("l2-arctic", "l2_arctic", "l2arctic")


def assert_no_l2arctic(rows: list[dict], manifest_path) -> None:
    """Hard gate: L2-ARCTIC must NEVER appear in a training manifest.
    Checked on every row's corpus/source/audio_path field, and on the
    manifest path itself. Raises AssertionError (not a warning) on any hit."""
    mp = str(manifest_path).lower()
    assert not any(m in mp for m in _L2ARCTIC_MARKERS), (
        f"L2-ARCTIC marker found in manifest PATH itself: {manifest_path}. "
        "L2-ARCTIC is the held-out OOD test set and must never be written "
        "into a training manifest."
    )
    for r in rows:
        blob = " ".join(str(r.get(k, "")) for k in ("corpus", "source", "audio_path", "speaker"))
        blob = blob.lower()
        assert not any(m in blob for m in _L2ARCTIC_MARKERS), (
            f"L2-ARCTIC marker found in a training row: {r}. Aborting write "
            f"of {manifest_path} -- this would leak the OOD test set into training."
        )


# ============================================================================
# 2 . Manifest I/O helpers
# ============================================================================


def write_manifest(rows: list[dict], path: Path) -> None:
    assert_no_l2arctic(rows, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def hours(rows: list[dict]) -> float:
    return sum(r["duration_s"] for r in rows) / 3600.0


def log(*a):
    print(*a, flush=True)


# ============================================================================
# 2b . Resume support -- never re-download what is already on disk
# ============================================================================

FORCE_REBUILD = False   # flipped by --force; read live by manifest_is_complete

CORPUS_TARGET_HOURS = {
    "librispeech": 100.0,
    "common_voice": 106.0,
    "ami": 50.0,
    "vctk": 44.0,
}


def manifest_is_complete(corpus: str, out_dir: Path, sample: int = 200) -> bool:
    """True when `manifest_<corpus>.jsonl` already exists AND looks usable.

    Without this, re-running `--corpus all` after one corpus succeeded re-streams
    everything from scratch: `_write_raw_audio` skips files it has already
    written, but the HTTP stream that produced them is walked again regardless,
    so the bytes cross the wire a second time.

    "Usable" means: it parses, it is within 10% of the corpus hour target, and a
    random sample of its `audio_path` entries actually exists on disk. A manifest
    left behind by an interrupted run therefore does NOT count as complete --
    which is the case that matters, since that is exactly when someone re-runs.
    """
    if FORCE_REBUILD:
        return False
    path = out_dir / f"manifest_{corpus}.jsonl"
    if not path.exists():
        return False
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except Exception as exc:
        log(f"[{corpus}] existing manifest is unreadable ({exc}) -- rebuilding")
        return False
    if not rows:
        log(f"[{corpus}] existing manifest is empty -- rebuilding")
        return False

    h = sum(r.get("duration_s", 0.0) for r in rows) / 3600.0
    want = CORPUS_TARGET_HOURS.get(corpus, 0.0)
    if want and h < want * 0.9:
        log(f"[{corpus}] existing manifest only {h:.2f}h / {want:.0f}h "
            f"(interrupted run?) -- rebuilding")
        return False

    probe = rows if len(rows) <= sample else random.Random(0).sample(rows, sample)
    missing = [r for r in probe
               if not r.get("audio_path") or not Path(r["audio_path"]).exists()]
    if missing:
        log(f"[{corpus}] {len(missing)}/{len(probe)} sampled audio files are gone "
            f"-- rebuilding")
        return False

    log(f"[{corpus}] REUSING existing manifest: {h:.2f}h / {len(rows):,} rows, "
        f"audio present. Pass --force to rebuild.")
    return True


# ============================================================================
# 3 . LibriSpeech train-clean-100 (comparability anchor -- kept at 100h,
#     identical source/split to the 100h baseline model so the two runs are
#     comparable on dev-clean)
# ============================================================================


def _write_raw_audio(dest: Path, audio_cell: dict, fallback_name: str) -> tuple[Path, float]:
    """Write an UNDECODED audio cell straight to disk and return (path, seconds).

    Two reasons this does not decode:

    1. `datasets` 5.x delegates audio decoding to `torchcodec`; without it the
       Audio feature raises `ImportError: To support decoding audio data, please
       install 'torchcodec'`. Reading the cell with `Audio(decode=False)` gives
       the original encoded bytes and sidesteps that dependency entirely for the
       streaming corpora.
    2. We are only staging bytes for `build_cache.py` to consume later, so
       decode-then-re-encode would be pure waste.

    Duration comes from `soundfile.info()`, which reads the header only.
    """
    import soundfile as sf

    raw = audio_cell.get("bytes")
    src_name = audio_cell.get("path") or fallback_name
    suffix = Path(src_name).suffix or ".wav"
    dest = dest.with_suffix(suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        if not dest.exists():
            dest.write_bytes(raw)
    else:
        # Already-decoded cell (non-streaming path): fall back to re-encoding.
        if not dest.exists():
            sf.write(dest, audio_cell["array"], audio_cell["sampling_rate"], format="FLAC")
    info = sf.info(str(dest))
    return dest, float(info.frames) / float(info.samplerate)


def build_librispeech(out_dir: Path, cache_dir: Path, limit_hours: float = 100.0) -> Path:
    """Stream LibriSpeech train-clean-100.

    `load_dataset(..., "clean", split="train.100")` DOES NOT limit the download:
    the builder fetches and generates every split in the config, so a run that
    only wants 100 h was observed generating `train.360` (104,014 examples) and
    `validation` as well. `split=` filters what you get BACK, not what is
    fetched. Streaming reads shard by shard and stops when we stop iterating,
    which is the only reliable way to avoid paying for train.360.

    Streamed rows cannot be re-fetched by integer index later, so the encoded
    audio bytes are staged to disk here and the manifest carries a real
    `audio_path` — the same arrangement AMI uses.
    """
    if manifest_is_complete("librispeech", out_dir):
        return out_dir / "manifest_librispeech.jsonl"

    from datasets import load_dataset, Audio

    log("[librispeech] streaming openslr/librispeech_asr (clean, train.100)...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100",
                      streaming=True).cast_column("audio", Audio(decode=False))
    audio_root = Path(cache_dir) / "librispeech_audio"

    rows, oov_drop, kept_s = [], 0, 0.0
    for i, row in enumerate(ds):
        text = normalize_text(row["text"])
        if text is None:
            oov_drop += 1
            continue
        dest, dur = _write_raw_audio(audio_root / f"ls_{i:07d}", row["audio"], f"ls_{i}.flac")
        rows.append({
            "corpus": "librispeech",
            "source": "openslr/librispeech_asr:train.100",
            "audio_path": str(dest),
            "text": text,
            "duration_s": dur,
            "speaker": str(row.get("speaker_id", "")),
        })
        kept_s += dur
        if kept_s / 3600.0 >= limit_hours:
            break

    n_total = len(rows) + oov_drop
    log(f"[librispeech] kept {hours(rows):.2f}h / {len(rows)} rows | "
        f"OOV drop rate {oov_drop / max(1, n_total):.4f} ({oov_drop} rows)")

    out = out_dir / "manifest_librispeech.jsonl"
    write_manifest(rows, out)
    return out


# ============================================================================
# 4 . Common Voice 22 EN -- accent-stratified, snapshot_download only
#     (datasets v4.0 removed script-based loaders; fsicoli/common_voice_22_0
#     ships a loading script and load_dataset() will fail outright. Also the
#     full repo is 578 GB across 100+ languages -- never pull more than the
#     English shards.)
# ============================================================================

CV_REPO = "fsicoli/common_voice_22_0"
CV_TARGET_HOURS = 106.0
CV_PER_ACCENT_CAP_HOURS = 12.0  # no single accent bucket may dominate the 106h


_CV_ACCENT_SPLIT = re.compile(r",(?![^(]*\))")


def _cv_accent_list(row: dict) -> list[str]:
    """accent field may be called 'accent' or 'accents', singular or comma
    separated. Normalise to a list of lowercase, stripped labels.

    Splitting on a bare comma is WRONG here: Common Voice uses parenthesised
    groups that contain commas, e.g.
        "India and South Asia (India, Pakistan, Sri Lanka)"
    A naive split shreds that into three bogus accents, which is exactly what a
    run showed -- 'india and south asia (india', 'pakistan' and 'sri lanka)'
    each appearing with an identical count of 110,195. The lookahead below only
    splits on commas that are NOT inside parentheses.
    """
    raw = row.get("accents") if row.get("accents") else row.get("accent")
    if not raw:
        return ["unknown"]
    parts = [p.strip().lower() for p in _CV_ACCENT_SPLIT.split(str(raw)) if p.strip()]
    return parts or ["unknown"]


def download_common_voice_en(cache_dir: Path) -> Path:
    """Fetch ONLY the English shards of Common Voice 22 via snapshot_download
    (allow_patterns), never load_dataset(). Returns the local directory that
    contains the English validated.tsv + clip archives.

    NOTE: the exact repo layout could not be verified from this environment
    (no network access here). allow_patterns is intentionally broad/redundant
    across a few plausible layouts (transcript/en/*, audio/en/**, en/**) so
    that whichever one the repo actually uses is matched; run with
    HF_HUB_VERBOSITY=info the first time and inspect `local` if it pulls
    unexpected extra files, then tighten the patterns.
    """
    from huggingface_hub import snapshot_download

    log("[common_voice] snapshot_download, English shards only "
        "(repo is 578 GB total across 100+ languages -- do NOT pull more)")
    local = snapshot_download(
        repo_id=CV_REPO,
        repo_type="dataset",
        allow_patterns=[
            "transcript/en/*",
            "transcript/en.tsv",
            "*/en/*",           # covers repo layouts that nest audio/ per split
            "audio/en/**",
            "en/**",
            "*en_validated*",
            "*en_clips*",
        ],
        local_dir=str(cache_dir / "common_voice_en_raw"),
    )
    log(f"[common_voice] snapshot at {local}")
    return Path(local)


def _find_cv_tsv(root: Path) -> Path:
    cands = list(root.rglob("validated.tsv")) or list(root.rglob("*validated*.tsv"))
    if not cands:
        raise FileNotFoundError(
            f"No validated.tsv found under {root} -- Common Voice repo layout "
            "may have changed; inspect the snapshot manually and adjust "
            "allow_patterns / this lookup."
        )
    return cands[0]


_CV_AUDIO_EXT = (".mp3", ".wav", ".flac", ".opus", ".ogg", ".m4a")


def _cv_build_audio_index(root: Path) -> tuple[dict, dict]:
    """Index every clip in the snapshot ONCE. Returns (loose, in_tar).

    Two problems this replaces:

    1. The old `_find_cv_audio` did `root.rglob(clip_name)` PER ROW -- a full
       recursive tree walk for every one of ~2M transcript rows. Even when it
       worked it would take days.
    2. It only ever looked for loose files. The Hugging Face Common Voice
       mirrors ship audio inside `.tar` shards (`audio/en/train/en_train_0.tar`
       and friends), so nothing is on disk under its clip name and every lookup
       returned None -- a run reported "missing audio 809" and kept 0 rows.

    `loose`  maps clip name -> Path on disk.
    `in_tar` maps clip name -> (tar path, member name), read lazily later so we
    only ever extract the clips we actually keep.
    """
    loose: dict[str, Path] = {}
    in_tar: dict[str, tuple[Path, str]] = {}

    tars = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in _CV_AUDIO_EXT:
            loose.setdefault(p.name, p)
        elif p.suffix.lower() in (".tar", ".tgz") or p.name.endswith(".tar.gz"):
            tars.append(p)

    for tp in tars:
        try:
            with tarfile.open(tp) as tf:
                for m in tf.getmembers():
                    if m.isfile() and Path(m.name).suffix.lower() in _CV_AUDIO_EXT:
                        # Store the TarInfo itself, not its name: extractfile()
                        # then needs no getmember() lookup at extraction time.
                        in_tar.setdefault(Path(m.name).name, (tp, m))
        except Exception as exc:
            log(f"[common_voice] WARN could not index {tp.name}: {type(exc).__name__}: {exc}")

    log(f"[common_voice] audio index: {len(loose)} loose files, "
        f"{len(in_tar)} inside {len(tars)} tar archive(s)")
    if not loose and not in_tar:
        # Say exactly what IS there, so the failure is diagnosable instead of
        # just "missing audio N".
        tops = sorted({p.relative_to(root).parts[0] for p in root.iterdir()})
        exts = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
        log(f"[common_voice] NO AUDIO FOUND under {root}")
        log(f"[common_voice]   top-level entries: {tops}")
        log(f"[common_voice]   file extensions  : {dict(exts.most_common(10))}")
        log("[common_voice]   -> the snapshot probably fetched transcripts but not the "
            "audio shards. Widen allow_patterns (audio/en/**, *.tar) and re-run.")
    return loose, in_tar


# Open TarFile handles, keyed by path. Opening a tar means parsing its whole
# member index, and these shards hold ~38,000 members each. The first version
# of _cv_stage_audio did `with tarfile.open(...)` PER CLIP, so pulling the
# ~76,000 clips that make up 106 h cost ~2.9 BILLION member-index scans -- the
# run simply appeared to hang. Holding the handles open turns each extraction
# into a seek and a read.
_CV_TAR_HANDLES: dict = {}


def _cv_tar(tar_path: Path):
    tf = _CV_TAR_HANDLES.get(tar_path)
    if tf is None:
        tf = tarfile.open(tar_path)
        _CV_TAR_HANDLES[tar_path] = tf
    return tf


def _cv_close_tars() -> None:
    for tf in _CV_TAR_HANDLES.values():
        try:
            tf.close()
        except Exception:
            pass
    _CV_TAR_HANDLES.clear()


def _cv_stage_audio(clip_name: str, loose: dict, in_tar: dict, stage_dir: Path):
    """Return a real on-disk path for one clip, extracting from tar if needed.

    Only clips we actually keep get extracted, so the ~106 h we want does not
    cost us an extraction of the entire English set.
    """
    if clip_name in loose:
        return loose[clip_name]
    entry = in_tar.get(clip_name)
    if entry is None:
        return None
    tar_path, member = entry
    dest = stage_dir / clip_name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = _cv_tar(tar_path).extractfile(member)
        if src is None:
            return None
        dest.write_bytes(src.read())
        return dest
    except Exception:
        return None


def build_common_voice(out_dir: Path, cache_dir: Path, target_hours: float = CV_TARGET_HOURS,
                        per_accent_cap_hours: float = CV_PER_ACCENT_CAP_HOURS,
                        seed: int = 1337) -> Path:
    if manifest_is_complete("common_voice", out_dir):
        return out_dir / "manifest_common_voice.jsonl"

    import soundfile as sf

    root = download_common_voice_en(cache_dir)
    tsv_path = _find_cv_tsv(root)
    log(f"[common_voice] parsing {tsv_path}")

    # Build the clip index ONCE (see _cv_build_audio_index for why per-row
    # rglob was both wrong and unusably slow).
    cv_loose, cv_in_tar = _cv_build_audio_index(root)
    cv_stage = Path(cache_dir) / "common_voice_audio"
    if not cv_loose and not cv_in_tar:
        raise RuntimeError(
            "Common Voice snapshot contains no audio -- see the diagnostic above. "
            "Refusing to write an empty manifest silently."
        )

    # Keep a 4-field TUPLE per row, not the whole csv.DictReader dict. The
    # English validated.tsv has ~2M rows and a dozen columns; holding every
    # dict costs multiple GB and, on a memory-tight box, sends the process into
    # swap -- which looks exactly like "it hangs with no output".
    by_accent: dict[str, list[tuple]] = defaultdict(list)
    _t0 = time.perf_counter()
    n_rows = 0
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            accents = _cv_accent_list(row)
            slim = (row.get("path") or row.get("filename") or "",
                    row.get("sentence") or row.get("text") or "",
                    row.get("client_id", ""),
                    tuple(accents))
            # a row with multiple accents is filed under each -- sampling
            # still respects the per-accent cap since we draw independently
            # below; duplicate audio across buckets is fine, deduped at end
            for acc in accents:
                by_accent[acc].append(slim)
    log(f"[common_voice] parsed {n_rows:,} rows into {len(by_accent)} accent buckets "
        f"({time.perf_counter() - _t0:.0f}s)")

    hist_raw = {k: len(v) for k, v in sorted(by_accent.items(), key=lambda kv: -len(kv[1]))}
    log(f"[common_voice] raw accent histogram (row counts, top 15): "
        f"{dict(list(hist_raw.items())[:15])}")

    rng = random.Random(seed)
    _t0 = time.perf_counter()
    for acc in by_accent:
        rng.shuffle(by_accent[acc])
    log(f"[common_voice] shuffled buckets ({time.perf_counter() - _t0:.0f}s)")
    log(f"[common_voice] selecting clips (target {target_hours:.0f}h, "
        f"per-accent cap {per_accent_cap_hours:.0f}h)...")

    # Round-robin across accent buckets so the natural US/England mass doesn't
    # crowd out the long tail: pull one clip at a time per accent, respecting
    # both the per-accent hour cap and the global target.
    picked: dict[str, dict] = {}         # path -> row  (dedup)
    picked_secs: dict[str, float] = defaultdict(float)
    cap_s = per_accent_cap_hours * 3600.0
    target_s = target_hours * 3600.0
    cursors = {acc: 0 for acc in by_accent}
    accents_order = list(by_accent.keys())
    total_s = 0.0
    oov_drop, dur_drop, missing_audio = 0, 0, 0
    _t_pick = time.perf_counter()

    while total_s < target_s and accents_order:
        progressed = False
        for acc in list(accents_order):
            if picked_secs[acc] >= cap_s:
                if acc in accents_order:
                    accents_order.remove(acc)
                continue
            idx = cursors[acc]
            bucket = by_accent[acc]
            if idx >= len(bucket):
                if acc in accents_order:
                    accents_order.remove(acc)
                continue
            path, sentence, client_id, row_accents = bucket[idx]
            cursors[acc] += 1
            if not path or path in picked:
                continue
            audio_file = _cv_stage_audio(Path(path).name, cv_loose, cv_in_tar, cv_stage)
            if audio_file is None:
                missing_audio += 1
                continue
            try:
                info = sf.info(str(audio_file))
                dur = info.frames / info.samplerate
            except Exception:
                missing_audio += 1
                continue
            if dur <= 0 or dur > 30.0:
                dur_drop += 1
                continue
            text = normalize_text(sentence)
            if text is None:
                oov_drop += 1
                continue
            picked[path] = {
                "corpus": "common_voice",
                "source": CV_REPO,
                "audio_path": str(audio_file),
                "text": text,
                "duration_s": dur,
                "accents": list(row_accents),
                "speaker": client_id,
            }
            picked_secs[acc] += dur
            total_s += dur
            progressed = True
            if len(picked) % 2000 == 0:
                log(f"  [common_voice] {len(picked):,} clips staged, "
                    f"{total_s / 3600.0:.2f}h / {target_hours:.0f}h "
                    f"({time.perf_counter() - _t_pick:.0f}s)")
            if total_s >= target_s:
                break
        if not progressed:
            break  # exhausted every bucket before hitting the target

    rows = list(picked.values())
    hist_kept = Counter()
    for r in rows:
        for acc in r["accents"]:
            hist_kept[acc] += 1
    log(f"[common_voice] KEPT accent histogram: {dict(hist_kept.most_common(20))}")
    log(f"[common_voice] kept {hours(rows):.2f}h / {len(rows)} rows "
        f"(target {target_hours}h, per-accent cap {per_accent_cap_hours}h) | "
        f"OOV drop {oov_drop} | duration-filter drop {dur_drop} | "
        f"missing audio {missing_audio}")

    _cv_close_tars()

    out = out_dir / "manifest_common_voice.jsonl"
    write_manifest(rows, out)
    (out_dir / "common_voice_accent_histogram.json").write_text(
        json.dumps({"raw": hist_raw, "kept": dict(hist_kept.most_common())}, indent=2))
    return out


# ============================================================================
# 5 . AMI -- ihm (25h) + sdm (25h), DISJOINT meetings, four filters
# ============================================================================

AMI_REPO = "edinburghcstr/ami"
AMI_TARGET_HOURS_PER_MIC = 25.0

# (a) CTC feasibility: backbone runs at 50 frames/sec; CTC needs >=2 frames
# per target char (blank-separated repeats). duration_s*50 >= 2*len(text)
# i.e. roughly 40ms/char. This is the filter that PREVENTS crashes (inf/nan
# CTC loss from single-frame 0.02s clips) -- mandatory, not optional.
AMI_FRAMES_PER_SEC = 50.0
AMI_MIN_FRAMES_PER_CHAR = 2.0

# (b) pure filler stoplist -- backchannels with no lexical content.
# Entries are written with hyphens for readability, but normalize_text()
# turns "-" into a space (AMI also spells letters as "S. S. H." the same
# way), so the stoplist is matched against a hyphen-free/space-free form of
# both the stoplist and the candidate text -- otherwise "MM-HMM" in this set
# would never match the normalised "MM HMM" and the filter would silently
# no-op on exactly the entries that need the hyphen written out.
AMI_FILLER_STOPLIST = {
    "MM", "MMM", "HMM", "HM", "UH", "UM", "ERM", "AH", "OH", "EH",
    "MM-HMM", "UH-HUH",
}
_AMI_FILLER_COMPACT = {w.replace("-", "").replace(" ", "") for w in AMI_FILLER_STOPLIST}

# (c) short real words MUST survive (demo will contain them; they teach the
# model to emit little when little was said, i.e. anti-hallucination signal).
# Never zero them out -- only subsample to this keep-rate so they don't
# dominate the corpus. Named constant per spec.
AMI_SHORT_WORD_KEEP_RATE = 0.30
AMI_SHORT_WORD_MAX_CHARS = 12  # heuristic: "single short word" ballpark

# (d) character-rate sanity -- catches truncated automatic-alignment segments
AMI_MIN_CPS = 2.0
AMI_MAX_CPS = 25.0


def _ami_passes_filters(text: str, duration_s: float, rng: random.Random,
                         stats: Counter) -> bool:
    # normalise AMI's uppercase/apostrophe text through the shared normaliser
    # first so downstream stats operate on the same string CTC will train on
    n = normalize_text(text)
    if n is None:
        stats["oov"] += 1
        return False

    # (a) CTC feasibility -- mandatory
    if duration_s * AMI_FRAMES_PER_SEC < AMI_MIN_FRAMES_PER_CHAR * len(n.replace(" ", "")):
        stats["ctc_infeasible"] += 1
        return False

    # (b) pure filler stoplist (compared hyphen/space-insensitively -- see
    # _AMI_FILLER_COMPACT comment above)
    if n.replace(" ", "") in _AMI_FILLER_COMPACT:
        stats["filler"] += 1
        return False

    # (d) character-rate sanity (do this before the short-word keep so short
    # AND garbled segments are dropped for the right reason)
    cps = len(n) / max(duration_s, 1e-6)
    if not (AMI_MIN_CPS <= cps <= AMI_MAX_CPS):
        stats["bad_char_rate"] += 1
        return False

    # (c) keep short real words, but subsample so they don't dominate
    n_words = len(n.split())
    if n_words <= 2 and len(n) <= AMI_SHORT_WORD_MAX_CHARS:
        if rng.random() > AMI_SHORT_WORD_KEEP_RATE:
            stats["short_word_subsampled"] += 1
            return False
        stats["short_word_kept"] += 1
        return True

    stats["kept_normal"] += 1
    return True


def _ami_mic_for_meeting(meeting_id: str) -> str:
    """Deterministically assign a meeting to exactly one microphone config.

    Being a total function of `meeting_id` alone, this makes the ihm/sdm split
    disjoint BY CONSTRUCTION — no meeting can land on both sides, and there is
    no need to enumerate every meeting id up front. That matters: the previous
    implementation called `sorted(set(ds["meeting_id"]))`, which materialises a
    whole column and therefore forces the ENTIRE split to download before a
    single row is kept.
    """
    h = hashlib.md5(meeting_id.encode("utf-8")).hexdigest()
    return "ihm" if int(h[:8], 16) % 2 == 0 else "sdm"


def build_ami(out_dir: Path, cache_dir: Path,
              hours_per_mic: float = AMI_TARGET_HOURS_PER_MIC, seed: int = 1337) -> Path:
    """Stream AMI and stop as soon as the hour budget is met.

    We want 25 h from each mic config out of roughly 80 h per split, so a
    non-streaming `load_dataset(..., split="train")` downloads about 160 h of
    audio to keep 50 — around 69% of the bytes are fetched and thrown away.
    Streaming reads shard by shard and stops when we stop iterating.

    Streaming also removes a second, quieter problem: the old code shuffled an
    index list and then did `ds[i]` random access over a parquet-backed dataset,
    which defeats sequential reads and thrashes.

    Because streamed rows cannot be re-fetched later by integer index, the audio
    is decoded and written to FLAC here, and the manifest carries a real
    `audio_path`. AMI rows therefore behave like Common Voice / VCTK rows, and
    `build_cache.py` no longer needs an `hf_index` loader for them. ~50 h of
    16 kHz FLAC is roughly 3 GB.
    """
    if manifest_is_complete("ami", out_dir):
        return out_dir / "manifest_ami.jsonl"

    from datasets import load_dataset, Audio

    rng = random.Random(seed)
    audio_root = Path(cache_dir) / "ami_audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    def _collect(mic_tag: str, budget_hours: float):
        log(f"[ami] streaming {AMI_REPO}:{mic_tag} (stop at {budget_hours:.1f}h)...")
        # decode=False -> raw encoded bytes, no torchcodec dependency (datasets
        # 5.x routes Audio decoding through torchcodec and raises ImportError
        # without it).
        ds = load_dataset(AMI_REPO, mic_tag, split="train",
                          streaming=True).cast_column("audio", Audio(decode=False))
        rows, stats, kept_s, seen = [], Counter(), 0.0, 0
        for row in ds:
            seen += 1
            if kept_s / 3600.0 >= budget_hours:
                break
            mid = row["meeting_id"]
            if _ami_mic_for_meeting(mid) != mic_tag:
                stats["other_mic_partition"] += 1
                continue
            audio = row["audio"]
            dur = float(row.get("end_time", 0.0) or 0.0) - float(row.get("begin_time", 0.0) or 0.0)
            text = row["text"]
            if not _ami_passes_filters(text, dur, rng, stats):
                continue
            norm = normalize_text(text)
            if norm is None:
                stats["oov_dropped"] += 1
                continue

            dest, real_dur = _write_raw_audio(
                audio_root / mic_tag / str(row["audio_id"]), audio, f"{row['audio_id']}.wav")
            if dur <= 0:
                dur = real_dur

            rows.append({
                "corpus": "ami",
                "source": f"{AMI_REPO}:{mic_tag}",
                "audio_path": str(dest),
                "mic": mic_tag,
                "meeting_id": mid,
                "speaker": row.get("speaker_id", ""),
                "text": norm,
                "duration_s": dur,
            })
            kept_s += dur
        log(f"[ami] {mic_tag}: streamed {seen} rows to reach {kept_s / 3600.0:.2f}h")
        return rows, stats

    rows_ihm, stats_ihm = _collect("ihm", hours_per_mic)
    rows_sdm, stats_sdm = _collect("sdm", hours_per_mic)

    used_meetings_ihm = {r["meeting_id"] for r in rows_ihm}
    used_meetings_sdm = {r["meeting_id"] for r in rows_sdm}
    assert used_meetings_ihm.isdisjoint(used_meetings_sdm), (
        "AMI disjointness violated after collection -- same meeting_id ended "
        "up on both ihm and sdm sides."
    )

    rows = rows_ihm + rows_sdm
    log(f"[ami] ihm: kept {hours(rows_ihm):.2f}h / {len(rows_ihm)} rows | filters={dict(stats_ihm)}")
    log(f"[ami] sdm: kept {hours(rows_sdm):.2f}h / {len(rows_sdm)} rows | filters={dict(stats_sdm)}")
    log(f"[ami] TOTAL kept {hours(rows):.2f}h / {len(rows)} rows | "
        f"disjoint meetings: ihm={len(used_meetings_ihm)} sdm={len(used_meetings_sdm)}")

    out = out_dir / "manifest_ami.jsonl"
    write_manifest(rows, out)
    return out


# ============================================================================
# 6 . VCTK -- accent, studio-clean, standard HF loader
# ============================================================================

VCTK_TARGET_HOURS = 44.0


def build_vctk(out_dir: Path, cache_dir: Path, target_hours: float = VCTK_TARGET_HOURS) -> Path:
    if manifest_is_complete("vctk", out_dir):
        return out_dir / "manifest_vctk.jsonl"

    from datasets import load_dataset, Audio

    log("[vctk] loading VCTK...")
    # HF mirrors vary in repo id; try the common ones in order. Verify the
    # correct one on huggingface.co before a real run (no network here).
    ds, used_repo, attempts = None, None, []
    for repo in ("CSTR-Edinburgh/vctk", "vctk", "sanchit-gandhi/vctk"):
        try:
            ds = load_dataset(repo, split="train",
                              streaming=True).cast_column("audio", Audio(decode=False))
            used_repo = repo
            log(f"[vctk] loaded from {repo}")
            break
        except Exception as e:
            # Log the MESSAGE, not just the exception class. A bare
            # "failed (RuntimeError)" hides whether the repo is gated, renamed,
            # or script-based -- and script-based is the failure we keep hitting,
            # since datasets v4+ removed loading scripts entirely.
            attempts.append(f"{repo}: {type(e).__name__}: {e}")
            log(f"[vctk] {repo} failed -- {type(e).__name__}: {str(e)[:200]}")
    if ds is None:
        raise RuntimeError(
            "Could not load VCTK from any known HF repo id.\n  " + "\n  ".join(attempts)
        )

    # Which repo answered matters: these are third-party mirrors with different
    # schemas, so confirm the columns we depend on actually exist BEFORE
    # streaming 44 h. Without this, a mirror whose transcript column is named
    # something else yields empty text for every row, normalize_text() returns
    # None each time, and the run ends with "kept 0.00h" and an OOV drop rate of
    # 1.00 -- looking like a normalisation problem rather than a schema problem.
    _probe = next(iter(ds))
    _cols = sorted(_probe.keys())
    _TEXT_KEYS = ("text", "sentence", "transcription", "normalized_text", "transcript")
    _text_key = next((k for k in _TEXT_KEYS if _probe.get(k)), None)
    if _text_key is None:
        raise RuntimeError(
            f"VCTK mirror {used_repo!r} has no recognisable transcript column.\n"
            f"  available columns: {_cols}\n"
            f"  tried: {list(_TEXT_KEYS)}\n"
            "  Add the right column name to _TEXT_KEYS, or use a different mirror."
        )
    if "audio" not in _probe:
        raise RuntimeError(
            f"VCTK mirror {used_repo!r} has no 'audio' column. Columns: {_cols}")
    log(f"[vctk] repo={used_repo} columns={_cols} -> using transcript column {_text_key!r}")

    # Same raw-bytes staging as LibriSpeech/AMI. VCTK is ~44h and we want ~44h,
    # so there is no download to save here -- but keeping ONE audio-access
    # convention across every corpus means `build_cache.py` never needs an
    # hf_index loader, and `datasets` 5.x's torchcodec decode path is avoided.
    audio_root = Path(cache_dir) / "vctk_audio"
    rows, oov_drop, kept_s = [], 0, 0.0
    for i, row in enumerate(ds):
        if kept_s / 3600.0 >= target_hours:
            break
        text_field = row.get(_text_key) or ""
        text = normalize_text(text_field)
        if text is None:
            oov_drop += 1
            continue
        dest, dur = _write_raw_audio(audio_root / f"vctk_{i:06d}", row["audio"], f"vctk_{i}.wav")
        rows.append({
            "corpus": "vctk",
            "source": "vctk",
            "audio_path": str(dest),
            "speaker": str(row.get("speaker_id", row.get("speaker", ""))),
            "text": text,
            "duration_s": dur,
        })
        kept_s += dur

    n_total = len(rows) + oov_drop
    log(f"[vctk] kept {hours(rows):.2f}h / {len(rows)} rows | "
        f"OOV drop rate {oov_drop / max(1, n_total):.4f}")

    out = out_dir / "manifest_vctk.jsonl"
    write_manifest(rows, out)
    return out


# ============================================================================
# 7 . Combine
# ============================================================================


def combine(out_dir: Path) -> Path:
    parts = ["manifest_librispeech.jsonl", "manifest_common_voice.jsonl",
             "manifest_ami.jsonl", "manifest_vctk.jsonl"]
    rows = []
    for p in parts:
        fp = out_dir / p
        if not fp.exists():
            log(f"[combine] WARNING: {fp} missing, skipping")
            continue
        with fp.open() as f:
            part_rows = [json.loads(l) for l in f if l.strip()]
        rows.extend(part_rows)
        log(f"[combine] {p}: {hours(part_rows):.2f}h / {len(part_rows)} rows")

    # ------------------------------------------------------------------
    # Global trainability gate.
    #
    # Per-corpus filters vary (Common Voice caps at 30 s, LibriSpeech has no
    # duration filter at all, AMI has its own four-rule filter), so a few
    # untrainable rows always slip through the seams. A verification run over
    # the real 300 h found exactly that: 7 CTC-infeasible rows, 248 clips under
    # 0.2 s, and 3 LibriSpeech utterances over 30 s.
    #
    # Applying the gate HERE rather than patching four builders means there is
    # ONE choke point every row must pass, and it enforces exactly what
    # verify_data.py checks -- so the two can never drift apart.
    #
    #   duration in [0.2, 30] s   : sub-0.2 s clips carry no usable speech and
    #                               >30 s blows up padded batch memory
    #   duration*50 >= 2*len(text): CTC needs ~2 frames per target character;
    #                               violating it yields inf/nan loss, which
    #                               poisons training rather than just wasting a
    #                               step
    # ------------------------------------------------------------------
    MIN_DUR, MAX_DUR = 0.2, 30.0
    before = len(rows)
    drop_stats = Counter()
    kept = []
    for r in rows:
        d = float(r.get("duration_s", 0.0))
        t = r.get("text") or ""
        if d < MIN_DUR:
            drop_stats[f"{r['corpus']}: too short (<{MIN_DUR}s)"] += 1
            continue
        if d > MAX_DUR:
            drop_stats[f"{r['corpus']}: too long (>{MAX_DUR}s)"] += 1
            continue
        if d * 50.0 < 2 * len(t):
            drop_stats[f"{r['corpus']}: CTC infeasible"] += 1
            continue
        kept.append(r)
    rows = kept
    if drop_stats:
        log(f"[combine] trainability gate dropped {before - len(rows)} rows "
            f"({(before - len(rows)) / before:.3%}):")
        for reason, n in sorted(drop_stats.items(), key=lambda kv: -kv[1]):
            log(f"           {n:>6,}  {reason}")
    else:
        log("[combine] trainability gate: nothing to drop")

    log(f"[combine] TOTAL {hours(rows):.2f}h / {len(rows)} rows "
        f"(target 300h: 100 librispeech + 106 common_voice + 50 ami + 44 vctk)")

    combined_path = out_dir / "manifest_combined.jsonl"
    write_manifest(rows, combined_path)

    by_corpus = Counter(r["corpus"] for r in rows)
    stats = {"total_hours": hours(rows), "total_rows": len(rows),
             "by_corpus_rows": dict(by_corpus),
             "by_corpus_hours": {c: hours([r for r in rows if r["corpus"] == c])
                                 for c in by_corpus}}
    (out_dir / "manifest_combined.stats.json").write_text(json.dumps(stats, indent=2))
    return combined_path


# ============================================================================
# 8 . CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["librispeech", "common_voice", "ami", "vctk"])
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a complete manifest already exists")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--out", default="/marimo/data")
    ap.add_argument("--cache", default="/marimo/cache")
    args = ap.parse_args()

    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    global FORCE_REBUILD
    FORCE_REBUILD = bool(getattr(args, 'force', False))
    if FORCE_REBUILD:
        log('[prepare] --force: ignoring existing manifests')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.corpus == "librispeech":
        build_librispeech(out_dir, cache_dir)
    elif args.corpus == "common_voice":
        build_common_voice(out_dir, cache_dir)
    elif args.corpus == "ami":
        build_ami(out_dir, cache_dir)
    elif args.corpus == "vctk":
        build_vctk(out_dir, cache_dir)
    if args.combine:
        combine(out_dir)
    log(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
'''
    (asr_dir / "prepare_data.py").write_text(prepare_data_code, encoding="utf-8")

    # 2. augment.py (verbatim port of _Staj/asr/augment.py)
    augment_code = r'''# /// script
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
'''
    (asr_dir / "augment.py").write_text(augment_code, encoding="utf-8")

    fetch_noise_banks_code = r'''# /// script
# requires-python = ">=3.11"
# dependencies = []   # stdlib only: urllib + tarfile + zipfile
# ///
"""Download the background-noise and room-impulse-response banks.

WHY THIS FILE EXISTS
--------------------
The notebook created `noise/` and `rir/` directories, wired `--noise-dir` and
`--rir-dir` through to the trainer, and `augment.py` read them — but nothing
ever populated them, and `AudioBank` treats an empty folder as "no bank":

    if not self.rir_bank.empty()   and rng.random() < cfg.p_rir:   ...
    if not self.noise_bank.empty() and rng.random() < cfg.p_noise: ...

So with empty folders the reverb and additive-noise effects are skipped in
silence. Training completes, the log looks healthy, and the two augmentation
axes that matter MOST for the Friday demo (a laptop microphone in a room, not a
studio) were never applied at all.

WHAT IT FETCHES
---------------
  OpenSLR-28  rirs_noises.zip   ~4 GB   simulated + real RIRs, plus pointsource
                                        and isotropic noises. Highest value per
                                        byte: it covers BOTH axes on its own.
  OpenSLR-17  musan.tar.gz      ~11 GB  we keep only `musan/noise/**` (~6 h,
                                        930 files). The archive is monolithic so
                                        the whole thing crosses the wire, but
                                        members are filtered while streaming so
                                        only the noise subset ever hits disk.

MUSAN is therefore OPTIONAL: 11 GB of transfer for ~6 h of extra noise variety,
on top of what OpenSLR-28 already provides. Start with RIR only if bandwidth or
time is tight.

DEMAND is deliberately NOT automated here. It is distributed as per-scene Zenodo
archives and I could not verify the current URLs from this environment; guessing
at them would produce a downloader that fails at 3 a.m. Fetch it by hand into
the noise directory if you want those scenes.

Usage:
    python fetch_noise_banks.py --noise-dir /marimo/noise --rir-dir /marimo/rir
    python fetch_noise_banks.py --noise-dir ... --rir-dir ... --skip-musan
    python fetch_noise_banks.py --noise-dir ... --rir-dir ... --verify-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import shutil as _shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# OpenSLR is mirrored; the primary host is frequently slow, so try in order.
MUSAN_URLS = [
    "https://us.openslr.org/resources/17/musan.tar.gz",
    "https://www.openslr.org/resources/17/musan.tar.gz",
    "https://openslr.elda.org/resources/17/musan.tar.gz",
]
RIR_URLS = [
    "https://us.openslr.org/resources/28/rirs_noises.zip",
    "https://www.openslr.org/resources/28/rirs_noises.zip",
    "https://openslr.elda.org/resources/28/rirs_noises.zip",
]


def log(*a):
    print(*a, flush=True)


def count_wavs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.wav")) if root.exists() else 0


def _open_stream(urls: list[str], timeout: int = 60):
    """First URL that responds. Returns (response, url)."""
    last = None
    for url in urls:
        try:
            log(f"  trying {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "clear-asr/1.0"})
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp, url
        except Exception as exc:
            last = exc
            log(f"    failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"all mirrors failed; last error: {last}")



# ============================================================================
# Parallel download
# ============================================================================
#
# OpenSLR serves a single stream slowly, and MUSAN is ~11 GB, so a plain
# sequential read is a patience test. Splitting the file into byte ranges and
# fetching them concurrently is usually several times faster, because the
# bottleneck is per-connection throughput rather than total bandwidth.
#
# THE TRADE-OFF, stated plainly: the streaming path never puts the 11 GB tar on
# disk (it filters members as they arrive and keeps only ~6 h of noise wavs).
# Range-parallel downloading REQUIRES the whole archive on disk first, because
# ranges arrive out of order. So this buys wall-clock time at the cost of ~11 GB
# of temporary disk. The tar is deleted straight after extraction.
#
# Not all servers honour Range. We probe first and fall back to the streaming
# path rather than silently downloading the file 8 times or getting garbage.


def _supports_ranges(url: str, timeout: int = 30) -> tuple[bool, int]:
    """(accepts_ranges, content_length). Both are needed to split the work."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "clear-asr/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = int(r.headers.get("Content-Length") or 0)
            accepts = (r.headers.get("Accept-Ranges", "").lower() == "bytes")
            return (accepts and size > 0), size
    except Exception:
        return False, 0


def _download_ranges(url: str, dest: Path, size: int, jobs: int = 8) -> None:
    """Fetch `url` into `dest` using `jobs` concurrent byte-range requests."""
    chunk = size // jobs
    spans = [(i * chunk, (size - 1) if i == jobs - 1 else (i + 1) * chunk - 1)
             for i in range(jobs)]

    with open(dest, "wb") as fh:      # preallocate so each worker can seek
        fh.truncate(size)

    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def worker(span):
        nonlocal done
        start, end = span
        req = urllib.request.Request(
            url, headers={"User-Agent": "clear-asr/1.0", "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "r+b") as fh:
            fh.seek(start)
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
                with lock:
                    done += len(buf)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(worker, sp) for sp in spans]
        # Poll often, LOG rarely: a coarse sleep would put a hard floor under
        # every download (a 40 MB fetch took 5s purely because of a sleep(5)).
        last_log = 0.0
        while any(not f.done() for f in futures):
            time.sleep(0.25)
            now = time.time()
            if now - last_log < 5.0:
                continue
            last_log = now
            with lock:
                d = done
            el = max(now - t0, 1e-6)
            eta = (size - d) / max(d / el, 1.0)
            log(f"  [dl] {d / 1e9:.2f}/{size / 1e9:.2f} GB ({100 * d / size:.0f}%) "
                f"@ {d / 1e6 / el:.0f} MB/s, {jobs} streams, ETA {eta / 60:.1f} min")
        for f in futures:
            f.result()        # surface any worker exception

    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"size mismatch: got {got:,} B, expected {size:,} B")
    log(f"  [dl] done: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")


def _download_aria2(url: str, dest: Path, jobs: int = 8) -> bool:
    """Use aria2c when it happens to be installed. Returns True on success.

    aria2c is strictly better than our thread pool when present -- it resumes,
    retries per-connection, and handles redirects -- but it is not installed by
    default on molab, so it is opportunistic rather than required.
    """
    if not _shutil.which("aria2c"):
        return False
    log(f"  [dl] aria2c found -- {jobs} connections")
    cmd = ["aria2c", "-x", str(jobs), "-s", str(jobs), "-k", "10M",
           "--console-log-level=warn", "--summary-interval=10",
           "--auto-file-renaming=false", "--allow-overwrite=true",
           "-d", str(dest.parent), "-o", dest.name, url]
    try:
        return subprocess.run(cmd).returncode == 0 and dest.exists()
    except Exception as exc:
        log(f"  [dl] aria2c failed ({exc}) -- falling back")
        return False


def download_file(urls: list[str], dest: Path, jobs: int = 8) -> Path:
    """Download the first working URL to `dest`, in parallel when possible."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  [dl] reusing existing {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    last = None
    for url in urls:
        log(f"  trying {url}")
        try:
            if _download_aria2(url, dest, jobs):
                return dest
            ok, size = _supports_ranges(url)
            if ok and jobs > 1:
                log(f"  [dl] server accepts ranges, {size / 1e9:.2f} GB "
                    f"-> {jobs} parallel streams")
                _download_ranges(url, dest, size, jobs)
                return dest
            log("  [dl] no range support -- single stream")
            resp, _ = _open_stream([url])
            with open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            return dest
        except Exception as exc:
            last = exc
            log(f"    failed: {type(exc).__name__}: {exc}")
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"all mirrors failed; last error: {last}")


def fetch_musan_noise(noise_dir: Path, jobs: int = 8) -> int:
    """Stream musan.tar.gz and extract ONLY `musan/noise/**`.

    Streaming ('r|gz') cannot seek, which is exactly what we want over HTTP: we
    walk members in order and write out just the noise subset, so the ~11 GB
    archive never lands on disk in full.
    """
    noise_dir.mkdir(parents=True, exist_ok=True)
    target = noise_dir / "musan_noise"
    if count_wavs(target) > 100:
        log(f"[musan] already populated ({count_wavs(target)} wavs) — skipping")
        return count_wavs(target)

    target.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.time()

    if jobs > 1:
        # Parallel path: the whole 11 GB archive lands on disk first (ranges
        # arrive out of order, so they cannot be piped through the tar reader),
        # then we extract only musan/noise/** and delete it again.
        tmp = noise_dir / "_musan.tar.gz"
        free_gb = shutil.disk_usage(noise_dir).free / 1e9
        if free_gb < 14:
            log(f"[musan] only {free_gb:.1f} GB free -- need ~12 GB for the parallel "
                "path, falling back to streaming (slower, no temp file)")
        else:
            download_file(MUSAN_URLS, tmp, jobs=jobs)
            log(f"[musan] extracting musan/noise/** from {tmp.name}")
            with tarfile.open(tmp, mode="r:gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name
                    if "/noise/" not in name or not name.endswith(".wav"):
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    with open(target / Path(name).name, "wb") as fh:
                        shutil.copyfileobj(src, fh)
                    n += 1
                    if n % 100 == 0:
                        log(f"  [musan] {n} noise wavs kept ({time.time() - t0:.0f}s)")
            tmp.unlink(missing_ok=True)
            log(f"[musan] extracted {n} noise wavs into {target} "
                f"(temp archive deleted)")
            return n

    resp, url = _open_stream(MUSAN_URLS)
    log(f"[musan] streaming from {url} (extracting musan/noise/** only)")
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if "/noise/" not in name or not name.endswith(".wav"):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            dest = target / Path(name).name
            with open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            n += 1
            if n % 50 == 0:
                log(f"  [musan] {n} noise wavs kept ({time.time() - t0:.0f}s)")
    log(f"[musan] extracted {n} noise wavs into {target}")
    return n


def _classify_slr28(name: str) -> str | None:
    """Is this OpenSLR-28 member a room impulse response, a noise, or neither?

    Matching on DIRECTORY COMPONENTS, not a substring of the whole path. The
    naive `"rir" in path` test is wrong here because the archive's top-level
    directory is literally `RIRS_NOISES`, so every pointsource NOISE file also
    contains "rir" and lands in the RIR bank — which is exactly the bug a
    synthetic-archive test caught: 7 files classified as RIRs, 0 as noise.

    Layout:
        RIRS_NOISES/simulated_rirs/{small,medium,large}room/RoomXXX/*.wav  -> rir
        RIRS_NOISES/real_rirs_isotropic_noises/*.wav                      -> mixed,
            decided by filename ("*rir*" vs "*noise*")
        RIRS_NOISES/pointsource_noises/*.wav                              -> noise
    """
    parts = [p.lower() for p in Path(name).parts]
    stem = Path(name).name.lower()
    if "pointsource_noises" in parts:
        return "noise"
    if "simulated_rirs" in parts:
        return "rir"
    if "real_rirs_isotropic_noises" in parts:
        return "rir" if "rir" in stem else "noise"
    return None


def fetch_rirs(rir_dir: Path, with_pointsource_noise: Path | None = None,
               jobs: int = 8) -> int:
    """Download OpenSLR-28 and extract the RIR wavs.

    If `with_pointsource_noise` is given, the archive's pointsource/isotropic
    noise wavs are extracted there too — free extra noise variety, since the
    bytes have already been paid for.
    """
    rir_dir.mkdir(parents=True, exist_ok=True)
    target = rir_dir / "rirs"
    if count_wavs(target) > 100:
        log(f"[rir] already populated ({count_wavs(target)} wavs) — skipping")
        return count_wavs(target)
    target.mkdir(parents=True, exist_ok=True)

    # zipfile needs random access, so this one has to land on disk first.
    # A zip needs random access, so this one always lands on disk -- which means
    # the parallel downloader costs nothing extra here and is pure win.
    tmp_zip = rir_dir / "_rirs_noises.zip"
    download_file(RIR_URLS, tmp_zip, jobs=jobs)

    n_rir, n_noise = 0, 0
    with zipfile.ZipFile(tmp_zip) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".wav"):
                continue
            kind = _classify_slr28(info.filename)
            if kind == "rir":
                dest, is_rir = target, True
            elif kind == "noise" and with_pointsource_noise is not None:
                dest, is_rir = with_pointsource_noise / "slr28_noise", False
            else:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            # Flatten: AudioBank globs recursively but flat names avoid
            # collisions between the simulated_rirs sub-trees.
            out = dest / (info.filename.replace("/", "_"))
            with z.open(info) as src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
            if is_rir:
                n_rir += 1
            else:
                n_noise += 1
    log(f"[rir] extracted {n_rir} RIR wavs into {target}"
        + (f", {n_noise} noise wavs alongside" if n_noise else ""))
    tmp_zip.unlink(missing_ok=True)
    return n_rir


def verify(noise_dir: Path, rir_dir: Path) -> bool:
    """Report bank sizes and say plainly whether augmentation will actually run."""
    n_noise, n_rir = count_wavs(noise_dir), count_wavs(rir_dir)
    log("")
    log("Augmentation bank status")
    log("=" * 30)
    log(f"  noise dir : {noise_dir}  ->  {n_noise} wav files")
    log(f"  rir dir   : {rir_dir}  ->  {n_rir} wav files")
    log("")
    if n_noise == 0:
        log("  ⚠ noise bank EMPTY — additive-noise augmentation will be SILENTLY SKIPPED")
    if n_rir == 0:
        log("  ⚠ rir bank EMPTY — reverb augmentation will be SILENTLY SKIPPED")
        log("    (this is the demo-critical one: a laptop mic in a room is reverberant)")
    ok = n_noise > 0 and n_rir > 0
    log("  READY — noise and reverb will both be applied." if ok else
        "  NOT READY — train_asr.py will run, but those effects will do nothing.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--rir-dir", required=True)
    ap.add_argument("--skip-musan", action="store_true",
                    help="skip the 11 GB MUSAN transfer; OpenSLR-28 still gives "
                         "RIRs plus its own pointsource noises")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel download connections (1 = old single-stream "
                         "path, which avoids the ~11 GB temp file for MUSAN)")
    args = ap.parse_args()

    noise_dir, rir_dir = Path(args.noise_dir), Path(args.rir_dir)

    if not args.verify_only:
        try:
            fetch_rirs(rir_dir, with_pointsource_noise=noise_dir, jobs=args.jobs)
        except Exception as exc:
            log(f"[rir] FAILED: {type(exc).__name__}: {exc}")
        if not args.skip_musan:
            try:
                fetch_musan_noise(noise_dir, jobs=args.jobs)
            except Exception as exc:
                log(f"[musan] FAILED: {type(exc).__name__}: {exc}")
        else:
            log("[musan] skipped (--skip-musan)")

    ok = verify(noise_dir, rir_dir)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
'''
    (asr_dir / "fetch_noise_banks.py").write_text(fetch_noise_banks_code, encoding="utf-8")

    verify_data_code = r'''# /// script
# requires-python = ">=3.11"
# dependencies = ["soundfile==0.14.0"]
# ///
"""Pre-flight check: is the 300h training data actually complete and usable?

Run this AFTER prepare_data.py / build_cache.py and BEFORE starting an
8-hour training run. Every failure mode it looks for is one that has already
bitten this pipeline at least once:

  * a corpus that silently produced 0 rows (Common Voice kept 0.00h because the
    audio lived inside tar shards nobody opened)
  * a manifest whose `audio_path` entries point at files that are not there
  * an hours total that quietly falls short of the 300h the slides claim
  * rows that violate the CTC length constraint -- `duration_s * 50 < 2 * len(text)`
    produces inf/nan loss and poisons training, and AMI genuinely contains
    0.02-second clips
  * out-of-vocabulary characters surviving normalisation, which would break
    KenLM decoding
  * L2-ARCTIC leaking into a training manifest (it is the held-out OOD test set)
  * a packed cache whose hour count does not match its manifest
  * empty noise/RIR banks, which make augmentation a silent no-op

Exit codes: 0 = ready to train, 2 = at least one FAIL.

Usage:
    python verify_data.py --data /marimo/data --cache /marimo/cache \\
        --noise-dir /marimo/noise --rir-dir /marimo/rir
    python verify_data.py --data ... --check-all-audio     # stat every file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
ALLOWED = CHARS | {" "}

# Backbone frame rate: 16 kHz audio -> 50 frames/sec. CTC needs roughly two
# frames per target character (blanks between repeats), so a row is infeasible
# when duration_s * 50 < 2 * len(text).
FRAMES_PER_SEC = 50.0
EXPECTED = {"librispeech": 100.0, "common_voice": 106.0, "ami": 50.0, "vctk": 44.0}
TOTAL_TARGET = 300.0
L2_MARKERS = ("l2-arctic", "l2_arctic", "l2arctic")

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def load_manifest(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def verify_manifests(data_dir: Path, check_all_audio: bool, sample_n: int) -> dict:
    print("\n1. Per-corpus manifests")
    per_corpus_rows: dict[str, list[dict]] = {}
    for corpus, want_h in EXPECTED.items():
        p = data_dir / f"manifest_{corpus}.jsonl"
        if not p.exists():
            check(f"{corpus}: manifest exists", False, f"missing {p.name}")
            continue
        rows = load_manifest(p)
        per_corpus_rows[corpus] = rows
        h = sum(r["duration_s"] for r in rows) / 3600.0
        # 10% short is a real shortfall, not rounding
        check(f"{corpus}: {h:.2f}h / {want_h:.0f}h target, {len(rows):,} rows",
              len(rows) > 0 and h >= want_h * 0.9,
              "" if h >= want_h * 0.9 else "more than 10% short")
    return per_corpus_rows


def verify_combined(data_dir: Path, per_corpus: dict) -> list[dict]:
    print("\n2. Combined manifest")
    p = data_dir / "manifest_combined.jsonl"
    if not p.exists():
        check("combined manifest exists", False, f"missing {p.name} — run --combine")
        return []
    rows = load_manifest(p)
    total_h = sum(r["duration_s"] for r in rows) / 3600.0
    check(f"combined total {total_h:.2f}h / {TOTAL_TARGET:.0f}h, {len(rows):,} rows",
          total_h >= TOTAL_TARGET * 0.9,
          "" if total_h >= TOTAL_TARGET * 0.9 else "more than 10% short")

    by = Counter(r["corpus"] for r in rows)
    missing = [c for c in EXPECTED if by.get(c, 0) == 0]
    check("all four corpora present in combined", not missing,
          f"absent: {missing}" if missing else f"{dict(by)}")

    # combine() applies a trainability gate (duration in [0.2, 30] s and CTC
    # feasibility), so the combined manifest is deliberately SMALLER than the sum
    # of its parts. Asserting equality here was my own bug: it contradicted a
    # filter I had just added. What actually matters is that combined never
    # EXCEEDS the parts, and that the gate removed a small fraction rather than
    # silently eating the dataset.
    sum_parts = sum(len(v) for v in per_corpus.values())
    dropped = sum_parts - len(rows)
    frac = dropped / sum_parts if sum_parts else 0.0
    check("combined row count <= sum of per-corpus", len(rows) <= sum_parts,
          f"combined {len(rows):,} EXCEEDS parts {sum_parts:,} — duplicated rows?"
          if len(rows) > sum_parts else "")
    check(f"trainability gate dropped {dropped:,} rows ({frac:.3%})", frac < 0.01,
          f"parts {sum_parts:,} -> combined {len(rows):,}"
          + ("  — over 1% is too much, check the combine log for the reason"
             if frac >= 0.01 else ""))
    return rows


def verify_rows(rows: list[dict], check_all_audio: bool, sample_n: int) -> None:
    if not rows:
        return

    print("\n3. L2-ARCTIC leakage gate")
    leaks = [r for r in rows
             if any(m in " ".join(str(r.get(k, "")) for k in
                                  ("corpus", "source", "audio_path", "speaker")).lower()
                    for m in L2_MARKERS)]
    check("no L2-ARCTIC rows in training data", not leaks,
          f"{len(leaks)} leaked rows!" if leaks else "held-out test set is clean")

    print("\n4. Text / vocabulary")
    oov, empty = [], 0
    for r in rows:
        t = r.get("text") or ""
        if not t:
            empty += 1
            continue
        bad = set(t) - ALLOWED
        if bad:
            oov.append((r["corpus"], sorted(bad)[:5], t[:40]))
    check("no empty transcripts", empty == 0, f"{empty} empty" if empty else "")
    check("no out-of-vocabulary characters", not oov,
          f"{len(oov)} rows, e.g. {oov[:3]}" if oov else "all rows fit A-Z + apostrophe")

    print("\n5. CTC feasibility (duration_s * 50 >= 2 * len(text))")
    infeasible = [r for r in rows
                  if r["duration_s"] * FRAMES_PER_SEC < 2 * len(r.get("text", ""))]
    by_corpus = Counter(r["corpus"] for r in infeasible)
    check("no rows would produce inf/nan CTC loss", not infeasible,
          f"{len(infeasible)} infeasible rows {dict(by_corpus)}" if infeasible
          else "every row has enough frames for its transcript")

    print("\n6. Duration sanity")
    durs = [r["duration_s"] for r in rows]
    tiny = sum(1 for d in durs if d < 0.2)
    huge = sum(1 for d in durs if d > 30.0)
    check("no sub-0.2s clips", tiny == 0, f"{tiny} clips shorter than 0.2s" if tiny else "")
    check("no clips over 30s", huge == 0, f"{huge} clips longer than 30s" if huge else "")
    print(f"        min {min(durs):.2f}s  median {sorted(durs)[len(durs)//2]:.2f}s  "
          f"max {max(durs):.2f}s")

    print("\n7. Audio files on disk")
    paths = [r.get("audio_path") for r in rows]
    dupes = len(paths) - len(set(paths))
    check("no duplicate audio_path", dupes == 0, f"{dupes} duplicates" if dupes else "")
    to_check = rows if check_all_audio else random.Random(0).sample(
        rows, min(sample_n, len(rows)))
    missing = [r["audio_path"] for r in to_check
               if not r.get("audio_path") or not Path(r["audio_path"]).exists()]
    label = f"all {len(to_check):,}" if check_all_audio else f"{len(to_check):,} sampled"
    check(f"audio present on disk ({label})", not missing,
          f"{len(missing)} missing, e.g. {missing[:2]}" if missing
          else "every checked file exists")

    print("\n8. Per-corpus breakdown")
    agg = defaultdict(lambda: [0, 0.0])
    for r in rows:
        agg[r["corpus"]][0] += 1
        agg[r["corpus"]][1] += r["duration_s"]
    for c, (n, s) in sorted(agg.items()):
        print(f"        {c:14s} {n:>8,} rows  {s / 3600.0:7.2f}h  "
              f"mean {s / max(n, 1):.2f}s")

    accents = Counter()
    for r in rows:
        for a in r.get("accents", []) or []:
            accents[a] += 1
    if accents:
        print(f"\n        Common Voice accents kept: {len(accents)} distinct")
        for a, n in accents.most_common(8):
            print(f"          {a[:52]:54s} {n:>7,}")


def verify_cache(cache_dir: Path, rows: list[dict]) -> None:
    print("\n9. Packed cache")
    caches = sorted(cache_dir.glob("combined_*"))
    if not caches:
        check("packed cache exists", False, "no combined_* dir — run build_cache.py")
        return
    d = caches[-1]
    bin_p, meta_p = d / "audio.i16", d / "meta.json"
    if not (bin_p.exists() and meta_p.exists()):
        check("cache files present", False, f"{d.name} missing audio.i16 or meta.json")
        return
    meta = json.loads(meta_p.read_text())
    n_utt = len(meta.get("texts", []))
    cache_h = meta["offsets"][-1] / 16000 / 3600.0
    size_gb = bin_p.stat().st_size / 1e9
    check(f"cache: {n_utt:,} utterances, {cache_h:.2f}h, {size_gb:.1f} GB",
          n_utt > 0 and cache_h > 0)
    if rows:
        # build_cache skips rows it cannot decode; a large gap means many were lost
        keep = n_utt / len(rows)
        check("cache kept >=95% of manifest rows", keep >= 0.95,
              f"only {keep:.1%} of {len(rows):,} manifest rows made it into the cache")
        expect_bytes = meta["offsets"][-1] * 2
        check("cache byte size matches offsets", abs(expect_bytes - bin_p.stat().st_size) < 1024,
              f"offsets imply {expect_bytes:,} B, file is {bin_p.stat().st_size:,} B")


def verify_banks(noise_dir: Path | None, rir_dir: Path | None) -> None:
    print("\n10. Augmentation banks")
    for label, d in (("noise", noise_dir), ("rir", rir_dir)):
        if d is None:
            continue
        n = sum(1 for _ in d.rglob("*.wav")) if d.exists() else 0
        check(f"{label} bank populated ({n:,} wavs)", n > 0,
              "empty -> this augmentation is a SILENT no-op" if n == 0 else "")


def verify_kenlm(lm_dir: Path | None) -> None:
    """The LM is not needed to TRAIN, but without it the +KenLM column of the
    results table is empty -- and that column carries the headline 10.1 -> 5.1
    WER improvement. Flagged here so it is noticed before the run, not after."""
    if lm_dir is None:
        return
    print("\n11. KenLM language model")
    arpa = lm_dir / "3-gram.pruned.1e-7.arpa"
    if not arpa.exists():
        check("KenLM ARPA present", False,
              f"missing {arpa} -- evaluation will be greedy-only")
        return
    # Structural check, NOT a size guess. An earlier version demanded >500 MB and
    # was wrong -- 3-gram.pruned.1e-7 is ~98 MB decompressed, so the threshold
    # rejected a perfectly good model. Every ARPA ends with `\\end\\`; a
    # truncated download still gunzips and still has a valid header, but cannot
    # have the terminator.
    size = arpa.stat().st_size
    head_ok = tail_ok = False
    try:
        with open(arpa, "rb") as fh:
            head = fh.read(4096).decode("utf-8", errors="replace")
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
        head_ok = "\\data\\" in head or "ngram 1=" in head
        tail_ok = "\\end\\" in tail
    except Exception:
        pass
    detail = ("" if (head_ok and tail_ok) else
              ("no ARPA header" if not head_ok else
               "no \\end\\ terminator -- truncated download"))
    check(f"KenLM ARPA present and complete ({size / 1e6:.0f} MB)",
          size >= 10 * 1024 * 1024 and head_ok and tail_ok, detail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--rir-dir", default=None)
    ap.add_argument("--lm-dir", default=None)
    ap.add_argument("--check-all-audio", action="store_true",
                    help="stat every audio file instead of a random sample "
                         "(slower, but proves nothing is missing)")
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()

    print("=" * 66)
    print("ASR — training data verification")
    print("=" * 66)

    data_dir = Path(args.data)
    per_corpus = verify_manifests(data_dir, args.check_all_audio, args.sample)
    rows = verify_combined(data_dir, per_corpus)
    verify_rows(rows, args.check_all_audio, args.sample)
    if args.cache:
        verify_cache(Path(args.cache), rows)
    verify_banks(Path(args.noise_dir) if args.noise_dir else None,
                 Path(args.rir_dir) if args.rir_dir else None)
    verify_kenlm(Path(args.lm_dir) if args.lm_dir else None)

    fails = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 66)
    if fails:
        print(f"NOT READY — {len(fails)} check(s) failed:")
        for f in fails:
            print(f"  - {f}")
        print("Fix these before starting an 8-hour run.")
    else:
        print(f"READY — all {len(_results)} checks passed. Safe to start training.")
    print("=" * 66)
    sys.exit(2 if fails else 0)


if __name__ == "__main__":
    main()
'''
    (asr_dir / "verify_data.py").write_text(verify_data_code, encoding="utf-8")

    fetch_kenlm_code = r'''# /// script
# requires-python = ">=3.11"
# dependencies = []   # stdlib only
# ///
"""Fetch the KenLM language model used for CTC beam-search decoding.

This is the SAME model the 100h baseline used, so the 300h numbers stay
comparable: LibriSpeech **3-gram.pruned.1e-7** from OpenSLR-11. `kenlm_grid.py`
hardcodes that exact URL, and this script reuses it rather than picking a
"better" LM -- swapping the LM would change the decoder underneath both rows of
the results table and make the 100h vs 300h comparison meaningless.

The 4-gram from the same resource is deliberately NOT fetched. It is several
times larger, and the project already decided against it.

WHY THE LM MUST MATCH THE ACOUSTIC VOCABULARY
---------------------------------------------
This LM was built on LibriSpeech-normalised text: A-Z, apostrophe, no digits, no
punctuation. That is exactly why `prepare_data.normalize_text` expands digits to
words and drops rows with out-of-vocabulary characters instead of extending the
CTC output vocabulary. Emitting a character the LM has never seen collapses the
beam search and throws away the 10.1 -> 5.1 WER gain that KenLM provides.

Usage:
    python fetch_kenlm.py --lm-dir /marimo/lm
    python fetch_kenlm.py --lm-dir /marimo/lm --verify-only
    python fetch_kenlm.py --lm-dir /marimo/lm --jobs 8
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Same file kenlm_grid.py uses. OpenSLR mirrors: the primary host is often slow
# and its us. subdomain has had a certificate hostname mismatch, so try in order.
LM_NAME = "3-gram.pruned.1e-7.arpa"
LM_URLS = [
    f"https://us.openslr.org/resources/11/{LM_NAME}.gz",
    f"https://www.openslr.org/resources/11/{LM_NAME}.gz",
    f"https://openslr.elda.org/resources/11/{LM_NAME}.gz",
]
# Sanity FLOOR only -- deliberately low. An earlier version guessed 500 MB and
# was simply wrong: 3-gram.pruned.1e-7 decompresses to about 98 MB (it is a
# PRUNED model, ~200k unigrams / ~2.45M bigrams), so the guess deleted a
# perfectly good file and re-downloaded it. Size is a bad completeness test.
MIN_ARPA_BYTES = 10 * 1024 * 1024


def arpa_is_complete(path: Path) -> tuple[bool, str]:
    """Structural check instead of a size guess.

    Every ARPA file ends with the literal line `\end\`. A download truncated
    mid-stream still gunzips and still has a valid-looking header, so the header
    alone proves nothing -- but it cannot have the terminator. Reading the last
    kilobyte is O(1) and is the actual definition of "complete".
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat: {exc}"
    if size < MIN_ARPA_BYTES:
        return False, f"only {size / 1e6:.0f} MB -- far too small even for a pruned model"

    with open(path, "rb") as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")
        fh.seek(max(0, size - 4096))
        tail = fh.read().decode("utf-8", errors="replace")

    if "\\data\\" not in head and "ngram 1=" not in head:
        return False, "no ARPA header (\\data\\ / ngram 1=) in the first 4 KB"
    if "\\end\\" not in tail:
        return False, ("no \\end\\ terminator in the last 4 KB -- the download was "
                       "truncated mid-file")
    ngrams = [l.strip() for l in head.splitlines() if l.strip().startswith("ngram ")]
    return True, f"{size / 1e6:.0f} MB, {', '.join(ngrams[:3])}, \\end\\ present"


def log(*a):
    print(*a, flush=True)


def _supports_ranges(url: str, timeout: int = 30) -> tuple[bool, int]:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "clear-asr/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = int(r.headers.get("Content-Length") or 0)
            return (r.headers.get("Accept-Ranges", "").lower() == "bytes" and size > 0), size
    except Exception:
        return False, 0


def _download_ranges(url: str, dest: Path, size: int, jobs: int) -> None:
    chunk = size // jobs
    spans = [(i * chunk, (size - 1) if i == jobs - 1 else (i + 1) * chunk - 1)
             for i in range(jobs)]
    with open(dest, "wb") as fh:
        fh.truncate(size)
    done, lock, t0 = 0, threading.Lock(), time.time()

    def worker(span):
        nonlocal done
        start, end = span
        req = urllib.request.Request(
            url, headers={"User-Agent": "clear-asr/1.0", "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "r+b") as fh:
            fh.seek(start)
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
                with lock:
                    done += len(buf)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(worker, sp) for sp in spans]
        last = 0.0
        while any(not f.done() for f in futures):
            time.sleep(0.25)
            now = time.time()
            if now - last < 5.0:
                continue
            last = now
            with lock:
                d = done
            el = max(now - t0, 1e-6)
            log(f"  [lm] {d / 1e6:.0f}/{size / 1e6:.0f} MB ({100 * d / size:.0f}%) "
                f"@ {d / 1e6 / el:.0f} MB/s, {jobs} streams")
        for f in futures:
            f.result()
    if dest.stat().st_size != size:
        raise RuntimeError(f"size mismatch: {dest.stat().st_size:,} != {size:,}")


def download_lm_archive(dest_gz: Path, jobs: int) -> None:
    if dest_gz.exists() and dest_gz.stat().st_size > 0:
        log(f"  [lm] reusing existing {dest_gz.name} ({dest_gz.stat().st_size / 1e6:.0f} MB)")
        return
    if shutil.which("aria2c"):
        log(f"  [lm] aria2c found -- {jobs} connections")
        for url in LM_URLS:
            r = subprocess.run(["aria2c", "-x", str(jobs), "-s", str(jobs), "-k", "10M",
                                "--console-log-level=warn", "--auto-file-renaming=false",
                                "--allow-overwrite=true", "-d", str(dest_gz.parent),
                                "-o", dest_gz.name, url])
            if r.returncode == 0 and dest_gz.exists():
                return
    last = None
    for url in LM_URLS:
        log(f"  trying {url}")
        try:
            ok, size = _supports_ranges(url)
            if ok and jobs > 1:
                log(f"  [lm] range support, {size / 1e6:.0f} MB -> {jobs} parallel streams")
                _download_ranges(url, dest_gz, size, jobs)
            else:
                log("  [lm] no range support -- single stream")
                req = urllib.request.Request(url, headers={"User-Agent": "clear-asr/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest_gz, "wb") as fh:
                    shutil.copyfileobj(r, fh, length=1 << 20)
            return
        except Exception as exc:
            last = exc
            log(f"    failed: {type(exc).__name__}: {exc}")
            dest_gz.unlink(missing_ok=True)
    raise RuntimeError(f"all mirrors failed; last error: {last}")


def fetch_lm(lm_dir: Path, jobs: int = 8) -> Path:
    lm_dir.mkdir(parents=True, exist_ok=True)
    arpa = lm_dir / LM_NAME
    if arpa.exists():
        ok, why = arpa_is_complete(arpa)
        if ok:
            log(f"[lm] already present and complete: {arpa}")
            log(f"[lm]   {why}")
            return arpa
        log(f"[lm] {arpa.name} is unusable ({why}) -- re-downloading")
        arpa.unlink()

    gz = lm_dir / f"{LM_NAME}.gz"
    download_lm_archive(gz, jobs)

    log(f"[lm] decompressing {gz.name} -> {arpa.name}")
    t0 = time.time()
    with gzip.open(gz, "rb") as f_in, open(arpa, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 22)
    gz.unlink(missing_ok=True)
    log(f"[lm] ready: {arpa} ({arpa.stat().st_size / 1e9:.2f} GB, "
        f"{time.time() - t0:.0f}s to decompress)")
    return arpa


def verify(lm_dir: Path) -> bool:
    arpa = lm_dir / LM_NAME
    log("")
    log("KenLM status")
    log("=" * 30)
    if not arpa.exists():
        log(f"  {arpa} -> MISSING")
        log("  NOT READY — evaluation will fall back to greedy decoding only, "
            "and the +KenLM column of the results table will be empty.")
        return False
    log(f"  {arpa}")
    ok, why = arpa_is_complete(arpa)
    log(f"  {why}")
    log("  READY — beam-search decoding with KenLM is available." if ok else
        "  NOT READY — " + why)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    lm_dir = Path(args.lm_dir)
    if not args.verify_only:
        try:
            fetch_lm(lm_dir, jobs=args.jobs)
        except Exception as exc:
            log(f"[lm] FAILED: {type(exc).__name__}: {exc}")
    sys.exit(0 if verify(lm_dir) else 2)


if __name__ == "__main__":
    main()
'''
    (asr_dir / "fetch_kenlm.py").write_text(fetch_kenlm_code, encoding="utf-8")

    # 3. gdrive_sync.py (Drive target reconciled to ASR-300, matching the
    #    actual 300h run -- was "ASR-350" before this fix)
    gdrive_sync_code = r'''"""Google Drive mirroring for ASR checkpoints.

Target: molab (marimo cloud), NOT Colab. There is no `/content/drive` mount
here, so the OAuth / service-account path is the one that actually runs.

WHY THIS WAS REWRITTEN
----------------------
The previous version failed silently in three separate ways at once:

  1. `googleapiclient` / `google-auth` were not in ANY dependency list, so
     `GAPI_AVAILABLE` was False and the upload path was never even attempted.
     A perfectly valid token would simply be ignored.
  2. `sync_checkpoint()` returned None whether it uploaded a file or did
     nothing at all, and every failure was swallowed by a bare
     `except Exception: pass`.
  3. Because it raised nothing, the caller in train_asr.py logged
     "[drive] mirrored: ep001.pt" for a no-op. A false confirmation is worse
     than a visible failure: it is exactly the state in which someone leaves
     an 8-hour run overnight believing the checkpoints are safe.

So every function here returns an explicit status and a human-readable reason,
and nothing is ever reported as mirrored unless bytes actually moved.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

try:
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GAPI_AVAILABLE = True
    GAPI_IMPORT_ERROR = ""
except ImportError as _exc:      # pragma: no cover - depends on the environment
    GAPI_AVAILABLE = False
    GAPI_IMPORT_ERROR = str(_exc)

SCOPES = ["https://www.googleapis.com/auth/drive"]

# The 300h retrain (100 librispeech + 106 common_voice + 50 ami + 44 vctk).
DRIVE_SUBPATH = ("CLEAR", "Phase 1", "ASR-300")

# Explicit override wins over any search. Set this if the credential file is
# somewhere unusual, or simply not named one of the conventional names.
ENV_CRED = "ECAD_GDRIVE_CREDENTIALS"

# Conventional names, in preference order. Service-account keys are tried
# before user OAuth tokens because they do not expire.
_SA_NAMES = ("service_account.json", "service-account.json", "sa.json")
_OAUTH_NAMES = ("token.json", "credentials.json", "oauth_token.json",
                "authorized_user.json")


def _is_file(p: Path) -> bool:
    """`Path.is_file()` raises PermissionError on directories we may not stat
    (e.g. /root when running unprivileged). A credential search must never take
    down the caller, so unreadable paths simply count as "not here"."""
    try:
        return p.is_file()
    except OSError:
        return False


def _search_roots() -> list[Path]:
    """Directories to look in, widest sensible set.

    Includes the filesystem root: on molab people commonly drop credentials at
    `/` or at a workspace root that is not the cwd, and the old list checked
    neither, so a token uploaded "to root" was invisible.
    """
    here = Path(__file__).resolve().parent
    roots = [
        Path.cwd(), Path.cwd().parent,
        here, here.parent, here.parent.parent,
        Path.home(), Path("/"), Path("/root"), Path("/home/user"),
        Path("/content"), Path("/marimo"), Path("/workspace"),
    ]
    seen, out = set(), []
    for r in roots:
        try:
            rr = r.resolve()
            is_dir = rr.is_dir()
        except OSError:
            continue
        if rr not in seen and is_dir:
            seen.add(rr)
            out.append(rr)
    return out


def find_credential_file() -> tuple[Path | None, str]:
    """Locate a credential file. Returns (path_or_None, how_it_was_found)."""
    override = os.environ.get(ENV_CRED, "").strip()
    if override:
        p = Path(override)
        if _is_file(p):
            return p, f"${ENV_CRED}={override}"
        return None, f"${ENV_CRED} is set to {override!r} but that file does not exist"

    for root in _search_roots():
        for name in (*_SA_NAMES, *_OAUTH_NAMES):
            p = root / name
            if _is_file(p):
                return p, f"found {p}"

    # Last resort: glob for anything that looks like a Google credential,
    # because Google's console hands out files named e.g.
    # `client_secret_1234-abcd.apps.googleusercontent.com.json`.
    for root in _search_roots():
        try:
            for pat in ("*service*account*.json", "*client_secret*.json", "*token*.json"):
                for p in sorted(root.glob(pat)):
                    if _is_file(p):
                        return p, f"glob match {p}"
        except Exception:
            continue
    return None, "no credential file found (see diagnose() for the search paths)"


def get_mounted_gdrive_path() -> Path | None:
    """A real mounted Drive, if one exists. On molab there usually is not one —
    that is expected, and the OAuth path below handles it."""
    for mount in ("/content/drive/MyDrive", "/gdrive/MyDrive", "/mnt/gdrive/MyDrive",
                  str(Path.home() / "gdrive" / "MyDrive")):
        p = Path(mount)
        if p.exists():
            target = p.joinpath(*DRIVE_SUBPATH)
            try:
                target.mkdir(parents=True, exist_ok=True)
                return target
            except Exception:
                continue
    return None


def get_gapi_service() -> tuple[object | None, str]:
    """Build a Drive API client. Returns (service_or_None, reason)."""
    if not GAPI_AVAILABLE:
        return None, (
            "google-api-python-client / google-auth are not installed "
            f"({GAPI_IMPORT_ERROR}). Install them in the venv cell — without them "
            "the OAuth upload path cannot run at all."
        )

    cred_path, how = find_credential_file()
    if cred_path is None:
        return None, how

    name = cred_path.name.lower()
    looks_service_account = any(k in name for k in ("service", "sa.json"))

    if looks_service_account:
        try:
            creds = service_account.Credentials.from_service_account_file(
                str(cred_path), scopes=SCOPES)
            return build("drive", "v3", credentials=creds), f"service account ({how})"
        except Exception as exc:
            return None, f"service-account load failed for {cred_path}: {type(exc).__name__}: {exc}"

    try:
        creds = Credentials.from_authorized_user_file(str(cred_path), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds), f"user OAuth token ({how})"
    except Exception as exc:
        return None, (
            f"OAuth token load failed for {cred_path}: {type(exc).__name__}: {exc}. "
            "Note this expects an AUTHORIZED-USER json (the one holding "
            "refresh_token/client_id/client_secret), not a raw client-secret file "
            "downloaded from the Google console."
        )


def get_or_create_folder(service, folder_name: str, parent_id: str | None = None) -> str:
    q = (f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
         f"and trashed = false")
    if parent_id:
        q += f" and '{parent_id}' in parents"
    resp = service.files().list(q=q, spaces="drive", fields="files(id)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    return service.files().create(body=meta, fields="id").execute()["id"]


def upload_file_gapi(service, file_path: Path, run_name: str) -> tuple[bool, str]:
    parent_id = None
    for folder_name in (*DRIVE_SUBPATH, run_name):
        parent_id = get_or_create_folder(service, folder_name, parent_id)
    q = f"name = '{file_path.name}' and '{parent_id}' in parents and trashed = false"
    existing = service.files().list(q=q, spaces="drive", fields="files(id)").execute().get("files", [])
    media = MediaFileUpload(str(file_path), resumable=True)
    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        return True, f"updated {file_path.name}"
    service.files().create(body={"name": file_path.name, "parents": [parent_id]},
                           media_body=media).execute()
    return True, f"uploaded {file_path.name}"


def sync_checkpoint(file_path: Path, run_name: str) -> tuple[bool, str]:
    """Mirror one file. Returns (ok, reason).

    Returning a STATUS rather than None is the whole point of this rewrite: the
    caller cannot otherwise distinguish "uploaded" from "did absolutely nothing",
    and the previous version reported both as success.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False, f"{file_path} does not exist"

    mounted = get_mounted_gdrive_path()
    if mounted is not None:
        try:
            dest_dir = mounted / run_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_dir / file_path.name)
            return True, f"copied to mounted Drive {dest_dir}"
        except Exception as exc:
            return False, f"mounted-Drive copy failed: {type(exc).__name__}: {exc}"

    service, reason = get_gapi_service()
    if service is None:
        return False, reason
    try:
        return upload_file_gapi(service, file_path, run_name)
    except Exception as exc:
        return False, f"Drive upload failed: {type(exc).__name__}: {exc}"


# ============================================================================
# Download side. The module only ever uploaded, which was fine while Drive was
# a backup -- but the 100h baseline checkpoint lives THERE and nowhere else, and
# the results table needs it re-decoded under the same protocol as the 300h row.
# Copying it down by hand is exactly the kind of step that gets done once, wrong.
# ============================================================================

# The baseline is not under DRIVE_SUBPATH: that constant points at where THIS
# project writes ("CLEAR/Phase 1/ASR-300"), while the finished 100h run sits in
# "CLEAR/Phase 1/runs/FINAL". Keeping them as separate constants avoids a
# tempting-but-wrong reuse.
BASELINE_SUBPATH = ("CLEAR", "Phase 1", "runs", "FINAL")

# What a run directory must contain for eval_asr.py / tune_lm.py to load it.
RUN_FILES = ("config.json", "adapter.pt", "head.pt")

# Small, and they answer questions the checkpoint cannot: how many epochs the
# baseline actually ran, where it stopped improving, and what its recorded
# hyperparameters were. Fetched when present, never required -- an older run that
# predates them must still be loadable.
RUN_FILES_OPTIONAL = ("summary.json", "history.jsonl")


def find_folder(service, subpath) -> tuple[str | None, str]:
    """Resolve a folder PATH, one component at a time. Never creates anything.

    `get_or_create_folder` is the wrong tool for reading: if a name is misspelled
    it would silently create an empty folder and the caller would then report
    "0 files found" instead of "that path does not exist". Downloads must fail
    loudly on a bad path.
    """
    parent = None
    for i, name in enumerate(subpath):
        q = (f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' "
             f"and trashed = false")
        if parent:
            q += f" and '{parent}' in parents"
        files = service.files().list(q=q, spaces="drive",
                                    fields="files(id,name)").execute().get("files", [])
        if not files:
            got = "/".join(subpath[:i]) or "My Drive root"
            return None, (f"folder '{name}' not found under {got} "
                          f"(looking for {'/'.join(subpath)})")
        if len(files) > 1:
            # Drive allows duplicate names in one parent. Guessing would make the
            # download non-deterministic, so say so instead.
            return None, (f"{len(files)} folders named '{name}' under "
                          f"{'/'.join(subpath[:i]) or 'root'} -- ambiguous, rename one")
        parent = files[0]["id"]
    return parent, f"resolved {'/'.join(subpath)}"


def _download_one(service, file_id: str, name: str, size: int, dest: Path) -> tuple[bool, str]:
    import io

    from googleapiclient.http import MediaIoBaseDownload

    out = dest / name
    # Skip work that is already done, but only on an exact size match. A partial
    # file from an interrupted download has a smaller size and must NOT count as
    # present -- that is how a truncated adapter.pt would reach torch.load.
    if out.exists() and size and out.stat().st_size == size:
        return True, f"{name}: already present ({size / 1e6:.1f} MB), size matches"
    if out.exists():
        out.unlink()

    tmp = dest / (name + ".part")
    req = service.files().get_media(fileId=file_id)
    with open(tmp, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=8 * 1024 * 1024)
        done = False
        last = -1
        while not done:
            status, done = dl.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                if pct >= last + 20:
                    last = pct
                    print(f"    [drive] {name}: {pct}%", flush=True)
    if size and tmp.stat().st_size != size:
        tmp.unlink(missing_ok=True)
        return False, (f"{name}: size mismatch, got {tmp.stat().st_size:,} "
                       f"expected {size:,} -- download truncated")
    # Rename only after the size check, so a failed download never leaves a file
    # that looks usable.
    tmp.replace(out)
    return True, f"{name}: downloaded {out.stat().st_size / 1e6:.1f} MB"


def download_run(dest_dir: Path, subpath=BASELINE_SUBPATH,
                 files=RUN_FILES) -> tuple[bool, str, list[str]]:
    """Fetch a run directory from Drive into `dest_dir`. Returns (ok, reason, log)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    mounted = None
    for mount in ("/content/drive/MyDrive", "/gdrive/MyDrive", "/mnt/gdrive/MyDrive",
                  str(Path.home() / "gdrive" / "MyDrive")):
        if Path(mount).exists():
            mounted = Path(mount).joinpath(*subpath)
            break
    if mounted is not None:
        if not mounted.is_dir():
            return False, f"mounted Drive found but {mounted} does not exist", lines
        for name in files:
            src = mounted / name
            if not src.is_file():
                return False, f"{src} missing on mounted Drive", lines
            shutil.copy2(src, dest_dir / name)
            lines.append(f"{name}: copied {src.stat().st_size / 1e6:.1f} MB from mount")
        return True, f"copied {len(files)} files from {mounted}", lines

    service, reason = get_gapi_service()
    if service is None:
        return False, reason, lines

    folder_id, why = find_folder(service, subpath)
    lines.append(why)
    if folder_id is None:
        return False, why, lines

    present = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false", spaces="drive",
        fields="files(id,name,size)", pageSize=1000).execute().get("files", [])
    by_name = {f["name"]: f for f in present}
    lines.append(f"folder contains {len(present)} items: "
                 f"{sorted(by_name)[:12]}{' ...' if len(present) > 12 else ''}")

    missing = [n for n in files if n not in by_name]
    if missing:
        return False, (f"{'/'.join(subpath)} is missing {missing}. Present: "
                       f"{sorted(by_name)}"), lines

    ok_all = True
    for name in files:
        f = by_name[name]
        good, msg = _download_one(service, f["id"], name, int(f.get("size") or 0), dest_dir)
        lines.append(msg)
        ok_all &= good
    for name in RUN_FILES_OPTIONAL:
        if name in by_name:
            f = by_name[name]
            _, msg = _download_one(service, f["id"], name, int(f.get("size") or 0), dest_dir)
            lines.append(f"(optional) {msg}")
        else:
            lines.append(f"(optional) {name}: not in the folder, skipped")
    return ok_all, ("all files present locally" if ok_all else
                    "at least one file failed -- see the log"), lines


def verify_run_dir(dest_dir: Path, files=RUN_FILES) -> tuple[bool, str]:
    """Confirm the downloaded run is actually loadable, not merely present.

    A run directory that exists but whose config.json is unparseable, or whose
    adapter.pt is a truncated tensor file, fails later inside eval_asr.py with a
    confusing traceback. Checking here keeps the failure next to its cause.
    """
    dest_dir = Path(dest_dir)
    problems = []
    for name in files:
        p = dest_dir / name
        if not p.is_file():
            problems.append(f"{name} missing")
        elif p.stat().st_size == 0:
            problems.append(f"{name} is empty")
    cfg_p = dest_dir / "config.json"
    if cfg_p.is_file():
        try:
            import json

            cfg = json.loads(cfg_p.read_text())
            for key in ("ws", "lora_layers", "lora_r", "lora_alpha"):
                if key not in cfg:
                    problems.append(f"config.json has no '{key}' -- eval_asr.py needs it")
            if "ws" in cfg:
                problems += [f"config.json ws={cfg['ws']} contains a layer outside 1..12"
                             for L in cfg["ws"] if not 1 <= int(L) <= 12][:1]
        except Exception as exc:
            problems.append(f"config.json is not valid JSON: {exc}")
    if problems:
        return False, "; ".join(problems)
    import json

    cfg = json.loads(cfg_p.read_text())
    return True, (f"loadable: ws={cfg.get('ws')} lora={cfg.get('lora_layers')} "
                  f"r={cfg.get('lora_r')}")


def diagnose() -> str:
    """Human-readable report of what the sync layer can and cannot do.

    Run this BEFORE starting an 8-hour training run. If it does not say
    'READY', nothing will be mirrored and the run's checkpoints exist only on
    ephemeral cloud disk.
    """
    lines = ["Google Drive sync diagnosis", "=" * 30]
    lines.append(f"google-api libs importable : {GAPI_AVAILABLE}"
                 + ("" if GAPI_AVAILABLE else f"  ({GAPI_IMPORT_ERROR})"))
    mounted = get_mounted_gdrive_path()
    lines.append(f"mounted Drive              : {mounted or 'none (expected on molab)'}")
    cred, how = find_credential_file()
    lines.append(f"credential file            : {cred or 'NOT FOUND'}")
    lines.append(f"  how                      : {how}")
    lines.append(f"  ${ENV_CRED}".ljust(28) + f": {os.environ.get(ENV_CRED, '<unset>')}")
    service, reason = get_gapi_service()
    lines.append(f"Drive API client           : {'built' if service else 'NOT built'}")
    lines.append(f"  reason                   : {reason}")
    lines.append("")
    lines.append("searched directories:")
    for r in _search_roots():
        lines.append(f"  {r}")
    lines.append("")
    ok = bool(mounted) or bool(service)
    lines.append("READY — checkpoints will be mirrored." if ok else
                 "NOT READY — nothing will be mirrored. Fix the above before an "
                 "unattended run, or the checkpoints live only on ephemeral disk.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-baseline", metavar="DEST",
                    help="download CLEAR/'Phase 1'/runs/FINAL into DEST")
    ap.add_argument("--subpath", default="/".join(BASELINE_SUBPATH),
                    help="slash-separated Drive folder path to fetch")
    args = ap.parse_args()

    if not args.fetch_baseline:
        print(diagnose())
        raise SystemExit(0)

    dest = Path(args.fetch_baseline)
    sub = tuple(x for x in args.subpath.split("/") if x)
    print(f"[drive] fetching {'/'.join(sub)} -> {dest}")
    ok, reason, lines = download_run(dest, subpath=sub)
    for line in lines:
        print(f"  {line}")
    print(f"[drive] {'OK' if ok else 'FAILED'}: {reason}")
    if not ok:
        print()
        print(diagnose())
        raise SystemExit(2)

    good, why = verify_run_dir(dest)
    print(f"[drive] verify: {'OK' if good else 'PROBLEM'} -- {why}")
    raise SystemExit(0 if good else 2)
'''
    (asr_dir / "gdrive_sync.py").write_text(gdrive_sync_code, encoding="utf-8")

    # 4. build_cache.py (verbatim port of _Staj/asr/build_cache.py) -- FIX:
    #    this blob used to be defined and then silently never written to
    #    disk because the write call below it wrote a nonexistent
    #    `train_asr_code` name instead. Now it is written to its own file.
    build_cache_code = r'''# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile==0.14.0", "datasets==5.0.0"]
# ///
"""
ASR -- 300h retrain, packed-cache builder.

Converts manifest_combined.jsonl (produced by prepare_data.py) into the SAME
packed int16 binary + meta.json cache format ablation_engine.py's `prepare()`
uses (audio.i16 + offsets + texts), so the training script's Dataset class
can stay close to SpeechDS in ablation_engine.py.

Design decision (not specified verbatim in the task, made here for
consistency with the existing codebase): LibriSpeech and AMI rows carry an
`hf_index` into their HF dataset split rather than a standalone audio file
(that's how ablation_engine.py and the AMI loader consume them -- parquet-backed,
audio decoded from in-memory arrow arrays, no extracted wav files on disk).
Common Voice and VCTK rows carry a real `audio_path` on disk. This script
handles both: it re-opens the relevant HF dataset split once per corpus and
indexes into it for hf_index rows, and reads directly from disk for
audio_path rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, flush=True)


def _decode_soundfile(path: str, sr_target: int) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32")
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if int(sr) != sr_target:
        w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr_target / sr)),
                      np.arange(len(w)), w).astype(np.float32)
    return w


def _decode_hf_cell(cell, sr_target: int) -> np.ndarray:
    import soundfile as sf

    if isinstance(cell, dict) and cell.get("array") is not None:
        w, sr = np.asarray(cell["array"], np.float32), cell.get("sampling_rate", sr_target)
    elif isinstance(cell, dict) and cell.get("bytes"):
        w, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
    elif isinstance(cell, dict) and cell.get("path"):
        w, sr = sf.read(cell["path"], dtype="float32")
    else:
        raise ValueError(f"unrecognised audio cell: {type(cell)}")
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if int(sr) != sr_target:
        w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr_target / sr)),
                      np.arange(len(w)), w).astype(np.float32)
    return w


_HF_SPLIT_CACHE = {}


def _hf_split(corpus: str, source: str):
    key = (corpus, source)
    if key in _HF_SPLIT_CACHE:
        return _HF_SPLIT_CACHE[key]
    from datasets import load_dataset

    if corpus == "librispeech":
        # Should be unreachable: prepare_data.py streams LibriSpeech and stages
        # the encoded bytes to disk, so its rows carry `audio_path`. Naming the
        # split does NOT limit the download -- a run was observed generating
        # train.360 (104,014 examples) despite split="train.100" -- so silently
        # falling back to load_dataset here would re-download hundreds of hours.
        raise ValueError(
            "LibriSpeech rows must carry `audio_path` (streamed bytes), not "
            "`hf_index`. Rebuild the manifest with `--corpus librispeech`."
        )
    elif corpus == "ami":
        # AMI no longer reaches this path. prepare_data.py streams AMI and
        # writes FLAC to disk, so its rows carry a real `audio_path` and are
        # read directly. Reaching here means a stale manifest built by the old
        # random-access code is being fed to a new cache builder -- that
        # mismatch would silently re-download ~160h of AMI, so fail instead.
        raise ValueError(
            "AMI rows must carry `audio_path` (streamed FLAC), not `hf_index`. "
            "This manifest was built by an older prepare_data.py -- rebuild it "
            "with `--corpus ami` before running build_cache."
        )
    else:
        raise ValueError(f"no hf_index loader for corpus={corpus}")
    _HF_SPLIT_CACHE[key] = ds
    return ds



def _stratified_subset(rows: list[dict], target_hours: float, seed: int = 1337) -> list[dict]:
    """Take `target_hours` while PRESERVING the corpus mix.

    Why not just slice the first N rows: the combined manifest is written corpus
    by corpus, so `rows[:N]` is pure LibriSpeech. A WS-layer ablation run on
    100% clean read speech would answer the wrong question entirely -- the whole
    reason for the probe is that accent, spontaneous speech and noise might move
    the optimal layers, and none of those appear in the first slice.

    Each corpus contributes the same PROPORTION it has in the full 300h mix, so
    a 50h probe cache is a scale model of the real training set.
    """
    import random as _random

    by_corpus: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_corpus[r["corpus"]].append(r)
    total_s = sum(r["duration_s"] for r in rows)
    rng = _random.Random(seed)

    out: list[dict] = []
    for corpus, crows in sorted(by_corpus.items()):
        corpus_s = sum(r["duration_s"] for r in crows)
        share = corpus_s / total_s if total_s else 0.0
        want_s = target_hours * 3600.0 * share
        rng.shuffle(crows)
        got = 0.0
        for r in crows:
            if got >= want_s:
                break
            out.append(r)
            got += r["duration_s"]
        log(f"  [subset] {corpus:14s} {got / 3600.0:6.2f}h "
            f"({share:.1%} of the mix, {len([x for x in out if x['corpus'] == corpus]):,} rows)")
    rng.shuffle(out)
    log(f"  [subset] TOTAL {sum(r['duration_s'] for r in out) / 3600.0:.2f}h "
        f"/ {len(out):,} rows (target {target_hours:.0f}h)")
    return out


def build_cache(manifest_path: Path, cache_dir: Path, sr: int = 16000,
                 max_rows: int | None = None, hours: float | None = None) -> Path:
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if max_rows:
        rows = rows[:max_rows]
    if hours:
        rows = _stratified_subset(rows, hours, seed=1337)

    key = hashlib.md5(f"{manifest_path}|{len(rows)}|{sr}|{hours}".encode()).hexdigest()[:12]
    tag = f"{hours:.0f}h" if hours else "full"
    d = cache_dir / f"combined_{tag}_{key}"
    bin_p, meta_p = d / "audio.i16", d / "meta.json"
    if bin_p.exists() and meta_p.exists():
        log(f"[cache] found existing cache at {d}")
        return d

    d.mkdir(parents=True, exist_ok=True)
    t0, offs, texts, corpora, pos = time.perf_counter(), [0], [], [], 0
    with open(bin_p, "wb") as f:
        for i, row in enumerate(rows):
            try:
                if row.get("audio_path") and Path(row["audio_path"]).exists():
                    w = _decode_soundfile(row["audio_path"], sr)
                else:
                    ds = _hf_split(row["corpus"], row["source"])
                    cell = ds[row["hf_index"]]["audio"]
                    w = _decode_hf_cell(cell, sr)
            except Exception as e:
                log(f"[cache] WARN skip row {i} ({row.get('corpus')}): {type(e).__name__}: {e}")
                continue
            q = np.clip(np.rint(w * 32768.0), -32768, 32767).astype(np.int16)
            f.write(q.tobytes())
            pos += q.size
            offs.append(pos)
            texts.append(row["text"])
            corpora.append(row["corpus"])
            if (i + 1) % 2000 == 0:
                log(f"  [cache] {i + 1}/{len(rows)} ({pos * 2 / 1e9:.1f} GB, "
                    f"{time.perf_counter() - t0:.0f}s)")

    meta_p.write_text(json.dumps({"offsets": offs, "texts": texts, "corpora": corpora}))
    total_h = (offs[-1]) / sr / 3600.0
    log(f"[cache] {len(texts)} utterances, {total_h:.2f}h -> {d} "
        f"({(time.perf_counter() - t0) / 60:.1f} min)")
    return d


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/marimo/data/manifest_combined.jsonl")
    ap.add_argument("--cache", default="/marimo/cache")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None,
                    help="build a corpus-stratified subset of this many hours "
                         "(e.g. 50 for the WS ablation probe) instead of the full mix")
    args = ap.parse_args()
    build_cache(Path(args.manifest), Path(args.cache),
                max_rows=args.max_rows, hours=args.hours)
'''
    (asr_dir / "build_cache.py").write_text(build_cache_code, encoding="utf-8")

    # 5. train_asr.py (verbatim port of _Staj/asr/train_asr.py) -- FIX: this
    #    variable did not exist anywhere in the broken notebook; the write
    #    call two lines below used to reference it and raise NameError,
    #    which is why the notebook could never finish writing its modules.
    #    This is the real 508-line trainer: frozen mHuBERT-147 + LoRA +
    #    weighted-sum + CTC head, GPU-batched augmentation, per-epoch
    #    immutable checkpoints (ep{N:03d}.pt) plus a resumable last.pt.
    train_asr_code = r'''# /// script
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
'''
    (asr_dir / "train_asr.py").write_text(train_asr_code, encoding="utf-8")

    # 6. eval_asr.py -- FIX: the previous occupant of this slot was a 34-line
    #    fraud that hardcoded the OLD 100h model's dev-clean numbers under
    #    the 300h run's name and invented L2-ARCTIC figures out of thin air.
    #    It has been replaced wholesale with the real 334-line evaluator
    #    (verbatim port of _Staj/asr/eval_asr.py): loads adapter.pt/head.pt,
    #    decodes, scores WER/CER with jiwer on both dev-clean and L2-ARCTIC,
    #    greedy and +KenLM separately, plus a real Whisper baseline and
    #    CPU/GPU RTF + peak RAM. Every failure path raises -- see the NOTICE
    #    comment at the top of the written file for the full story.
    eval_asr_code = r'''# /// script
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
'''
    # 10. tune_lm.py (verbatim port of _Staj/asr/tune_lm.py)
    tune_lm_code = r'''# /// script
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
'''
    (asr_dir / "tune_lm.py").write_text(tune_lm_code, encoding="utf-8")

    (asr_dir / "eval_asr.py").write_text(eval_asr_code, encoding="utf-8")

    # Verify claim vs reality instead of asserting it in prose: check that
    # every module we just tried to write is actually on disk before saying
    # so (this replaces the old unconditional "All 6 pipeline and sync
    # engines are written" markdown, which was true only by accident even
    # when this cell worked, and outright false while bug #1/#2 were live).
    # The list was missing three modules this cell actually writes
    # (fetch_noise_banks, verify_data, fetch_kenlm) and the count was hardcoded
    # "/6", so it reported "All 7/6 modules" -- a check that cannot fail is not
    # a check. Both the list and the total now come from one place.
    _expected_modules = [
        "prepare_data.py", "augment.py", "gdrive_sync.py", "build_cache.py",
        "fetch_noise_banks.py", "verify_data.py", "fetch_kenlm.py",
        "train_asr.py", "tune_lm.py", "eval_asr.py",
    ]
    _n = len(_expected_modules)
    # Print each written file's fingerprint. The scripts print their own on
    # startup as `[src] name sha1:xxxxxxxx`; if a traceback's line numbers do not
    # match the code you are reading, compare the two hashes -- a mismatch means
    # this cell has not been re-run since the notebook changed, which has now
    # cost debugging time more than once.
    import hashlib as _hl
    for _m in _expected_modules:
        _f = asr_dir / _m
        if _f.exists():
            print(f"  [src] {_m} sha1:{_hl.sha1(_f.read_bytes()).hexdigest()[:8]}",
                  flush=True)
    _written = [m for m in _expected_modules if (asr_dir / m).exists()]
    _missing = [m for m in _expected_modules if m not in _written]
    if _missing:
        mo.md(f"❌ **{len(_written)}/{_n} modules written.** Missing: {_missing}")
    else:
        mo.md(f"✓ All {len(_written)}/{_n} pipeline modules confirmed written to "
              f"`{asr_dir}`: {_written}")


@app.cell
def _():
    import torch
    cuda_available = torch.cuda.is_available()
    return cuda_available,


@app.cell
def _(mo):
    install_btn = mo.ui.run_button(label="Initialize Virtual Environment", kind="success")
    return install_btn,


@app.cell
def _(mo, install_btn):
    mo.md(
        f"""
        ## 2 · Python 3.11 virtual environment
        Click this button to start installing Python 3.11 virtualenv with PyTorch cu128 and secondary stacks:
        
        {install_btn}
        """
    )


@app.cell
def _(mo, install_btn, py_bin, base_dir, subprocess, cuda_available):
    mo.stop(not install_btn.value, mo.md("*Virtual environment setup is pending. Click the 'Initialize' button above to trigger.*"))
    
    venv_dir = base_dir / "asr311"
    if not venv_dir.exists():
        try:
            subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)
        except subprocess.CalledProcessError as e:
            mo.stop(True, mo.md(f"❌ Failed to create virtual environment (exit {e.returncode}): {str(e)}"))
        except Exception as e:
            mo.stop(True, mo.md(f"❌ Failed to create virtual environment: {str(e)}"))
            
    env_msg = None
    try:
        pip_bin = str(venv_dir / "bin" / "pip")
        torch_index = "https://download.pytorch.org/whl/cpu" if not cuda_available else "https://download.pytorch.org/whl/cu128"
        # torch/torchaudio/torchvision MUST come from the same cu128 index at
        # the same version -- a mismatch previously crashed transformers with
        # a torchvision::nms error.
        subprocess.run([pip_bin, "install", "--upgrade", "pip"], check=True)
        subprocess.run(
            [pip_bin, "install", "torch", "torchaudio", "torchvision", "--index-url", torch_index],
            check=True,
        )
        # Dependency pins, verified against PyPI (pip index versions <pkg>):
        #   - soundfile==0.14.0 exists (0.14.0/0.13.x/0.12.x are all real
        #     releases; 0.14.0 is current) -- exact pin kept for
        #     reproducibility, matching build_cache.py / eval_asr.py.
        #   - datasets==5.0.0 exists and is used WITHOUT the old
        #     "<4.0.0" ceiling: that ceiling was a workaround for
        #     script-based dataset repos (trust_remote_code), but nothing in
        #     this pipeline needs it -- Common Voice is fetched via
        #     huggingface_hub.snapshot_download (never load_dataset), and
        #     LibriSpeech/AMI/VCTK are all parquet-native/standard loaders
        #     that work fine on datasets>=4. Pinning exactly to a version
        #     verified to exist is safer here than reintroducing an
        #     unnecessary ceiling.
        try:
            subprocess.run(
                [pip_bin, "install", "transformers>=4.44", "peft>=0.11", "jiwer", "hf_transfer", "torchcodec",
                 "google-api-python-client", "google-auth", "google-auth-oauthlib", 
                 "soundfile==0.14.0", "datasets==5.0.0", "numpy", "pandas", "tqdm",
                 "psutil", "pyctcdecode"],
                check=True,
            )
            # kenlm was in every PEP-723 header and in NO install list, so
            # pyctcdecode imported fine and then died at decode time with
            # `NameError: name 'kenlm' is not defined` -- pyctcdecode degrades to
            # a warning when the bindings are missing rather than refusing to
            # build, so the failure surfaced only once the LM was needed.
            #
            # There is often no wheel for the current Python, so this is a source
            # build and can genuinely fail. It is installed in its OWN step and
            # NOT under check=True: without KenLM the pipeline still produces the
            # greedy column, and losing that to a compiler error would be worse
            # than losing the +KenLM column.
            def _try_pip(*spec):
                _r = subprocess.run([pip_bin, "install", *spec],
                                    capture_output=True, text=True)
                return _r.returncode == 0, (_r.stderr or _r.stdout)

            def _sh_run(cmd):
                _r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
                return _r.returncode == 0, (_r.stderr or _r.stdout)

            # kenlm has no wheels for recent Pythons, so every route is a source
            # build and needs a full toolchain. What that build actually needs, in
            # the order it usually goes wrong:
            #   cmake, make, g++      -- the build itself
            #   Python.h              -- the bindings; a venv does NOT provide it,
            #                            it comes from the interpreter's dev headers
            #   zlib/bz2/lzma headers -- kenlm links them unconditionally
            # Missing Python.h is the classic one and produces a cmake error that
            # says nothing about Python, which is why it is checked by hand here.
            import shutil as _sh
            import sysconfig as _sc

            _inc = _sc.get_paths().get("include", "")
            _pyh = (Path(_inc) / "Python.h").is_file() if _inc else False
            _tools = {t: bool(_sh.which(t)) for t in ("cmake", "make", "g++")}
            print(f"[venv] kenlm prerequisites: {_tools} Python.h={_pyh} ({_inc})",
                  flush=True)

            if not all(_tools.values()) or not _pyh:
                _pkgs = "cmake build-essential zlib1g-dev libbz2-dev liblzma-dev"
                print(f"[venv] installing: {_pkgs}", flush=True)
                _ok, _out = _sh_run(f"apt-get update -qq && apt-get install -y -qq {_pkgs}")
                print(f"[venv] apt-get: {'OK' if _ok else 'FAILED'}", flush=True)
                if not _ok:
                    print(_out[-800:], flush=True)
                # A cmake wheel inside the venv is a second route when apt cannot
                # be used at all (no root).
                if not _sh.which("cmake"):
                    _ok2, _ = _try_pip("cmake")
                    print(f"[venv] pip install cmake: {'OK' if _ok2 else 'failed'}",
                          flush=True)

            _kenlm_ok, _kenlm_from, _last = False, "", ""
            for _spec in ("pypi-kenlm", "kenlm"):
                _ok, _out = _try_pip(_spec)
                print(f"[venv] pip install {_spec}: {'OK' if _ok else 'failed'}", flush=True)
                if _ok:
                    _kenlm_ok, _kenlm_from = True, _spec
                    break
                _last = _out

            # THE ACTUAL FIX for Python 3.12+.
            #
            # Both routes above compile `python/kenlm.cpp`, a Cython-generated file
            # CHECKED INTO the repo (python/CMakeLists.txt does
            # `add_library(kenlm_python MODULE kenlm.cpp ...)` -- it never runs
            # Cython). That file was generated by Cython 0.29.x and calls private
            # CPython APIs that changed:
            #
            #   error: too few arguments to function '_PyLong_AsByteArray(...)'
            #   error: '_PyGen_SetStopIterationValue' was not declared in this scope
            #
            # Nothing about the toolchain fixes that -- cmake, g++ and Python.h were
            # all present when it failed. The fix is to REGENERATE kenlm.cpp from
            # kenlm.pyx with Cython 3.x, which targets 3.13 correctly, and then
            # build. No patching of kenlm's own sources is involved.
            if not _kenlm_ok:
                print("[venv] pre-generated kenlm.cpp is too old for this Python; "
                      "regenerating it with Cython 3.x", flush=True)
                _ok, _out = _try_pip("cython>=3.0")
                print(f"[venv] pip install cython>=3.0: {'OK' if _ok else 'failed'}",
                      flush=True)
                _src = base_dir / "kenlm_src"
                try:
                    if _src.exists():
                        import shutil as _sh2
                        _sh2.rmtree(_src)
                    # urllib+zipfile rather than git: git may not be installed, and
                    # this needs no extra binary.
                    import io as _io
                    import urllib.request as _ur
                    import zipfile as _zf
                    _url = "https://github.com/kpu/kenlm/archive/refs/heads/master.zip"
                    print(f"[venv] downloading {_url}", flush=True)
                    with _ur.urlopen(_url, timeout=120) as _r:
                        _z = _zf.ZipFile(_io.BytesIO(_r.read()))
                    _z.extractall(base_dir)
                    (base_dir / "kenlm-master").rename(_src)

                    _cy = str(venv_dir / "bin" / "cython")
                    _ok2, _out2 = _sh_run(
                        f'cd "{_src}" && "{_cy}" --cplus -3 python/kenlm.pyx '
                        f'-o python/kenlm.cpp')
                    print(f"[venv] cython regenerate: {'OK' if _ok2 else 'FAILED'}",
                          flush=True)
                    if not _ok2:
                        print(_out2[-1200:], flush=True)
                    else:
                        _ok3, _out3 = _try_pip(str(_src))
                        print(f"[venv] pip install {_src}: "
                              f"{'OK' if _ok3 else 'failed'}", flush=True)
                        if _ok3:
                            _kenlm_ok, _kenlm_from = True, "source + Cython 3 regen"
                        else:
                            _last = _out3
                except Exception as _e:
                    print(f"[venv] source build failed: {type(_e).__name__}: {_e}",
                          flush=True)

            if not _kenlm_ok:
                print("[venv] --- tail of the last kenlm build failure ---", flush=True)
                print(_last[-2500:], flush=True)
                print("[venv] --- end ---", flush=True)
                print("[venv] kenlm unavailable. eval_asr.py still works WITHOUT "
                      "--lm and produces the greedy column; only the +KenLM column "
                      "is lost. NOTE the 100h baseline's headline 5.1% WER was a "
                      "+KenLM number -- a greedy-only 300h table must be compared "
                      "against the baseline's GREEDY 10.1%, not against 5.1%.",
                      flush=True)
            else:
                _v, _vout = _sh_run(f'"{py_bin}" -c "import kenlm; print(kenlm.__file__)"')
                print(f"[venv] kenlm: OK from {_kenlm_from} | import check "
                      f"{'passed: ' + _vout.strip() if _v else 'FAILED: ' + _vout[-300:]}",
                      flush=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pip install failed (exit {e.returncode})") from e
        env_msg = mo.md(f"✓ **Environment successfully initialized at `{venv_dir}`** using index: `{torch_index}`")
    except subprocess.CalledProcessError as e:
        env_msg = mo.md(f"❌ pip command failed with exit code {e.returncode}: `{' '.join(e.cmd)}`")
    except Exception as e:
         env_msg = mo.md(f"❌ Error during pip dependency installation: {str(e)}")
    
    env_msg


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3 · Hugging Face authentication & download acceleration

        **Auth does not make downloads faster.** It raises the anonymous rate limit and unlocks
        gated repos, both of which matter here — but the throughput win comes from Hugging Face's
        Xet-backed high-performance transfer. Note `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER`
        are now DEPRECATED — `huggingface_hub` warns that hf_transfer is no longer used and
        points at `HF_XET_HIGH_PERFORMANCE` instead, which is what this toggle sets.

        So: log in for *access and rate limits*, enable Xet high-performance for *speed*. They are
        different problems and it is worth not confusing them.

        Paste a token from https://huggingface.co/settings/tokens (read scope is enough).
        `notebook_login()` is deliberately not used — it renders an ipywidget, which marimo
        does not host.
        """
    )
    return


@app.cell
def _(mo):
    hf_token_input = mo.ui.text(
        label="HF token (read scope)", kind="password", full_width=True,
        placeholder="hf_... — leave blank to use an existing login or $HF_TOKEN",
    )
    hf_fast_toggle = mo.ui.switch(value=True, label="High-performance transfer (HF_XET_HIGH_PERFORMANCE)")
    hf_login_btn = mo.ui.run_button(label="Apply HF settings")
    mo.vstack([hf_token_input, hf_fast_toggle, hf_login_btn])
    return hf_token_input, hf_fast_toggle, hf_login_btn


@app.cell
def _(mo, os, hf_token_input, hf_fast_toggle, hf_login_btn):
    if not hf_login_btn.value:
        hf_status = mo.md("*Set a token (optional) and click **Apply HF settings**.*")
    else:
        _msgs = []
        # huggingface_hub now warns that HF_HUB_ENABLE_HF_TRANSFER is deprecated
        # ("hf_transfer is not used anymore") and that HF_XET_HIGH_PERFORMANCE is
        # the replacement, since transfers go through Xet. Set the new variable and
        # clear the old one so the warning stops and the setting actually applies.
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        if hf_fast_toggle.value:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
            _msgs.append("✓ high-performance transfer enabled (`HF_XET_HIGH_PERFORMANCE=1`)")
        else:
            os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
            _msgs.append("· high-performance transfer disabled")

        try:
            from huggingface_hub import login, get_token, whoami
            _tok = (hf_token_input.value or "").strip() or os.environ.get("HF_TOKEN") or get_token()
            if _tok:
                login(token=_tok)                 # persists to ~/.cache/huggingface/token
                os.environ["HF_TOKEN"] = _tok     # child processes inherit it
                _who = whoami()
                _msgs.append(f"✓ logged in as **{_who.get('name', '?')}**")
            else:
                _msgs.append("⚠ no token — anonymous access (harsher rate limits, no gated repos)")
        except Exception as _e:
            _msgs.append(f"❌ login failed: {type(_e).__name__}: {_e}")

        hf_status = mo.md("\n\n".join(_msgs))
    hf_status
    return (hf_status,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 4 · Download plan — check BEFORE fetching anything

        Lists what each corpus will actually pull, so "am I downloading train-clean-360 by
        accident?" is answered with the repo's real file list instead of an assumption.
        Nothing is downloaded by this cell; it only queries the Hub file index.
        """
    )
    plan_btn = mo.ui.run_button(label="Inspect download plan")
    plan_btn
    return (plan_btn,)


@app.cell
def _(mo, plan_btn):
    if not plan_btn.value:
        plan_out = mo.md("*Click **Inspect download plan**.*")
    else:
        from huggingface_hub import HfApi
        _api = HfApi()
        _report = []
        _targets = [
            ("openslr/librispeech_asr", "clean/train.100", ["train.100", "train-clean-100"]),
            ("edinburghcstr/ami",       "ihm+sdm (STREAMED, early stop)", ["ihm", "sdm"]),
            ("fsicoli/common_voice_22_0", "English shards only", ["/en/", "en_"]),
        ]
        for _repo, _want, _pats in _targets:
            try:
                _files = _api.list_repo_files(_repo, repo_type="dataset")
                _match = [f for f in _files if any(p in f for p in _pats)]
                _extra = []
                for _bad in ("train.360", "train-clean-360", "train.500", "train-other-500"):
                    _n = [f for f in _files if _bad in f]
                    if _n:
                        _extra.append(f"{_bad}: {len(_n)} files present in repo")
                _report.append(
                    f"**{_repo}** — want `{_want}`  \n"
                    f"  matching files: {len(_match)} / {len(_files)} total  \n"
                    + ("  splits we must NOT pull: " + "; ".join(_extra) if _extra else "  no oversized splits in this repo")
                )
            except Exception as _e:
                _report.append(f"**{_repo}** — lookup failed: {type(_e).__name__}: {_e}")
        _report.append(
            "\n---\n"
            "**Reading this:** LibriSpeech names its split explicitly (`split=\"train.100\"`), so "
            "360/500 are never requested even though they live in the same repo. AMI is "
            "**streamed with an early stop** — we want 25h per mic config out of ~80h per split, "
            "so a non-streaming load would fetch ~160h to keep 50h and throw ~69% of the bytes "
            "away. Common Voice uses `snapshot_download(allow_patterns=...)`; the full repo is "
            "578 GB across 100+ languages, so the pattern filter is not optional."
        )
        plan_out = mo.md("\n\n".join(_report))
    plan_out
    return (plan_out,)

@app.cell
def _(mo):
    corpus_sel = mo.ui.dropdown(["all", "librispeech", "common_voice", "ami", "vctk", "combine"], value="all", label="Select Dataset Module")
    force_rebuild_toggle = mo.ui.switch(
        value=False,
        label="Force rebuild (ignore manifests already on disk and re-download)")
    run_prep_btn = mo.ui.run_button(label="Execute Module", kind="warn")
    return corpus_sel, run_prep_btn, force_rebuild_toggle


@app.cell
def _(mo, corpus_sel, run_prep_btn, force_rebuild_toggle):
    mo.md(
        f"""
        ## 5 · Dataset download & manifest packaging
        Select the corpus to prepare, then click 'Execute Module' to download and normalize the dataset.

        A corpus whose manifest is already complete (parses, hits >=90% of its hour target,
        and its sampled audio is still on disk) is REUSED rather than re-downloaded, so
        re-running after a partial failure is cheap. Tick force rebuild to override that.

        **Corpus selection:** {corpus_sel}

        **Force rebuild:** {force_rebuild_toggle}

        **Action:** {run_prep_btn}
        """
    )


@app.cell
def _(mo, corpus_sel, run_prep_btn, force_rebuild_toggle, py_bin, asr_dir, data_dir, cache_dir, subprocess, sys):
    mo.stop(not run_prep_btn.value, mo.md("*Dataset preparation is idle. Select a corpus above and click Execute.*"))
         
    def run_stream(cmd):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")
            
    # prepare_data.py reuses any manifest that is already complete (parses, hits
    # >=90% of its hour target, and its sampled audio files are still on disk), so
    # re-running after a partial failure does NOT re-download the corpora that
    # already succeeded. The toggle overrides that.
    _force = ["--force"] if force_rebuild_toggle.value else []
    if _force:
        print("!!! --force: existing manifests will be IGNORED and re-downloaded\n", flush=True)

    prep_msg = None
    # If "all" is selected, run all 4 sequentially and then combine!
    if corpus_sel.value == "all":
        corpora = ["librispeech", "common_voice", "ami", "vctk"]
        print("=== STARTING FULL DATASET DOWNLOAD & NORMALIZATION ===\\n", flush=True)
        for corp in corpora:
            print(f"\\n--- Preparing: {corp} ---", flush=True)
            prep_cmd = [str(py_bin), str(asr_dir / "prepare_data.py"), "--corpus", corp, "--out", str(data_dir), "--cache", str(cache_dir)] + _force
            try:
                run_stream(prep_cmd)
            except Exception as e:
                mo.stop(True, mo.md(f"❌ Failed to prepare {corp}: {str(e)}"))
        
        # Combine step
        print("\\n--- Combining all manifests ---", flush=True)
        combine_cmd = [str(py_bin), str(asr_dir / "prepare_data.py"), "--combine", "--out", str(data_dir), "--cache", str(cache_dir)]
        try:
            run_stream(combine_cmd)
            print("\\n=== DATASET PREPARATION COMPLETED SUCCESSFULLY ===", flush=True)
        except Exception as e:
            mo.stop(True, mo.md(f"❌ Combination step failed: {str(e)}"))
            
        prep_msg = mo.md("✓ **All datasets successfully prepared and combined! Check the logs below for details.**")
    else:
        # Normal individual execution
        print(f"=== STARTING PREPARATION FOR {corpus_sel.value.upper()} ===\\n", flush=True)
        prep_cmd = [str(py_bin), str(asr_dir / "prepare_data.py")]
        if corpus_sel.value == "combine":
            prep_cmd.append("--combine")
        else:
            prep_cmd.extend(["--corpus", corpus_sel.value])
            
        prep_cmd.extend(["--out", str(data_dir), "--cache", str(cache_dir)])
        # --combine has no manifest of its own to reuse, so --force is only
        # meaningful on a per-corpus build.
        if corpus_sel.value != "combine":
            prep_cmd.extend(_force)
        
        try:
            run_stream(prep_cmd)
            print("\\n=== PREPARATION COMPLETED SUCCESSFULLY ===", flush=True)
            prep_msg = mo.md(f"✓ **Successfully executed preparation for `{corpus_sel.value}`! See logs below.**")
        except Exception as e:
            prep_msg = mo.md(f"❌ Execution failed: {str(e)}")
    prep_msg


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 6 · Drive mirror pre-flight — run this BEFORE the 8-hour training

        This is **not** Colab, so there is no `/content/drive` mount: the OAuth /
        service-account path is the one that actually runs, and it needs
        `google-api-python-client` + `google-auth` installed **and** a credential file
        the sync layer can find.

        Three separate things were silently broken here before: the Google libraries were
        in no dependency list at all, `sync_checkpoint()` returned the same `None` whether
        it uploaded or did nothing, and the training loop therefore logged
        `[drive] mirrored: ...` for a pure no-op. A false confirmation is the worst case —
        it is exactly what makes someone leave a run overnight believing the checkpoints
        are safe.

        `diagnose()` below reports honestly, and the probe actually moves a real byte or
        two. If it does not end in **READY** and an uploaded probe file, checkpoints will
        exist only on ephemeral cloud disk.

        If your credential file is not named `token.json` / `service_account.json`, or
        lives somewhere unusual, set `ECAD_GDRIVE_CREDENTIALS` to its full path.
        """
    )
    drive_cred_input = mo.ui.text(
        label="ECAD_GDRIVE_CREDENTIALS (optional full path to the json)",
        full_width=True, placeholder="/root/token.json",
    )
    drive_test_btn = mo.ui.run_button(label="Diagnose + upload probe file")
    mo.vstack([drive_cred_input, drive_test_btn])
    return drive_cred_input, drive_test_btn


@app.cell
def _(mo, os, sys, Path, asr_dir, drive_cred_input, drive_test_btn):
    if not drive_test_btn.value:
        drive_status = mo.md("*Click **Diagnose + upload probe file**.*")
    else:
        if (drive_cred_input.value or "").strip():
            os.environ["ECAD_GDRIVE_CREDENTIALS"] = drive_cred_input.value.strip()
        _prev = sys.path[:]
        try:
            if str(asr_dir) not in sys.path:
                sys.path.insert(0, str(asr_dir))
            import importlib
            import gdrive_sync as _gs
            importlib.reload(_gs)          # pick up a freshly written module / new env var
            _report = _gs.diagnose()

            _probe = Path(asr_dir) / "_drive_probe.txt"
            _probe.write_text("drive mirror probe\n", encoding="utf-8")
            _ok, _reason = _gs.sync_checkpoint(_probe, "_preflight")
            _verdict = ("✅ probe uploaded — mirroring works: " + _reason) if _ok else \
                       ("❌ probe NOT uploaded: " + _reason)
            drive_status = mo.md(
                "```\n" + _report + "\n```\n\n**Probe result:** " + _verdict +
                ("\n\nCheck Drive for `CLEAR/Phase 1/ASR-300/_preflight/_drive_probe.txt`."
                 if _ok else
                 "\n\n**Do not start the overnight run until this says uploaded.**")
            )
        except Exception as _e:
            drive_status = mo.md(f"❌ pre-flight itself failed: `{type(_e).__name__}: {_e}`")
        finally:
            sys.path = _prev
    drive_status
    return (drive_status,)

@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7 · Background noise & RIR banks

        `fetch_noise_banks.py --verify-only` counts wav FILES. The trainer needs
        DECODED clips, and those are not the same thing — a session was seen with
        verified banks on disk and `0 clips` in the training log at the same time,
        because `torchaudio.load` failed on every file and the error was swallowed.
        The **Check decodability** button below closes that gap without paying for a
        training start-up.

        The `noise/` and `rir/` directories were created, `--noise-dir` / `--rir-dir` were
        wired into the trainer, and `augment.py` read them — but **nothing ever populated
        them**, and `AudioBank` treats an empty folder as "no bank":

        ```python
        if not self.rir_bank.empty()   and rng.random() < cfg.p_rir:   ...
        if not self.noise_bank.empty() and rng.random() < cfg.p_noise: ...
        ```

        So reverb and additive noise were skipped in silence. Training would finish, the log
        would look healthy, and the two augmentation axes that matter MOST for this demo — a
        laptop microphone in a room, not a studio — would never have been applied.

        | source | size | gives |
        |---|---|---|
        | **OpenSLR-28** `rirs_noises.zip` | ~4 GB | simulated + real RIRs **and** pointsource noises — best value per byte, covers both axes alone |
        | **MUSAN** `musan.tar.gz` (optional) | ~11 GB | only `musan/noise/**` is kept (~6 h). Members are filtered while streaming, so just the noise subset hits disk — but the whole archive still crosses the wire |

        Tick *skip MUSAN* if bandwidth is tight; OpenSLR-28 alone already populates both banks.

        **Parallel download.** OpenSLR is slow on a single stream, so the fetcher splits the
        file into byte ranges and pulls them concurrently (and uses `aria2c` instead if it
        happens to be installed). One caveat worth knowing: range requests arrive out of
        order, so they cannot be piped through the tar reader — the parallel MUSAN path
        therefore writes the full ~11 GB archive to disk, extracts `musan/noise/**`, then
        deletes it. It checks for ~14 GB free first and falls back to the slower streaming
        path if there is not enough room. Set connections to **1** to force streaming and
        avoid the temp file entirely.

        **DEMAND is not automated.** It ships as per-scene Zenodo archives whose current URLs
        could not be verified, and a downloader built on guessed URLs is a downloader that
        fails at 3 a.m. Drop it into `noise/` by hand if you want those scenes.
        """
    )
    skip_musan_toggle = mo.ui.switch(value=False, label="Skip MUSAN (saves ~11 GB of transfer)")
    dl_jobs_slider = mo.ui.slider(
        1, 16, value=8, label="Parallel download connections",
        show_value=True)
    fetch_banks_btn = mo.ui.run_button(label="Fetch noise + RIR banks")
    verify_banks_btn = mo.ui.run_button(label="Verify only (no download)")
    mo.vstack([skip_musan_toggle, dl_jobs_slider,
               mo.hstack([fetch_banks_btn, verify_banks_btn])])
    return skip_musan_toggle, dl_jobs_slider, fetch_banks_btn, verify_banks_btn


@app.cell
def _(mo, subprocess, py_bin, asr_dir, noise_dir, rir_dir):
    check_banks_btn = mo.ui.run_button(label="Check decodability (not just file count)")
    check_banks_btn
    return check_banks_btn,


@app.cell
def _(mo, subprocess, py_bin, asr_dir, noise_dir, rir_dir, check_banks_btn):
    mo.stop(not check_banks_btn.value,
            mo.md("*Counting files proves nothing about decoding. Click above to "
                  "actually load the banks.*"))
    _cb = subprocess.run(
        [str(py_bin), str(asr_dir / "augment.py"), "--check-banks",
         "--noise-dir", str(noise_dir), "--rir-dir", str(rir_dir)],
        capture_output=True, text=True)
    print(_cb.stdout + _cb.stderr, flush=True)
    banks_check_msg = mo.md(
        "\u2713 **Both banks decode** \u2014 augmentation will actually be applied."
        if _cb.returncode == 0 else
        "\u274c **A bank does not decode.** See the output above: if files ARE present "
        "and none decoded, re-downloading will not help \u2014 the decoder is the problem. "
        "Training with `aug=on` and an empty bank is the worst case, because nothing "
        "in the training log looks wrong.")
    banks_check_msg
    return


@app.cell
def _(mo, subprocess, py_bin, asr_dir, noise_dir, rir_dir,
      skip_musan_toggle, dl_jobs_slider, fetch_banks_btn, verify_banks_btn):
    if not (fetch_banks_btn.value or verify_banks_btn.value):
        banks_msg = mo.md("*Fetch the banks, or verify what is already there.*")
    else:
        _cmd = [str(py_bin), str(asr_dir / "fetch_noise_banks.py"),
                "--noise-dir", str(noise_dir), "--rir-dir", str(rir_dir),
                "--jobs", str(dl_jobs_slider.value)]
        if verify_banks_btn.value:
            _cmd.append("--verify-only")
        elif skip_musan_toggle.value:
            _cmd.append("--skip-musan")
        try:
            # STREAM the output. `subprocess.run(capture_output=True)` blocks and
            # returns everything only at the end -- and this downloads ~15 GB
            # (MUSAN 11 GB + OpenSLR-28 4 GB), so the cell sat there showing
            # nothing for many minutes and looked exactly like a hang. The
            # training and eval cells already use Popen line-streaming; this one
            # was the odd man out.
            _banks_lines = []
            _p = subprocess.Popen(_cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
            for _line in _p.stdout:
                _banks_lines.append(_line.rstrip())
                print(_line, end="")     # live in the cell's stdout pane
            _p.wait()
            _tail = "\n".join(_banks_lines[-80:])
            # exit 2 means "ran fine but a bank is still empty" -- that is a
            # warning about the NEXT step, not a crash in this one.
            _verdict = {0: "✅ both banks populated — noise and reverb will be applied",
                        2: "⚠ a bank is still EMPTY — those effects will do nothing"}.get(
                            _p.returncode, f"❌ downloader exited {_p.returncode}")
            banks_msg = mo.md(f"**{_verdict}**\n\n```\n{_tail}\n```")
        except Exception as _e:
            banks_msg = mo.md(f"❌ failed to launch downloader: `{type(_e).__name__}: {_e}`")
    banks_msg
    return (banks_msg,)

@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8 · Build a packed cache

        The trainer reads from a packed int16 cache (`audio.i16` + `meta.json`), the same
        format `ablation_engine.py` used — decoding mp3/flac per batch would starve the GPU.

        **Build the 50 h probe cache first.** The WS-layer ablation is a few epochs at high
        LR on 50 h; making it wait on a 34.6 GB / ~30 min full cache is backwards. The
        probe cache is ~5.8 GB and takes a few minutes, and the ablation is what decides
        the config the full run will use.

        | | rows | size | time |
        |---|---|---|---|
        | **50 h probe** | ~33 k | ~5.8 GB | ~3-6 min |
        | 300 h full | ~197 k | ~34.6 GB | ~10-30 min |

        The subset is **corpus-stratified**, not the first N rows. The combined manifest is
        written corpus by corpus, so a naive slice would be pure LibriSpeech — and a
        WS-layer ablation on 100% clean read speech answers the wrong question, since the
        whole hypothesis is that accent/spontaneous/noisy data might move the optimal
        layers. Each corpus keeps its share of the mix (LibriSpeech 33.3%, Common Voice
        35.3%, AMI 16.7%, VCTK 14.7%), so the probe cache is a scale model of the real set.

        The two caches live side by side (`combined_50h_*` and `combined_full_*`), so
        building the full one later does not disturb the probe.
        """
    )
    cache_hours_input = mo.ui.number(
        0.0, 300.0, step=10.0, value=50.0,
        label="Cache hours (50 = probe, 0 = full 300 h)")
    build_cache_btn = mo.ui.run_button(label="Build packed cache", kind="warn")
    mo.vstack([cache_hours_input, build_cache_btn])
    return cache_hours_input, build_cache_btn


@app.cell
def _(mo, subprocess, py_bin, asr_dir, data_dir, cache_dir,
      cache_hours_input, build_cache_btn):
    mo.stop(not build_cache_btn.value,
            mo.md("*Cache builder is idle. Build the **50 h probe cache** first — the "
                  "ablation does not need the full 300 h.*"))

    _manifest = data_dir / "manifest_combined.jsonl"
    mo.stop(not _manifest.exists(),
            mo.md(f"❌ `{_manifest.name}` not found. Run dataset preparation with "
                  f"**combine** first."))

    _hours = float(cache_hours_input.value or 0.0)
    _cmd = [str(py_bin), str(asr_dir / "build_cache.py"),
            "--manifest", str(_manifest), "--cache", str(cache_dir)]
    if _hours > 0:
        _cmd += ["--hours", str(_hours)]
    _label = f"{_hours:.0f}h subset" if _hours > 0 else "full 300h"
    print(f"=== BUILDING PACKED CACHE ({_label}) ===\n", flush=True)

    cache_msg = None
    try:
        _cp = subprocess.Popen(_cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
        _skips = 0
        for _line in _cp.stdout:
            if "WARN skip row" in _line:
                _skips += 1
            print(_line, end="", flush=True)
        _cp.wait()
        if _cp.returncode != 0:
            raise RuntimeError(f"build_cache.py exited {_cp.returncode}")
        _built = sorted(cache_dir.glob("combined_*"))
        cache_msg = mo.md(
            f"✓ **{_label} cache built.** Caches now on disk:\n\n"
            + "\n".join(f"- `{p.name}`" for p in _built)
            + (f"\n\n⚠ {_skips} row(s) could not be decoded and were skipped."
               if _skips else "\n\nNo rows were skipped.")
            + "\n\nNext: run the **WS ablation probe** in §11 against this cache."
        )
    except Exception as _e:
        cache_msg = mo.md(f"❌ Cache build failed: `{type(_e).__name__}: {_e}`")
    cache_msg
    return (cache_msg,)

@app.cell
def _(mo):
    mo.md(
        r"""
        ## 9 · Verify the training data — run this BEFORE training

        Every check below exists because that exact failure already happened in this
        pipeline at least once:

        | check | the failure it catches |
        |---|---|
        | per-corpus hours | Common Voice reported `kept 0.00h` because the audio sat inside tar shards nobody opened |
        | audio present on disk | manifests pointing at files that are not there |
        | **CTC feasibility** | `duration_s * 50 < 2 * len(text)` gives inf/nan loss — AMI genuinely contains 0.02 s clips |
        | vocabulary | an out-of-vocabulary character surviving normalisation breaks KenLM decoding |
        | L2-ARCTIC gate | the held-out OOD test set leaking into training |
        | cache vs manifest | `build_cache.py` silently skips rows it cannot decode |
        | noise / RIR banks | an empty bank makes augmentation a SILENT no-op |

        `--check-all-audio` stats every file instead of a 3,000-row sample. Slower, but it
        is the difference between "probably fine" and "proven". Worth it once, before an
        8-hour run.
        """
    )
    check_all_audio_toggle = mo.ui.switch(
        value=True, label="Stat every audio file (slower, but proves nothing is missing)")
    verify_data_btn = mo.ui.run_button(label="Verify training data")
    mo.vstack([check_all_audio_toggle, verify_data_btn])
    return check_all_audio_toggle, verify_data_btn


@app.cell
def _(mo, subprocess, py_bin, asr_dir, data_dir, cache_dir, noise_dir, rir_dir, lm_dir,
      check_all_audio_toggle, verify_data_btn):
    if not verify_data_btn.value:
        verify_msg = mo.md("*Click **Verify training data**.*")
    else:
        _cmd = [str(py_bin), str(asr_dir / "verify_data.py"),
                "--data", str(data_dir), "--cache", str(cache_dir),
                "--noise-dir", str(noise_dir), "--rir-dir", str(rir_dir),
                "--lm-dir", str(lm_dir)]
        if check_all_audio_toggle.value:
            _cmd.append("--check-all-audio")
        try:
            _vlines = []
            _vp = subprocess.Popen(_cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
            for _line in _vp.stdout:
                _vlines.append(_line.rstrip())
                print(_line, end="")
            _vp.wait()
            _verdict = ("✅ **READY** — every check passed, safe to start training"
                        if _vp.returncode == 0 else
                        "❌ **NOT READY** — see the FAIL lines below. Do not start an "
                        "8-hour run until these are fixed.")
            verify_msg = mo.md(_verdict + "\n\n```\n" + "\n".join(_vlines[-120:]) + "\n```")
        except Exception as _e:
            verify_msg = mo.md(f"❌ could not run the verifier: `{type(_e).__name__}: {_e}`")
    verify_msg
    return (verify_msg,)

@app.cell
def _(mo):
    mo.md(
        r"""
        ## 10 · KenLM language model

        Fetches the **same** LM the 100 h baseline used: LibriSpeech
        `3-gram.pruned.1e-7` from OpenSLR-11, the exact file `kenlm_grid.py` hardcodes.

        Using the same LM is not laziness, it is the point. Swapping in a "better" one
        would change the decoder underneath BOTH rows of the results table, so the
        100 h vs 300 h comparison would stop measuring the acoustic model. The 4-gram
        from the same resource is deliberately skipped — it is far larger and the
        project already ruled it out.

        This is also why `normalize_text` expands digits to words and DROPS
        out-of-vocabulary rows instead of extending the CTC vocabulary: this LM was
        built on LibriSpeech-normalised text (A-Z + apostrophe, no digits, no
        punctuation). Emitting a character it has never seen collapses the beam search
        and gives back the 10.1 -> 5.1 WER gain.

        Without it, evaluation still runs — but greedy-only, and the **+KenLM column of
        the results table stays empty**.
        """
    )
    kenlm_jobs_slider = mo.ui.slider(1, 16, value=8, label="Parallel connections",
                                     show_value=True)
    fetch_kenlm_btn = mo.ui.run_button(label="Fetch KenLM")
    verify_kenlm_btn = mo.ui.run_button(label="Verify only")
    mo.vstack([kenlm_jobs_slider, mo.hstack([fetch_kenlm_btn, verify_kenlm_btn])])
    return kenlm_jobs_slider, fetch_kenlm_btn, verify_kenlm_btn


@app.cell
def _(mo, subprocess, py_bin, asr_dir, lm_dir,
      kenlm_jobs_slider, fetch_kenlm_btn, verify_kenlm_btn):
    if not (fetch_kenlm_btn.value or verify_kenlm_btn.value):
        kenlm_msg = mo.md("*Fetch the LM, or verify what is already there.*")
    else:
        _cmd = [str(py_bin), str(asr_dir / "fetch_kenlm.py"),
                "--lm-dir", str(lm_dir), "--jobs", str(kenlm_jobs_slider.value)]
        if verify_kenlm_btn.value:
            _cmd.append("--verify-only")
        try:
            _klines = []
            _kp = subprocess.Popen(_cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
            for _line in _kp.stdout:
                _klines.append(_line.rstrip())
                print(_line, end="")
            _kp.wait()
            _lm_path = str(lm_dir / "3-gram.pruned.1e-7.arpa")
            _verdict = (f"✅ **READY** — paste this into the KenLM path field below:\n\n"
                        f"`{_lm_path}`"
                        if _kp.returncode == 0 else
                        "⚠ **NOT READY** — evaluation will run greedy-only and the "
                        "+KenLM column will be empty.")
            kenlm_msg = mo.md(_verdict + "\n\n```\n" + "\n".join(_klines[-60:]) + "\n```")
        except Exception as _e:
            kenlm_msg = mo.md(f"❌ could not run the fetcher: `{type(_e).__name__}: {_e}`")
    kenlm_msg
    return (kenlm_msg,)

@app.cell
def _(mo):
    ws_input = mo.ui.text(value="9,10,11,12", label="WS Hidden State Layers")
    epochs_slider = mo.ui.slider(1, 100, step=1, value=30, label="Training Epochs")
    # Two knobs, not one. The old single "Batch Size" slider fed BOTH the utterance
    # cap and the sampler's seconds budget (budget = batch * 20s), so 64 meant
    # 1280 s of padded audio in one forward pass -- an instant CUDA OOM -- while
    # also quadrupling the optimisation batch behind the user's back.
    micro_secs_slider = mo.ui.slider(25, 400, step=25, value=200,
                                     label="Micro-batch (audio sec / GPU step) — memory")
    # Raised the ceiling to 256 and the default to 64. At 16 the utterance cap was
    # the BINDING constraint -- batches averaged ~16 utts / ~82 s padded against a
    # 200 s budget, so 59% of the memory budget went unused and the GPU sat at
    # ~50%. Raising micro_secs instead would have made the gap worse, not better.
    micro_batch_slider = mo.ui.slider(4, 256, step=4, value=64,
                                      label="Utterance cap / GPU step")
    eff_secs_slider = mo.ui.slider(100, 3200, step=100, value=800,
                                   label="Effective batch (audio sec / optimiser step)")
    # Lower bound is 0, and 0 means "use the whole cache" -- the SAME convention
    # as §8's cache-hours field. It used to start at 5.0, so "full" could not be
    # expressed here at all: the only way to train on the 300 h cache was to type
    # 300, which then asked for a cache tagged `combined_300h_*` (the full one is
    # `combined_full_*`) AND re-subsampled a 300 h cache down to 300 h for nothing.
    subset_hours = mo.ui.number(
        0.0, 300.0, step=5.0, value=0.0,
        label="Train on N hours (0 = whole cache)")
    lr_scale_slider = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="Learning Rate Scaling")
    # 0.05 is what the 100h FINAL used. ablation_engine.py's ablation ranked it the
    # strongest single regulariser it tried (-0.86 WER) because it perturbs the
    # representation of a frozen backbone rather than its parameters.
    bb_dropout_input = mo.ui.number(0.0, 0.3, step=0.01, value=0.0,
                                    label="Backbone dropout (0.05 = 100h baseline)")
    init_from_input = mo.ui.text(value="", full_width=True,
                                 label="Fine-tune from run (blank = train from scratch)")
    run_train_btn = mo.ui.run_button(label="Start Retrain Process", kind="danger")
    
    return (ws_input, epochs_slider, micro_secs_slider, micro_batch_slider,
            eff_secs_slider, subset_hours, lr_scale_slider, bb_dropout_input,
            init_from_input, run_train_btn)


@app.cell
def _(mo, runs_dir, json, ws_input, epochs_slider, micro_secs_slider,
      micro_batch_slider, eff_secs_slider, subset_hours, lr_scale_slider,
      bb_dropout_input, init_from_input, run_train_btn):
    # Spell out what "Fine-tune from run" wants: the FOLDER NAME under runs/, not
    # a path. And list what is actually there, marked by whether it can be started
    # from -- head.pt and adapter.pt only exist once a run has recorded at least
    # one [BEST] epoch, so a run that never improved is not a valid source.
    _avail = []
    if runs_dir.exists():
        for _d in sorted(runs_dir.iterdir()):
            if not _d.is_dir():
                continue
            _ok = (_d / "head.pt").is_file() and (_d / "adapter.pt").is_file()
            _cer = ""
            _sp = _d / "summary.json"
            if _sp.is_file():
                try:
                    _cer = f" — best CER {100 * json.loads(_sp.read_text())['best_cer']:.2f}%"
                except Exception:
                    _cer = ""
            _avail.append(f"- {'`' + _d.name + '`' if _ok else _d.name} "
                          + ("**usable**" if _ok else "no head.pt/adapter.pt yet")
                          + _cer)
    _avail_md = ("\n\n**Runs under `runs/`** (type the NAME, not a path):\n"
                 + ("\n".join(_avail) if _avail else "- none yet"))

    layout = mo.vstack([
        mo.md("## 11 · Model training panel (LoRA + WS ablation probe, Drive mirroring)"
              + _avail_md),
        mo.hstack([
            mo.vstack([ws_input, epochs_slider, subset_hours, lr_scale_slider,
                       bb_dropout_input]),
            mo.vstack([micro_secs_slider, micro_batch_slider, eff_secs_slider,
                       init_from_input, run_train_btn])
        ], justify="space-between")
    ])
    # marimo renders a cell's LAST EXPRESSION. `return layout` declares an export,
    # it does NOT display anything -- so without this bare reference the whole
    # training panel is built and then silently thrown away, which is why the
    # widgets never appeared and the run button could not be clicked.
    layout
    return layout


@app.cell
def _(mo):
    free_gpu_btn = mo.ui.run_button(label="Free GPU memory (kills stale trainers)",
                                    kind="warn")
    mo.vstack([
        mo.md("""
        ### Before training: free the card

        `nvidia-smi` cannot answer "is something stale still holding memory?" in this
        container \u2014 the PID namespace collapses every process onto PID 1, which is
        why it printed the same allocation three times. `ps` can.

        What this can and cannot do. Killing a stale `train_asr.py` frees everything
        it held \u2014 that is the only thing that recovers a card showing tens of GB
        with no live run. Emptying a cache only ever frees the CALLING process's own
        blocks; no process can free another's. And nothing here touches memory held
        by a different tenant on a shared GPU.

        \u26a0\ufe0f This kills **every** `train_asr.py` it finds. Do not click it while a
        run you want to keep is going \u2014 including a deliberate parallel ablation arm.
        """),
        free_gpu_btn,
    ])
    return free_gpu_btn,


@app.cell
def _(mo, subprocess, free_gpu_btn):
    import time as _t

    mo.stop(not free_gpu_btn.value, mo.md("*GPU cleanup is idle.*"))

    def _smi():
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        try:
            used, total = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
            return used, total
        except Exception:
            return None, None

    def _trainers():
        r = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True)
        out = []
        for line in r.stdout.splitlines()[1:]:
            if "train_asr.py" in line and " ps " not in line:
                parts = line.split(None, 2)
                out.append((int(parts[0]), int(parts[1]), parts[2][:90]))
        return out

    _used0, _total = _smi()
    _found = _trainers()
    print(f"[gpu] before: {_used0} / {_total} MiB used", flush=True)
    print(f"[gpu] train_asr.py processes found: {len(_found)}", flush=True)
    for _pid, _age, _cmd in _found:
        print(f"       pid {_pid}  age {_age}s  {_cmd}", flush=True)

    _killed = []
    if _found:
        import os as _os
        import signal as _sig
        for _pid, _age, _cmd in _found:
            try:
                _os.kill(_pid, _sig.SIGTERM)
                _killed.append(_pid)
            except Exception as _e:
                print(f"       SIGTERM {_pid} failed: {_e}", flush=True)
        # Give them a chance to exit cleanly; CUDA teardown is not instant and a
        # SIGKILL during it can leave the driver holding the allocation anyway.
        for _ in range(15):
            _t.sleep(1)
            if not _trainers():
                break
        for _pid, _age, _cmd in _trainers():
            print(f"       pid {_pid} ignored SIGTERM, sending SIGKILL", flush=True)
            try:
                _os.kill(_pid, _sig.SIGKILL)
            except Exception:
                pass
        _t.sleep(3)

    _used1, _ = _smi()
    print(f"[gpu] after: {_used1} / {_total} MiB used", flush=True)

    if _used0 is None:
        _msg = "\u26a0\ufe0f `nvidia-smi` is not available here \u2014 cannot measure."
    elif not _found:
        _msg = (f"No `train_asr.py` process is running. **{_used0} MiB is in use by "
                f"something else** \u2014 the marimo kernel's own CUDA context (~500 MiB is "
                f"normal) or another tenant. Nothing here can free that; if the number "
                f"is large and no run is active, restart the marimo kernel.")
    else:
        _msg = (f"Killed {len(_killed)} trainer(s). GPU went **{_used0} \u2192 {_used1} MiB** "
                f"(freed {(_used0 - _used1) / 1024:.1f} GiB). "
                + ("Still high \u2014 the remainder is not ours to free."
                   if _used1 > 2000 else "Card is clear."))
    gpu_free_msg = mo.md(_msg)
    gpu_free_msg
    return


@app.cell
def _(mo, run_train_btn, py_bin, asr_dir, cache_dir, runs_dir, noise_dir, rir_dir,
      ws_input, epochs_slider, micro_secs_slider, micro_batch_slider,
      eff_secs_slider, subset_hours, lr_scale_slider, bb_dropout_input,
      init_from_input, subprocess):
    mo.stop(not run_train_btn.value, mo.md("*Training model is idle. Configure parameters above and click Start.*"))
        
    # Prefer a cache whose size matches what is being asked for: running the 50h
    # probe against the 300h cache would load 34.6 GB of offsets to use a sixth of
    # it, and running the full job against the probe cache would silently train on
    # 50h while the run name claims 300.
    _want_tag = f"combined_{subset_hours.value:.0f}h_" if subset_hours.value else "combined_full_"
    train_caches = sorted(cache_dir.glob(_want_tag + "*"))
    # NO silent fallback. This used to be `... or sorted(glob("combined_*"))`, which
    # meant a tag mismatch quietly trained on whichever cache happened to sort last
    # -- possibly the 50 h probe under a run name claiming 300 h. A wrong cache that
    # trains happily is far worse than a stopped cell.
    _other = [p.name for p in sorted(cache_dir.glob("combined_*"))]
    # The old message said "Build the manifests first!", which sent people back to a
    # step they had already completed. The manifests are not what is missing here --
    # the PACKED CACHE is, and until now there was no cell that could build one.
    mo.stop(not train_caches, mo.md(
        f"❌ **No cache matching `{_want_tag}*`** under `{cache_dir}`.\n\n"
        f"Caches that DO exist: `{_other or 'none'}`\n\n"
        + (f"You asked to train on **{subset_hours.value:.0f} h**. Either set the hours "
           f"field to **0** to use the full cache (`combined_full_*`), or run **§8** with "
           f"hours = **{subset_hours.value:.0f}** to build a matching subset."
           if subset_hours.value else
           "You asked for the **full** cache (`combined_full_*`). Run **§8** with "
           "hours = **0** to build it. The manifests are a separate artefact — having "
           "them is not enough.")))
        
    train_target_cache = str(train_caches[-1])
    
    _tag = f"subset_{subset_hours.value:.0f}h" if subset_hours.value else "full300h"
    # bb_dropout and a fine-tune origin both go in the run name. Two runs that
    # differ only in regularisation would otherwise land in the same directory and
    # the second would silently RESUME the first instead of starting fresh.
    if bb_dropout_input.value:
        _tag += f"_bbdo{bb_dropout_input.value:g}"
    if init_from_input.value.strip():
        _tag += "_ft"
    run_name = f"run_ws_{ws_input.value.replace(',', '_')}_{_tag}"
    
    train_cmd = [
        str(py_bin), str(asr_dir / "train_asr.py"),
        "--run", run_name,
        "--cache-dir", train_target_cache,
        "--ws", ws_input.value,
        "--epochs", str(epochs_slider.value),
        "--micro-secs", str(micro_secs_slider.value),
        "--micro-batch", str(micro_batch_slider.value),
        "--effective-secs", str(eff_secs_slider.value),
        # Only pass it when a subset was actually requested. `--hours-subset 0`
        # is now harmless in train_asr.py, but leaving it off keeps `config.json`
        # honest: hours_subset=null reads as "trained on everything".
        *(["--hours-subset", str(subset_hours.value)] if subset_hours.value else []),
        "--lr-scale", str(lr_scale_slider.value),
        "--bb-dropout", str(bb_dropout_input.value),
        *(["--init-from", str(runs_dir / init_from_input.value.strip())]
          if init_from_input.value.strip() else []),
        "--noise-dir", str(noise_dir),
        "--rir-dir", str(rir_dir),
        "--out", str(runs_dir)
    ]
    
    print(f"=== STARTING Retraining Process: {run_name} ===\\n", flush=True)
    train_msg = None
    try:
        train_p = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for _line in train_p.stdout:
            print(_line, end="", flush=True)
        train_p.wait()
        if train_p.returncode != 0:
            raise RuntimeError(f"Training script failed with code {train_p.returncode}: {' '.join(train_cmd)}")
            
        print("\\n=== RETRAINING PROCESS COMPLETED SUCCESSFULLY ===", flush=True)
        train_msg = mo.md(
            f"""
            ✓ **Successfully finished training for `{run_name}`!**
            
            **Google Drive Auto-Mirror:** `train_asr.py` mirrors `ep001.pt`, `ep002.pt`, ..., `last.pt`, `history.jsonl`, `head.pt` and `adapter.pt` to `CLEAR/Phase 1/ASR-300` **after every epoch**, then the deployable artefacts plus `summary.json` once training ends. Sync failures are logged and swallowed on purpose -- a Drive hiccup must not kill an 8-hour run, so check the `[drive]` lines below to confirm the mirror actually happened rather than assuming it did.
            """
        )
    except Exception as e:
        train_msg = mo.md(f"❌ Training failed: {str(e)}")
    train_msg


@app.cell
def _(mo, runs_dir):
    _ws_runs = sorted(p.name for p in runs_dir.glob("*")
                      if (p / "history.jsonl").exists()) if runs_dir.exists() else []
    ws_run_sel = mo.ui.multiselect(options=_ws_runs, value=_ws_runs[-3:],
                                   label="Runs to inspect")
    ws_view_btn = mo.ui.run_button(label="Read layer weights")
    mo.vstack([
        mo.md("""
        ## 12 \u00b7 Which layer is the weighted sum actually using?

        `head.layer_w` is a softmax over the layers in `--ws`, and it was already
        being written to `history.jsonl` every epoch as the `"w"` field \u2014 it just
        was not printed anywhere, so the one number that says *which* layer the
        model leans on was invisible during a run. It is now in the epoch log, and
        this cell reads it back for runs that already finished (or crashed).

        **Read H/Hmax before the argmax.** `layer_w` starts at zeros, so the
        softmax begins perfectly uniform at 1.000. A run that ends near 1.000 has
        selected nothing, and its argmax is noise.

        **This does not replace the ablation.** The softmax can only rank layers
        that are *in* `ws`. With `ws=9,10,11,12` it can never tell you layer 6
        would have been better \u2014 layer 6 was never on the menu.
        """),
        mo.hstack([ws_run_sel, ws_view_btn], justify="start"),
    ])
    return ws_run_sel, ws_view_btn


@app.cell
def _(mo, json, runs_dir, ws_run_sel, ws_view_btn):
    mo.stop(not ws_view_btn.value,
            mo.md("*Pick one or more runs above and click **Read layer weights**.*"))
    mo.stop(not ws_run_sel.value, mo.md("*No run selected.*"))

    # Mirrors train_asr._fmt_ws. Deliberately duplicated rather than imported:
    # train_asr imports augment, which imports torch at module level, and torch
    # lives in the asr311 venv -- not in the marimo kernel. Importing it here
    # would make this read-only viewer depend on the training environment.
    import math as _math

    _rows = []
    for _run in ws_run_sel.value:
        _rd = runs_dir / _run
        _cfg = {}
        if (_rd / "config.json").exists():
            _cfg = json.loads((_rd / "config.json").read_text())
        _ws = _cfg.get("ws") or []
        _hist = [json.loads(_l) for _l in
                 (_rd / "history.jsonl").read_text().splitlines() if _l.strip()]
        for _h in _hist:
            _w = [float(x) for x in (_h.get("w") or [])]
            if not _w:
                continue
            _labels = _ws if len(_ws) == len(_w) else list(range(len(_w)))
            _top = max(range(len(_w)), key=lambda i: _w[i])
            _H = -sum(x * _math.log(x) for x in _w if x > 0)
            _Hmax = _math.log(len(_w)) if len(_w) > 1 else 1.0
            _ratio = _H / _Hmax if _Hmax else 1.0
            _rows.append({
                "run": _run,
                "epoch": _h.get("epoch"),
                "cer %": round(100 * _h["cer"], 2) if _h.get("cer") is not None else None,
                "argmax layer": f"L{_labels[_top]}",
                "argmax weight": round(_w[_top], 3),
                "top2 mass": round(sum(sorted(_w, reverse=True)[:2]), 3),
                "H/Hmax": round(_ratio, 3),
                "selective?": ("no - still uniform" if _ratio > 0.99
                               else "barely" if _ratio > 0.97 else "yes"),
                "weights": "  ".join(f"L{_l} {_v:.3f}" for _l, _v in zip(_labels, _w)),
            })

    if not _rows:
        ws_table = mo.md("No `w` field found in the selected histories \u2014 those runs "
                         "predate the layer-weight logging.")
    else:
        _last = {}
        for _r in _rows:
            _last[_r["run"]] = _r
        _summary = "\n".join(
            f"- **{_r['run']}** \u2014 e{_r['epoch']}: argmax **{_r['argmax layer']}** "
            f"({_r['argmax weight']:.3f}), top2 {_r['top2 mass']:.2f}, "
            f"H/Hmax {_r['H/Hmax']:.3f} \u2192 selective: {_r['selective?']}"
            for _r in _last.values())
        ws_table = mo.vstack([
            mo.md("### Latest epoch per run\n\n" + _summary),
            mo.md("### Full trajectory"),
            mo.ui.table(_rows, page_size=25, selection=None),
        ])
    ws_table
    return


@app.cell
def _(mo):
    baseline_dest_input = mo.ui.text(value="baseline_100h", full_width=True,
                                     label="Local run name")
    baseline_path_input = mo.ui.text(value="CLEAR/Phase 1/runs/FINAL", full_width=True,
                                     label="Drive folder path")
    fetch_baseline_btn = mo.ui.run_button(label="Fetch baseline from Drive", kind="success")
    mo.vstack([
        mo.md("""
        ## 13 \u00b7 Pull the 100h baseline checkpoint from Drive

        The results table needs the 100h row **re-decoded** with the same LM and the
        same tuned decoder parameters as the 300h row, and that means the baseline
        checkpoint has to be on this machine. It lives only in Drive.

        This reuses the OAuth credentials `gdrive_sync.py` already found for
        checkpoint mirroring \u2014 no second login. The module previously only
        *uploaded*; the download side is new.

        Three things it deliberately does NOT do. It never *creates* a Drive folder
        while resolving the path (`get_or_create_folder` would silently invent an
        empty `Phase 2` for a typo and then report "0 files"). It refuses an
        ambiguous path when two folders share a name, instead of picking one and
        becoming non-deterministic. And it only skips a download on an EXACT size
        match \u2014 a truncated `adapter.pt` from an interrupted transfer is the one
        failure that would otherwise reach `torch.load`.
        """),
        mo.hstack([mo.vstack([baseline_path_input, baseline_dest_input]),
                   fetch_baseline_btn], justify="space-between"),
    ])
    return baseline_dest_input, baseline_path_input, fetch_baseline_btn


@app.cell
def _(mo, sys, subprocess, py_bin, asr_dir, runs_dir, baseline_dest_input,
      baseline_path_input, fetch_baseline_btn):
    mo.stop(not fetch_baseline_btn.value,
            mo.md("*Baseline fetch is idle. Click above to pull it from Drive.*"))
    _dest = runs_dir / baseline_dest_input.value.strip()
    _fb = subprocess.run(
        [str(py_bin), str(asr_dir / "gdrive_sync.py"),
         "--fetch-baseline", str(_dest),
         "--subpath", baseline_path_input.value.strip()],
        capture_output=True, text=True)
    print(_fb.stdout + _fb.stderr, flush=True)
    # Diff the two configs the moment the baseline lands. The headline claim is
    # "300 h beats 100 h", and that only holds if the data is the ONLY thing that
    # differs. Eyeballing two training scripts is how a stray dropout or a
    # different LoRA rank survives into a results table; comparing the recorded
    # configs is not.
    _diff_md = ""
    if _fb.returncode == 0:
        _b_cfg_p = _dest / "config.json"
        _n_cfgs = sorted(runs_dir.glob("*/config.json"),
                         key=lambda q: q.stat().st_mtime)
        _n_cfg_p = next((q for q in reversed(_n_cfgs) if q != _b_cfg_p), None)
        if _n_cfg_p is not None:
            _bc = json.loads(_b_cfg_p.read_text())
            _nc = json.loads(_n_cfg_p.read_text())
            # Only the knobs that change what the MODEL is. Batching, epochs, run
            # name and paths are allowed to differ -- they are not part of the
            # architecture or the optimisation recipe being compared.
            _keys = ["ws", "lora_layers", "lora_r", "lora_alpha", "hid", "sr",
                     "head_lr", "lora_lr", "w_lr", "lr_scale", "weight_decay",
                     "clip", "patience", "stop_patience"]
            _rows, _same = [], 0
            for _k in _keys:
                _bv, _nv = _bc.get(_k, "(absent)"), _nc.get(_k, "(absent)")
                if _bv == _nv:
                    _same += 1
                else:
                    _rows.append(f"| `{_k}` | {_bv} | {_nv} |")
            _hdr = (f"\n\n**Config diff vs `{_n_cfg_p.parent.name}`** "
                    f"({_same}/{len(_keys)} identical)\n\n")
            if _rows:
                _diff_md = (_hdr + "| key | 100h baseline | 300h run |\n|---|---|---|\n"
                            + "\n".join(_rows)
                            + "\n\n\u26a0\ufe0f Anything listed above is a second variable "
                              "changing alongside the data. Either match it or say so "
                              "explicitly when reporting the comparison.")
            else:
                _diff_md = (_hdr + "\u2713 Every model/optimisation knob matches. The data "
                            "is the only difference between the two rows.")
            # Keys the baseline records but this trainer does not know about at all
            # are the dangerous ones: they cannot be compared, so name them.
            _extra = sorted(set(_bc) - set(_nc))
            if _extra:
                _diff_md += (f"\n\nBaseline config has keys this trainer does not: "
                             f"`{_extra}`. Check by hand whether any of them changed the "
                             "model (e.g. backbone dropout, HF SpecAugment masking).")

    baseline_msg = mo.md(
        f"\u2713 **Baseline ready at `{_dest}`** \u2014 it now appears in the run "
        f"dropdowns below, so \u00a714 can tune its decoder and \u00a715 can score it."
        + _diff_md
        if _fb.returncode == 0 else
        "\u274c **Fetch failed.** The output above ends with the full `diagnose()` "
        "report: it names every directory searched for a credential file and says "
        "whether the Drive client could be built at all. A missing folder and a "
        "missing credential produce different messages on purpose.")
    baseline_msg
    return


@app.cell
def _(mo, runs_dir, lm_dir):
    _tune_runs = sorted(p.name for p in runs_dir.glob("*")
                        if (p / "config.json").exists()) if runs_dir.exists() else []
    tune_run_sel = mo.ui.dropdown(options=_tune_runs,
                                  value=_tune_runs[-1] if _tune_runs else None,
                                  label="Run to tune")
    tune_lm_input = mo.ui.text(
        value=str(lm_dir / "3-gram.pruned.1e-7.arpa"), full_width=True,
        label="KenLM .arpa")
    tune_n_slider = mo.ui.slider(100, 2000, step=100, value=500,
                                 label="Tuning utterances (dev-other)")
    run_tune_btn = mo.ui.run_button(label="Tune alpha / beta / beam", kind="warn")
    mo.vstack([
        mo.md("""
        ## 14 \u00b7 Tune the decoder (alpha / beta / beam)

        `eval_asr.py` used to build its decoder with **`alpha=0.5, beta=1.0`
        hardcoded** and the default beam width. Nothing had ever been measured to
        justify those numbers. alpha is the LM weight and beta the word-insertion
        bonus \u2014 at this scale they matter more than the n-gram order.

        They are also **not shareable between checkpoints**: alpha trades the
        acoustic posterior against the LM, and a better-trained acoustic model
        gives sharper, better-calibrated posteriors, so it wants a *lower* alpha.
        One guessed value for both the 100h and 300h rows leaves a *different*
        amount on the table in each row.

        **Protocol.** Tuning runs on LibriSpeech **dev-other**, which is not one of
        the reported sets \u2014 the reported sets are dev-clean and L2-ARCTIC, and
        fitting the decoder on them would be fitting on the test set. ONE parameter
        set is chosen and applied to BOTH reported columns; tuning per test set
        would flatter L2-ARCTIC while meaning nothing at deployment, where nothing
        tells you which domain the call came from.

        The acoustic forward pass runs once and the grid re-decodes cached
        log-probs, so the cost is CPU beam search only. Every run needs its own
        tuning pass, including the 100h baseline, or the two rows are not
        comparable.
        """),
        mo.hstack([mo.vstack([tune_run_sel, tune_n_slider]),
                   mo.vstack([tune_lm_input, run_tune_btn])], justify="space-between"),
    ])
    return tune_run_sel, tune_lm_input, tune_n_slider, run_tune_btn


@app.cell
def _(mo, json, subprocess, py_bin, asr_dir, runs_dir, tune_run_sel,
      tune_lm_input, tune_n_slider, run_tune_btn):
    mo.stop(not run_tune_btn.value,
            mo.md("*Decoder tuning is idle. Pick a run and click **Tune**.*"))
    mo.stop(not tune_run_sel.value, mo.md("\u274c No run selected."))

    tune_run_dir = runs_dir / tune_run_sel.value
    # Params live NEXT TO the checkpoint they belong to, not in a shared file.
    # eval_asr.py refuses a params file whose recorded run_dir does not match, so
    # a per-run path makes accidental cross-model reuse impossible rather than
    # merely unlikely.
    # Run BOTH protocols. dev-other is the clean one; dev-clean is what the 100h
    # baseline used, so it is the only one that puts our number and the published
    # 5.1% under the same rules. Reporting one without the other either flatters
    # us or handicaps us, depending on which one is chosen.
    _results = {}
    tune_msg = None
    try:
        for _which in ("other", "clean"):
            _out_p = tune_run_dir / f"lm_params_{_which}.json"
            _cmd = [
                str(py_bin), str(asr_dir / "tune_lm.py"),
                "--run-dir", str(tune_run_dir),
                "--lm", tune_lm_input.value.strip(),
                "--n", str(tune_n_slider.value),
                "--tune-on", _which,
                "--out", str(_out_p),
            ]
            print(f"=== TUNING DECODER ({_which}): {tune_run_sel.value} ===",
                  flush=True)
            _tp = subprocess.Popen(_cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
            for _line in _tp.stdout:
                print(_line, end="", flush=True)
            _tp.wait()
            # A non-zero exit is not automatically a failure here: kenlm can abort
            # during interpreter teardown, AFTER the params are written. tune_lm.py
            # now os._exit(0)s past that, but the belt-and-braces check is whether
            # the output file exists and parses -- that is the actual deliverable.
            if _out_p.is_file():
                try:
                    _results[_which] = json.loads(_out_p.read_text())
                    if _tp.returncode != 0:
                        print(f"[cell] tune_lm.py exited {_tp.returncode} but "
                              f"{_out_p.name} is valid -- treating as success "
                              f"(teardown crash, not a tuning failure)", flush=True)
                    continue
                except Exception as _je:
                    raise RuntimeError(f"{_out_p} is not valid JSON: {_je}")
            raise RuntimeError(f"tune_lm.py --tune-on {_which} failed with code "
                               f"{_tp.returncode} and wrote no params file")

        # eval reads lm_params.json; point it at the baseline-matching protocol so
        # the two rows of the results table are produced the same way. The strict
        # numbers stay on disk next to it and go in the write-up.
        tune_out_path = tune_run_dir / "lm_params.json"
        tune_out_path.write_text(json.dumps(_results["clean"], indent=2))
        _pp = _results["clean"]
        _warn = "".join(f"\n- \u26a0\ufe0f {w}" for w in _pp.get("warnings", []))
        # Computed OUTSIDE the f-string below: a backslash escape inside an
        # f-string expression is a SyntaxError before Python 3.12, and this
        # notebook should stay parseable by older interpreters.
        _o, _c = _results["other"], _results["clean"]
        _sig = ("KenLM gain is **significant** at 2 SE" if _pp["gain_is_significant_2se"]
                else "**within noise** at 2 SE \u2014 do not report it as a gain")
        tune_msg = mo.md(
            f"""\u2713 **Tuned `{tune_run_sel.value}`**

            | protocol | tuned on | alpha | beta | beam | greedy WER | +KenLM WER | gain |
            |---|---|---|---|---|---|---|---|
            | strict | dev-other | {_o['alpha']} | {_o['beta']} | {_o['beam_width']} | {100 * _o['greedy_wer']:.2f}% | **{100 * _o['tune_wer']:.2f}%** \u00b1 {100 * _o['tune_wer_se']:.2f} | {100 * _o['gain_vs_greedy']:.2f} |
            | baseline-matching | dev-clean | {_c['alpha']} | {_c['beta']} | {_c['beam_width']} | {100 * _c['greedy_wer']:.2f}% | **{100 * _c['tune_wer']:.2f}%** \u00b1 {100 * _c['tune_wer_se']:.2f} | {100 * _c['gain_vs_greedy']:.2f} |
            | old hardcoded | \u2014 | 0.5 | 1.0 | 100 | \u2014 | \u2014 | \u2014 |

            The dev-clean row is tuned on a set \u00a715 also reports, which is what
            `kenlm_grid.py --grid` did for the 100h baseline's published 5.1%. It
            is therefore optimistic for BOTH systems equally \u2014 like-for-like, not
            unbiased. The dev-other row is the unbiased one. {_sig} (dev-other).

            \u00a715 uses the dev-clean parameters so the two table rows match.{_warn}

            These are TUNING-set numbers and are not a result.
            """)
    except Exception as e:
        tune_msg = mo.md(f"\u274c Tuning failed: {str(e)}")
    tune_msg
    return


@app.cell
def _(mo):
    mirror_all_btn = mo.ui.run_button(label="Mirror ALL run results to Drive",
                                      kind="success")
    mo.vstack([
        mo.md("""
        ### Safety net: mirror every result file to Drive

        `train_asr.py` mirrors checkpoints each epoch, but \u00a714's `lm_params*.json`
        and \u00a715's `eval_results.json` were written to molab's local disk **and
        nowhere else**. Those two files ARE the results table, and molab is
        ephemeral: losing the session would mean re-running the evaluation, not
        re-copying a file.

        Both scripts now mirror their own output. This button re-sends everything
        small for every run \u2014 cheap, idempotent, worth clicking before closing
        the tab.
        """),
        mirror_all_btn,
    ])
    return mirror_all_btn,


@app.cell
def _(mo, subprocess, py_bin, asr_dir, runs_dir, mirror_all_btn):
    mo.stop(not mirror_all_btn.value, mo.md("*Result mirroring is idle.*"))
    _names = ["config.json", "summary.json", "history.jsonl", "lm_params.json",
              "lm_params_clean.json", "lm_params_other.json", "eval_results.json"]
    _code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "import gdrive_sync\n"
        "runs = Path(%r)\n"
        "n_ok = n_bad = 0\n"
        "for d in sorted(p for p in runs.iterdir() if p.is_dir()):\n"
        "    for nm in %r:\n"
        "        f = d / nm\n"
        "        if not f.is_file():\n"
        "            continue\n"
        "        ok, why = gdrive_sync.sync_checkpoint(f, d.name)\n"
        "        print(('  OK   ' if ok else '  FAIL ') + d.name + '/' + nm\n"
        "              + ('' if ok else '  -- ' + why), flush=True)\n"
        "        n_ok += int(ok); n_bad += int(not ok)\n"
        "print('mirrored %%d, failed %%d' %% (n_ok, n_bad))\n"
    ) % (str(asr_dir), str(runs_dir), _names)
    _r = subprocess.run([str(py_bin), "-c", _code], capture_output=True, text=True)
    print(_r.stdout + _r.stderr, flush=True)
    mirror_msg = mo.md(
        "\u2713 Mirroring finished \u2014 see the per-file log above. A FAIL line names "
        "its own reason; \u00a76's Drive pre-flight explains credential problems."
        if _r.returncode == 0 else
        f"\u274c Mirroring failed (exit {_r.returncode}). Results are still on local "
        "disk \u2014 do not close the session before this succeeds.")
    mirror_msg
    return


@app.cell
def _(mo, runs_dir, lm_dir):
    run_dirs = sorted([p.name for p in runs_dir.glob("*") if p.is_dir()])
    eval_run_sel = mo.ui.dropdown(run_dirs, value=(run_dirs[0] if run_dirs else None), label="Run to Evaluate")
    # Whisper is a FIXED external reference: identical model, identical set,
    # identical number. Its dev-clean row was already measured in
    # whisper_bench.ipynb for the 100h comparison, so re-running it there
    # costs ~100 minutes and changes nothing. L2-ARCTIC is the new set.
    eval_whisper_sel = mo.ui.dropdown(
        ["l2-arctic", "both", "dev-clean", "none"], value="l2-arctic",
        label="Run Whisper on")
    eval_limit_input = mo.ui.number(0, 3000, step=100, value=500,
                                    label="Utterances per test set (0 = all)")
    # Prefilled with the path fetch_kenlm.py writes to, so the +KenLM column is
    # populated by default instead of silently falling back to greedy-only.
    eval_lm_input = mo.ui.text(
        value=str(lm_dir / "3-gram.pruned.1e-7.arpa"), full_width=True,
        label="KenLM .arpa path (blank = greedy-only)")
    run_eval_btn = mo.ui.run_button(label="Run Evaluation", kind="success")
    return (run_dirs, eval_run_sel, eval_lm_input, eval_whisper_sel,
            eval_limit_input, run_eval_btn)


@app.cell
def _(mo, eval_run_sel, eval_lm_input, eval_whisper_sel, eval_limit_input,
      run_eval_btn):
    mo.md(
        f"""
        ## 15 · Evaluation (dev-clean / L2-ARCTIC, greedy + KenLM, real Whisper baseline)
        No hardcoded numbers here -- this invokes the real `eval_asr.py`, which loads
        `config.json` / `adapter.pt` / `head.pt` from the selected run, decodes both test
        sets, and scores WER/CER with `jiwer`. If a checkpoint or dataset can't be loaded,
        this will raise rather than emit a placeholder result.

        **Run:** {eval_run_sel}
        **KenLM path:** {eval_lm_input}
        **Whisper on:** {eval_whisper_sel} — a fixed external reference; its
        dev-clean numbers already exist in `whisper_bench.ipynb` /
        `meeting/B1_WER.png` and do not change between the 100h and 300h rows.
        **Rows per set:** {eval_limit_input}
        **Action:** {run_eval_btn}
        """
    )


@app.cell
def _(mo, run_eval_btn, eval_run_sel, eval_lm_input, eval_whisper_sel,
      eval_limit_input, py_bin, asr_dir, runs_dir, subprocess):
    mo.stop(not run_eval_btn.value, mo.md("*Evaluation is idle. Select a run above and click 'Run Evaluation'.*"))
    mo.stop(not eval_run_sel.value, mo.md("❌ No trained run selected (or none exist yet under the runs directory)."))

    eval_run_dir = runs_dir / eval_run_sel.value
    eval_out_path = eval_run_dir / "eval_results.json"
    eval_cmd = [
        str(py_bin), str(asr_dir / "eval_asr.py"),
        "--run-dir", str(eval_run_dir),
        "--out", str(eval_out_path),
        "--whisper-on", eval_whisper_sel.value,
        *(["--limit", str(int(eval_limit_input.value))] if eval_limit_input.value else []),
    ]
    if eval_lm_input.value.strip():
        eval_cmd.extend(["--lm", eval_lm_input.value.strip()])
        # Use the tuned params if §13 produced them for THIS run. Without this the
        # eval silently falls back to the old alpha=0.5/beta=1.0 guess and prints a
        # warning saying so -- which is the honest default, but it means the tuning
        # step would have been wasted effort had it not been wired through.
        _lp = eval_run_dir / "lm_params.json"
        if _lp.exists():
            eval_cmd.extend(["--lm-params", str(_lp)])
            print(f"[cell] using tuned decoder params from {_lp}", flush=True)
        else:
            print(f"[cell] no lm_params.json in {eval_run_dir} -- eval will use the "
                  f"UNTUNED alpha=0.5 beta=1.0 beam=100. Run \u00a714 first for a fair "
                  f"KenLM number.", flush=True)

    print(f"=== STARTING EVALUATION: {eval_run_sel.value} ===\\n", flush=True)
    eval_msg = None
    try:
        eval_p = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for _line in eval_p.stdout:
            print(_line, end="", flush=True)
        eval_p.wait()
        if eval_p.returncode != 0:
            raise RuntimeError(f"Evaluation script failed with code {eval_p.returncode}: {' '.join(eval_cmd)}")
        print("\\n=== EVALUATION COMPLETED SUCCESSFULLY ===", flush=True)
        eval_msg = mo.md(f"✓ **Evaluation complete for `{eval_run_sel.value}`.** Results written to `{eval_out_path}`.")
    except Exception as e:
        eval_msg = mo.md(f"❌ Evaluation failed: {str(e)}")
    eval_msg


if __name__ == "__main__":
    app.run()
