# /// script
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

# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
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
