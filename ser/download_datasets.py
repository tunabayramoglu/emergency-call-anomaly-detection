"""
Download open SER datasets for training.

Downloads and extracts to data/audio/datasets/<name>/
Each dataset gets a companion CSV at data/labels/<name>_labels.csv
with columns: path, label, arousal, source, speaker_id

Run from the repository root:
    python ser/download_datasets.py
    python ser/download_datasets.py --datasets cremad,ravdess
"""

import argparse, csv, os, sys, zipfile, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotion_taxonomy import arousal

# The repository root, i.e. the parent of ser/. Everything this script writes
# stays inside it; an earlier version resolved one level too high and created
# data/ next to the checkout.
ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio" / "datasets"
LABELS_DIR = ROOT / "data" / "labels"
TMP_DIR = ROOT / "tmp"
for d in (AUDIO_DIR, LABELS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)
CSV_FIELDS = ["path", "label", "arousal", "source", "speaker_id"]

# ── CREMA-D ─────────────────────────────────────────────────────

CREMAD_URL = "https://github.com/CheyneyComputerScience/CREMA-D/raw/master/Audio.zip"
CREMAD_LABEL_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happiness", "NEU": "neutral", "SAD": "sadness",
}

def download_cremad():
    dest = AUDIO_DIR / "cremad"
    label_csv = LABELS_DIR / "cremad_labels.csv"
    if dest.exists() and label_csv.exists():
        print("[CREMA-D] Already downloaded. SKIP.")
        return

    zip_path = TMP_DIR / "cremad_audio.zip"
    if not zip_path.exists():
        print(f"[CREMA-D] Downloading from {CREMAD_URL} ...")
        import urllib.request
        urllib.request.urlretrieve(CREMAD_URL, zip_path)
        print(f"  Downloaded {zip_path.stat().st_size / 1024 / 1024:.0f} MB")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(AUDIO_DIR)
    extracted = AUDIO_DIR / "AudioWAV"
    if extracted.exists():
        shutil.move(str(extracted), str(dest))

    print("  Building label CSV...")
    rows = []
    for fpath in sorted(dest.glob("*.wav")):
        parts = fpath.stem.split("_")
        if len(parts) < 3:
            continue
        raw = parts[1]
        label = CREMAD_LABEL_MAP.get(raw)
        if label is None:
            continue
        aro = arousal(label, "cremad")
        if aro is None:
            continue
        rows.append({"path": str(fpath), "label": label, "arousal": aro,
                     "source": "cremad", "speaker_id": fpath.parent.name})

    _write_csv(label_csv, rows)
    print(f"  {len(rows)} clips -> {label_csv}")
    print("[CREMA-D] Done.\n")


# ── RAVDESS ─────────────────────────────────────────────────────

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

def download_ravdess():
    dest = AUDIO_DIR / "ravdess"
    label_csv = LABELS_DIR / "ravdess_labels.csv"
    if dest.exists() and label_csv.exists():
        print("[RAVDESS] Already downloaded. SKIP.")
        return

    zip_path = TMP_DIR / "ravdess_speech.zip"
    if not zip_path.exists():
        print(f"[RAVDESS] Downloading from {RAVDESS_URL} ...")
        import urllib.request
        urllib.request.urlretrieve(RAVDESS_URL, zip_path)
        print(f"  Downloaded {zip_path.stat().st_size / 1024 / 1024:.0f} MB")

    print("  Extracting (this may take a while)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    print("  Building label CSV...")
    rows = []
    for wav in sorted(dest.rglob("*.wav")):
        # 03-01-06-01-02-01-12.wav  →  mod-voice-emotion-int-stmt-rep-actor
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        label = RAVDESS_EMOTION_MAP.get(emotion_code)
        if label is None:
            continue
        aro = arousal(label, "ravdess")
        if aro is None:
            continue
        speaker_id = parts[-1] if len(parts) >= 8 else "unknown"
        rows.append({"path": str(wav), "label": label, "arousal": aro,
                     "source": "ravdess", "speaker_id": speaker_id})

    _write_csv(label_csv, rows)
    print(f"  {len(rows)} clips -> {label_csv}")
    print("[RAVDESS] Done.\n")


# ── JL Corpus ───────────────────────────────────────────────────

JL_URL = "https://github.com/tli725/JL-Corpus/archive/refs/heads/master.zip"
JL_LABEL_MAP = {
    "Angry": "angry", "Sad": "sad", "Neutral": "neutral",
    "Happy": "happy", "Excited": "excited", "Anxious": "anxious",
    "Worried": "worried", "Apologetic": "apologetic",
    "Pensive": "pensive", "Enthusiastic": "enthusiastic",
}

def download_jl():
    dest = AUDIO_DIR / "jl"
    label_csv = LABELS_DIR / "jl_labels.csv"
    if dest.exists() and label_csv.exists():
        print("[JL] Already downloaded. SKIP.")
        return

    zip_path = TMP_DIR / "jl_corpus.zip"
    if not zip_path.exists():
        print(f"[JL] Downloading from {JL_URL} ...")
        try:
            import urllib.request
            urllib.request.urlretrieve(JL_URL, zip_path)
            print(f"  Downloaded {zip_path.stat().st_size / 1024 / 1024:.0f} MB")
        except Exception as e:
            print(f"  [WARN] Download failed: {e}. SKIP JL.")
            return

    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    src_dir = dest / "JL-Corpus-master"
    if src_dir.exists():
        for item in src_dir.iterdir():
            shutil.move(str(item), str(dest / item.name))
        src_dir.rmdir()

    print("  Building label CSV...")
    rows = []
    meta_path = dest / "metadata.csv"
    if meta_path.exists():
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_raw = row.get("emotion", "").strip()
                label = JL_LABEL_MAP.get(label_raw)
                if label is None:
                    continue
                aro = arousal(label, "jl")
                if aro is None:
                    continue
                wav_name = row.get("filename", "").strip()
                wav_path = dest / "wav" / wav_name
                if wav_path.exists():
                    rows.append({"path": str(wav_path.resolve()), "label": label,
                                 "arousal": aro, "source": "jl",
                                 "speaker_id": "jl_default"})

    _write_csv(label_csv, rows)
    print(f"  {len(rows)} clips -> {label_csv}")
    print("[JL] Done.\n")


# ── ASVP-ESD ────────────────────────────────────────────────────

ASVP_URL = "https://zenodo.org/record/7132783/files/ASVP-ESD.zip"

def download_asvp():
    dest = AUDIO_DIR / "asvp_esd"
    label_csv = LABELS_DIR / "asvp_esd_labels.csv"
    if dest.exists() and label_csv.exists():
        print("[ASVP-ESD] Already downloaded. SKIP.")
        return

    zip_path = TMP_DIR / "asvp_esd.zip"
    if not zip_path.exists():
        print(f"[ASVP-ESD] Downloading from {ASVP_URL} ...")
        import urllib.request
        urllib.request.urlretrieve(ASVP_URL, zip_path)
        print(f"  Downloaded {zip_path.stat().st_size / 1024 / 1024:.0f} MB")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    print("  Building label CSV...")
    rows = []
    for emo_dir in sorted(dest.iterdir()):
        if not emo_dir.is_dir():
            continue
        label = emo_dir.name.lower()
        aro = arousal(label, "asvp_esd")
        if aro is None:
            continue
        for wav in sorted(emo_dir.glob("*.wav")):
            rows.append({"path": str(wav), "label": label, "arousal": aro,
                         "source": "asvp_esd", "speaker_id": "asvp_default"})

    _write_csv(label_csv, rows)
    print(f"  {len(rows)} clips -> {label_csv}")
    print("[ASVP-ESD] Done.\n")


# ── Helpers ─────────────────────────────────────────────────────

def _write_csv(path, rows):
    if not rows:
        print(f"  WARNING: 0 rows for {path}")
        return
    for r in rows:
        r.setdefault("speaker_id", "unknown")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

def summary():
    print("\n=== Dataset Download Summary ===\n")
    total = 0
    for csv_path in sorted(LABELS_DIR.glob("*_labels.csv")):
        with open(csv_path) as f:
            count = max(0, sum(1 for _ in f) - 1)
        name = csv_path.stem.replace("_labels", "")
        total += count
        print(f"  {name:>20}: {count:>6} clips")
    print(f"  {'TOTAL':>20}: {total:>6} clips")


# ── Main ────────────────────────────────────────────────────────

AVAILABLE = {
    "cremad": download_cremad,
    "ravdess": download_ravdess,
    "jl": download_jl,
    "asvp_esd": download_asvp,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SER datasets")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated: cremad,ravdess,jl,asvp_esd (default: all)")
    args = parser.parse_args()

    keys = [k.strip() for k in args.datasets.split(",")] if args.datasets else list(AVAILABLE)
    for key in keys:
        fn = AVAILABLE.get(key)
        if fn:
            fn()
        else:
            print(f"[WARN] Unknown '{key}'. Available: {list(AVAILABLE)}")

    summary()
