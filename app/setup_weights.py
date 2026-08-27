"""
Lay out `app/models/` from the artefacts sitting in `_Staj/`.

    python setup_weights.py            # build it
    python setup_weights.py --check    # verify only

Sources (all already local):
    weights/ASR-300.zip                                    -> models/asr/
    weights/SER.zip                                        -> models/ser/
    weights/WINNER_intermediate_attn_bert_full_p2_seed1.pt -> models/fusion/
    *.arpa anywhere in the repo (optional)                 -> models/lm/

    weights/mHuBERT-147.zip                                -> models/backbone/

`mHuBERT-147.zip` also carries the HuggingFace-format files, so the backbone is
extracted from it rather than downloaded. Two of its five members are skipped:
`checkpoint_best.pt` (1.14 GB, fairseq-era, unused by transformers) and
`pytorch_model.bin` (redundant once `model.safetensors` is present). ~360 MB
lands on disk instead of ~1.75 GB, and no 380 MB download happens.

`bert-base-uncased` is not in any zip and is still fetched by `demo_prefetch.py`.
"""

from __future__ import annotations

import argparse, os
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The checkpoint archives live under `weights/` in the repo. During the internship
# they sat at the repo root, so both locations are accepted.
STAJ = REPO / "weights" if (REPO / "weights").is_dir() else REPO
DEST = HERE / "models"

NEEDED = {
    "asr": ("config.json", "adapter.pt", "head.pt", "lm_params_clean.json"),
    "ser": ("config.json", "adapter.pt", "head.pt"),
}

# What transformers actually reads from a local model directory. Anything else
# in the zip is dead weight for this demo.
BACKBONE_KEEP = {"config.json", "preprocessor_config.json", "model.safetensors"}
BACKBONE_SKIP_REASON = {
    "checkpoint_best.pt": "fairseq-era export; transformers never reads it",
    "pytorch_model.bin": "redundant once model.safetensors is present",
}


# Directories that are large, machine-local, and never hold a project artefact.
# REPO.glob("**/*.arpa") without this walks app/.venv and .git on every run.
_PRUNE = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}


def _find_arpa() -> list[Path]:
    """Every *.arpa in the repository, skipping build and VCS directories."""
    out: list[Path] = []
    for p in REPO.rglob("*.arpa"):
        if _PRUNE.isdisjoint(p.relative_to(REPO).parts):
            out.append(p)
    return sorted(out)


def _extract(zip_path: Path, dest: Path, strip_top: bool) -> int:
    """Extract a zip flat into `dest`, optionally dropping one leading directory
    (SER.zip nests everything under `ser_ws7-8-9-11-12_L1-12/`)."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if strip_top and "/" in name:
                name = name.split("/", 1)[1]
            target = dest / Path(name).name if "/" not in name else dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


def _human(n: int) -> str:
    """Byte count that does not read as 0 for a 593-byte config."""
    for unit, div in (("MB", 2**20), ("KB", 2**10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def _hf_cached(repo_id: str) -> bool:
    """True when repo_id has a materialised snapshot in the HuggingFace cache.

    Checked without importing huggingface_hub, because --check must work before
    the dependencies are installed.
    """
    # HF_HOME, when set, REPLACES the default cache. Falling back to the default
    # as well would report a hit for a cache the library will not read.
    home = os.environ.get("HF_HOME")
    roots = [Path(home) / "hub"] if home else [Path.home() / ".cache" / "huggingface" / "hub"]
    name = "models--" + repo_id.replace("/", "--")
    for r in roots:
        snaps = r / name / "snapshots"
        if snaps.is_dir() and any(d.is_dir() and any(d.iterdir()) for d in snaps.iterdir()):
            return True
    return False


def _existing_backbone() -> Path | None:
    """Reuse pipeline's search so this script and the demo agree on where
    a backbone counts as present. Two independent answers here would mean
    `--check` could pass while the pipeline still downloads, or vice versa."""
    try:
        sys.path.insert(0, str(HERE))
        from pipeline import Paths as _P

        return _P(DEST).backbone()
    except Exception:
        return None


def _zip_member_size(zip_path: Path, basename: str) -> int | None:
    """Uncompressed size of a member, by basename. None if unavailable."""
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as z:
            for i in z.infolist():
                if Path(i.filename).name == basename:
                    return i.file_size
    except (zipfile.BadZipFile, OSError):
        return None
    return None


def build() -> int:
    problems = []

    for zname, sub in (("ASR-300.zip", "asr"), ("SER.zip", "ser")):
        src = STAJ / zname
        if not src.exists():
            problems.append(f"missing {src}")
            continue
        with zipfile.ZipFile(src) as z:
            nested = all("/" in i.filename for i in z.infolist() if not i.is_dir())
        n = _extract(src, DEST / sub, strip_top=nested)
        print(f"  {zname:16s} -> models/{sub}/   ({n} files)")

    fdest = DEST / "fusion"
    fdest.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(STAJ.glob("WINNER_*.pt")) + sorted(STAJ.glob("BEST_*.pt"))
    if not ckpts:
        problems.append(f"no WINNER_*.pt / BEST_*.pt in {STAJ}")
    for c in ckpts:
        shutil.copy2(c, fdest / c.name)
        print(f"  {c.name[:38]:38s} -> models/fusion/")

    bz = STAJ / "mHuBERT-147.zip"
    if (found := _existing_backbone()):
        print(f"  backbone already extracted at {found} — not unpacking the zip")
    elif bz.exists():
        dest = DEST / "backbone"
        dest.mkdir(parents=True, exist_ok=True)
        kept = skipped = 0
        with zipfile.ZipFile(bz) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                base = Path(info.filename).name
                if base not in BACKBONE_KEEP:
                    print(f"      skip {base}  ({BACKBONE_SKIP_REASON.get(base, 'not needed')})")
                    skipped += 1
                    continue
                target = dest / base
                if target.exists() and target.stat().st_size == info.file_size:
                    kept += 1
                    continue
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                kept += 1
        print(f"  mHuBERT-147.zip  -> models/backbone/  ({kept} kept, {skipped} skipped)")
        if not (dest / "model.safetensors").exists():
            problems.append("models/backbone/model.safetensors was not extracted")
    else:
        print(f"  no mHuBERT-147.zip in {STAJ} — the backbone will be downloaded "
              "from HuggingFace on first run")

    arpas = _find_arpa()
    if arpas:
        (DEST / "lm").mkdir(parents=True, exist_ok=True)
        target = DEST / "lm" / arpas[0].name
        if arpas[0].resolve() == target.resolve():
            # Second run: the file we found IS the one we would write. Copying it
            # onto itself raises SameFileError and kills the whole install.
            print(f"  {arpas[0].name:38s} already in models/lm/")
        else:
            shutil.copy2(arpas[0], target)
            print(f"  {arpas[0].name:38s} -> models/lm/")
    else:
        print("  no .arpa found — the demo will decode greedily "
              "(DEMO_BRIEF.md §4 says that is fine)")

    for p in problems:
        print(f"  PROBLEM  {p}")
    return 1 if problems else 0


def _report_decoder(arpa: list[Path]) -> None:
    """Say which decoder will actually run, and why.

    Downloading the .arpa is only half of KenLM decoding: the file is data, the
    decoder is code. `pyctcdecode` does the beam search and `kenlm` reads the
    ARPA, and `kenlm` is the one that will not build on Windows Python >= 3.12.
    Without this, the only clue is one line of startup log that scrolls past
    under Gradio's output, and the demo quietly decodes at ~10.6% WER instead
    of ~5.1%.
    """
    print()
    missing = []
    for mod, why in (("pyctcdecode", "beam search"),
                     ("kenlm", "reads the .arpa")):
        try:
            __import__(mod)
            print(f"  ok       {mod} importable ({why})")
        except ImportError:
            print(f"  MISSING  {mod} ({why})")
            missing.append(mod)

    if not arpa:
        print("  -> decoder: GREEDY (no language model on disk)")
        return
    if missing:
        print(f"  -> decoder: GREEDY — {' and '.join(missing)} not installed, so the "
              f"language model in lm/ cannot be used.")
        print("     Fix:  pip install pyctcdecode pypi-kenlm")
        print("     On Windows, `kenlm` needs Python <= 3.11 and a C++ toolchain; if it "
              "will not build, greedy is a supported fallback (see README §6).")
        return
    print("  -> decoder: KENLM")


def check() -> int:
    ok = True
    for sub, names in NEEDED.items():
        for n in names:
            f = DEST / sub / n
            if not f.is_file():
                print(f"  MISSING  {sub}/{n}")
                ok = False
            elif f.stat().st_size == 0:
                print(f"  EMPTY    {sub}/{n}")
                ok = False
            else:
                print(f"  ok       {sub}/{n}  ({_human(f.stat().st_size)})")

    ck = sorted((DEST / "fusion").glob("*.pt")) if (DEST / "fusion").is_dir() else []
    if not ck:
        print("  MISSING  fusion/*.pt")
        ok = False
    else:
        for c in ck:
            print(f"  ok       fusion/{c.name}  ({_human(c.stat().st_size)})")

    # Ask the pipeline where IT would find a backbone, so this report cannot
    # disagree with what actually happens at load time.
    found = _existing_backbone()
    if found:
        w = next((found / n for n in ("model.safetensors", "pytorch_model.bin")
                  if (found / n).is_file()), None)
        rel = found.relative_to(HERE) if HERE in found.parents else found
        print(f"  ok       backbone at {rel}  "
              f"({w.stat().st_size // 2**20} MB — no HF download needed)")
    elif _hf_cached("utter-project/mHuBERT-147"):
        print("  ok       backbone  in the HuggingFace cache — no download needed")
    else:
        print("  MISSING  backbone  — not on disk and not in the HuggingFace cache; "
              "the demo cannot start offline")
        ok = False

    # The fusion head runs on top of a frozen text encoder that is always fetched
    # from HuggingFace. A check that ignores it can pass on a machine where the
    # demo still has ~420 MB to download.
    if _hf_cached("bert-base-uncased"):
        print("  ok       text encoder bert-base-uncased in the HuggingFace cache")
    else:
        print("  MISSING  text encoder bert-base-uncased — not in the HuggingFace "
              "cache; run `python bench/demo_prefetch.py --encoder bert`")
        ok = False

    # Report a truncated extraction even though `found` may have skipped past it:
    # a 206 MB file sitting in models/backbone/ is confusing dead weight and the
    # user should be told it is safe to delete.
    st = DEST / "backbone" / "model.safetensors"
    if st.is_file():
        want = _zip_member_size(STAJ / "mHuBERT-147.zip", "model.safetensors")
        got = st.stat().st_size
        if want and got != want:
            print(f"  stale    models/backbone/model.safetensors is truncated "
                  f"({got // 2**20} MB of {want // 2**20} MB) — ignored"
                  f"{', a complete copy was found elsewhere' if found else ''}. "
                  "Safe to delete.")

    arpa = sorted((DEST / "lm").glob("*.arpa")) if (DEST / "lm").is_dir() else []
    print(f"  {'ok      ' if arpa else 'absent  '} lm/"
          f"{arpa[0].name if arpa else '(none) — greedy decoding'}")

    # lm_params without an LM is harmless; an LM without params is not, because
    # pyctcdecode's defaults are not the tuned values.
    if arpa and not (DEST / "asr" / "lm_params_clean.json").is_file():
        print("  PROBLEM  an .arpa is present but lm_params_clean.json is not; "
              "the pipeline refuses to guess alpha/beta and will decode greedily")
        ok = False

    _report_decoder(arpa)

    print("\n" + ("all present" if ok else "incomplete"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without building")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    rc = build()
    print()
    return check() or rc


if __name__ == "__main__":
    sys.exit(main())
