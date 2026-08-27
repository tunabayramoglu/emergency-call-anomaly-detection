#!/usr/bin/env python3
"""Get a freshly cloned repository into a working state.

On a bare machine it does, in order:

    1. checks for uv and installs it if missing (asks first)
    2. creates a Python 3.11 virtualenv under app/.venv
    3. installs dependencies (torch CPU + transformers + peft + gradio, then
       optionally pyctcdecode + kenlm)
    4. unpacks the archives in weights/ into the app/models/ layout
    5. downloads the KenLM language model (94 MB, skippable)
    6. pre-fetches mHuBERT-147 and bert-base-uncased from HuggingFace
    7. verifies that everything is in place

Every step is idempotent: running it a second time skips what is already done.

Usage:
    python setup.py                 # full install, prompts before installing uv
    python setup.py --yes           # never prompts
    python setup.py --check         # changes nothing, only reports state
    python setup.py --skip-kenlm    # no 94 MB language model (greedy decoding)
    python setup.py --skip-prefetch # do not pre-fetch the HuggingFace models
    python setup.py --skip-deps     # skip the pip installs

Python 3.11 is preferred but not mandatory. The one place it matters is KenLM:
the bindings do not build on 3.12+ because they ship a pre-generated Cython file.
Without KenLM the demo still runs and simply decodes greedily -- the transcript
degrades from 5.13% to 10.65% WER, which does not change the verdict for
demo-length utterances (see docs/DEMO_BRIEF.md section 4).

This script uses the standard library only, so it needs no installation itself.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
VENV = REPO / "app" / ".venv"
MODELS = REPO / "app" / "models"
LM_DIR = MODELS / "lm"

WINDOWS = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# "av" decodes the .m4a demo clips; libsndfile alone cannot read AAC/MP4.
# scikit-learn, sentence-transformers, pandas and matplotlib are imported by
# fusion/benchmark_modules/, so without them the benchmark cannot be re-run in
# the environment this script builds.
CORE_DEPS = ["transformers", "peft>=0.11", "gradio", "soundfile", "numpy", "av",
             "pytest", "scikit-learn", "sentence-transformers", "pandas",
             "matplotlib"]
LM_DEPS = ["pyctcdecode", "kenlm"]

UV_INSTALL_WINDOWS = (
    'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
)
UV_INSTALL_POSIX = "curl -LsSf https://astral.sh/uv/install.sh | sh"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")


def run(cmd: list[str] | str, *, shell: bool = False, check: bool = True) -> int:
    """Run a command, streaming its output. With check=False a failure is returned."""
    printable = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
    print(f"    $ {printable}")
    rc = subprocess.call(cmd, shell=shell, cwd=REPO)
    if rc != 0 and check:
        raise SystemExit(f"\nCommand failed with {rc}:\n  {printable}")
    return rc


def ask(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"    {question} -> stdin is not a terminal, not proceeding without --yes")
        return False
    return input(f"    {question} [y/N] ").strip().lower() in {"y", "yes"}


def cxx_toolchain() -> tuple[bool, str]:
    """Building kenlm from source needs BOTH cmake and a C++ compiler.

    Having cmake alone is not enough: without a compiler, cmake itself fails at
    the configure stage and the install dies in a wall of build output. The two
    are reported separately so it is clear which one is missing.
    """
    if not shutil.which("cmake"):
        return False, "no cmake"

    if WINDOWS:
        if shutil.which("cl"):
            return True, "cmake + MSVC (cl.exe)"
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
            / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            out = subprocess.run(
                [str(vswhere), "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "displayName"],
                capture_output=True, text=True).stdout.strip()
            if out:
                return True, f"cmake + {out.splitlines()[0]}"
            return False, "cmake present but the C++ workload is not installed in Visual Studio"
        return False, "cmake present but Visual Studio / MSVC is not installed"

    for cc in ("cc", "gcc", "clang", "c++"):
        if shutil.which(cc):
            return True, f"cmake + {cc}"
    return False, "cmake present but no C++ compiler"


def uv_path() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    # The installer does not update PATH for the current session, so also look
    # in the places it installs to.
    for cand in (
        Path.home() / ".local" / "bin" / ("uv.exe" if WINDOWS else "uv"),
        Path.home() / ".cargo" / "bin" / ("uv.exe" if WINDOWS else "uv"),
    ):
        if cand.exists():
            return str(cand)
    return None


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #

def ensure_uv(assume_yes: bool) -> str:
    uv = uv_path()
    if uv:
        ver = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"    already installed: {uv} ({ver})")
        return uv

    installer = UV_INSTALL_WINDOWS if WINDOWS else UV_INSTALL_POSIX
    print("    uv not found. Install command:")
    print(f"      {installer}")
    if not ask("Install uv?", assume_yes):
        raise SystemExit(
            "\nCannot continue without uv. Run the command above by hand and\n"
            "start setup.py again."
        )
    run(installer, shell=True)

    uv = uv_path()
    if not uv:
        raise SystemExit(
            "\nuv was installed but is not on PATH. Open a new terminal and run\n"
            "setup.py again."
        )
    print(f"    installed: {uv}")
    return uv


def ensure_venv(uv: str) -> None:
    if VENV_PY.exists():
        ver = subprocess.run([str(VENV_PY), "--version"], capture_output=True, text=True)
        print(f"    already present: {VENV} ({ver.stdout.strip()})")
        return
    run([uv, "venv", "--python", "3.11", str(VENV)])


def install_deps(uv: str) -> bool:
    """Core dependencies are mandatory, the KenLM ones are not.

    Returns whether the packages needed for KenLM decoding got installed.
    """
    base = [uv, "pip", "install", "--python", str(VENV_PY)]
    run(base + ["torch", "--index-url", TORCH_CPU_INDEX])
    run(base + CORE_DEPS)

    print("\n    KenLM beam search packages (optional):")
    ok, why = cxx_toolchain()
    if not ok:
        # kenlm's sdist builds with cmake and a C++ compiler. Attempting it takes
        # a while and ends in a wall of compiler errors, so say it up front.
        print(f"    {why} — kenlm builds from source, so this is not attempted.")
        print("    To install it, see the prerequisites table in README.md and")
        print("    then run setup.py again.")
        print("    The demo works fine with greedy decoding; this is not an error.")
        return False
    print(f"    build environment: {why}")

    rc = run(base + LM_DEPS, check=False)
    if rc == 0:
        return True
    print(
        "\n    installation failed — the demo will decode greedily, which is not an error.\n"
        "    kenlm publishes no wheels on PyPI and builds from source with cmake and\n"
        "    a C++ compiler. On Windows those are usually missing; on Linux/macOS the\n"
        "    system compiler is normally enough. The bindings also fail to build on\n"
        "    Python 3.12+.\n"
        "\n    If you need beam search:\n"
        "      Windows  -> install Visual Studio Build Tools (C++ workload) + cmake\n"
        "      Linux    -> apt install build-essential cmake\n"
        "    then retry this step with `python setup.py --skip-prefetch`.\n"
        "    If you do not need it, do nothing: greedy decoding takes the transcript\n"
        "    from 5.13% to 10.65% WER and does not change the verdict on demo clips."
    )
    return False


def build_weights() -> None:
    # check=False: a weights problem is reported by the verification step, which
    # is more useful than dying here and never reaching the HuggingFace pre-fetch.
    if run([str(VENV_PY), str(REPO / "app" / "setup_weights.py")], check=False) != 0:
        print("    weights step reported a problem — see the verification below")


def fetch_kenlm() -> None:
    arpa = list(LM_DIR.glob("*.arpa"))
    if arpa:
        print(f"    already present: {arpa[0].name} ({arpa[0].stat().st_size / 1e6:.0f} MB)")
        return
    LM_DIR.mkdir(parents=True, exist_ok=True)
    # KenLM is optional: the demo runs without it, only greedy decoding instead
    # of beam search. A blocked download or a missing C++ toolchain must not
    # abort the install before the HF pre-fetch and the verification step.
    rc = run([str(VENV_PY), str(REPO / "asr" / "phase2_300h" / "fetch_kenlm.py"),
              "--lm-dir", str(LM_DIR)], check=False)
    if rc != 0:
        print("    KenLM not installed. The demo still works; see the README.")


def prefetch_hf() -> int:
    # The return code matters: demo_prefetch exits 2 when a download failed, and
    # a silent failure here only surfaces when the demo is started offline.
    rc = run([str(VENV_PY), str(REPO / "bench" / "demo_prefetch.py"),
              "--weights", str(MODELS), "--encoder", "bert"], check=False)
    if rc != 0:
        print("    pre-fetch did not complete — the demo will need the network "
              "on first run, or re-run this step once you are online")
    return rc


def verify() -> int:
    rc = run([str(VENV_PY), str(REPO / "app" / "setup_weights.py"), "--check"],
             check=False)

    # setup_weights only looks at the weights. The demo cannot start without the
    # packages, so the core imports are checked separately here.
    core = ("torch", "transformers", "peft", "gradio", "soundfile", "numpy", "av")
    missing = [m for m in core
               if subprocess.call([str(VENV_PY), "-c", f"import {m}"],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) != 0]
    if missing:
        print(f"    MISSING  packages: {', '.join(missing)}")
        rc = 1
    else:
        print(f"    ok       {', '.join(core)}")
    return rc


# --------------------------------------------------------------------------- #
# status report
# --------------------------------------------------------------------------- #

def report() -> int:
    problems: list[str] = []

    def line(ok: bool, label: str, detail: str = "", optional: bool = False) -> None:
        if not ok and not optional:
            problems.append(label)
        print(f"  {'ok      ' if ok else 'missing '} {label}{('  ' + detail) if detail else ''}")

    print(f"repo      {REPO}")
    print(f"platform  {platform.system()} {platform.machine()}, python {sys.version.split()[0]}")
    print()

    uv = uv_path()
    line(bool(uv), "uv", uv or "not installed")

    tc_ok, tc_why = cxx_toolchain()
    line(tc_ok, "C++ build environment (optional)",
         tc_why if tc_ok else f"{tc_why} — kenlm cannot be built, greedy decoding",
         optional=True)

    have_venv = VENV_PY.exists()
    detail = ""
    if have_venv:
        detail = subprocess.run([str(VENV_PY), "--version"],
                                capture_output=True, text=True).stdout.strip()
    line(have_venv, "app/.venv", detail)

    if have_venv:
        for mod, label in (("torch", "torch"), ("transformers", "transformers"),
                           ("peft", "peft"), ("gradio", "gradio"),
                           ("kenlm", "kenlm (optional)"),
                           ("pyctcdecode", "pyctcdecode (optional)")):
            rc = subprocess.call([str(VENV_PY), "-c", f"import {mod}"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            line(rc == 0, label, optional="(optional)" in label)

    for sub in ("asr", "ser", "fusion"):
        d = MODELS / sub
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        line(n > 0, f"app/models/{sub}", f"{n} files" if n else "")

    arpa = list(LM_DIR.glob("*.arpa")) if LM_DIR.is_dir() else []
    line(bool(arpa), "KenLM arpa (optional)",
         f"{arpa[0].stat().st_size / 1e6:.0f} MB" if arpa else "absent, greedy decoding",
         optional=True)

    # The verdict has to agree with the lines printed above it. It used to test
    # uv, the venv and models/asr only, so it could print "missing torch" and
    # then "ready" on the same screen.
    if problems:
        print("\nincomplete (" + ", ".join(problems) + ") — run python setup.py")
        return 1
    print("\nready")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repository setup",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="never prompt, including for the uv install")
    ap.add_argument("--check", action="store_true",
                    help="change nothing, only report the current state")
    ap.add_argument("--skip-deps", action="store_true",
                    help="skip the pip installs (if the environment is already set up)")
    ap.add_argument("--skip-kenlm", action="store_true",
                    help="do not download the 94 MB language model")
    ap.add_argument("--skip-prefetch", action="store_true",
                    help="do not pre-fetch the HuggingFace models")
    a = ap.parse_args()

    if a.check:
        return report()

    if not (REPO / "app" / "setup_weights.py").exists():
        raise SystemExit(f"{REPO} does not look like this repository "
                         "(app/setup_weights.py is missing)")

    total = 7
    step(1, total, "uv")
    uv = ensure_uv(a.yes)

    step(2, total, "virtualenv (Python 3.11)")
    ensure_venv(uv)

    step(3, total, "dependencies")
    if a.skip_deps:
        print("    skipped (--skip-deps)")
        have_lm_pkgs = False
    else:
        have_lm_pkgs = install_deps(uv)

    step(4, total, "weights (weights/ -> app/models/)")
    build_weights()

    step(5, total, "KenLM language model")
    if a.skip_kenlm:
        print("    skipped (--skip-kenlm) — the demo will decode greedily")
    elif not have_lm_pkgs and not a.skip_deps:
        print("    skipped — pyctcdecode/kenlm are not installed, so the arpa file "
              "would be useless")
    else:
        fetch_kenlm()

    step(6, total, "HuggingFace pre-fetch (mHuBERT-147 + bert-base-uncased)")
    if a.skip_prefetch:
        print("    skipped (--skip-prefetch) — ~820 MB will download on first run")
    else:
        prefetch_hf()

    step(7, total, "verification")
    rc = verify()

    activate = r"app\.venv\Scripts\activate" if WINDOWS else "source app/.venv/bin/activate"
    print("\n" + "-" * 70)
    if rc == 0:
        print("Setup complete. To run the demo:\n")
        print(f"    {activate}")
        print("    python app/app.py")
    else:
        print("Setup finished incomplete. Look at the MISSING lines above, then")
        print("run `python setup.py --check` to see the state again.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
