# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "gdrivefs", "fsspec", "google-auth-oauthlib"]
# ///
#
# molab Drive pull (marimo). Downloads and extracts the academic SER zips + label CSVs.
#   molab: upload, then press "Download the data". Academic corpora only.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Drive pull")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # Drive pull (academic SER data)

        Pulls the **academic** zips and the label CSVs from `CLEAR/emotion_data`.
        Downloads them into `DATA_ROOT`, extracts them, then deletes each zip. It uses
        `gdrive_fsspec`, so the first pull triggers OAuth once. After that `train_ser` reads
        from the same `DATA_ROOT`.
        """
    )
    return (mo,)


@app.cell
def _():
    import os
    import zipfile
    import csv
    from pathlib import Path

    REMOTE_ROOT = "CLEAR/emotion_data"
    ZIPS = ["cremad.zip", "ravdess.zip", "savee.zip", "tess.zip",
            "jl.zip", "asvp_esd.zip", "kaggle_emergency.zip"]
    SOURCES = [z[:-4] for z in ZIPS]
    LABEL_CSVS = [f"{s}_labels.csv" for s in SOURCES]

    _root = Path("/marimo") if Path("/marimo").exists() else Path.cwd()
    DATA_ROOT = Path(os.environ.get("ECAD_DATA_ROOT", str(_root / "ser_data")))
    LABEL_DIR = DATA_ROOT / "labels"

    def get_fs():
        try:
            from gdrive_fsspec import GoogleDriveFileSystem
        except Exception:
            from gdrivefs import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                     auth_kwargs={"use_local_webserver": False})

    def _get(fs, remote, local):
        (fs.get_file if hasattr(fs, "get_file") else fs.get)(remote, str(local))

    def pull():
        fs = get_fs()
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        LABEL_DIR.mkdir(parents=True, exist_ok=True)

        for c in LABEL_CSVS:                     # 1) label CSVs
            dst = LABEL_DIR / c
            if dst.exists():
                continue
            print(f"[label] {c}", flush=True)
            try:
                _get(fs, f"{REMOTE_ROOT}/labels/{c}", dst)
            except Exception as e:
                print(f"  ⚠ {c} did not download ({type(e).__name__})", flush=True)

        for z in ZIPS:                           # 2) zips: download -> extract -> delete
            name = z[:-4]
            marker = DATA_ROOT / name / ".extracted"
            if marker.exists():
                print(f"[skip] {name} is already extracted", flush=True)
                continue
            local_zip = DATA_ROOT / z
            print(f"[pull] {z}", flush=True)
            _get(fs, f"{REMOTE_ROOT}/zips/{z}", local_zip)
            print(f"[extract] {z} ({local_zip.stat().st_size/1e6:.0f} MB)", flush=True)
            with zipfile.ZipFile(local_zip) as zf:
                zf.extractall(DATA_ROOT)
            (DATA_ROOT / name).mkdir(parents=True, exist_ok=True)
            marker.write_text("ok")
            local_zip.unlink()
            print(f"  ✓ {name}", flush=True)

        print("\n[verify] sample path check:", flush=True)   # 3) do the paths resolve
        for c in LABEL_CSVS:
            p = LABEL_DIR / c
            if not p.exists():
                continue
            with open(p) as f:
                r = list(csv.reader(f))
            if len(r) < 2 or "path" not in r[0]:
                continue
            rel = r[1][r[0].index("path")]
            ok = (DATA_ROOT / rel).exists()
            print(f"  {c:28s} '{rel[:38]}' -> {'FOUND' if ok else 'MISSING (different extract layout?)'}",
                  flush=True)
        print(f"\ndone -> {DATA_ROOT}", flush=True)

    return DATA_ROOT, pull


@app.cell
def _(DATA_ROOT, mo):
    run_btn = mo.ui.run_button(label="▶ Download the data (academic, ~4.2 GB)")
    mo.vstack([mo.md(f"**DATA_ROOT:** `{DATA_ROOT}`  -  academic corpora only"),
               run_btn])
    return (run_btn,)


@app.cell
def _(mo, pull, run_btn):
    if not run_btn.value:
        _out = mo.md("Press **Download the data** (the first pull triggers OAuth once).")
    else:
        try:
            pull()
            _out = mo.md("Download and extract finished. You can now run `train_ser`.")
        except Exception as _e:
            _out = mo.md(f"**ERROR:** {type(_e).__name__}: {_e}")
    _out
    return


if __name__ == "__main__":
    app.run()
