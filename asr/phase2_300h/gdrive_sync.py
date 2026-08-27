# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
"""Google Drive mirroring for ASR checkpoints.

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
