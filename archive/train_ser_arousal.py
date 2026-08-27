"""
SER Arousal Regression Training — mHuBERT-147 backbone.

Two-stage, same pattern as Phase 1 ASR:
  Stage A: Extract mHuBERT-147 features for all audio → cache to disk.
  Stage B: Train regression head on cached features → MSE loss.

Usage:
    python scripts/training/train_ser.py
    python scripts/training/train_ser.py --datasets cremad,ravdess --max_train 512
"""

import argparse, csv, json, os, sys, time, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from packaging.version import Version

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.emotion_taxonomy import arousal

# ── CONFIG ──────────────────────────────────────────────────────

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "tmp" / "ser_feat_cache"
OUTPUT_DIR = ROOT / "models" / "ser_head"
LABELS_DIR = ROOT / "data" / "labels"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
BACKBONE_ID = "utter-project/mHuBERT-147"
SR = 16_000          # mHuBERT sample rate
HID = 768            # mHuBERT-147 hidden dim

# Training
TRAIN_BATCH = 256
EVAL_BATCH = 512
NUM_EPOCHS = 100
LR = 1e-3
LR_PATIENCE = 5      # ReduceLROnPlateau (val MAE)
LR_FACTOR = 0.5
STOP_PATIENCE = 10   # early stopping (val MAE)
NUM_WORKERS = 0       # 0 = no multiprocessing (safer on Windows)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Arousal Regression Head ─────────────────────────────────────

class ArousalHead(nn.Module):
    """Lightweight regression head on top of frozen mHuBERT features.

    Input:  [B, T, HID]  (time × features from mHuBERT)
    Output: [B, 1]       (arousal score 0-1)
    """
    def __init__(self, hidden_size: int = HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, T, HID]  → mean pool over time → [B, HID]
        x = x.mean(dim=1)  # [B, HID]
        return self.net(x).squeeze(-1)  # [B]


# ── Dataset: cached features ────────────────────────────────────

class CachedFeatureDataset(Dataset):
    """Reads pre-computed features + arousal targets from disk.

    Each item is a .npy file + a .csv row with the arousal target.
    """
    def __init__(self, cache_dir: Path, split_csv: str):
        self.samples = []
        csv_path = cache_dir / split_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((
                    row["feat_path"],      # path to .npy feature file
                    float(row["arousal"]), # target
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, target = self.samples[idx]
        feat = np.load(feat_path).astype(np.float32)
        return torch.from_numpy(feat), torch.tensor(target, dtype=torch.float32)


# ── Stage A: Feature Extraction ─────────────────────────────────

def stage_a_extract(datasets: list[str], max_samples: int | None = None):
    """Extract mHuBERT features for all clips in the given datasets.

    Loads mHuBERT-147 once, runs forward on all audio,
    saves per-clip .npy features to CACHE_DIR / <dataset>_feats / .
    Creates train/val/test split CSVs in CACHE_DIR.
    """
    from datasets import load_dataset, Audio
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    print(f"\n{'='*60}")
    print(f"Stage A: Feature Extraction")
    print(f"  Backbone: {BACKBONE_ID}")
    print(f"  Device:   {DEVICE}")
    print(f"  Datasets: {datasets}")
    print(f"{'='*60}\n")

    # Load backbone
    t0 = time.time()
    print("Loading mHuBERT-147...")
    backbone = HubertModel.from_pretrained(BACKBONE_ID)
    backbone.eval()
    backbone.to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s ({sum(p.numel() for p in backbone.parameters())/1e6:.0f}M params)")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BACKBONE_ID)

    all_samples = []  # list of (feat_path, arousal, dataset)

    for ds_name in datasets:
        csv_path = LABELS_DIR / f"{ds_name}_labels.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {ds_name} — labels CSV not found. Run download_datasets.py first.")
            continue

        # Read labels
        with open(csv_path) as f:
            clips = list(csv.DictReader(f))
        if max_samples:
            clips = clips[:max_samples]
        print(f"\n  [{ds_name}] {len(clips)} clips")

        feat_dir = CACHE_DIR / f"{ds_name}_feats"
        os.makedirs(feat_dir, exist_ok=True)

        n_ok = 0
        for i, clip in enumerate(clips):
            audio_path = clip["path"]
            target = float(clip["arousal"])
            feat_path = str(feat_dir / f"feat_{i:06d}.npy")

            # Skip if already cached
            if os.path.exists(feat_path):
                all_samples.append((feat_path, target, ds_name))
                n_ok += 1
                continue

            # Load audio
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=SR, mono=True)
            except Exception as e:
                print(f"    [WARN] Failed to load {audio_path}: {e}")
                continue

            if len(audio) < 4000:  # too short
                continue

            # Truncate to 8s max (typical mHuBERT limit)
            if len(audio) > 8 * SR:
                audio = audio[:8 * SR]

            # Extract features
            inputs = feature_extractor(
                audio, sampling_rate=SR, return_tensors="pt",
                padding=True, do_normalize=True,
            ).to(DEVICE)

            with torch.no_grad():
                outputs = backbone(**inputs)
                # outputs.last_hidden_state: [1, T, HID]
                feat = outputs.last_hidden_state[0].cpu().numpy()  # [T, HID]

            np.save(feat_path, feat)
            all_samples.append((feat_path, target, ds_name))
            n_ok += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(clips)}] {elapsed/60:.1f}min | {n_ok} ok")

        print(f"  [{ds_name}] {n_ok}/{len(clips)} clips extracted")

    if not all_samples:
        print("  No samples extracted. Check dataset paths.")
        return None

    # Create train/val/test split (80/10/10)
    print(f"\n  Creating train/val/test split ({len(all_samples)} total)...")
    rng = np.random.RandomState(42)
    rng.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.9)

    splits = {
        "train.csv": all_samples[:n_train],
        "val.csv":   all_samples[n_train:n_val],
        "test.csv":  all_samples[n_val:],
    }

    for split_name, samples in splits.items():
        csv_out = CACHE_DIR / split_name
        with open(csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feat_path", "arousal", "dataset"])
            for fp, aro, ds in samples:
                writer.writerow([fp, aro, ds])
        print(f"    {split_name}: {len(samples)} samples")

    # Cleanup
    del backbone
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\nStage A complete. Features cached at {CACHE_DIR}")
    return CACHE_DIR


# ── Stage B: Head Training ──────────────────────────────────────

def stage_b_train(cache_dir: Path):
    """Train arousal regression head on cached features."""
    print(f"\n{'='*60}")
    print(f"Stage B: Head Training")
    print(f"  Cache: {cache_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  LR:     {LR}")
    print(f"{'='*60}\n")

    # Datasets
    train_ds = CachedFeatureDataset(cache_dir, "train.csv")
    val_ds   = CachedFeatureDataset(cache_dir, "val.csv")
    test_ds  = CachedFeatureDataset(cache_dir, "test.csv")

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if len(train_ds) == 0:
        print("  ERROR: No training samples. Run Stage A first.")
        return

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model, loss, optimizer
    model = ArousalHead(HID).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    best_val_mae = float("inf")
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_train = 0

        for feats, targets in train_loader:
            feats = feats.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(feats)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            train_mae += (preds - targets).abs().sum().item()
            n_train += feats.size(0)

        train_loss /= n_train
        train_mae /= n_train

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val = 0

        with torch.no_grad():
            for feats, targets in val_loader:
                feats = feats.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(feats)
                loss = criterion(preds, targets)

                val_loss += loss.item() * feats.size(0)
                val_mae += (preds - targets).abs().sum().item()
                n_val += feats.size(0)

        val_loss /= n_val
        val_mae /= n_val
        scheduler.step(val_mae)

        epoch_time = time.time() - t0

        # Log
        history.append((epoch, train_loss, train_mae, val_loss, val_mae))
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_MAE={train_mae:.4f} | "
              f"val_loss={val_loss:.4f} val_MAE={val_mae:.4f} | "
              f"{epoch_time:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e}")

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_ser_head.pt")
            print(f"    -> saved best model (val_MAE={val_mae:.4f})")
        else:
            no_improve += 1
            if no_improve >= STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {STOP_PATIENCE})")
                break

    # ── Test evaluation ──
    print(f"\n  Loading best model (epoch {best_epoch}) for test evaluation...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_ser_head.pt", map_location=DEVICE))
    model.eval()

    test_loss = 0.0
    test_mae = 0.0
    test_r2_numer = 0.0
    test_r2_denom = 0.0
    all_preds = []
    all_targets = []
    n_test = 0

    with torch.no_grad():
        for feats, targets in test_loader:
            feats = feats.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(feats)

            test_loss += criterion(preds, targets).item() * feats.size(0)
            test_mae += (preds - targets).abs().sum().item()
            n_test += feats.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    test_loss /= n_test
    test_mae /= n_test

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    test_r2 = 1 - ((all_targets - all_preds).pow(2).sum() / (all_targets - all_targets.mean()).pow(2).sum())
    test_r2 = test_r2.item()

    # Threshold accuracy (0.2/0.5/0.8 as panic/fear/neutral/calm boundaries)
    thresholds = [0.2, 0.5, 0.8]
    thresh_acc = []
    for i, th in enumerate(thresholds):
        if i == 0:
            pred_class = (all_preds < th).int()
            true_class = (all_targets < th).int()
        else:
            prev = thresholds[i-1]
            pred_class = ((all_preds >= prev) & (all_preds < th)).int()
            true_class = ((all_targets >= prev) & (all_targets < th)).int()
        acc = (pred_class == true_class).float().mean().item()
        thresh_acc.append((f"<{th}", acc))

    # Also threshold for >0.8 (panic)
    pred_panic = (all_preds >= 0.8).int()
    true_panic = (all_targets >= 0.8).int()
    panic_acc = (pred_panic == true_panic).float().mean().item()

    print(f"\n{'='*60}")
    print(f"Test Results (best epoch={best_epoch})")
    print(f"{'='*60}")
    print(f"  Test MSE:  {test_loss:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")
    print(f"  Threshold accuracies:")
    for label, acc in thresh_acc:
        print(f"    {label:>8}: {acc:.3f}")
    print(f"    panic>=0.8: {panic_acc:.3f}")

    # Save model info
    info = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "datasets": list(set(s[2] for s in
            list(csv.DictReader(open(CACHE_DIR / "train.csv"))) +
            list(csv.DictReader(open(CACHE_DIR / "val.csv"))) +
            list(csv.DictReader(open(CACHE_DIR / "test.csv"))))),
        "threshold_accuracies": {f"<{t}": acc for (lbl, acc) in thresh_acc},
        "panic_accuracy": panic_acc,
    }
    with open(OUTPUT_DIR / "ser_head_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  Model saved to {OUTPUT_DIR}")
    print(f"  Info saved to {OUTPUT_DIR / 'ser_head_info.json'}")

    return info


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SER arousal regression model")
    parser.add_argument("--datasets", default="cremad,ravdess",
                        help="Comma-separated datasets to train on")
    parser.add_argument("--max_train", type=int, default=None,
                        help="Max samples per dataset (for smoke testing)")
    parser.add_argument("--stage", choices=["A", "B", "both"], default="both",
                        help="Run stage A (extract), B (train), or both")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    print(f"mHuBERT-147 SER Arousal Regression")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device:  {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")

    if args.stage in ("A", "both"):
        cache_dir = stage_a_extract(datasets, max_samples=args.max_train)
        if args.stage == "A":
            sys.exit(0)
    else:
        cache_dir = CACHE_DIR
        # Verify cache exists
        if not (cache_dir / "train.csv").exists():
            print("ERROR: No cached features found. Run Stage A first.")
            sys.exit(1)

    stage_b_train(cache_dir)
