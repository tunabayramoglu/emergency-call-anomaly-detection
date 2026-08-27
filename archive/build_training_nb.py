import json
from pathlib import Path

base = Path("C:/Users/Asus/Documents/_Projects/_Staj/notebooks/colab")

def cell(source, ctype="code"):
    return {
        "cell_type": ctype,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }

CLASS6 = [
    "panic", "fear", "urgency", "distress", "confusion", "neutral"
]

cells = []

cells.append(cell("""# SER Training -- 6-Class Classification (mHuBERT-147)

Two stages:
  Stage A: Extract mHuBERT-147 features for all audio clips -> cache to Drive
  Stage B: Train classification head on cached features -> CrossEntropyLoss, 6-way softmax

Prerequisite: Run ser_dataset_setup.ipynb first.

Drive input: MyDrive/CLEAR/emotion_data/audio/ + labels/
Drive output: MyDrive/CLEAR/emotion_data/feat_cache/ + CLEAR/models/""", "markdown"))

cells.append(cell("""# Cell 1: Mount Drive + install deps
import os, sys, csv, time, gc, json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/CLEAR/emotion_data")
AUDIO_DIR = DRIVE_ROOT / "audio"
LABELS_DIR = DRIVE_ROOT / "labels"
CACHE_DIR = DRIVE_ROOT / "feat_cache"
MODEL_DIR = DRIVE_ROOT.parent / "models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Drive root: {DRIVE_ROOT}")
print(f"Cache: {CACHE_DIR}")
print(f"Models: {MODEL_DIR}")

!pip install -q librosa soundfile transformers datasets jiwer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")""))

cells.append(cell("## Config", "markdown"))

cells.append(cell(f"""# Cell 2: Config
CLASSES = {json.dumps(CLASS6)}
N_CLASSES = len(CLASSES)
DATASETS = ["cremad", "kaggle_emergency", "ravdess", "jl", "asvp_esd"]

MAX_TRAIN_SAMPLES = None
TRAIN_BATCH = 256
EVAL_BATCH = 512
NUM_EPOCHS = 100
LR = 1e-3
LR_PATIENCE = 5
LR_FACTOR = 0.5
STOP_PATIENCE = 10
NUM_WORKERS = 2

BACKBONE_ID = "utter-project/mHuBERT-147"
SR = 16000
HID = 768

print("Config:")
print(f"  Classes: {CLASSES}")
print(f"  Datasets: {DATASETS}")
print(f"  Max samples: {MAX_TRAIN_SAMPLES or 'all'}")""))

cells.append(cell("## Classification Head", "markdown"))

cells.append(cell(f"""# Cell 3: Model definition
class EmotionHead(nn.Module):
    def __init__(self, hidden_size=HID, num_classes={len(CLASS6)}):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x)

m = EmotionHead()
print(f"EmotionHead: {{sum(p.numel() for p in m.parameters())}} params")""))

cells.append(cell("## Dataset: Cached Features", "markdown"))

cells.append(cell("""# Cell 4: Dataset class
class CachedFeatureDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["feat_path"], int(row["class_idx"])))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        feat = np.load(self.samples[idx][0]).astype(np.float32)
        return torch.from_numpy(feat), self.samples[idx][1]"""))

cells.append(cell("## Stage A: Feature Extraction", "markdown"))

cells.append(cell("""# Cell 5: Stage A
print("=" * 60)
print("Stage A: Feature Extraction")
print("=" * 60)

from transformers import HubertModel, Wav2Vec2FeatureExtractor
import librosa

t_start = time.time()
backbone = None
feature_extractor = None
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

for ds_name in DATASETS:
    csv_path = LABELS_DIR / f"{ds_name}_labels.csv"
    if not csv_path.exists():
        print(f"SKIP {ds_name} - labels CSV not found.")
        continue
    with open(csv_path) as f:
        clips = list(csv.DictReader(f))
    if MAX_TRAIN_SAMPLES:
        clips = clips[:MAX_TRAIN_SAMPLES]

    feat_dir = CACHE_DIR / f"{ds_name}_feats"
    os.makedirs(feat_dir, exist_ok=True)

    if backbone is None:
        print("Loading mHuBERT-147...")
        t0 = time.time()
        backbone = HubertModel.from_pretrained(BACKBONE_ID)
        backbone.eval()
        backbone.to(DEVICE)
        print(f"  Loaded in {time.time() - t0:.1f}s")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BACKBONE_ID)

    n_ok = 0
    n_skip = 0
    print(f"[{ds_name}] {len(clips)} clips")

    for i, clip in enumerate(clips):
        audio_path = clip["path"]
        feat_path = str(feat_dir / f"feat_{i:06d}.npy")
        if os.path.exists(feat_path):
            n_skip += 1
            continue
        try:
            audio, sr = librosa.load(audio_path, sr=SR, mono=True)
        except Exception:
            print(f"  [WARN] Load failed: {Path(audio_path).name}")
            continue
        if len(audio) < 4000:
            continue
        if len(audio) > 8 * SR:
            audio = audio[:8 * SR]
        inputs = feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt",
            padding=True, do_normalize=True,
        ).to(DEVICE)
        with torch.no_grad():
            feat = backbone(**inputs).last_hidden_state[0].cpu().numpy()
        np.save(feat_path, feat)
        n_ok += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(clips)}] {elapsed:.0f}s | OK={n_ok}")
    print(f"  [{ds_name}] OK={n_ok} SKIP={n_skip}")

print("Stage A complete.")""))

cells.append(cell("## Train/Val/Test Split", "markdown"))

cells.append(cell(f"""# Cell 6: Create split
print("Creating train/val/test split...")
class_to_idx = {{c: i for i, c in enumerate(CLASSES)}}
all_samples = []

for ds_name in DATASETS:
    feat_dir = CACHE_DIR / f"{{ds_name}}_feats"
    csv_path = LABELS_DIR / f"{{ds_name}}_labels.csv"
    if not csv_path.exists() or not feat_dir.exists():
        continue
    with open(csv_path) as f:
        clips = list(csv.DictReader(f))
    if MAX_TRAIN_SAMPLES:
        clips = clips[:MAX_TRAIN_SAMPLES]
    for i, clip in enumerate(clips):
        feat_path = feat_dir / f"feat_{{i:06d}}.npy"
        cls_name = clip.get("class_6", "")
        if cls_name in class_to_idx and feat_path.exists():
            all_samples.append((str(feat_path), class_to_idx[cls_name], ds_name))

print(f"Total samples: {{len(all_samples)}}")

if len(all_samples) == 0:
    print("ERROR: No features found. Run Stage A first.")
else:
    rng = np.random.RandomState(42)
    rng.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.9)

    for name, samples in [("train.csv", all_samples[:n_train]),
                          ("val.csv", all_samples[n_train:n_val]),
                          ("test.csv", all_samples[n_val:])]:
        with open(CACHE_DIR / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feat_path", "class_idx", "dataset"])
            w.writerows(samples)
        print(f"  {{name}}: {{len(samples)}}")

    train_classes = Counter(s[1] for s in all_samples[:n_train])
    print("Class distribution in train:")
    for i, c in enumerate(CLASSES):
        print(f"  {{c:>12}}: {{train_classes.get(i, 0)}}")""))

cells.append(cell("## Stage B: Head Training", "markdown"))

cells.append(cell(f"""# Cell 7: Stage B
print("=" * 60)
print("Stage B: Head Training")
print("=" * 60)

all_samples = locals().get("all_samples", [])
if len(all_samples) == 0:
    print("ERROR: No samples. Run Cell 6 first.")
else:
    train_ds = CachedFeatureDataset(CACHE_DIR / "train.csv")
    val_ds = CachedFeatureDataset(CACHE_DIR / "val.csv")
    test_ds = CachedFeatureDataset(CACHE_DIR / "test.csv")
    print(f"Train: {{len(train_ds)}} | Val: {{len(val_ds)}} | Test: {{len(test_ds)}}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = EmotionHead(HID, N_CLASSES).to(DEVICE)

    # Class weights
    train_classes = Counter(s[1] for s in all_samples[:n_train])
    cls_counts = [train_classes.get(i, 1) for i in range(N_CLASSES)]
    total_c = sum(cls_counts)
    w = torch.tensor([total_c / max(c, 1) for c in cls_counts], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR)

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, n_train = 0, 0, 0
        for feats, targets in train_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
            train_correct += (logits.argmax(1) == targets).sum().item()
            n_train += feats.size(0)
        train_loss /= n_train
        train_acc = train_correct / n_train

        model.eval()
        val_loss, val_correct, n_val = 0, 0, 0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                logits = model(feats)
                loss = criterion(logits, targets)
                val_loss += loss.item() * feats.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
                n_val += feats.size(0)
        val_loss /= n_val
        val_acc = val_correct / n_val
        scheduler.step(val_loss)

        print(f"  Epoch {{epoch:3d}}: train_loss={{train_loss:.4f}} acc={{train_acc:.4f}} | "
              f"val_loss={{val_loss:.4f}} acc={{val_acc:.4f}} | {{time.time()-t0:.1f}}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_emotion_head.pt")
        else:
            no_improve += 1
            if no_improve >= STOP_PATIENCE:
                print(f"  Early stopping at epoch {{epoch}}")
                break

    # Test evaluation
    print(f"Loading best model (epoch {{best_epoch}})...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_emotion_head.pt", map_location=DEVICE))
    model.eval()

    test_loss, test_correct, n_test = 0, 0, 0
    conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int64)
    with torch.no_grad():
        for feats, targets in test_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, targets)
            preds = logits.argmax(1)
            test_loss += loss.item() * feats.size(0)
            test_correct += (preds == targets).sum().item()
            n_test += feats.size(0)
            for p, t in zip(preds, targets):
                conf[t, p] += 1

    test_loss /= n_test
    test_acc = test_correct / n_test

    print(f"Test Results (best epoch={{best_epoch}})")
    print(f"  Test Loss: {{test_loss:.4f}}")
    print(f"  Test Acc:  {{test_acc:.4f}}")
    print()
    print("Per-class accuracy:")
    for i, c in enumerate(CLASSES):
        total_c = conf[i].sum().item()
        correct_c = conf[i, i].item()
        print(f"  {c:>12}: {{correct_c:>4}}/{{total_c:<4}} ({{correct_c/max(total_c,1):.3f}})")

    print()
    print("Confusion matrix:")
    header = " ".join(f"{c[:4]:>4}" for c in CLASSES)
    print(f"{'':>12} {header}")
    for i, c in enumerate(CLASSES):
        row = " ".join(f"{conf[i, j].item():>4}" for j in range(N_CLASSES))
        print(f"  {c:>10}: {row}")

    info = {{
        "best_epoch": best_epoch, "test_acc": test_acc, "test_loss": test_loss,
        "classes": CLASSES,
        "per_class_acc": {c: (conf[i,i].item() / max(conf[i].sum().item(), 1))
                          for i, c in enumerate(CLASSES)},
        "datasets": DATASETS,
    }}
    with open(MODEL_DIR / "emotion_head_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Model saved to {MODEL_DIR / 'best_emotion_head.pt'}")""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

path = base / "ser_training.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
data = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
cc = sum(1 for c in data["cells"] if c["cell_type"] == "code")
mc = sum(1 for c in data["cells"] if c["cell_type"] == "markdown")
print(f"Wrote {len(data['cells'])} cells ({cc} code, {mc} markdown)")
