"""
Module 4: INTERMEDIATE (feature-level) fusion for the fusion benchmark.

Extended (without changing `run_intermediate`'s observable behaviour) with:
  - a class-weighted loss option, gated by common.CLASS_WEIGHTING,
  - a parametrised `_run_intermediate_core` used both by `run_intermediate`
    (fixed defaults matching the original contract exactly) and by
    sweep.py (varies lr / fc1 hidden size / dropout),
  - a `run_intermediate_logits` variant returning val/test logits for the
    post-hoc class-bias correction step.
"""

import numpy as np
import copy
from sklearn.metrics import f1_score
import common as _common
from common import set_seed, y_of, emotion_features, class_weights
# CLASS_WEIGHTING read live off `_common.CLASS_WEIGHTING` — see early.py.


class IntermediateFusionModel:
    """Placeholder for type hints only; the real nn.Module is built lazily
    inside _run_intermediate_core so this module stays torch-free at import
    time."""
    pass


def run_intermediate(splits, emb, seed, device, encoder_name):
    """
    INTERMEDIATE fusion: concatenate learned emotion embedding with frozen text embedding
    as features, then jointly transform through MLP.

    Args:
        splits: dict with "train", "val", "test" keys, each containing list[Row]
        emb: dict with "train", "val", "test" keys, each containing np.ndarray embeddings
        seed: int, random seed
        device: str, "cpu" or "cuda"
        encoder_name: str, encoder identifier (not used here)

    Returns:
        tuple[np.ndarray, np.ndarray]: (y_true_test, y_pred_test) as int arrays
    """
    result = _run_intermediate_core(splits, emb, seed, device, encoder_name)
    return result["y_test"], result["test_pred"]


def run_intermediate_logits(splits, emb, seed, device, encoder_name):
    """
    Same computation as `run_intermediate`, also returning val/test logits
    for the post-hoc class-bias correction step. No retraining.

    Returns (y_val, val_logits, y_test, test_logits).
    """
    result = _run_intermediate_core(splits, emb, seed, device, encoder_name)
    return result["y_val"], result["val_logits"], result["y_test"], result["test_logits"]


def _run_intermediate_core(
    splits, emb, seed, device, encoder_name,
    hidden: int = 256, dropout: float = 0.3, lr: float = 1e-3,
) -> dict:
    """
    Shared core used by `run_intermediate` (defaults = original contract
    hyperparameters) and by sweep.py (varies hidden/dropout/lr).

    `hidden` is the fc1 output size (32 + text_dim -> hidden); fc2 (-> 128)
    and the emotion embedding (6 -> 32) are fixed, matching the contract
    shape and matching how "hidden" is defined for `late`'s text branch.

    Returns a dict with y_val, val_pred, val_logits, y_test, test_pred, test_logits.
    """
    set_seed(seed)

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam

    emotion_feat_train = emotion_features(splits["train"], "train", seed)
    emotion_feat_val = emotion_features(splits["val"], "val", seed)
    emotion_feat_test = emotion_features(splits["test"], "test", seed)

    text_emb_train = emb["train"]
    text_emb_val = emb["val"]
    text_emb_test = emb["test"]

    y_train = y_of(splits["train"])
    y_val = y_of(splits["val"])
    y_test = y_of(splits["test"])

    D = text_emb_train.shape[1]

    class _IntermediateFusionModel(nn.Module):
        def __init__(self, text_dim, hidden):
            super().__init__()
            self.emotion_embedding = nn.Linear(6, 32, bias=False)
            self.fc1 = nn.Linear(32 + text_dim, hidden)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 3)

        def forward(self, emo_idx, text_emb):
            emotion_emb = self.emotion_embedding(emo_idx)
            combined = torch.cat([emotion_emb, text_emb], dim=1)
            x = self.fc1(combined)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            logits = self.fc3(x)
            return logits

    model = _IntermediateFusionModel(D, hidden).to(device)

    emotion_feat_train_t = torch.FloatTensor(emotion_feat_train).to(device)
    emotion_feat_val_t = torch.FloatTensor(emotion_feat_val).to(device)
    emotion_feat_test_t = torch.FloatTensor(emotion_feat_test).to(device)

    text_emb_train_t = torch.FloatTensor(text_emb_train).to(device)
    text_emb_val_t = torch.FloatTensor(text_emb_val).to(device)
    text_emb_test_t = torch.FloatTensor(text_emb_test).to(device)

    y_train_t = torch.LongTensor(y_train).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    train_dataset = TensorDataset(emotion_feat_train_t, text_emb_train_t, y_train_t)
    val_dataset = TensorDataset(emotion_feat_val_t, text_emb_val_t, y_val_t)
    test_dataset = TensorDataset(emotion_feat_test_t, text_emb_test_t)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if _common.CLASS_WEIGHTING:
        w = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1
    best_state_dict = None
    best_val_logits = None
    patience = 8
    patience_counter = 0

    for epoch in range(60):
        model.train()
        for emotion_batch, text_emb_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(emotion_batch, text_emb_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logit_chunks = []
            for emotion_batch, text_emb_batch, _ in val_loader:
                logits = model(emotion_batch, text_emb_batch)
                val_logit_chunks.append(logits.cpu().numpy())
            val_logits_epoch = np.concatenate(val_logit_chunks, axis=0)
            y_val_pred = np.argmax(val_logits_epoch, axis=1)

        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_logits = val_logits_epoch
            patience_counter = 0
            print(f"Epoch {epoch+1}: F1={val_f1:.4f}")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Hand the trained model to the driver if it asked for one (no-op
    # during sweeps, which train hundreds of throwaway models).
    _common.offer_checkpoint(
        f"intermediate_{encoder_name}_{seed}", best_state_dict, best_val_f1,
        meta={"method": "intermediate", "encoder": encoder_name, "seed": seed,
              "regime": _common.EMOTION_REGIME,
              "class_weighting": _common.CLASS_WEIGHTING},
    )
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        test_logit_chunks = []
        for emotion_batch, text_emb_batch in test_loader:
            logits = model(emotion_batch, text_emb_batch)
            test_logit_chunks.append(logits.cpu().numpy())
        test_logits = np.concatenate(test_logit_chunks, axis=0)
        y_test_pred = np.argmax(test_logits, axis=1)

    return {
        "y_val": y_val.astype(np.int64),
        "val_pred": np.argmax(best_val_logits, axis=1).astype(np.int64),
        "val_logits": best_val_logits,
        "y_test": y_test.astype(np.int64),
        "test_pred": y_test_pred.astype(np.int64),
        "test_logits": test_logits,
    }
