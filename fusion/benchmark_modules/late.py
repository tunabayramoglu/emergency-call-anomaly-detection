"""
Module 3: LATE (decision-level) fusion for the fusion benchmark.

Two independent unimodal classifiers trained separately, combined only at the decision level
via sklearn LogisticRegression over concatenated softmax probabilities.

Extended (without changing `run_late`'s observable behaviour) with:
  - a class-weighted loss option, gated by common.CLASS_WEIGHTING,
  - a parametrised `_run_late_core` used both by `run_late` (fixed defaults,
    matching the original contract hyperparameters exactly) and by
    sweep.py (varies lr / text-branch hidden size / dropout),
  - a `run_late_logits` variant that also returns val/test combiner scores
    for the post-hoc class-bias correction step.
"""

import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import common as _common
from common import set_seed, y_of, emotion_features, LABELS, class_weights
# CLASS_WEIGHTING is read live off `_common.CLASS_WEIGHTING` at call time (see
# early.py's comment for why: a `from ... import CLASS_WEIGHTING`
# would freeze the flag's value at import time and ignore later toggles).


def run_late(
    splits: dict,
    emb: dict,
    seed: int,
    device: str,
    encoder_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    LATE (decision-level) fusion: two independent branches combined at the final layer.

    Returns (y_true_test, y_pred_test), both int arrays of shape (N_test,),
    values in 0..2 indexing LABELS.
    """
    result = _run_late_core(splits, emb, seed, device, encoder_name)
    return result["y_test"], result["test_pred"]


def run_late_logits(splits, emb, seed, device, encoder_name):
    """
    Same computation as `run_late`, also returning combiner scores (the
    LogisticRegression combiner's `decision_function` output, i.e. its
    pre-softmax scores) on val and test, for the post-hoc class-bias
    correction step. No retraining involved.

    Returns (y_val, val_logits, y_test, test_logits).
    """
    result = _run_late_core(splits, emb, seed, device, encoder_name)
    return result["y_val"], result["val_logits"], result["y_test"], result["test_logits"]


def _run_late_core(
    splits: dict,
    emb: dict,
    seed: int,
    device: str,
    encoder_name: str,
    text_hidden: int = 128,
    text_dropout: float = 0.2,
    lr: float = 1e-3,
) -> dict:
    """
    Shared core used by `run_late` (defaults = original contract hyperparameters,
    so `run_late`'s behaviour is unchanged) and by the sweep in
    sweep.py (which varies text_hidden / text_dropout / lr).

    `text_hidden` / `text_dropout` / `lr` apply ONLY to the text branch — the
    emotion branch (6 -> 32 -> 3) is fixed per the contract, since the sweep
    spec calls "hidden" specifically "the text-branch hidden size" for late.

    Returns a dict with y_val, val_pred, val_logits, y_test, test_pred,
    test_logits (logits = combiner decision_function scores, i.e. pre-argmax).
    """
    set_seed(seed)

    # Lazy torch import
    import torch
    import torch.nn as nn

    # Extract data
    y_train = y_of(splits["train"])
    y_val = y_of(splits["val"])
    y_test = y_of(splits["test"])

    X_emotion_train = emotion_features(splits["train"], "train", seed).astype(np.float32)  # (N_tr, 6)
    X_emotion_val = emotion_features(splits["val"], "val", seed).astype(np.float32)      # (N_val, 6)
    X_emotion_test = emotion_features(splits["test"], "test", seed).astype(np.float32)    # (N_test, 6)

    X_text_train = emb["train"].astype(np.float32)  # (N_tr, D)
    X_text_val = emb["val"].astype(np.float32)      # (N_val, D)
    X_text_test = emb["test"].astype(np.float32)    # (N_test, D)

    D = X_text_train.shape[1]

    # Define emotion branch architecture (fixed shape per contract)
    class EmotionBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 3)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define text branch architecture (hidden/dropout parametrised for the sweep)
    class TextBranch(nn.Module):
        def __init__(self, input_dim, hidden, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, 3)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    class_w = class_weights(y_train) if _common.CLASS_WEIGHTING else None

    # Train emotion branch
    emotion_branch = _train_branch(
        X_emotion_train, y_train, X_emotion_val, y_val,
        EmotionBranch(), device, batch_size=64, max_epochs=40, patience=5,
        lr=1e-3, class_w=class_w,
    )
    emotion_branch = emotion_branch.to(device)
    emotion_branch.eval()

    # Train text branch
    text_branch = _train_branch(
        X_text_train, y_train, X_text_val, y_val,
        TextBranch(D, text_hidden, text_dropout), device, batch_size=64, max_epochs=40, patience=5,
        lr=lr, class_w=class_w,
    )
    text_branch = text_branch.to(device)
    text_branch.eval()

    # Get probabilities on val set (for combiner training)
    with torch.no_grad():
        emotion_val_logits = emotion_branch(torch.tensor(X_emotion_val, device=device))
        emotion_val_probs = torch.softmax(emotion_val_logits, dim=1).cpu().numpy()

        text_val_logits = text_branch(torch.tensor(X_text_val, device=device))
        text_val_probs = torch.softmax(text_val_logits, dim=1).cpu().numpy()

    # Concatenate val probabilities: [emotion_probs (3), text_probs (3)] -> (N_val, 6)
    X_combiner_val = np.hstack([emotion_val_probs, text_val_probs])

    # Fit combiner on val probabilities. class_weight="balanced" honours the
    # same all-or-nothing CLASS_WEIGHTING switch as the branch losses above.
    combiner = LogisticRegression(
        max_iter=1000, class_weight=("balanced" if _common.CLASS_WEIGHTING else None)
    )
    combiner.fit(X_combiner_val, y_val)

    # Late fusion has no single model: it is two independent branches plus an
    # sklearn combiner, so all three are offered together. No-op unless the
    # driver installed a sink.
    _common.offer_checkpoint(
        f"late_{encoder_name}_{seed}",
        {
            "emotion_branch": emotion_branch.state_dict(),
            "text_branch": text_branch.state_dict(),
            "combiner_coef": combiner.coef_,
            "combiner_intercept": combiner.intercept_,
        },
        f1_score(y_val, combiner.predict(X_combiner_val), average="macro", zero_division=0),
        meta={"method": "late", "encoder": encoder_name, "seed": seed,
              "regime": _common.EMOTION_REGIME,
              "class_weighting": _common.CLASS_WEIGHTING},
    )

    # Get probabilities on test set (for combiner prediction)
    with torch.no_grad():
        emotion_test_logits = emotion_branch(torch.tensor(X_emotion_test, device=device))
        emotion_test_probs = torch.softmax(emotion_test_logits, dim=1).cpu().numpy()

        text_test_logits = text_branch(torch.tensor(X_text_test, device=device))
        text_test_probs = torch.softmax(text_test_logits, dim=1).cpu().numpy()

    X_combiner_test = np.hstack([emotion_test_probs, text_test_probs])

    val_logits = combiner.decision_function(X_combiner_val)
    test_logits = combiner.decision_function(X_combiner_test)
    val_pred = combiner.predict(X_combiner_val).astype(np.int64)
    test_pred = combiner.predict(X_combiner_test).astype(np.int64)

    return {
        "y_val": y_val.astype(np.int64),
        "val_pred": val_pred,
        "val_logits": val_logits,
        "y_test": y_test.astype(np.int64),
        "test_pred": test_pred,
        "test_logits": test_logits,
    }


def _train_branch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model,
    device: str,
    batch_size: int = 64,
    max_epochs: int = 40,
    patience: int = 5,
    lr: float = 1e-3,
    class_w: np.ndarray | None = None,
):
    """
    Train a single branch model with early stopping on val macro-F1.

    Returns the model with best state dict restored.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = model.to(device)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, device=device),
        torch.tensor(y_train, device=device, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.tensor(X_val, device=device),
        torch.tensor(y_val, device=device, dtype=torch.long)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer. class_w is None unless CLASS_WEIGHTING is on, in
    # which case it is the same inverse-frequency weight vector for both
    # branches (computed once on train labels, in the caller).
    if class_w is not None:
        weight_tensor = torch.tensor(class_w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping state
    best_val_f1 = -np.inf
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                val_preds.append(preds.cpu().numpy())

        val_preds_all = np.concatenate(val_preds)
        val_f1 = f1_score(y_val, val_preds_all, average="macro", zero_division=0)

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Restore best state
    model.load_state_dict(best_state_dict)
    model.eval()

    return model
