"""
film.py — FiLM-conditioned intermediate fusion.

Deliverable #5 of the extension: a SEPARATE method row from `run_intermediate`
(inter.py), which that module is left untouched by.

Rationale (from the benchmark spec): in `run_intermediate`, the emotion
embedding is 32 dims concatenated against a 768-dim (bert) or 384-dim
(minilm) text embedding — only ~4-8% of the input to the first Linear layer.
A plain MLP can learn to all but ignore that slice. FiLM (Feature-wise Linear
Modulation) instead uses the emotion embedding to produce a per-dimension
scale (`gamma`) and shift (`beta`) that multiplicatively/additively modulate
the FULL text embedding, so emotion has leverage over every text dimension
rather than competing for capacity as 4% of concatenated width.

Architecture:
  emotion embedding (nn.Linear(6, 32, bias=False))
    -> Linear(32, D) = gamma_head  -> gamma (N, D)
    -> Linear(32, D) = beta_head   -> beta  (N, D)
  h = gamma * text_emb + beta                      # FiLM modulation
  h -> Linear(D, hidden) -> ReLU -> Dropout -> Linear(hidden, 128) -> ReLU -> Linear(128, 3)
  (same MLP trunk shape as run_intermediate, after the fusion point)

Same training loop, early stopping, class-weighting switch, and equal sweep
budget as `run_intermediate` (see sweep.py).
"""

import numpy as np
import copy
from sklearn.metrics import f1_score
import common as _common
from common import set_seed, y_of, emotion_features, class_weights
# CLASS_WEIGHTING read live off `_common.CLASS_WEIGHTING` — see early.py.


def run_intermediate_film(splits, emb, seed, device, encoder_name):
    """
    FiLM-conditioned intermediate fusion. Registered as a SEPARATE method row
    from `run_intermediate` — does not modify or call into that module.

    Returns (y_true_test, y_pred_test), both int arrays of shape (N_test,),
    values in 0..2 indexing LABELS.
    """
    result = _run_film_core(splits, emb, seed, device, encoder_name)
    return result["y_test"], result["test_pred"]


def run_intermediate_film_logits(splits, emb, seed, device, encoder_name):
    """Same computation as `run_intermediate_film`, also returning val/test
    logits for the post-hoc class-bias correction step."""
    result = _run_film_core(splits, emb, seed, device, encoder_name)
    return result["y_val"], result["val_logits"], result["y_test"], result["test_logits"]


def _run_film_core(
    splits, emb, seed, device, encoder_name,
    hidden: int = 256, dropout: float = 0.3, lr: float = 1e-3,
) -> dict:
    """
    Core FiLM training loop. Hyperparameters default to the same values as
    `run_intermediate`'s defaults so the two are compared on equal footing;
    sweep.py varies hidden/dropout/lr with an identical grid.

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

    # --- FIX 2: mean-centre the text features -------------------------------
    # BERT CLS embeddings are strongly anisotropic: after L2 normalisation the
    # vectors all sit in a narrow cone (cosine similarity ~0.9 between random
    # sentences), so most of each vector is a large shared component and only a
    # small residual actually distinguishes utterances. FiLM is MULTIPLICATIVE,
    # so `gamma * text` amplifies that shared component and the product becomes
    # almost a pure function of the emotion — the model collapses onto the
    # emotion-only solution (val macro-F1 0.4141, which is exactly the
    # emotion-only score) and never recovers. Removing the mean leaves the
    # discriminative residual for gamma to modulate.
    # Statistics come from TRAIN ONLY — computing them over val/test would leak.
    text_mean = emb["train"].mean(axis=0, keepdims=True)
    text_emb_train = emb["train"] - text_mean
    text_emb_val = emb["val"] - text_mean
    text_emb_test = emb["test"] - text_mean

    y_train = y_of(splits["train"])
    y_val = y_of(splits["val"])
    y_test = y_of(splits["test"])

    D = text_emb_train.shape[1]

    class _FiLMFusionModel(nn.Module):
        def __init__(self, text_dim, hidden):
            super().__init__()
            self.emotion_embedding = nn.Linear(6, 32, bias=False)
            self.gamma_head = nn.Linear(32, text_dim)
            self.beta_head = nn.Linear(32, text_dim)
            self.fc1 = nn.Linear(text_dim, hidden)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 3)

            # --- FIX 1: initialise FiLM as the identity transform ------------
            # With default nn.Linear init, gamma starts near 0, so
            # `h = gamma * text + beta` is ~0 on the first step: the text
            # signal is annihilated before the trunk ever sees it and there is
            # no gradient path back to it. Zeroing the weights and setting
            # gamma's bias to 1 makes the layer start as exactly `h = text`
            # (plain passthrough, i.e. the same starting point as a text-only
            # model) and lets the network LEARN modulation away from identity.
            # This is the standard FiLM initialisation.
            nn.init.zeros_(self.gamma_head.weight)
            nn.init.ones_(self.gamma_head.bias)
            nn.init.zeros_(self.beta_head.weight)
            nn.init.zeros_(self.beta_head.bias)

        def forward(self, emo_idx, text_emb):
            emotion_emb = self.emotion_embedding(emo_idx)      # (N, 32)
            gamma = self.gamma_head(emotion_emb)                # (N, D)
            beta = self.beta_head(emotion_emb)                  # (N, D)
            h = gamma * text_emb + beta                          # FiLM modulation, full width
            x = self.fc1(h)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            logits = self.fc3(x)
            return logits

    model = _FiLMFusionModel(D, hidden).to(device)

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
            print(f"[film] Epoch {epoch+1}: F1={val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Hand the trained model to the driver if it asked for one (no-op
    # during sweeps, which train hundreds of throwaway models).
    _common.offer_checkpoint(
        f"intermediate_film_{encoder_name}_{seed}", best_state_dict, best_val_f1,
        meta={"method": "intermediate_film", "encoder": encoder_name, "seed": seed,
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
