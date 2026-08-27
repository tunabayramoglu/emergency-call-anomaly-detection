"""
attn.py — emotion-conditioned attention pooling over token-level
text features.

Deliverable #6 of the extension. Separate method row from both
`run_intermediate` and `run_intermediate_film`.

Where `run_intermediate` fuses a single pooled (CLS) text vector with the
emotion embedding, this method lets the emotion embedding act as an
attention QUERY over the text encoder's TOKEN-level hidden states — so
emotion can pick out which words in the utterance it is most relevant to,
rather than being concatenated onto one fixed pooled summary.

Requires `encoders.get_token_embeddings`, which returns
per-token BERT hidden states (bert only — see that module's docstring for
why minilm is not supported here).

Architecture:
  emotion embedding (nn.Linear(6, 32, bias=False)) -> Linear(32, D) = query projection
    -> query (N, 1, D)
  token states (N, T, D) = keys and values (already produced by the encoder)
  nn.MultiheadAttention(embed_dim=D, num_heads=4, batch_first=True),
    key_padding_mask = ~attention_mask.bool()   (True = ignore, per torch's convention)
  attended output (N, 1, D) -> squeeze -> (N, D)
    -> same MLP trunk as run_intermediate: Linear(D, hidden) -> ReLU -> Dropout
       -> Linear(hidden, 128) -> ReLU -> Linear(128, 3)

Same training loop, early stopping, class-weighting switch, and equal sweep
budget as `run_intermediate`.
"""

import numpy as np
import copy
from sklearn.metrics import f1_score
import common as _common
from common import set_seed, y_of, emotion_features, class_weights
# CLASS_WEIGHTING read live off `_common.CLASS_WEIGHTING` — see early.py.


def run_intermediate_attn(splits, emb, seed, device, encoder_name):
    """
    Emotion-conditioned attention pooling over token-level text features.

    NOTE on the uniform signature: like every other `run_*` method this
    accepts `emb` (the POOLED embeddings dict) for signature uniformity, but
    this method ignores it and instead calls
    `encoders.get_token_embeddings` itself to obtain per-token
    features — the pooled `emb` cannot be used for attention over tokens.
    Only supported for encoder_name == "bert" (see encoders.py);
    calling it with "minilm" raises ValueError from that function.

    Returns (y_true_test, y_pred_test), both int arrays of shape (N_test,),
    values in 0..2 indexing LABELS.
    """
    result = _run_attn_core(splits, emb, seed, device, encoder_name)
    return result["y_test"], result["test_pred"]


def run_intermediate_attn_logits(splits, emb, seed, device, encoder_name):
    """Same computation as `run_intermediate_attn`, also returning val/test
    logits for the post-hoc class-bias correction step."""
    result = _run_attn_core(splits, emb, seed, device, encoder_name)
    return result["y_val"], result["val_logits"], result["y_test"], result["test_logits"]


def _run_attn_core(
    splits, emb, seed, device, encoder_name,
    hidden: int = 256, dropout: float = 0.3, lr: float = 1e-3,
    cache_dir: str | None = None,
) -> dict:
    """
    Core training loop for emotion-conditioned attention pooling.

    Returns a dict with y_val, val_pred, val_logits, y_test, test_pred, test_logits.
    """
    set_seed(seed)

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam
    from encoders import get_token_embeddings

    # Read the cache location live off common so the driver can
    # point it at Drive; otherwise the ~2 GB token cache is rebuilt every session.
    tok = get_token_embeddings(
        splits, encoder_name, device,
        cache_dir=_common.TOKEN_CACHE_DIR if cache_dir is None else cache_dir,
    )
    tok_train, mask_train = tok["train"]
    tok_val, mask_val = tok["val"]
    tok_test, mask_test = tok["test"]

    D = tok_train.shape[2]

    emotion_feat_train = emotion_features(splits["train"], "train", seed)
    emotion_feat_val = emotion_features(splits["val"], "val", seed)
    emotion_feat_test = emotion_features(splits["test"], "test", seed)

    y_train = y_of(splits["train"])
    y_val = y_of(splits["val"])
    y_test = y_of(splits["test"])

    class _AttnFusionModel(nn.Module):
        def __init__(self, text_dim, hidden, num_heads=4):
            super().__init__()
            self.emotion_embedding = nn.Linear(6, 32, bias=False)
            self.query_proj = nn.Linear(32, text_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=num_heads, batch_first=True
            )
            self.fc1 = nn.Linear(text_dim, hidden)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 3)

        def forward(self, emo_idx, token_states, attn_mask):
            # emo_idx: (N,); token_states: (N, T, D); attn_mask: (N, T) 1=real, 0=pad
            emotion_emb = self.emotion_embedding(emo_idx)          # (N, 32)
            query = self.query_proj(emotion_emb).unsqueeze(1)      # (N, 1, D)
            key_padding_mask = attn_mask == 0                       # True = ignore this position
            attended, _ = self.attn(
                query, token_states, token_states, key_padding_mask=key_padding_mask
            )
            h = attended.squeeze(1)                                 # (N, D)
            x = self.fc1(h)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            logits = self.fc3(x)
            return logits

    model = _AttnFusionModel(D, hidden).to(device)

    emotion_feat_train_t = torch.FloatTensor(emotion_feat_train).to(device)
    emotion_feat_val_t = torch.FloatTensor(emotion_feat_val).to(device)
    emotion_feat_test_t = torch.FloatTensor(emotion_feat_test).to(device)

    tok_train_t = torch.FloatTensor(tok_train).to(device)
    tok_val_t = torch.FloatTensor(tok_val).to(device)
    tok_test_t = torch.FloatTensor(tok_test).to(device)

    mask_train_t = torch.LongTensor(mask_train).to(device)
    mask_val_t = torch.LongTensor(mask_val).to(device)
    mask_test_t = torch.LongTensor(mask_test).to(device)

    y_train_t = torch.LongTensor(y_train).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    train_dataset = TensorDataset(emotion_feat_train_t, tok_train_t, mask_train_t, y_train_t)
    val_dataset = TensorDataset(emotion_feat_val_t, tok_val_t, mask_val_t, y_val_t)
    test_dataset = TensorDataset(emotion_feat_test_t, tok_test_t, mask_test_t)

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Smaller batch than run_intermediate: (N, T, D) token tensors are much
    # larger than pooled (N, D) ones, and this only needs to run on frozen
    # features, so trading batch size for memory headroom costs nothing here.
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        for emotion_batch, tok_batch, mask_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(emotion_batch, tok_batch, mask_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logit_chunks = []
            for emotion_batch, tok_batch, mask_batch, _ in val_loader:
                logits = model(emotion_batch, tok_batch, mask_batch)
                val_logit_chunks.append(logits.cpu().numpy())
            val_logits_epoch = np.concatenate(val_logit_chunks, axis=0)
            y_val_pred = np.argmax(val_logits_epoch, axis=1)

        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_logits = val_logits_epoch
            patience_counter = 0
            print(f"[attn] Epoch {epoch+1}: F1={val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Hand the trained model to the driver if it asked for one (no-op
    # during sweeps, which train hundreds of throwaway models).
    _common.offer_checkpoint(
        f"intermediate_attn_{encoder_name}_{seed}", best_state_dict, best_val_f1,
        meta={"method": "intermediate_attn", "encoder": encoder_name, "seed": seed,
              "regime": _common.EMOTION_REGIME,
              "class_weighting": _common.CLASS_WEIGHTING},
    )
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        test_logit_chunks = []
        for emotion_batch, tok_batch, mask_batch in test_loader:
            logits = model(emotion_batch, tok_batch, mask_batch)
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
