"""
encoders.py

Encoder abstraction: converts text to embeddings for the fusion benchmark.
Supports "minilm" (sentence-transformers, 384-dim) and "bert" (transformers, 768-dim with CLS pooling).

Models are cached by (encoder_name, device) to avoid reloading during the 12-run matrix.
Embeddings are cached by encoder_name, split name, row count, and SHA1 hash of uid list,
ensuring different data variants (full vs filtered) get separate cache files.

All embeddings are returned as np.float32.
"""

import os
import hashlib
import numpy as np

# Module-level model cache: {(encoder_name, device): model}
# This is the one exception to the "no global mutable state" rule — intentional,
# because model loading is expensive and we run a 12-run matrix (6 methods x 2 encoders x 1 variant = 12 runs).
_model_cache: dict = {}


def embed_texts(texts: list[str], encoder_name: str, device: str,
                batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts using the specified encoder.

    Lazy-imports torch/transformers/sentence_transformers inside the function so the module
    can be imported on machines without the deep-learning stack.

    Args:
        texts: list of strings to encode
        encoder_name: "minilm" (sentence-transformers, 384-dim) or "bert" (transformers, 768-dim CLS)
        device: "cpu" or "cuda"
        batch_size: batch size for encoding (default 64)

    Returns:
        np.ndarray of shape (len(texts), D) with dtype float32
        D = 384 for minilm, 768 for bert

    Raises:
        ValueError: if encoder_name is not "minilm" or "bert"
    """
    if encoder_name not in ["minilm", "bert"]:
        raise ValueError(f"Unknown encoder_name: {encoder_name}")

    cache_key = (encoder_name, device)

    if encoder_name == "minilm":
        # Lazy import
        if cache_key not in _model_cache:
            from sentence_transformers import SentenceTransformer
            from common import ENCODER_IDS

            model = SentenceTransformer(ENCODER_IDS["minilm"])
            model.to(device)
            _model_cache[cache_key] = model

        model = _model_cache[cache_key]
        # SentenceTransformer.encode handles batching and returns normalized embeddings
        embeddings = model.encode(texts, batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    elif encoder_name == "bert":
        # Lazy imports
        if cache_key not in _model_cache:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from common import ENCODER_IDS

            tokenizer = AutoTokenizer.from_pretrained(ENCODER_IDS["bert"])
            model = AutoModel.from_pretrained(ENCODER_IDS["bert"])
            model.to(device)
            model.eval()
            _model_cache[cache_key] = (tokenizer, model)

        tokenizer, model = _model_cache[cache_key]
        import torch

        embeddings = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, max_length=64, truncation=True,
                                   padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Take CLS token (first token of last hidden state)
                cls_tokens = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, 768)

                # L2 normalize
                cls_normalized = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
                embeddings.append(cls_normalized.cpu().numpy())

        embeddings_array = np.vstack(embeddings).astype(np.float32)
        return embeddings_array


def embed_texts_tokenwise(texts: list[str], encoder_name: str, device: str,
                          max_length: int = 64, batch_size: int = 32):
    """
    Token-level embeddings for the attention-pooling method (`run_intermediate_attn`).

    Supports BOTH encoders. `all-MiniLM-L6-v2` is distributed as a
    sentence-transformers checkpoint, but the underlying network is an ordinary
    BERT-family transformer, so `AutoModel.from_pretrained` loads it directly and
    exposes `last_hidden_state`. Routing both encoders through AutoModel here is
    what closes the `intermediate_attn / minilm` gap — the first run reported NaN
    for that cell purely because this function used to refuse anything but bert,
    and a method that only ever runs on one encoder cannot support a robustness
    claim (we already know early-vs-intermediate flips between encoders).

    NOTE on a deliberate asymmetry: the POOLED path (`embed_texts`) keeps using
    sentence-transformers for minilm, with its mean-pooling + normalisation,
    because that is what produced the already-reported pooled results. The
    token-level path here is raw `last_hidden_state` for both encoders. So minilm
    has two slightly different representations depending on the path. That is
    intentional — changing the pooled path would invalidate existing numbers.

    Returns (embeddings, attention_mask):
        embeddings: np.ndarray (N, T, D) float32, last_hidden_state (no CLS pooling)
        attention_mask: np.ndarray (N, T) int, 1 for real tokens, 0 for padding
    T = max_length for every row (padded/truncated), so batches concatenate cleanly.
    """
    from common import ENCODER_IDS

    if encoder_name not in ENCODER_IDS:
        raise ValueError(
            f"unknown encoder {encoder_name!r}, expected one of {sorted(ENCODER_IDS)}"
        )

    import torch
    from transformers import AutoTokenizer, AutoModel

    # Separate cache namespace from the pooled loader: for minilm the pooled
    # path stores a SentenceTransformer under ("minilm", device), and handing
    # that object to this function would break it.
    cache_key = ("tokenwise", encoder_name, device)
    if cache_key not in _model_cache:
        model_id = ENCODER_IDS[encoder_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        _model_cache[cache_key] = (tokenizer, model)
    tokenizer, model = _model_cache[cache_key]

    all_hidden = []
    all_mask = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts, max_length=max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_hidden.append(outputs.last_hidden_state.cpu().numpy())
            all_mask.append(attention_mask.cpu().numpy())

    hidden = np.concatenate(all_hidden, axis=0).astype(np.float32)
    mask = np.concatenate(all_mask, axis=0).astype(np.int64)
    return hidden, mask


def get_token_embeddings(splits: dict, encoder_name: str, device: str,
                         cache_dir: str = "/content/emb_cache_tok",
                         max_length: int = 64) -> dict:
    """
    Token-level embedding cache for `run_intermediate_attn`.

    Mirrors `get_embeddings`'s caching discipline (per split, keyed on row
    count + SHA1 of the uid list) but writes to a SEPARATE cache directory
    and uses a different filename prefix ("tok_" vs the pooled cache's bare
    encoder name), so it never collides with or silently reuses a pooled
    cache file. Does NOT touch or alter `get_embeddings`.

    Returns {"train"/"val"/"test": (embeddings (N, T, D) float32, attention_mask (N, T) int)}.
    Cached to disk as float16 to keep the cache directory small; loaded back
    as float32 for use, per the contract's float32 requirement for in-memory Emb.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = {}

    for split_name in ["train", "val", "test"]:
        rows = splits[split_name]
        n = len(rows)
        uids = [row["uid"] for row in rows]
        uid_hash = hashlib.sha1(",".join(str(u) for u in uids).encode()).hexdigest()

        emb_file = os.path.join(cache_dir, f"tok_{encoder_name}_{split_name}_{n}_{uid_hash}_emb.npy")
        mask_file = os.path.join(cache_dir, f"tok_{encoder_name}_{split_name}_{n}_{uid_hash}_mask.npy")

        if os.path.exists(emb_file) and os.path.exists(mask_file):
            hidden = np.load(emb_file).astype(np.float32)
            mask = np.load(mask_file)
        else:
            texts = [row["text"] for row in rows]
            hidden, mask = embed_texts_tokenwise(texts, encoder_name, device, max_length=max_length)
            np.save(emb_file, hidden.astype(np.float16))  # cache compactly on disk
            np.save(mask_file, mask)

        assert hidden.shape[0] == n, (
            f"Token embedding row count {hidden.shape[0]} != split row count {n} for {split_name}"
        )
        out[split_name] = (hidden.astype(np.float32), mask)

    return out


def get_embeddings(splits: dict, encoder_name: str, device: str,
                  cache_dir: str = "/content/emb_cache") -> dict:
    """
    Get embeddings for all three splits (train, val, test).

    Caches embeddings to disk with a key that includes encoder_name, split name, row count,
    and a SHA1 hash of the uid list. This ensures that different data variants (full vs filtered)
    produce different cache files and never silently reuse a cache from a different variant.

    Args:
        splits: {"train": list[Row], "val": list[Row], "test": list[Row]}
                where Row = dict with at least "uid" and "text" fields
        encoder_name: "minilm" or "bert"
        device: "cpu" or "cuda"
        cache_dir: directory for embedding cache files (default "/content/emb_cache")

    Returns:
        {"train": np.ndarray, "val": np.ndarray, "test": np.ndarray}
        All arrays are float32. Train/val/test shapes are (N_train, D), (N_val, D), (N_test, D).
    """
    os.makedirs(cache_dir, exist_ok=True)

    embeddings = {}

    for split_name in ["train", "val", "test"]:
        rows = splits[split_name]
        n = len(rows)

        # Build cache key: encoder_name, split name, row count, uid hash
        # The uid hash ensures different data variants get different cache files
        uids = [row["uid"] for row in rows]
        uid_hash = hashlib.sha1(",".join(str(u) for u in uids).encode()).hexdigest()

        cache_file = os.path.join(cache_dir, f"{encoder_name}_{split_name}_{n}_{uid_hash}.npy")

        if os.path.exists(cache_file):
            # Load from cache
            emb = np.load(cache_file)
        else:
            # Encode texts
            texts = [row["text"] for row in rows]
            emb = embed_texts(texts, encoder_name, device)

            # Save to cache
            np.save(cache_file, emb)

        # Verify row count matches split size before returning
        assert emb.shape[0] == n, \
            f"Embedding row count {emb.shape[0]} != split row count {n} for {split_name}"

        embeddings[split_name] = emb.astype(np.float32)

    return embeddings
