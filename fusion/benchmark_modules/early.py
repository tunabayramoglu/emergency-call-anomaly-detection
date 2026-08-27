import numpy as np
import copy
from sklearn.metrics import f1_score
import common as _common
from common import set_seed, LABELS, EMOTIONS, ENCODER_IDS, y_of, class_weights
# CLASS_WEIGHTING is read live off `_common.CLASS_WEIGHTING` (not imported by
# name) so that flipping the flag on the `common` module object
# AFTER this module has already been imported still takes effect the next
# time run_early() is called. `from common import CLASS_WEIGHTING`
# would instead freeze whatever value was true at import time.


def run_early(splits, emb, seed, device, encoder_name):
    """
    Early fusion: inject emotion as a special token at the front of text,
    then fine-tune the encoder end-to-end for 3-way classification.

    The defining property of early fusion is that emotion enters at the INPUT,
    before any learned representation is formed, allowing the encoder's attention
    to condition every token on it.

    Args:
        splits: dict with "train", "val", "test" lists of Row dicts
        emb: dict with embeddings (ignored for early fusion)
        seed: random seed
        device: torch device string (e.g., "cuda" or "cpu")
        encoder_name: "minilm" or "bert", key to ENCODER_IDS

    Returns:
        tuple[np.ndarray, np.ndarray]: (y_true_test, y_pred_test),
            both int arrays of shape (N_test,) with values in 0..2 indexing LABELS
    """
    y_true, y_pred, _val_logits, _test_logits = _run_early_or_textft(
        splits, emb, seed, device, encoder_name, use_emotion=True
    )
    return y_true, y_pred


def _run_early_or_textft(splits, emb, seed, device, encoder_name, use_emotion: bool):
    """
    Shared implementation for `run_early` (use_emotion=True) and
    `run_text_only_finetuned` (use_emotion=False, see text_ft.py).

    Identical hyperparameters, training loop, and early stopping in both
    cases. The ONLY difference is whether the emotion token is prepended to
    the input string — this isolates "emotion visible to the encoder" from
    "encoder was fine-tuned", which the `early` vs `text_only` (frozen
    embedding, LogisticRegression) comparison could not do on its own.

    Also returns (val_logits, test_logits) alongside (y_true_test, y_pred_test)
    so the post-hoc class-bias correction (report.tune_class_bias)
    can be applied to this method without retraining.
    """
    set_seed(seed)

    # Lazy imports to avoid requiring torch/transformers at module load time
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # emb is ignored for early fusion / text-only-finetuned — the point of both
    # is that they build their own representation from the raw string, not
    # from frozen features.

    train_texts = [row["text"] for row in splits["train"]]
    train_emotions = [row["gen_emotion"] for row in splits["train"]]
    train_labels = y_of(splits["train"])

    val_texts = [row["text"] for row in splits["val"]]
    val_emotions = [row["gen_emotion"] for row in splits["val"]]
    val_labels = y_of(splits["val"])

    test_texts = [row["text"] for row in splits["test"]]
    test_emotions = [row["gen_emotion"] for row in splits["test"]]
    test_labels = y_of(splits["test"])

    # Emotion channel as a (N, 6) distribution, honouring the active
    # EMOTION_REGIME. Under "oracle" these are one-hot, so everything below
    # reduces exactly to prepending the true emotion token — the previously
    # reported hard-label numbers stay reproducible.
    train_emo_probs = _common.emotion_features(splits["train"], "train", seed)
    val_emo_probs = _common.emotion_features(splits["val"], "val", seed)
    test_emo_probs = _common.emotion_features(splits["test"], "test", seed)

    if use_emotion:
        # A FIXED placeholder emotion token is prepended to every row, so the
        # tokenisation is identical across rows and the emotion always lands at
        # position 1 (right after [CLS]). The placeholder's EMBEDDING is then
        # overwritten at forward time with the probability-weighted mixture of
        # the six emotion-token embeddings (see `_soft_inputs_embeds`).
        #
        # Why not just prepend the argmax token? Because the SER emits a
        # distribution, and collapsing it to one token throws away its
        # uncertainty — which is exactly the information a combiner needs in
        # order to distrust an unreliable emotion reading. Mixing in embedding
        # space is the input-level equivalent of feeding a soft label.
        placeholder = f"[{EMOTIONS[0].upper()}]"
        train_texts_aug = [f"{placeholder} {t}" for t in train_texts]
        val_texts_aug = [f"{placeholder} {t}" for t in val_texts]
        test_texts_aug = [f"{placeholder} {t}" for t in test_texts]
    else:
        # text_only_finetuned: plain text, no emotion signal anywhere in the input
        train_texts_aug = list(train_texts)
        val_texts_aug = list(val_texts)
        test_texts_aug = list(test_texts)

    # Load tokenizer and model
    model_id = ENCODER_IDS[encoder_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # --- Checkpoint loading guard -----------------------------------------
    # Loading bert-base-uncased for sequence classification prints two kinds
    # of warning; both are EXPECTED here, but the same message would also
    # fire if the encoder body itself silently failed to load, so we check
    # the categories explicitly instead of trusting eyeballed log output:
    #   - UNEXPECTED "cls.predictions.*"/"cls.seq_relationship.*" (or
    #     "pooler.*"): the pretraining MLM/NSP heads in the checkpoint have
    #     no home in AutoModelForSequenceClassification and are discarded —
    #     harmless, we never wanted those heads.
    #   - MISSING "classifier.weight"/"classifier.bias" (or "bert.pooler"/
    #     "pooler"): the 3-way classification head does not exist in a
    #     pretrained MLM checkpoint and is randomly initialised — expected,
    #     because we are about to fine-tune it from scratch.
    # Anything else missing would mean the ENCODER BODY itself failed to
    # load (e.g. a name mismatch / truncated checkpoint) — that must raise,
    # not be silently swallowed as "just the usual warning".
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, output_loading_info=True
    )
    unexpected_ok = all(
        k.startswith(("cls.", "pooler.")) for k in loading_info["unexpected_keys"]
    )
    assert unexpected_ok, (
        f"unexpected keys outside the known-benign MLM/NSP/pooler heads: "
        f"{loading_info['unexpected_keys']}"
    )
    missing_bad = [
        k for k in loading_info["missing_keys"]
        if not k.startswith(("classifier", "bert.pooler", "pooler"))
    ]
    assert not missing_bad, f"encoder weights missing from checkpoint: {missing_bad}"

    # Add emotion tokens as additional special tokens (added even when
    # use_emotion is False, so the tokenizer/vocab is identical between the
    # two runs and any difference in result is attributable only to whether
    # the token is actually used in the input string, not to a vocab-size
    # side effect).
    emotion_tokens = [f"[{e.upper()}]" for e in EMOTIONS]
    tokenizer.add_special_tokens({"additional_special_tokens": emotion_tokens})

    # CRITICAL: resize_token_embeddings MUST be called AFTER both:
    # (1) tokenizer gains the new tokens, and (2) model is loaded.
    # Calling it in the wrong order produces silent garbage embeddings for new tokens.
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    # Initialise the new emotion-token embedding rows from the MEAN of the
    # existing input embeddings, rather than leaving them at whatever
    # `resize_token_embeddings` defaults to (effectively random). With only
    # 4 epochs at lr 2e-5 those rows barely move from their initial value,
    # so a random start would leave them close to noise all the way through
    # training; the mean-of-existing-rows start is a much better prior for
    # "a token that behaves like an ordinary token".
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        mean_embedding = input_embeddings.weight[:old_vocab_size].mean(dim=0)
        input_embeddings.weight[old_vocab_size:] = mean_embedding
        # Some architectures tie input/output embeddings for the LM head;
        # AutoModelForSequenceClassification has no LM head so there is
        # nothing further to tie here, but keep the call in case of a
        # future architecture change.
        model.tie_weights()

    model.to(device)

    # Row indices of the six emotion tokens in the (resized) embedding table.
    emotion_token_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(emotion_tokens), dtype=torch.long, device=device
    )
    assert (emotion_token_ids >= old_vocab_size).all(), (
        "emotion tokens were not added as NEW vocabulary entries — they must not "
        "collide with pre-existing ids, or the soft mixture would blend unrelated "
        f"wordpieces. ids={emotion_token_ids.tolist()} old_vocab={old_vocab_size}"
    )

    def _soft_inputs_embeds(input_ids, emo_probs):
        """Embed `input_ids`, then replace position 1 with the soft emotion mix.

        `emo_probs` is (B, 6) summing to 1. The mixture is
        `emo_probs @ E[emotion_token_ids]`, i.e. a convex combination of the six
        emotion-token embeddings. With a one-hot input this returns exactly the
        embedding of the true emotion token, so the "oracle" regime is bit-for-bit
        the old hard-token behaviour.

        Gradients flow into the emotion token rows through this mixture, so those
        rows still get trained exactly as they did in the hard-token version.
        """
        emb_layer = model.get_input_embeddings()
        base = emb_layer(input_ids)                       # (B, T, H)
        emotion_matrix = emb_layer.weight[emotion_token_ids]   # (6, H)
        mixed = emo_probs @ emotion_matrix                # (B, H)
        return torch.cat(
            [base[:, :1], mixed.unsqueeze(1), base[:, 2:]], dim=1
        )

    def _forward(input_ids, attention_mask, emo_probs):
        """One forward pass, routing through inputs_embeds only when the emotion
        channel is in use. text_only_finetuned keeps the plain input_ids path so
        the two arms differ in the emotion signal and nothing else."""
        if not use_emotion:
            return model(input_ids=input_ids, attention_mask=attention_mask)
        return model(
            inputs_embeds=_soft_inputs_embeds(input_ids, emo_probs),
            attention_mask=attention_mask,
        )

    # Tokenize datasets with batching
    def tokenize_texts(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

    train_encoded = tokenize_texts(train_texts_aug)
    val_encoded = tokenize_texts(val_texts_aug)
    test_encoded = tokenize_texts(test_texts_aug)

    # Move all tensors in batch dicts to device
    for key in train_encoded:
        train_encoded[key] = train_encoded[key].to(device)
    for key in val_encoded:
        val_encoded[key] = val_encoded[key].to(device)
    for key in test_encoded:
        test_encoded[key] = test_encoded[key].to(device)

    # Convert labels to long tensors on device
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long, device=device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Class-weighted loss, all-or-nothing via the module-level CLASS_WEIGHTING
    # switch in common. Weights are computed on TRAIN labels
    # only and never touch val/test.
    if _common.CLASS_WEIGHTING:
        w = torch.tensor(class_weights(train_labels), dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=w)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    batch_size = 32
    num_epochs = _common.MAX_FINETUNE_EPOCHS

    best_val_f1 = -1.0
    best_state_dict = None
    best_val_logits = None

    n_train = len(train_texts_aug)
    generator = torch.Generator().manual_seed(seed)

    train_emo_t = torch.tensor(train_emo_probs, dtype=torch.float32, device=device)
    val_emo_t = torch.tensor(val_emo_probs, dtype=torch.float32, device=device)
    test_emo_t = torch.tensor(test_emo_probs, dtype=torch.float32, device=device)

    def forward_logits(encoded, emo_t, eval_batch_size: int = 128) -> np.ndarray:
        """Batched inference returning raw logits (N, 3)."""
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, encoded["input_ids"].shape[0], eval_batch_size):
                stop = start + eval_batch_size
                out = _forward(
                    encoded["input_ids"][start:stop],
                    encoded["attention_mask"][start:stop],
                    emo_t[start:stop],
                )
                chunks.append(out.logits.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict(encoded, emo_t, eval_batch_size: int = 128) -> np.ndarray:
        """Batched inference. Never run a whole split in one forward pass —
        the test split is ~1.5k sequences and would spike GPU memory."""
        return np.argmax(forward_logits(encoded, emo_t, eval_batch_size), axis=1)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Shuffle every epoch. The dataset is ordered by seed group, so without
        # this each batch would be near-duplicate paraphrases of one scenario —
        # correlated batches, unstable gradients, and an effectively much
        # smaller batch diversity than 32.
        perm = torch.randperm(n_train, generator=generator).to(device)

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]

            optimizer.zero_grad()
            outputs = _forward(
                train_encoded["input_ids"][idx],
                train_encoded["attention_mask"][idx],
                train_emo_t[idx],
            )
            loss = criterion(outputs.logits, train_labels_tensor[idx])
            loss.backward()
            optimizer.step()

        # Validation phase — model selection happens here, never on test
        val_logits = forward_logits(val_encoded, val_emo_t)
        val_preds = np.argmax(val_logits, axis=1)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_logits = val_logits

        print(f"Epoch {epoch+1}/{num_epochs} - Val F1: {val_f1:.4f}")

    # Restore best state dict before test inference
    # Hand the trained model to the driver if it asked for one (no-op
    # during sweeps, which train hundreds of throwaway models).
    _common.offer_checkpoint(
        f"early_or_textft_{encoder_name}_{seed}", best_state_dict, best_val_f1,
        meta={"method": "early_or_textft", "encoder": encoder_name, "seed": seed,
              "regime": _common.EMOTION_REGIME,
              "class_weighting": _common.CLASS_WEIGHTING},
    )
    model.load_state_dict(best_state_dict)

    test_logits = forward_logits(test_encoded, test_emo_t)
    y_pred_test = np.argmax(test_logits, axis=1)

    return (
        test_labels.astype(np.int64),
        y_pred_test.astype(np.int64),
        best_val_logits,
        test_logits,
    )


def run_early_logits(splits, emb, seed, device, encoder_name):
    """
    Same computation as `run_early` but also returns the val/test logits, for
    the post-hoc class-bias correction step (no retraining involved — this
    just exposes what `run_early` already computed).

    Returns (y_val, val_logits, y_test, test_logits).
    """
    set_seed(seed)
    from common import y_of as _y_of

    y_test, _y_pred, val_logits, test_logits = _run_early_or_textft(
        splits, emb, seed, device, encoder_name, use_emotion=True
    )
    y_val = _y_of(splits["val"])
    return y_val, val_logits, y_test, test_logits
