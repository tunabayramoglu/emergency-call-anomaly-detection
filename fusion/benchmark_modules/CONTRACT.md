# Fusion Benchmark — module contract

Every module in this directory implements one fusion method against a fixed
interface, and `fusion_benchmark.ipynb` is the driver that imports them
and runs the matrix at the bottom of this file. The modules were written
independently against this document, which is why the interface is specified
down to the signature.

Anyone modifying a module should keep to it. Names and parameters are load
bearing — the driver calls them positionally — and the only permitted import
between modules is `common`. The directory started with the six
methods in the run matrix and grew to eleven as FiLM, attention, text
fine-tuning, the SER-noise study and the sweep were added; those follow the same
rules.

## Target environment
Google Colab, Python 3.11, single T4/A100 GPU, PyTorch + HuggingFace transformers +
sentence-transformers + scikit-learn + numpy + pandas + matplotlib.
Write plain `.py` module files. No CLI, no argparse, no `if __name__ == "__main__"`.
The notebook driver imports the functions and calls them.

## The task
Dataset: 9,740 synthetic emergency-call utterances. Each row has a transcript (`text`) and a
speech-emotion label (`gen_emotion`). Predict `anomaly` — whether the emotion MISMATCHES the
content severity. This is a fusion-combiner benchmark: we compare LATE vs INTERMEDIATE vs EARLY
fusion of the two channels (emotion, text).

## Data schema (dataset_final.jsonl, one JSON object per line)
Fields you MAY use as model input:
  - `text`        : str, the utterance
  - `gen_emotion` : str, one of EMOTIONS
Field that is the TARGET:
  - `anomaly`     : str, one of LABELS
Field used ONLY for splitting (never as a feature):
  - `seed_id`     : int
Field used ONLY as a row identifier / cache key:
  - `uid`         : int
Field used ONLY for the data-variant filter:
  - `source_model`: str

### BANNED FIELDS — using any of these as a model feature is a correctness bug
`judge_content_risk`, `judge_voice_risk`, `gen_content_risk`, `content_risk_seed`, `event`,
`profile`, `target_emotion`, `reason`, `judge_model`, `judge_count`, `judge_agreement`,
`judge_models`, `anomaly_votes`, `source_model`.
Reason: the judge risk fields are outputs of the same LLM call that produced the label — a
lookup table over them scores 91.5% and is pure leakage. `event` is ~6.7% wrong (measured).
Never touch these.

## Shared constants (defined in common, import them — do not redefine)
```python
LABELS   = ["normal", "borderline", "anomaly"]        # class index = position in this list
EMOTIONS = ["neutral", "confusion", "fear", "panic", "urgency", "distress"]
ENCODER_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # 384-dim
    "bert":   "bert-base-uncased",                        # 768-dim, CLS pooling
}
VARIANTS = ["full", "filtered"]   # filtered = drop source_model in {"qwen3.6-flash","qwen3.5-flash"}
```

## Universal types
```python
Row    = dict          # one parsed JSON line
Splits = dict          # {"train": list[Row], "val": list[Row], "test": list[Row]}
Emb    = dict          # {"train": np.ndarray (N_tr, D), "val": ..., "test": ...}  float32
```

## THE UNIFORM METHOD SIGNATURE
Every benchmark method — the three fusion levels AND the three baselines — has this exact
signature and returns this exact tuple. The driver calls them in a loop, so any deviation breaks
the run.

```python
def run_<name>(splits: Splits, emb: Emb, seed: int, device: str, encoder_name: str
              ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true_test, y_pred_test), both int arrays of shape (N_test,),
    values in 0..2 indexing LABELS."""
```
Methods that do not need `emb` (majority, emotion-only, early fusion) still accept it and ignore it.
Every method MUST call `set_seed(seed)` from `common` as its first statement.
Every method that trains must select its checkpoint / early-stop on the VAL split by macro-F1,
then report on TEST. Never touch test during training.

## Module assignments

### 1. `common.py`
```python
LABELS, EMOTIONS, ENCODER_IDS, VARIANTS, BANNED_FIELDS   # constants above
def set_seed(seed: int) -> None
def load_rows(path: str, variant: str = "full") -> list[Row]
def make_splits(rows: list[Row], split_seed: int = 42,
                val_frac: float = 0.15, test_frac: float = 0.15) -> Splits
def assert_no_seed_overlap(splits: Splits) -> None
def y_of(rows: list[Row]) -> np.ndarray            # int labels, shape (N,)
def emotion_onehot(rows: list[Row]) -> np.ndarray  # float32, shape (N, 6)
def emotion_idx(rows: list[Row]) -> np.ndarray     # int, shape (N,)
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict
def make_result(method: str, encoder: str, variant: str, seed: int,
                y_true: np.ndarray, y_pred: np.ndarray) -> dict
```
- `make_splits` MUST split on **`seed_id` groups**, not rows: every utterance sharing a `seed_id`
  goes entirely to one side. Utterances of the same seed are near-paraphrases; a row-level split
  leaks and inflates scores. Shuffle the unique seed_ids with `split_seed` and slice.
  The split must be IDENTICAL for every method/encoder/variant, so it depends only on
  `split_seed` — never on the per-run `seed`.
- `assert_no_seed_overlap` raises AssertionError if any seed_id appears in two splits.
- `compute_metrics` returns
  `{"acc": float, "macro_f1": float, "f1_normal": float, "f1_borderline": float,
    "f1_anomaly": float, "confusion": list[list[int]]}` (confusion is a 3x3 nested list, JSON-safe).
- `make_result` returns `{"method","encoder","variant","seed", **compute_metrics(...)}`.

### 2. `encoders.py`
```python
def embed_texts(texts: list[str], encoder_name: str, device: str,
                batch_size: int = 64) -> np.ndarray
def get_embeddings(splits: Splits, encoder_name: str, device: str,
                   cache_dir: str = "/content/emb_cache") -> Emb
```
- `minilm`: use `sentence_transformers.SentenceTransformer`, `.encode(..., normalize_embeddings=True)`.
- `bert`: use `transformers.AutoTokenizer` + `AutoModel`, truncation `max_length=64`, take the
  CLS token (`last_hidden_state[:, 0]`), then L2-normalize. Run under `torch.no_grad()` and `.eval()`.
- `get_embeddings` caches to `{cache_dir}/{encoder_name}_{split}_{n}.npy` keyed also on a hash of
  the uid list, and reuses the cache when present. Returns float32 arrays.
- These embeddings are FROZEN features — used by late, intermediate, and the text-only baseline.
  Early fusion does NOT use them.

### 3. `late.py` — `run_late(splits, emb, seed, device, encoder_name)`
Two independent unimodal classifiers, combined only at the decision level.
- Emotion branch: 6-dim one-hot -> Linear(6,32) -> ReLU -> Linear(32,3). Softmax -> 3 probs.
- Text branch:    frozen embedding (D) -> Linear(D,128) -> ReLU -> Dropout(0.2) -> Linear(128,3). Softmax -> 3 probs.
- Train the two branches SEPARATELY (each on its own channel, cross-entropy, Adam lr=1e-3,
  batch 64, up to 40 epochs, early stop patience 5 on val macro-F1).
- Combiner: `sklearn.linear_model.LogisticRegression(max_iter=1000, multi_class="multinomial")`
  fitted on the 6 concatenated val-set probabilities [emotion_probs, text_probs] -> label.
  Fit the combiner on VAL (the branches never saw val labels for fitting), predict on TEST.
- Key property to preserve: the two channels NEVER see each other before the final decision.

### 4. `inter.py` — `run_intermediate(splits, emb, seed, device, encoder_name)`
Feature-level fusion.
- Learned emotion embedding: `nn.Embedding(6, 32)` (input = `emotion_idx`).
- Concatenate `[emotion_emb (32), text_emb (D)]` -> Linear(32+D, 256) -> ReLU -> Dropout(0.3)
  -> Linear(256, 128) -> ReLU -> Linear(128, 3).
- Cross-entropy, Adam lr=1e-3, weight_decay=1e-4, batch 64, up to 60 epochs,
  early stop patience 8 on val macro-F1, restore best state dict before predicting test.

### 5. `early.py` — `run_early(splits, emb, seed, device, encoder_name)`
Input-level fusion: the emotion is injected into the text before the encoder sees it.
- Build the input string as `f"[{emotion.upper()}] {text}"`, e.g. `"[PANIC] there's a fire"`.
- Load `ENCODER_IDS[encoder_name]` via `transformers.AutoModelForSequenceClassification`
  with `num_labels=3`, and FINE-TUNE it end-to-end (this is the point of early fusion — do not
  freeze it). Add the six tokens `[NEUTRAL] [CONFUSION] [FEAR] [PANIC] [URGENCY] [DISTRESS]`
  as additional_special_tokens and call `model.resize_token_embeddings(len(tokenizer))`.
- AdamW lr=2e-5, batch 32, max_length=64, 4 epochs, evaluate on val each epoch,
  keep the best state dict by val macro-F1, predict test with it.
- Use a plain PyTorch loop, not `transformers.Trainer`.
- `emb` is ignored. Works for both `minilm` and `bert` encoder names.

### 6. `report.py`
```python
def run_majority(splits, emb, seed, device, encoder_name) -> tuple[np.ndarray, np.ndarray]
def run_emotion_only(splits, emb, seed, device, encoder_name) -> tuple[np.ndarray, np.ndarray]
def run_text_only(splits, emb, seed, device, encoder_name) -> tuple[np.ndarray, np.ndarray]
def results_table(results: list[dict]) -> "pd.DataFrame"
def plot_results(df, out_path: str) -> None
def plot_confusion(results: list[dict], method: str, encoder: str, variant: str,
                   out_path: str) -> None
```
- `run_majority`: predict the train-set majority class for every test row. Ignores `emb`.
- `run_emotion_only`: `LogisticRegression(max_iter=1000)` on the 6-dim one-hot ONLY. Ignores `emb`.
- `run_text_only`: `LogisticRegression(max_iter=1000)` on the frozen text embedding ONLY.
- These three are the ablations that prove fusion is doing something. They are mandatory —
  a reviewer's first question is "how much of this is just the text channel?".
- `results_table`: takes the flat list of result dicts, returns a tidy DataFrame with one row per
  (method, encoder, variant) aggregated over run seeds, with columns
  `acc_mean, acc_std, macro_f1_mean, macro_f1_std, f1_normal_mean, f1_borderline_mean,
  f1_anomaly_mean, n_seeds`. Sort by `variant, encoder, macro_f1_mean` descending.
- `plot_results`: grouped bar chart of macro-F1 with std error bars, methods on the x axis,
  one subplot per (encoder, variant). Save to `out_path` at dpi=150, tight bbox.
- `plot_confusion`: 3x3 confusion matrix heatmap (summed over run seeds) for the requested
  combination, annotated with counts, axis ticks = LABELS. Save to `out_path`.
- Use matplotlib only — no seaborn.

## Run matrix the driver will execute
methods = [majority, emotion_only, text_only, late, intermediate, early]
encoders = ["minilm", "bert"]
variants = ["full", "filtered"]
run seeds = [0, 1, 2]   -> mean ± std reported

## Rules
- numpy/pandas/sklearn/torch/transformers/sentence-transformers/matplotlib only.
- No global mutable state. No file writes except the caches and plot paths specified.
- Print at most one short progress line per epoch.
- Comments in English. Type hints on every public function.
- Assume `common` is importable as a top-level module: `from common import ...`.
- One complete file per method, with no placeholders or `TODO` stubs left in.
