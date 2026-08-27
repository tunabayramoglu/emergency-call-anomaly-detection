# Archive

Superseded versions. They are kept so that anyone curious about how the project
got here can look, but **do not use any of them**; the active equivalents live
elsewhere in the repo.

| file | what it was | use this instead |
|---|---|---|
| `aug_sweep_v1.py` | First augmentation ablation attempt, a cumulative A0–A5 chain | `asr/phase1/ablation_engine.py` |
| `aug_night_v2.py` | 19-axis overnight runner with HF pull/push and a deadline plan | `asr/phase1/ablation_engine.py` |
| `overnight_nb.py` | marimo front end with a separate engine file; dropped in favour of a single file | `asr/phase1/ablation_engine.py` |
| `anomaly_flag.py` | The first version of the anomaly definition, where voice emotion came from a ground-truth label | `fusion/` |
| `build_notebooks.py`, `build_training_nb.py` | Notebook generators, abandoned in favour of editing by hand | — |
| `train_ser_arousal.py` | SER as **arousal regression**. The shipped SER is a 6-class classifier instead | `ser/train_ser.py` |

`train_ser_arousal.py` is also broken as it stands: it resolves its data root
one level above the checkout and imports `emotion_taxonomy` through a
`training.` package prefix that does not exist — the module itself is alive at
`ser/emotion_taxonomy.py`. Both bugs are left in place; fixing dead code would
only make it look usable.

The honesty warning inside `anomaly_flag.py` still holds: CREMA-D and RAVDESS use
fixed, neutral sentences, so in that setup the text channel carried no real
signal. The fusion dataset was built precisely to fix that.
