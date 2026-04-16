# legacy_tf/ — original TensorFlow / Keras implementation (reference only)

The four files in this directory are the **unmodified** original
TensorFlow 1.9 / Keras 2.2 implementation of the PIE paper:

- `pie_intent.py`       — ConvLSTM encoder-decoder intent model
- `pie_predict.py`      — LSTM+attention encoder-decoder trajectory & speed models
- `train_test.py`       — entry point (trains intent → speed → trajectory, then tests)
- `utils.py`            — image/bbox preprocessing helpers

**Do not run this code.** TF 1.9 + Keras 2.2.1 is effectively
unbuildable on modern Python/CUDA. The files are kept solely as a
parity reference for the PyTorch port (see `plan.md` → Phase 4).

This directory will be **deleted** once the PyTorch port hits parity
(see `plan.md` → Phase 6).

## What this depends on that isn't here
- `pie_data.py` from `github.com/aras62/PIE` (imported at
  `train_test.py:22` but never committed to this repo).

## Hyperparameters already catalogued in `notepad.md` (D2)
If you need architecture or hyperparameter facts, read `notepad.md`
first — it is the distilled source of truth.
