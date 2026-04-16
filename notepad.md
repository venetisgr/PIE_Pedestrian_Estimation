# Notepad — Gotchas, Discoveries, Mistakes

Free-form scratchpad. Append new entries dated at the top. Short, blunt,
honest. Each entry: **what I hit**, **why it happened**, **fix**.

---

## 2026-04-16 — Initial discoveries during exploration

### D1. `pie_data.py` is NOT in the repo
- `train_test.py:22` does `from pie_data import PIE`, but there is no
  such file in this repository.
- It lives in the upstream PIE dataset repo
  (`github.com/aras62/PIE`). We will **vendor** it into
  `pie_pytorch/data/pie_data.py` (plus any small fixups for Python 3),
  not `pip install` from a moving target.
- This is a *prerequisite* for anything data-related; Phase 1.1 blocks
  Phase 1.3 onwards.

### D2. Hyperparameters differ per model — record the exact numbers
Explore-agent findings (authoritative — confirmed against source):

**Intent** (`pie_intent.py`):
- `num_hidden=128`, `reg=0.001`, `lstm_dropout=0.4`, `lstm_recurrent_dropout=0.2`
- `convlstm_num_filters=64`, `convlstm_kernel_size=2`, `activation='tanh'`
- `RMSprop(lr=1e-5, decay=0, clipvalue=0)`, batch=128, epochs=400
- Callbacks: EarlyStop patience=5 min_delta=1e-4; ReduceLROnPlateau factor=0.5 patience=5 min_lr=1e-7; ModelCheckpoint best `val_loss`.

**Trajectory** (`pie_predict.py`):
- `num_hidden=256`, `reg=1e-4`, `activation='softsign'`, `embed_size=64`, `embed_dropout=0`
- `RMSprop(lr=1e-3)`, batch=64, epochs=60, loss='mse'
- Callbacks: EarlyStop patience=10 min_delta=1.0 (note the *huge* min_delta — loss is raw pixel MSE, not normalized); ReduceLROnPlateau factor=0.2 patience=5 min_lr=1e-7.

**Speed**: same training knobs as trajectory, feature sizes are 1D.

### D3. Normalization quirk in trajectory data
- `get_tracks` in `pie_predict.py:128-139` subtracts the first frame
  from bbox/center sequences → observation length effectively
  **shrinks by one** (`observe_length -= 1` at line 192).
- Must mirror this exactly in the PyTorch `Dataset.__getitem__`,
  otherwise shapes flow through wrong and models diverge silently.

### D4. VGG16 feature caching vs Colab disk
- The intent pipeline pre-extracts VGG16 features and pickles one file
  per frame (`pie_intent.py:244-267`). On full PIE this is huge; on
  Colab we'll blow the 100 GB tmpfs easily.
- Mitigations to bake in from day 1:
  1. `shard` features into `.safetensors` files per `set/vid`, not per
     frame (fewer inodes, streamable).
  2. On Colab, default to caching to Drive if mounted, else tmpfs with
     a hard size ceiling + eviction.
  3. Respect the subset flag before precomputing.

### D5. `K.set_image_dim_ordering('tf')` — channels-last
- Source uses NHWC everywhere. PyTorch default is NCHW. When we port
  the ConvLSTM, **be explicit** about axis order in docstrings and
  tests, and keep our fixtures in NHWC→NCHW conversion so parity
  checks line up.

### D6. `tensorflow-gpu==1.9.0` / `keras==2.2.1` is basically unbuildable today
- Don't try to `pip install` the old stack. For Phase 4 weight parity,
  use a **TF 2.x compat load**: `tf.keras.models.load_model` can read
  many TF 1.x `.h5` files; where it can't, read HDF5 directly via
  `h5py` and fetch tensors by layer-name path. Document per-layer
  name mapping in this file when we do the port.

### D7. License note
- Apache 2.0 — fine to fork, modify, relicense derivatives under
  compatible terms. Keep the original LICENSE file and add a NOTICE
  referencing the ICCV 2019 paper.

### D8. Pretrained checkpoints already present
```
data/pie/intention/context_loc_pretrained/model.h5   (reported ~50 MB)
data/pie/speed/speed_pretrained/model.h5             (~5 MB)
data/pie/trajectory/loc_intent_speed_pretrained/model.h5 (~20 MB)
```
Those are our **ground-truth reference** for Phase 4 parity.

### D9. `ConvLSTM2D` — the biggest porting risk
- PyTorch has no stdlib ConvLSTM. Third-party packages (`convlstm-pytorch`)
  vary in quality. We will **implement our own** cell:
  - 4 gates fused into one conv
  - `hard_sigmoid` gate activation and `tanh` state activation to
    match Keras defaults
  - Same initializer families (glorot for kernel, orthogonal for
    recurrent kernel, zeros for bias)
- Write a unit test that compares forward-pass outputs to a reference
  NumPy implementation with fixed weights. Do this before using it in
  the intent model.

### D10. `RMSprop` defaults differ
- Keras `RMSprop` defaults: `rho=0.9`, `epsilon=1e-7`.
- PyTorch `torch.optim.RMSprop` defaults: `alpha=0.99`, `eps=1e-8`.
- When we chase parity, set **`alpha=0.9, eps=1e-7`** explicitly.

### D11. Regularizers → weight decay
- Keras `regularizers.l2(v)` applies `v * sum(w**2)` to the loss (note:
  **not** halved). PyTorch's `weight_decay` in optimizers is the
  coefficient of `w` in the gradient of `0.5 * v * sum(w**2)`.
- Therefore to match Keras `l2(v)` we must pass `weight_decay=2*v`, or
  manually add the regularization term to the loss. Easy to get wrong
  — add a comment next to every optimizer construction.

### D12. Binary sigmoid output + `binary_crossentropy`
- Keras `binary_crossentropy` with sigmoid output == PyTorch
  `BCELoss(sigmoid(z))`. Prefer `BCEWithLogitsLoss` for numerical
  stability and drop the final `Sigmoid()` from forward pass; apply
  sigmoid only at inference.

### D13. Reproducibility targets
- Seed torch, numpy, random, `torch.backends.cudnn.deterministic=True`,
  `benchmark=False`. Do this in `trainer.py` entry point.

---

## Running list of mistakes / fixups (add as they happen)

_(nothing yet — first commit is docs only)_

---

## Open questions / decisions to revisit
- **Config framework**: `hydra` (rich composition) vs plain `argparse + yaml`
  (simpler). Start with the latter; revisit if configs grow.
- **Weight porting (Phase 4)**: is exact numerical parity worth the
  engineering time vs just retraining with the PyTorch code? If a
  PyTorch retrain reproduces paper numbers, parity matters less.
- **Dataset redistribution**: we cannot host PIE images. Sample must be
  either (a) a tiny user-downloaded subset, or (b) derived features
  only. Confirm licensing before hosting a Colab "sample" zip.
