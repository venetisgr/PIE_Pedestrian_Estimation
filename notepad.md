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

### 2026-04-17 — Phase 4.1: Keras ConvLSTM2D parity quirks
Two parity bugs found by inspecting the paper .h5 files:

1. **Keras ConvLSTM2D defaults to `padding='valid'`, not `'same'`**. With
   kernel=2 and 7×7 VGG features, that means 6×6 output (not 7×7).
   Flat = 64·6·6 = 2304; decoder LSTM input = 2308 = 2304 + 4 (bbox).
   Fixed by adding `convlstm_padding: str` to `IntentModelConfig` and
   computing `convlstm_out_hw()` for flat-size math.

2. **Keras ConvLSTM2D hardcodes `padding='same'` on the recurrent
   conv**, regardless of the user's `padding` setting. The input conv
   can shrink spatial; the hidden conv must preserve it. My
   implementation used the same padding for both, which only "worked"
   because I'd been using same everywhere. Fixed by decoupling:
   ``conv_x`` uses user-supplied padding, ``conv_h`` uses ``"same"``.
   Also moved the initial-hidden allocation to the conv_x output
   spatial size so both valid and same work.

Weight map (intent, from `data/pie/intention/context_loc_pretrained/model.h5`):
    conv_lst_m2d_1.kernel           (2,2,512,256)  -> encoder.cell.conv_x.weight (256,512,2,2)
    conv_lst_m2d_1.recurrent_kernel (2,2,64,256)   -> encoder.cell.conv_h.weight (256,64,2,2)
    conv_lst_m2d_1.bias             (256,)         -> encoder.cell.conv_x.bias
    decoder_network.kernel          (2308,512)     -> decoder.cell.W_x.weight (512,2308)
    decoder_network.recurrent_kernel(128,512)      -> decoder.cell.W_h.weight (512,128)
    decoder_network.bias            (512,)         -> decoder.cell.W_x.bias
    decoder_dense.kernel            (128,1)        -> head.weight (1,128)
    decoder_dense.bias              (1,)           -> head.bias

ConvLSTM kernel permute: Keras (kH, kW, C_in, 4*C_out) -> torch (4*C_out, C_in, kH, kW) via permute(3, 2, 0, 1). Dense kernels just transpose.

Intent param count from the loaded .h5: **1 837 953**. Matches the
paper (Keras model.summary would show the same).

### 2026-04-17 — test_downloader http_root fixture leaked chdir
`http_root` called `os.chdir(root)` without restoring. Every test after
it ran with CWD in a tmp dir, so any test using relative paths (e.g.
`test_keras_to_torch.py` skipping because `data/pie/...` looked
missing) silently skipped under the full pytest run but passed in
isolation. Fixed by saving/restoring CWD in the fixture teardown and
by making `_skip_if_h5_missing` use absolute paths based on `__file__`.
Lesson: fixtures that mutate process state (CWD, env vars, logging
handlers) MUST restore in teardown.

### 2026-04-17 — BCELoss + AMP is unsafe (caught mid-Colab-run)
- Intent training on Colab (T4, AMP auto-on for CUDA) crashed in the
  first step with:
      RuntimeError: torch.nn.functional.binary_cross_entropy and
      torch.nn.BCELoss are unsafe to autocast.
- Standard fix: switch to `binary_cross_entropy_with_logits` and remove
  the sigmoid from the model head. Logits-out is the PyTorch convention
  anyway.
- Changes:
    - `IntentConvLSTMEncDec.forward` returns raw logits (no sigmoid).
    - `cli/train.py::_build_intent` uses BCEWithLogitsLoss; metrics
      (accuracy, f1) are wrapped in a `_sigmoid_wrap` so they still
      see probabilities.
    - Intent docstring updated to say "apply torch.sigmoid(out) for
      P(crossing)".
    - `test_intent_forward_shape_and_range` -> `_shape_and_logits`;
      asserts `sigmoid(logits)` is in [0, 1] instead of `logits` itself.
- Lesson: AMP compatibility is a cross-cutting concern. Any loss that
  needs the sigmoid/softmax fused in should say so up front.

### 2026-04-17 — VGG extractor "auto" device + CPU-bound VGG net (caught mid-Colab-run)
- After fixing the dataclass field name, the intent run on Colab
  crashed again in `IntentFeatureDataset.__getitem__`:
      RuntimeError: Expected one of cpu, cuda, ipu, xpu, ... at start
      of device string: auto
- Two layered bugs:
    1. `_build_intent` forwarded the training config's `device` string
       ("auto") straight into `ExtractorConfig`. That's a Trainer-level
       shorthand; `ExtractorConfig.device` is used as a raw torch device.
    2. The VGG16 `_net` was never moved to any device. Even once we
       resolved "auto", CPU weights + GPU inputs would still mismatch.
- Fixes:
    - `VGG16FeatureExtractor.__init__` now raises `ValueError` if
      `cfg.device == "auto"` and moves `_net` to `self.cfg.device`.
    - `pie_pytorch/cli/train.py::_resolve_device` turns "auto" into a
      concrete string (cuda > mps > cpu) before constructing the
      extractor.
- Regression test `test_extractor_rejects_auto_device` covers the
  guard. Full suite now 166 passed.
- Lesson: each module that reads a `device` string should either
  accept only concrete devices (and enforce it), or resolve "auto"
  itself. Don't forward "auto" silently between layers.

### 2026-04-17 — intent YAML vs dataclass field name mismatch (caught mid-Colab-run)
- `intent_colab.yaml` has `model.observe_length` but
  `IntentModelConfig` declared `sequence_length` (a dead field, never
  used inside the model forward). `IntentModelConfig(**cfg["model"])`
  raised `TypeError: got an unexpected keyword argument
  'observe_length'`. Renamed the dataclass field to `observe_length`
  for consistency with `IntentConfig` / `TrajectoryConfig` /
  `SpeedConfig`.
- Unit tests that only covered the dataclass in isolation missed it.
  Added `test_packaged_configs_instantiate_model_config`: parameterized
  over all three Colab configs, actually calls `IntentModelConfig(
  **cfg["model"])` / `AttnEncDecConfig(**cfg["model"])` so the
  field-name contract between YAML and code is verified in CI.
- Lesson: "YAML loads" is not the same as "YAML produces a valid
  kwarg-set for the class it's meant to feed". Assert both.

### 2026-04-17 — PyYAML shorthand-scientific footgun (caught mid-Colab-run)
- `--override training.lr=5e-3` on Colab crashed because
  `yaml.safe_load("5e-3")` returns the string `"5e-3"`, not `0.005`.
  PyYAML's float resolver regex requires either a decimal point before
  the exponent (`5.0e-3`) or a leading dot (`.5e-3`). Bare mantissa + `e-3`
  is rejected.
- Fix: when `yaml.safe_load` returns a string identical to the raw token,
  attempt `int(raw)` then `float(raw)` before committing the value.
  Regression test `test_override_coerces_shorthand_scientific` covers
  both `lr=5e-3` (float) and `epochs=30` (int).
- Lesson #2 for config tests: YAML quirks hit both config *files* and
  command-line override tokens. Test both paths.

### 2026-04-17 — YAML `Infinity` footgun (caught mid-Colab-run)
- All three Colab configs had `height_rng: [0, Infinity]`. YAML does
  NOT treat `Infinity` as special; it parses as the **string**
  `'Infinity'`. `pie_data._get_trajectories` then does
  `height_rng[1] < float('inf')` which raises
  `TypeError: '<' not supported between instances of 'str' and 'float'`.
- Fix: use `.inf` (YAML 1.1 spec for float infinity). Verified all three
  configs now parse `height_rng[1]` as `math.inf`.
- Added `tests/test_cli_and_overfit.py::test_packaged_configs_load_cleanly`
  — parameterized over all three configs, asserts `height_rng[1]` is a
  real float and `isinf`. This catches the same class of typo if someone
  adds a new config.
- Lesson: config smoke-loads deserve their own unit tests. Unit-tested
  override plumbing isn't enough when the actual shipped YAMLs have
  string↔numeric bugs.

### 2026-04-17 — Colab live validation (user-run)
- Full pipeline validated on a Colab T4: 95 pytest passes, annotations
  downloaded + parsed, set05 videos downloaded (~2 GB), frames
  extracted, Dataset classes return real-shaped tensors on real PIE
  tracks.
- **set05 is in the val split, not test.** Paper's fixed split:
  train=set01+02+04, val=set05+06, test=set03. Filtering the `test`
  split by `set05` correctly yields 0 tracks (not a bug — by design).
  Add a clearer error/warning if someone filters to an empty set.
- **PIE positional-arg footgun**: `PIE("/path/to/data")` sets
  `regen_database=True` and leaves `data_path=''` because the vendored
  `__init__` signature is `PIE(regen_database=False, data_path='')`.
  Always call `PIE(data_path="...")` as a kwarg. Consider adding a
  wrapper helper in `pie_pytorch.data.__init__` (e.g. `load_pie(path)`)
  so users can't miss this.
- **Don't clone code into Drive.** Drive's filesystem is slow and
  permission-quirky for git operations. Clone to `/content/` (fast
  ephemeral) and keep only the dataset on Drive. Documented in the
  Colab checklist.
- Subset histogram for real val split matched expectations: set05=16,
  set06=227 pedestrians. 16 tracks → 132 windows at obs=15 / overlap=0.5.
  Sample ped_id `5_1_1731` and label `1.0` (crossing) round-tripped
  through `IntentRawDataset`.
- Frame extraction for set05 (`PIE.extract_and_save_images('annotated')`)
  finished in the low-minutes range; progress bar rounds down so it
  shows 99.97% at completion. Cosmetic only.

### 2026-04-16 — Annotations layout surprise (self-caught via live run)
- Initial `download_annotations` assumed the tarball had three top-level
  dirs (`annotations/`, `annotations_attributes/`, `annotations_vehicle/`)
  each containing its own `{dir}.zip`. That is **wrong**. The real
  upstream repo only has one top-level `annotations/` dir and stashes
  **all three** zips inside it along with README + shell scripts:
      PIE-master/annotations/annotations.zip
      PIE-master/annotations/annotations_attributes.zip
      PIE-master/annotations/annotations_vehicle.zip
- Caught only by running the CLI against the real upstream (unit tests
  with my fake tarball happily passed). Took ~2 iterations to nail down
  the true layout via `WebFetch` of the GitHub tree.
- Rewrote the logic: pull the three zips into a `_annot_staging/` dir,
  unzip each into `dest/` (creates the three expected output dirs as
  zip roots), then delete the staging dir + zips (or move to
  `_annotation_zips/` when `--keep-zips`).
- Updated the pytest fixture `_make_fake_pie_tarball()` to mirror the
  real layout so the next regression of this shape fails CI, not
  production.
- Added `docs/colab_checklist.md` with a step-by-step Colab playbook.
- Verified end-to-end: `PIE.get_annotated_frame_numbers('set05')`
  returns real frame ranges after download.
- Lesson: for anything that touches a remote system I can't fully mock,
  do at least one real-network integration run before declaring it
  done. Tests alone are not enough.

### 2026-04-16 — Phase 1 loose ends
- **Annotations missing from repo.** `PIE_dataset/` has only `.gitkeep`. The
  PIE XML annotations live at `github.com/aras62/PIE/tree/master/annotations*`.
  Two paths forward (option 1 is default; option 2 in Phase 1.6 if requested):
    1. User downloads from aras62/PIE and points `$PIE_PATH` there.
    2. Ship `scripts/download_annotations.sh` that clones aras62/PIE and
       copies the three `annotations*` dirs into `$PIE_PATH`.
- **`pie_data.py` actual path.** Both `master` and `main` 404 on the repo root.
  Real location is `utilities/pie_data.py`. Vendored with attribution header.
- **Dataset size clarification.** Videos-only download is ~74 GB (6 sets);
  the 1.1 TB figure is for all frames extracted. Colab: download one set.
- **Phase 0 guard test was too loose.** Matched the string "keras" inside
  docstrings. Rewrote to walk each module's globals and only check
  `isinstance(v, ModuleType)` against `{tensorflow, keras}` top-level names.
  Retroactive: this would not have flagged the issue if the guard also
  triggered on `sys.modules` entries from transitive imports; it
  currently checks only *direct* imports of the forbidden top-level
  modules, which is what we actually care about.
- **Trajectory normalization order.** Legacy code windows FIRST, then
  does `w[1:] - w[0]` on bbox (and trims other fields to `w[1:]`). I
  originally did normalize-first-window-second which produced 46-step
  targets instead of 45. Fixed; tests now green.

### 2026-04-16 — Phase 1 test-fixture bug (self-caught, not prod code)
- `test_subset.py::test_fraction_different_seeds_differ` failed at first
  run. Cause: synthetic data generator reused the *same* per-track image
  paths (only frame index varied), so sampling different tracks still
  produced identical `image[0][0]` entries. Production `SubsetConfig` was
  correct; test was too weak to notice. Fixed by embedding the track
  index in the video dir (`video_{t:04d}`). All 10 subset tests pass now.
- Lesson for later tests: when asserting "these two outputs differ",
  make sure the synthetic fixture actually *can* differ on the field you're
  comparing.

### 2026-04-16 — Phase 0 environment surprises
- **Env had no PyTorch installed.** `pip install torch pytest pyyaml` pulled `torch 2.11.0+cu130` (CPU build, `cuda=False`). Acceptable for Phase 0/1/2 validation; we'll need a real GPU for Phase 4 parity and full training.
- **NumPy missing too.** After installing torch, a plain `import torch` emitted "Failed to initialize NumPy"; had to `pip install numpy`. Requirements file already lists numpy, but users may skip `pip install -r requirements.txt`. Docs should say so loudly.
- **pytest collected the `tests/` dir fine** once `pyproject.toml` set `testpaths = ["tests"]`. `23 passed in 3.34s`.

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
