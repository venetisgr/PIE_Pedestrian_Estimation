# PIE Pedestrian Estimation — TF/Keras → PyTorch Port

## Context

The repo implements the PIE (Pedestrian Intention Estimation) paper
(Rasouli et al., ICCV 2019) in TensorFlow 1.9 + Keras 2.2. The stack is
unmaintained and frozen to CUDA/cuDNN versions that do not install on
modern hardware or on Google Colab. The goal is a **faithful, modern
PyTorch port** of all three models, with **Weights & Biases logging**,
that can be run on a local GPU box *and* on Google Colab (where the
1.1 TB PIE dataset cannot live in full).

### Scoping decisions confirmed with the user
| Question | Decision |
|---|---|
| Models to port | All three: **Intent**, **Trajectory**, **Speed** |
| Dataset on Colab | Local = whole dataset; Drive users mount; add a **subset/sampling flag** so local users can also train on a portion; Colab defaults to a **small curated sample** |
| Code layout | Keep TF code for reference in `legacy_tf/`; new PyTorch code lives at the repo root (or in `pie_pytorch/`). Delete `legacy_tf/` once PyTorch parity is verified. |
| Fidelity | **Faithful** port of model architectures and data flow, with **modern training** niceties: `torch.amp`, W&B logging, `Accelerate`-style device-agnostic code, checkpointing via `safetensors`, config via `hydra`/YAML. |

---

## Models to port (from TF/Keras source)

### 1. Intent model — `pie_intent.py` → `pie_pytorch/models/intent.py`
- **Architecture** (`pie_intent.py:441`): `ConvLSTM2D` encoder over VGG16 feature maps + LSTM decoder + `Dense(1, sigmoid)`. VGG16 is ImageNet-pretrained, no top, used as a frozen feature extractor over 1.5×-enlarged, pad-resized 224×224 context crops around the pedestrian bbox.
- **Inputs**: `[VGG16_features_seq (T=15, H, W, C), bbox_seq (T=15, 4)]`.
- **Output**: `P(crossing) ∈ [0,1]`.
- **Training**: RMSprop, binary cross-entropy, `ReduceLROnPlateau`, `EarlyStopping`, `ModelCheckpoint` (Keras callbacks at `pie_intent.py:605-617`).
- **Hyperparams**: `num_hidden=128, reg=0.001, lstm_dropout=0.4, lstm_recurrent_dropout=0.2, convlstm_num_filters=64, convlstm_kernel_size=2`, batch=128, epochs=400, lr=1e-5.

### 2. Trajectory model — `pie_predict.py::pie_encdec` → `pie_pytorch/models/trajectory.py`
- **Architecture** (`pie_predict.py:632`): encoder LSTM with temporal attention on the input, decoder LSTM with element-wise self-attention on the concatenated `[encoder_embedding, decoder_input]`. Encoder input = bbox sequence (normalized by subtracting first frame). Decoder input = `[intention_prob, obd_speed]`. Output = future bbox deltas over `predict_length=45`.
- **Custom layers to reimplement**:
  - `attention_temporal` (`pie_predict.py:701`) — Permute → Dense(seq_len, sigmoid) → Permute → element-wise multiply.
  - `attention_element` (`pie_predict.py:714`) — Dense(dim, sigmoid) over feature dim → multiply.
- **Hyperparams**: `num_hidden=256, reg=1e-4, activation='softsign', embed_size=64`.

### 3. Speed model — `pie_predict.py` with `enc_input_type=['obd_speed']` → `pie_pytorch/models/speed.py`
- Same encoder-decoder skeleton as trajectory, but consumes OBD speed scalars, predicts future speeds, outputs 1-dim target. Shares architecture code with the trajectory model; differs by feature sizes only.

---

## Target repo layout

```
PIE_Pedestrian_Estimation/
├── legacy_tf/                     # original TF/Keras code, moved for reference
│   ├── pie_intent.py
│   ├── pie_predict.py
│   ├── train_test.py
│   ├── utils.py
│   └── README_LEGACY.md           # note that this is reference only
├── pie_pytorch/
│   ├── __init__.py
│   ├── data/
│   │   ├── pie_dataset.py         # torch Dataset wrapping pie_data.PIE
│   │   ├── samplers.py            # sequence sampling, overlap stride
│   │   ├── transforms.py          # jitter_bbox, squarify, pad_resize (ported from utils.py)
│   │   └── subset.py              # fraction / N-sample / set-id filter
│   ├── models/
│   │   ├── layers.py              # ConvLSTM2D, temporal_attention, element_attention
│   │   ├── intent.py              # convlstm_encdec
│   │   ├── trajectory.py          # attention encdec (bbox → bbox)
│   │   └── speed.py               # attention encdec (speed → speed)
│   ├── features/
│   │   └── vgg16_features.py      # VGG16 feature cache (.pt shards)
│   ├── training/
│   │   ├── trainer.py             # device-agnostic train loop, AMP, W&B
│   │   ├── callbacks.py           # EarlyStopping, ReduceLROnPlateau equivalents (torch)
│   │   ├── checkpoint.py          # save/load best, resume
│   │   └── metrics.py             # accuracy, F1, MSE, C-MSE
│   ├── configs/
│   │   ├── intent.yaml
│   │   ├── trajectory.yaml
│   │   ├── speed.yaml
│   │   └── colab_sample.yaml      # lower batch, small subset, fewer epochs
│   └── cli/
│       ├── train.py               # python -m pie_pytorch.cli.train --config …
│       └── eval.py
├── notebooks/
│   └── colab_quickstart.ipynb     # mounts Drive, installs deps, pulls sample, runs smoke train
├── scripts/
│   ├── download_pie_sample.sh     # tiny sample (1-2 set_XX/video_XXXX) for Colab
│   └── build_vgg16_cache.py
├── requirements.txt               # updated for torch/wandb
├── requirements-dev.txt
├── pyproject.toml                 # or setup.cfg
├── plan.md
├── checklist.md
├── notepad.md
└── README.md                      # updated usage (local + Colab)
```

---

## Step-by-step plan

Each step ends with **Validation** — the checks we run before ticking it
off in `checklist.md`. If a check fails, fix it, and log the failure +
root cause in `notepad.md`.

### Phase 0 — Scaffolding & housekeeping (no model code yet)
0.1 Move TF files into `legacy_tf/`. Add `legacy_tf/README_LEGACY.md` saying "reference only, to be deleted after parity".
0.2 Create the `pie_pytorch/` package skeleton (empty modules with docstrings).
0.3 Rewrite `requirements.txt` for PyTorch: `torch`, `torchvision`, `wandb`, `numpy`, `opencv-python`, `Pillow`, `scikit-learn`, `pyyaml`, `prettytable`, `tqdm`, `einops`, `safetensors`, `accelerate` (optional). Pin only lower bounds.
0.4 Add `requirements-dev.txt`: `pytest`, `ruff`, `black`, `mypy`.
0.5 Create `.gitignore` entries for `wandb/`, `*.pt`, `checkpoints/`, `data/features/`, `PIE_dataset/images/`.

**Validation 0**
- `python -c "import torch, torchvision, wandb; print(torch.__version__, torch.cuda.is_available())"` succeeds.
- `pytest -q` collects zero tests without errors.
- `python -c "import pie_pytorch"` imports cleanly.

### Phase 1 — Data layer
1.1 Vendor / depend on `pie_data.py` (the official PIE data API from `github.com/aras62/PIE`). It is not in this repo but `train_test.py:22` imports it. Add it under `pie_pytorch/data/pie_data.py` or install from git; prefer vendor so we can fix Python-3 issues.
1.2 Port `utils.py` helpers (`img_pad`, `squarify`, `jitter_bbox`, `bbox_sanity_check`) into `pie_pytorch/data/transforms.py`. Replace Keras `load_img`/`img_to_array` with PIL + `numpy`.
1.3 Implement `PIEIntentDataset`, `PIETrajectoryDataset`, `PIESpeedDataset` (`torch.utils.data.Dataset`) that reproduce the sampling in `get_tracks` from each model file (`pie_intent.py:274`, `pie_predict.py:89`) including the `overlap_stride = int((1-overlap) * seq_length)` logic and the first-frame subtraction normalization for bbox/center.
1.4 Implement the **subset selector**: `subset.py` accepts one of
    - `fraction: 0.1` (random subset of tracks)
    - `max_tracks: N`
    - `set_ids: ['set01', 'set02']` (Colab-friendly — a single set is ~tens of GB)
    and composes with the split (train/val/test).
1.5 Implement `VGG16FeatureExtractor` (`torchvision.models.vgg16(weights=IMAGENET1K_V1).features`) with the last pool layer kept. Precompute & cache features to disk as `.pt` or `.safetensors` shards, keyed by `set/vid/frame_pid`. Port the `pie_intent.load_images_and_process` path (`pie_intent.py:208`).

**Validation 1**
- Unit test: given a synthetic sequence list, `get_tracks` yields identical shapes to the TF version for overlap ∈ {0, 0.5, 1}. (Compare with a deterministic fixture.)
- `img_pad(mode='pad_resize')` output matches the TF version pixel-for-pixel on 3 sample images (assert `np.allclose` on arrays).
- VGG16 feature tensor shape = `(T, 7, 7, 512)` for a 224×224 input, matches `context_model.output_shape[1:]` used at `pie_intent.py:446`.
- End-to-end: `DataLoader` for 1 small set (set01, 1 video) yields a batch whose tensors have the documented shapes.

### Phase 2 — Model layer
2.1 Implement `ConvLSTM2DCell` + `ConvLSTM2D` in `layers.py`, matching Keras defaults (`hard_sigmoid` recurrent act, `tanh` act, kernel/recurrent/bias regularizers via weight decay). Match Keras weight shapes so we *could* load `.h5` weights (stretch goal).
2.2 Implement `TemporalAttention` and `ElementAttention` modules that mirror `pie_predict.py:701` / `pie_predict.py:714`.
2.3 Implement `IntentConvLSTMEncDec` using the modules above + a standard `nn.LSTM` decoder + `nn.Linear(1) + Sigmoid`. Include L2 reg applied via optimizer `weight_decay`.
2.4 Implement `TrajectoryAttnEncDec` and `SpeedAttnEncDec` (they share backbone; parametrize feature sizes).
2.5 Add `torchinfo.summary` in each model's `__init__` under a verbose flag.

**Validation 2**
- Forward-pass shape tests for each of the 3 models with documented input shapes → assert output shapes match the Keras originals:
  - Intent out: `(B, 1)`.
  - Trajectory out: `(B, 45, 4)`.
  - Speed out: `(B, 45, 1)`.
- Parameter counts within ~5% of the Keras originals (sanity; document exact numbers in `notepad.md`).
- Gradient check: one forward+backward on random input produces finite grads in every parameter.
- Deterministic test: with `torch.manual_seed(0)` the output of each model on a fixed fixture is reproducible.

### Phase 3 — Training / eval (with W&B)
3.1 `trainer.py` — device-agnostic loop (`cuda`/`mps`/`cpu`), `torch.amp.GradScaler`, gradient clipping (Keras `clipvalue`), `RMSprop` (match Keras init), LR plateau scheduler (`ReduceLROnPlateau`), early stopping.
3.2 `checkpoint.py` — save best by `val_loss`, keep top-K, `safetensors` format, plus a JSON sidecar with config & metrics.
3.3 `metrics.py` — `accuracy`, `f1_score` (match `pie_intent.test_chunk` @ 722-723), `MSE`, `C-MSE` (center MSE, from `pie_predict.test_final`).
3.4 W&B integration:
   - `wandb.init(project='pie-pytorch', config=cfg, mode=os.environ.get('WANDB_MODE','online'))`.
   - Log `train_loss`, `val_loss`, `lr`, metrics each epoch; log model graph once; log config; log a small prediction visualization every N epochs.
   - Gate W&B behind `--no-wandb` flag (Colab users without accounts can still run).
3.5 `cli/train.py` — `argparse` or `hydra`: `python -m pie_pytorch.cli.train --config pie_pytorch/configs/intent.yaml --subset.fraction 0.1 --wandb.enabled true`.
3.6 `cli/eval.py` — mirror `train_test.py` main.

**Validation 3**
- **Smoke test (CPU)**: 1 epoch on 8 synthetic samples completes without error for each of the 3 models.
- **Overfit test**: each model can drive training loss < 10% of initial on a batch of 32 samples within 200 steps — proves the training loop wires gradients correctly.
- W&B dashboard shows scalar curves and config for a dry run (verify in UI or with `wandb.Api` after run).
- Checkpoint → reload → eval gives the same metric value (tolerance 1e-5).
- Ctrl-C mid-run leaves a resumable checkpoint; `--resume` picks it up.

### Phase 4 — Parity vs. TF reference
4.1 Pretrained Keras weights live under `data/pie/{intention,trajectory,speed}/.../model.h5`. Load them with `tensorflow==2.x` in a one-off script, dump per-layer weight tensors to `.npz`, then port into PyTorch state dicts. (Optional but ideal.)
4.2 Run Keras pretrained model inference on a small set (e.g. 100 test tracks) and store outputs.
4.3 Run PyTorch port on the same inputs; compare numerically.

**Validation 4**
- Per-layer max-abs-diff between TF and ported PyTorch weights < 1e-6.
- On the same 100 test tracks:
  - Intent: PyTorch accuracy within ±1 pp of the Keras pretrained (paper ~0.81).
  - Trajectory: MSE within ±5% of Keras (paper ~473 for MSE-45).
  - Speed: MSE within ±5% of Keras (paper ~2.65).
- If numerical parity fails, document in `notepad.md` which layer diverged and whether the source is padding/activation/init differences — then fix.

### Phase 5 — Local + Colab UX
5.1 `README.md` updates:
   - Local install (CUDA / CPU / MPS), `PIE_PATH` env var, dataset download pointer (external link, since the data can't be redistributed).
   - Subset flag examples.
   - Colab section.
5.2 `notebooks/colab_quickstart.ipynb`:
   - `!pip install -r requirements.txt`.
   - Mount Drive cell (optional).
   - Download sample cell (`scripts/download_pie_sample.sh`).
   - Run `intent` smoke train (1 epoch, subset.fraction=0.01).
   - Run `eval.py` with pretrained weights.
   - Log to W&B (cell prompts for API key via `wandb.login()`).
5.3 `scripts/download_pie_sample.sh` — pulls one set directory (or a zipped curated subset hosted elsewhere) to `PIE_dataset/`.
5.4 Colab config (`configs/colab_sample.yaml`): batch=16, epochs=2, subset.max_tracks=2000, precache VGG features to Drive.

**Validation 5**
- Fresh Colab runtime: notebook runs top-to-bottom without manual edits in < 20 min (T4 GPU).
- Fresh local box (CUDA): `python -m pie_pytorch.cli.train --config configs/intent.yaml` runs for 1 epoch without OOM on a single RTX-class GPU.
- Both runs produce a W&B run URL (or `--no-wandb` works silently).

### Phase 6 — Cleanup
6.1 When parity is confirmed and both UX paths work, **delete `legacy_tf/`** and references. Remove TF from `requirements.txt` (already absent by this point).
6.2 Final `README.md` pass. Tag `v1.0-pytorch`.

**Validation 6**
- `grep -r "import tensorflow\|from keras\|import keras" .` returns nothing.
- `pytest -q` full suite green.
- `pie_pytorch.cli.train` and `pie_pytorch.cli.eval` work end-to-end for all 3 models with pretrained or freshly-trained weights.

---

## Global conventions

- **Config**: single YAML per model; CLI can override any key.
- **Seeds**: a global `seed_everything(cfg.seed)` helper; set `torch`, `numpy`, `random`, and `torch.backends.cudnn.deterministic`.
- **Logging**: `logging` stdlib module; W&B is an additional sink.
- **Tests**: `tests/` folder, `pytest`. Shape + smoke + overfit tests are the minimum bar.
- **Commits**: work lives on branch `claude/tensorflow-to-pytorch-conversion-0SlwI`.

## Out of scope (intentionally)

- Retraining from scratch to reproduce paper numbers exactly (we aim for the pretrained-parity bar in Phase 4).
- JAAD dataset support (PIE only).
- Deployment/inference serving.
