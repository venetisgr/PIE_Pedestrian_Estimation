# Progress Checklist — TF/Keras → PyTorch Port

This checklist mirrors the phases in [`plan.md`](./plan.md). Keep it up
to date as work progresses. When something fails or surprises us, log
the detail in [`notepad.md`](./notepad.md).

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked (see notepad)

---

## Phase 0 — Scaffolding & housekeeping
- [x] 0.1 Move TF files into `legacy_tf/` (+ `README_LEGACY.md`)
- [x] 0.2 Create `pie_pytorch/` package skeleton
- [x] 0.3 Rewrite `requirements.txt` for PyTorch stack
- [x] 0.4 Add `requirements-dev.txt`
- [x] 0.5 Update `.gitignore` (`wandb/`, `*.pt`, checkpoints, feature caches)
- [x] **Validation 0**: `torch 2.11.0+cu130` imports; `import pie_pytorch` works; `pytest tests/` → 23/23 passed

## Phase 1 — Data layer
- [ ] 1.1 Vendor `pie_data.py` (from aras62/PIE) into `pie_pytorch/data/`
- [ ] 1.2 Port `utils.py` → `pie_pytorch/data/transforms.py`
- [ ] 1.3 `PIEIntentDataset`, `PIETrajectoryDataset`, `PIESpeedDataset`
- [ ] 1.4 Subset selector (`fraction`, `max_tracks`, `set_ids`)
- [ ] 1.5 VGG16 feature extractor + disk cache
- [ ] **Validation 1**: shape tests vs TF, pixel-level parity on `img_pad`, VGG out = `(T,7,7,512)`, end-to-end DataLoader smoke

## Phase 2 — Model layer
- [ ] 2.1 `ConvLSTM2DCell` / `ConvLSTM2D` (Keras-equivalent defaults)
- [ ] 2.2 `TemporalAttention`, `ElementAttention`
- [ ] 2.3 `IntentConvLSTMEncDec`
- [ ] 2.4 `TrajectoryAttnEncDec`, `SpeedAttnEncDec`
- [ ] 2.5 `torchinfo.summary` hookup
- [ ] **Validation 2**: shape tests, param-count parity (±5%), grad finiteness, deterministic forward

## Phase 3 — Training / eval with W&B
- [ ] 3.1 `trainer.py` (AMP, clipping, RMSprop, plateau, early stop)
- [ ] 3.2 `checkpoint.py` (safetensors, resume, top-K)
- [ ] 3.3 `metrics.py` (accuracy, F1, MSE, C-MSE)
- [ ] 3.4 W&B integration (+ `--no-wandb` flag)
- [ ] 3.5 `cli/train.py`
- [ ] 3.6 `cli/eval.py`
- [ ] **Validation 3**: CPU smoke, overfit 32-batch, W&B run visible, reload-identity, resume works

## Phase 4 — Parity vs. TF reference
- [ ] 4.1 Script to dump Keras `.h5` weights → `.npz`
- [ ] 4.2 Run Keras pretrained inference on 100-track fixture
- [ ] 4.3 Run PyTorch port on same fixture, compare
- [ ] **Validation 4**: per-layer diff < 1e-6 (if weights are ported); intent acc within ±1 pp; traj/speed MSE within ±5%

## Phase 5 — Local + Colab UX
- [ ] 5.1 `README.md` updates (local install, Colab section, subset flags)
- [ ] 5.2 `notebooks/colab_quickstart.ipynb`
- [ ] 5.3 `scripts/download_pie_sample.sh`
- [ ] 5.4 `configs/colab_sample.yaml` (small batch, subset, fewer epochs)
- [ ] **Validation 5**: fresh Colab run top-to-bottom < 20 min; fresh local run 1 epoch no OOM; W&B URL produced

## Phase 6 — Cleanup
- [ ] 6.1 Delete `legacy_tf/`; remove TF mentions from requirements/README
- [ ] 6.2 Final README pass, tag `v1.0-pytorch`
- [ ] **Validation 6**: no `tensorflow`/`keras` imports anywhere; full `pytest` green; end-to-end CLI works for all 3 models

---

## Currently in progress
- Phase 0 complete. Starting Phase 1 (data layer) next.

## Next up
- Phase 1.1 — vendor `pie_data.py` from `github.com/aras62/PIE`.
- Phase 1.2 — port `utils.py` helpers into `pie_pytorch/data/transforms.py` with pixel-level parity tests.
