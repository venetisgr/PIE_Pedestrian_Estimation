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
- [x] 1.1 Vendor `pie_data.py` (from aras62/PIE `utilities/pie_data.py`) into `pie_pytorch/data/pie_data.py` (1355 lines, MIT)
- [x] 1.2 Port `utils.py` → `pie_pytorch/data/transforms.py` + `sequences.py`
- [x] 1.3 `IntentRawDataset`, `TrajectoryDataset`, `SpeedDataset`
- [x] 1.4 `SubsetConfig` (`fraction`, `max_tracks`, `set_ids`, seeded)
- [x] 1.5 VGG16 feature extractor + per-video shard cache (`VideoShardCache`)
- [x] **Validation 1**: 78 passed, 1 deselected (slow VGG-weights test) — transforms parity (29), subset (10), datasets (8), features (8), scaffold (23). NumPy-writability warning fixed. Annotations download path TBD.

## Phase 2 — Model layer
- [x] 2.1 `ConvLSTM2DCell` / `ConvLSTM2D` with Keras defaults (hard_sigmoid gates, unit_forget_bias, xavier/orthogonal init)
- [x] 2.2 `TemporalAttention`, `ElementAttention`; `KerasLSTM` w/ configurable tanh/softsign
- [x] 2.3 `IntentConvLSTMEncDec` (VGG features → ConvLSTM → RepeatVector+concat → KerasLSTM → sigmoid)
- [x] 2.4 `AttnEncDec` shared class + `trajectory_model()` / `speed_model()` factories (TemporalAttention → KerasLSTM(softsign) → embed+dropout → ElementAttention → KerasLSTM(softsign) → Dense linear)
- [ ] 2.5 `torchinfo.summary` hookup (deferred; param-count sanity covered by tests)
- [x] **Validation 2**: 38 new tests, 133 total passing. Intent param count ~1.8M (Keras ref ~1.8M), batch-invariance within 1e-6, E2E Dataset→model smoke for trajectory.

## Phase 3 — Training / eval with W&B
- [x] 3.1 `trainer.py` (AMP, grad clipping, RMSprop α=0.9 ε=1e-7, plateau, early stop)
- [x] 3.2 `checkpoint.py` (safetensors + state.pt, top-K, best/latest pointers, Drive-safe fallback)
- [x] 3.3 `metrics.py` (accuracy, f1, mse, center_mse + `MetricsAccumulator`)
- [x] 3.4 W&B integration — soft-dep, `--no-wandb` flag, project/run_name in config
- [x] 3.5 `cli/train.py` — YAML-driven, `--override key=value`, env-var expansion
- [x] 3.6 `cli/eval.py` — resolves best/latest/epoch dir and runs on any split
- [x] YAML configs: `intent_colab.yaml`, `trajectory_colab.yaml`, `speed_colab.yaml`
- [x] **Validation 3**: 158 total passing. Trainer overfits toy linear regression (val MSE < 0.05 in 60 ep). CLI-run speed + trajectory overfit on synthetic PIE-shaped data (loss drops >10%). Checkpoints land on disk, `--no-wandb` CLI flag works.

## Phase 4 — Parity vs. TF reference
- [x] 4.1 `pie_pytorch/io/keras_to_torch.py` — Keras .h5 → PyTorch state dict for all 3 models. Auto-sizes the config from the .h5 shapes. Found + fixed 2 parity bugs: valid padding default, hardcoded same on the recurrent conv.
- [x] 4.1 `pie_pytorch/cli/convert.py` — ``pie-convert --task --h5 --out`` CLI that writes a portable ``.safetensors`` + ``.config.json`` beside it.
- [x] 4.1 ``pie-eval --keras-h5 path/to/model.h5`` — load paper weights directly and evaluate on any split.
- [ ] 4.2 End-to-end paper-weights evaluation on PIE test split (set03). Requires downloading set03 videos + annotations; run on Colab.
- [ ] 4.3 Pluggable-backbone registry (for tracks 3-7).
- [ ] **Validation 4**: `pie-eval --keras-h5 ... --split test` on set03 should yield intent acc ≈ 0.79, F1 ≈ 0.87 (paper numbers).

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
- Phase 0-3 + downloader + Colab 3-model validation + Phase 4.1 weight
  port all shipped. 178 tests green.

## Experiment bucket list (5–7 tracks on the same eval split)
| Track | Backbone | Backbone state | Head |
|---|---|---|---|
| 1 | VGG16 | frozen | paper weights (Phase 4.1 ✅) |
| 2 | VGG16 | frozen | scratch |
| 3 | ResNet50 | frozen | scratch |
| 4 | DINOv2 ViT-B/14 | frozen | scratch |
| 5 | VGG16 | unfrozen | scratch |
| 6 | ResNet50 | unfrozen | scratch |
| 7 | DINOv2 ViT-B/14 | unfrozen | scratch |

## Next up
- Phase 4.2 — pluggable backbone registry + configs for tracks 3-4.
- Or: Colab eval of paper weights on set03 to get the reproducible
  baseline number before any other work.

## Out-of-phase (Phase 1.6) — Asset downloader (2026-04-16)
- [x] `pie_pytorch/data/downloader.py` with video + annotation sub-commands.
      Videos: 53 files / 6 sets hard-coded inventory, Range-resumable,
      atomic rename, `--workers` parallel, `--dry-run`, skip-if-complete.
      Annotations: pull aras62/PIE master tarball, extract the 3 zips
      from inside `annotations/`, unzip each to the expected layout,
      then clean up (unless `--keep-zips`). Idempotent.
- [x] 17 tests (local HTTP server with Range support + in-memory
      tarball). **95 passed, 1 deselected** total suite.
- [x] **Live validated**: downloaded set05 (~2.1 GB) and all annotations
      to a sandbox, loaded `PIE.get_annotated_frame_numbers('set05')`
      via the vendored parser — returned real frame ranges.
- [x] `docs/colab_checklist.md`: top-to-bottom Colab playbook.
