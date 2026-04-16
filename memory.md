# Memory — Resume-from-cold context

This file is the "wake-up briefing" if the session restarts. It should
be enough context to pick the work up without rereading the whole repo.

---

## The task in one paragraph

Convert the TensorFlow 1.9 / Keras 2.2 implementation of the PIE
(Pedestrian Intention Estimation, Rasouli et al. ICCV 2019) paper to
**modern PyTorch** with **Weights & Biases** logging. All **three
models** (intent, trajectory, speed) must be ported. The result must
run both **locally** (full 1.1 TB PIE dataset) and on **Google Colab**
(small curated sample + optional Drive mount). Keep TF code under
`legacy_tf/` for reference; delete it after parity is confirmed. Aim
for a **faithful** port with **modern training niceties** (AMP, proper
checkpointing, config files, seeds, W&B).

## Where we are right now

- **Branch**: `claude/tensorflow-to-pytorch-conversion-0SlwI`
- **Code changes**: none yet. Documentation only.
- **Docs written this session**:
  - `plan.md` — full step-by-step plan with per-phase validations
  - `checklist.md` — status tracker synced with `plan.md`
  - `notepad.md` — gotchas, hyperparameters, porting risks
  - `memory.md` — this file

## User's explicit scoping decisions (do not re-litigate)

1. Port **all three** models.
2. Dataset:
   - Local users → whole dataset supported.
   - Drive users on Colab → mount Drive.
   - **Add a subset/fraction/N-tracks flag** so even local users can
     train on a portion.
   - Colab default → ship a small curated sample (one or two `set_XX/`
     dirs or a hosted zip).
3. Keep TF code in `legacy_tf/` *for reference only*; delete once the
   PyTorch port hits parity.
4. **Faithful + modern**: mirror the architectures and data flow, but
   modernize training (AMP, W&B, YAML configs, safetensors, seeding).

## Repo facts (as of this session)

- Repo root: `/home/user/PIE_Pedestrian_Estimation`
- Source files (all TF/Keras today):
  - `pie_intent.py` — ConvLSTM encoder-decoder on VGG16 features; binary intent.
  - `pie_predict.py` — attention encoder-decoder for trajectory *and* speed.
  - `train_test.py` — entry point; trains intent, then speed, then trajectory; tests all three.
  - `utils.py` — `img_pad`, `squarify`, `jitter_bbox`, `bbox_sanity_check`, `update_progress`.
- **Missing** from this repo but imported: `pie_data.py` from `github.com/aras62/PIE`. Must be vendored in Phase 1.1.
- Pretrained weights already present:
  - `data/pie/intention/context_loc_pretrained/model.h5`
  - `data/pie/speed/speed_pretrained/model.h5`
  - `data/pie/trajectory/loc_intent_speed_pretrained/model.h5`
  These are the ground-truth baselines for Phase 4 parity.
- `PIE_dataset/` is a `.gitkeep`-only mount point — users point `PIE_PATH` at their actual dataset.

## Architecture summary (for quick reference)

**Intent** (`pie_intent.py:441`)
```
frames(T=15, 224, 224, 3)
  → VGG16 (frozen, imagenet)
  → feature maps (T, 7, 7, 512)
  → ConvLSTM2D(filters=64, kernel=2)
  → Flatten
  → RepeatVector(T) ⊕ bbox_decoder_input(T, 4)
  → LSTM(128, tanh, dropout=0.4)
  → Dense(1, sigmoid) → P(crossing)
```

**Trajectory** (`pie_predict.py:632`)
```
bbox_obs(T=14 after normalization, 4)
  → temporal_attention (Permute, Dense(T, sigmoid), Permute, Multiply)
  → LSTM(256, softsign) → hidden h
  → RepeatVector(45) → Dense(64, relu) → Dropout
  → concat with decoder_input(intention_prob, obd_speed) (45, 2)
  → element_attention (Dense(dim, sigmoid), Multiply)
  → LSTM(256, softsign, init_state=h)
  → Dense(4, linear) → bbox deltas (45, 4)
```

**Speed** — same skeleton as trajectory with 1D features in/out.

## Step-by-step intent of the plan (headlines)

- **Phase 0** scaffolding (move TF → `legacy_tf/`, build `pie_pytorch/` package, refresh requirements)
- **Phase 1** data layer (vendor `pie_data.py`, port transforms, Dataset classes, subset selector, VGG16 cache)
- **Phase 2** models (custom ConvLSTM2D, two attention modules, 3 end-to-end models)
- **Phase 3** training (trainer, checkpointing, metrics, W&B, CLI)
- **Phase 4** parity vs TF pretrained weights (dump `.h5` → `.npz` → load into PyTorch, compare)
- **Phase 5** UX (README, Colab notebook, sample download script, colab config)
- **Phase 6** cleanup (delete `legacy_tf/`, verify no TF imports remain)

Every phase ends with explicit validation checks in `plan.md`. Failures
get root-caused and logged in `notepad.md`.

## Immediate next actions on resume

1. Read `plan.md` (source of truth) and `checklist.md` (current
   status) — skim `notepad.md` for porting hazards already noted.
2. Before writing code, re-check for any user follow-ups in the
   conversation.
3. **Start Phase 0**:
   - `mkdir legacy_tf && git mv pie_intent.py pie_predict.py train_test.py utils.py legacy_tf/`
   - Create `pie_pytorch/` package skeleton per the layout in
     `plan.md`.
   - Rewrite `requirements.txt` for PyTorch stack.
   - Commit: "Phase 0: scaffold PyTorch package layout".
4. Then Phase 1 — vendor `pie_data.py`, port transforms, write
   `PIEIntentDataset`.

## Porting hazards to keep front of mind (full list in `notepad.md`)

- **ConvLSTM2D** has no PyTorch stdlib equivalent — implement our
  own; unit-test with fixed weights vs a NumPy reference.
- **Keras `l2(v)`** ≠ PyTorch `weight_decay=v` (factor of 2
  difference).
- **Keras RMSprop defaults** differ from PyTorch (`rho=0.9` vs
  `alpha=0.99`, `eps=1e-7` vs `1e-8`).
- **Channel ordering**: source is NHWC, PyTorch default is NCHW. Be
  explicit at boundaries.
- **Trajectory normalization shrinks obs length by 1**
  (`observe_length -= 1` at `pie_predict.py:192`).
- **`BCEWithLogitsLoss`** instead of sigmoid+BCE for stability.
- **TF 1.9 is unbuildable today** — use `h5py` + layer-name mapping
  to port pretrained weights, not the old Keras stack.

## Files to create next (Phase 0-1 concrete)

```
legacy_tf/{pie_intent,pie_predict,train_test,utils}.py      (moved)
legacy_tf/README_LEGACY.md

pie_pytorch/__init__.py
pie_pytorch/data/{__init__,pie_data,transforms,pie_dataset,subset}.py
pie_pytorch/features/{__init__,vgg16_features}.py
pie_pytorch/models/{__init__,layers,intent,trajectory,speed}.py
pie_pytorch/training/{__init__,trainer,checkpoint,metrics,callbacks}.py
pie_pytorch/configs/{intent,trajectory,speed,colab_sample}.yaml
pie_pytorch/cli/{__init__,train,eval}.py

requirements.txt             (rewritten)
requirements-dev.txt         (new)
.gitignore                   (add wandb/, *.pt, checkpoints/, features/)
tests/conftest.py            (pytest)
```

## Validation milestones (summary)

- **Phase 0 done when**: `import pie_pytorch` works, `pytest` collects 0 tests cleanly, torch+cuda+wandb import.
- **Phase 1 done when**: Dataset emits tensors with the TF shapes, `img_pad` is pixel-identical, VGG16 feature shape matches.
- **Phase 2 done when**: 3 models forward-pass with correct shapes, grads finite, param counts ≈ Keras.
- **Phase 3 done when**: 1-epoch CPU smoke runs, overfit test passes, W&B URL produced, resume works.
- **Phase 4 done when**: ported weights yield metrics within target bands vs Keras pretrained.
- **Phase 5 done when**: fresh Colab runs notebook top-to-bottom in < 20 min.
- **Phase 6 done when**: no TF imports anywhere, full test suite green, `legacy_tf/` deleted.

## Commands cheat sheet

```bash
# Run tests
pytest -q

# Train intent (local)
python -m pie_pytorch.cli.train --config pie_pytorch/configs/intent.yaml

# Train intent on a subset (10% of tracks)
python -m pie_pytorch.cli.train --config pie_pytorch/configs/intent.yaml \
    --data.subset.fraction 0.1

# Evaluate with pretrained
python -m pie_pytorch.cli.eval --config pie_pytorch/configs/intent.yaml \
    --checkpoint data/pie/intention/context_loc_pretrained/model_pt.safetensors

# Disable W&B
WANDB_MODE=disabled python -m pie_pytorch.cli.train ...
```
