# Colab Checklist — PIE PyTorch Port

Use this as a playbook when you open a new Colab notebook. Every box
is a single cell unless noted. Copy/paste top-to-bottom.

## 0. Runtime

- [ ] **Runtime → Change runtime type → GPU (T4)**. CPU works for tests,
      but training in Phases 2–3 will need a GPU.
- [ ] Check you got a GPU:
      ```python
      !nvidia-smi | head -20
      ```

## 1. Mount Google Drive (for dataset + checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Pick a root that will survive across sessions. Suggested:
```
/content/drive/MyDrive/pie_data/
```

```python
import os
PIE_PATH = "/content/drive/MyDrive/pie_data"
os.makedirs(PIE_PATH, exist_ok=True)
os.environ["PIE_PATH"] = PIE_PATH
print("Using PIE_PATH =", PIE_PATH)
```

> ⚠️ **Drive quota**: free Colab Drive is 15 GB. That holds
> **set01 + set02 + set05** (≈9 GB) comfortably; one more set will
> blow it out. Colab-Pro Drive gives 100 GB+.

## 2. Clone the repo on the feature branch

```bash
%cd /content
!git clone -b claude/tensorflow-to-pytorch-conversion-0SlwI \
    https://github.com/venetisgr/PIE_Pedestrian_Estimation.git
%cd PIE_Pedestrian_Estimation
```

## 3. Install dependencies

```bash
!pip install -q -r requirements.txt
!pip install -q -r requirements-dev.txt
```

Colab runtimes ship with `torch` and `numpy` pre-installed; pip will
no-op on those.

## 4. Run the test suite (no dataset needed)

```bash
!python -m pytest tests/ -q
```

Expected: **93 passed, 1 deselected in ~12 s**. If anything fails here,
stop and share the output — something in the Colab runtime differs
from this sandbox.

## 5. Download annotations (fast; < 1 min)

```bash
!python -m pie_pytorch.data.downloader annotations --dest "$PIE_PATH"
```

This pulls only the three `annotations*` dirs (~100 MB) from
`aras62/PIE`. Idempotent — safe to re-run.

## 6. Download videos — pick a plan

Videos land at `$PIE_PATH/PIE_clips/setXX/video_YYYY.mp4`.

### A) Smoke test (2 videos, ~3 GB — fastest) — **recommended first**

```bash
!python -m pie_pytorch.data.downloader videos \
    --dest "$PIE_PATH" --sets set05 --workers 2
```

### B) Medium sample (set01 + set05, ~8 GB)

```bash
!python -m pie_pytorch.data.downloader videos \
    --dest "$PIE_PATH" --sets set01 set05 --workers 4
```

### C) Full dataset (~74 GB, hours) — **only on Colab-Pro Drive**

```bash
!python -m pie_pytorch.data.downloader videos \
    --dest "$PIE_PATH" --all --workers 4
```

### Dry-run first to see the size

```bash
!python -m pie_pytorch.data.downloader videos \
    --dest "$PIE_PATH" --sets set01 set05 --dry-run
```

### Resume after a disconnect

Just re-run the same command. Partials live in `*.part`, completed
files are skipped. `--no-resume` forces a restart.

## 7. Verify the download

```bash
!ls -lh "$PIE_PATH/PIE_clips/" 2>&1 | head
!du -sh "$PIE_PATH"/*
```

You should see one `.mp4` per video in each set dir, sizes matching
the server (e.g. `set05/video_0001.mp4 ≈ 1.5 GB`).

## 8. Quick sanity: load one frame via pie_data

```python
import sys, os
sys.path.insert(0, "/content/PIE_Pedestrian_Estimation")
from pie_pytorch.data.pie_data import PIE

imdb = PIE(data_path=os.environ["PIE_PATH"])
print("Annotated frame counts per set (first pass may parse XML — slow):")
try:
    print(imdb.get_annotated_frame_numbers("set05"))
except Exception as e:
    print("  -> ", e)
```

First call parses every XML once and caches a pickle under
`$PIE_PATH/data_cache/`, so later calls are fast.

## 9. Extract frames from a video (on-demand)

`pie_data` has its own extractor:

```python
imdb.extract_and_save_images(extract_frame_type="annotated")
```

This takes a while — for set05 it's ~10 minutes. Frames land at
`$PIE_PATH/images/setXX/video_YYYY/NNNNN.png`. **Required before
training**, since the models consume frames not video.

## 10. Planned next steps (will fill in as phases ship)

- [ ] Phase 2 — models land → `from pie_pytorch.models.intent import IntentConvLSTMEncDec`
- [ ] Phase 3 — `python -m pie_pytorch.cli.train --config configs/intent_colab.yaml`
- [ ] W&B login: `wandb.login()` then `export WANDB_MODE=online`

---

## Common Colab gotchas

| Symptom | Fix |
|---|---|
| `gdrive: no space left` | Delete previous checkpoints or use Colab-Pro. |
| Download stalls / disconnects | Re-run the same `videos` command — it resumes. |
| `ModuleNotFoundError: torch` | Reinstall: `pip install -q -r requirements.txt`. |
| Session timeout in middle of download | Mount Drive *before* starting; partials survive on Drive. |
| Extraction slower than expected | `opencv-python-headless` is fine; just wait. |

## What NOT to do

- ❌ Don't download the Models tile from the PIE website in Colab; the
  `.h5` Keras weights are Phase 4 parity work and live under
  `data/pie/*_pretrained/` already in this repo.
- ❌ Don't move `$PIE_PATH` mid-project. Everything caches paths; you'll
  have to re-parse XML and re-extract frames.
- ❌ Don't edit `legacy_tf/` — it's reference-only and is deleted in Phase 6.
