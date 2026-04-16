"""PyTorch port of the PIE (Pedestrian Intention Estimation) models.

Subpackages:
    data       - Dataset classes, transforms, subset selection, PIE data API
    features   - VGG16 feature extraction / caching
    models     - ConvLSTM, attention modules, intent / trajectory / speed nets
    training   - Trainer, metrics, checkpointing, callbacks
    configs    - YAML config files (not Python code)
    cli        - Command-line entry points (train, eval)

See plan.md, checklist.md, notepad.md, memory.md at the repo root.
"""

__version__ = "0.0.1"
