"""VGG16 feature extractor + on-disk cache.

Phase 1.5. Uses torchvision vgg16 (IMAGENET1K_V1) features and caches
per set/video shards to avoid the one-file-per-frame explosion from
the original Keras pipeline (see notepad.md D4).
"""

from __future__ import annotations
