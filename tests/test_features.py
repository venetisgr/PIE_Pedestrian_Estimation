"""Tests for pie_pytorch.features.vgg16_features.

Two tiers:
 1. Always-on tests for crop/preprocess/cache (no network, no weights).
 2. "slow" test (``-m slow``) that actually loads VGG16 weights and
    checks feature shape. Skipped by default so CI / Colab free-tier
    don't timeout downloading 500+ MB of weights.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import PIL
import pytest
import torch

from pie_pytorch.features.vgg16_features import (
    VideoShardCache,
    context_crop,
    preprocess_image,
)


# ---------------------------------------------------------------------------
# crop + preprocess
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_rgb_image(tmp_path):
    arr = (np.arange(400 * 600 * 3, dtype=np.uint8) % 255).reshape(400, 600, 3)
    p = tmp_path / "frame.png"
    PIL.Image.fromarray(arr, mode="RGB").save(p)
    return str(p)


def test_context_crop_shape_and_mode(tmp_rgb_image):
    # bbox well inside the image
    crop = context_crop(tmp_rgb_image, [200.0, 150.0, 350.0, 250.0])
    assert crop.size == (224, 224)
    assert crop.mode == "RGB"


def test_context_crop_edge_bbox(tmp_rgb_image):
    # bbox near right edge -> squarify+jitter should still yield a valid crop
    crop = context_crop(tmp_rgb_image, [550.0, 200.0, 595.0, 280.0])
    assert crop.size == (224, 224)


def test_preprocess_torchvision_stats():
    pil = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))
    x = preprocess_image(pil, mode="torchvision")
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32
    # 128/255 = 0.502, then ImageNet normalization makes mean~0 (but not exact for grey)
    assert -3.0 <= x.min() and x.max() <= 3.0


def test_preprocess_keras_matches_legacy_semantics():
    """BGR reorder + Caffe mean subtraction, no /255."""
    pil = PIL.Image.new("RGB", (2, 2), color=(10, 20, 30))  # R=10, G=20, B=30
    x = preprocess_image(pil, mode="keras")
    assert x.shape == (3, 2, 2)
    # After RGB->BGR the first channel should be the B value (30),
    # minus the Caffe B-mean (103.939).
    assert torch.allclose(x[0], torch.full((2, 2), 30.0 - 103.939))
    assert torch.allclose(x[1], torch.full((2, 2), 20.0 - 116.779))
    assert torch.allclose(x[2], torch.full((2, 2), 10.0 - 123.68))


def test_preprocess_invalid_mode():
    pil = PIL.Image.new("RGB", (4, 4))
    with pytest.raises(ValueError):
        preprocess_image(pil, mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# VideoShardCache
# ---------------------------------------------------------------------------
def test_cache_put_flush_get_roundtrip(tmp_path):
    cache = VideoShardCache(tmp_path)

    img1 = "/PIE_data/set01/video_0001/00123.png"
    img2 = "/PIE_data/set01/video_0001/00124.png"
    feat1 = torch.randn(512, 7, 7)
    feat2 = torch.randn(512, 7, 7)

    with cache:
        cache.put(img1, "ped_A", feat1)
        cache.put(img2, "ped_A", feat2)

    # Rehydrate in a fresh cache instance
    cache2 = VideoShardCache(tmp_path)
    assert torch.equal(cache2.get(img1, "ped_A"), feat1)
    assert torch.equal(cache2.get(img2, "ped_A"), feat2)
    assert cache2.get(img1, "ped_B") is None  # different ped


def test_cache_cross_video_flush(tmp_path):
    """Switching video shards must flush the previous one."""
    cache = VideoShardCache(tmp_path)
    a = "/PIE/set02/video_0003/00001.png"
    b = "/PIE/set02/video_0004/00001.png"
    cache.put(a, "ped_X", torch.ones(512, 7, 7))
    cache.put(b, "ped_Y", torch.full((512, 7, 7), 2.0))
    cache.flush()

    cache2 = VideoShardCache(tmp_path)
    assert torch.equal(cache2.get(a, "ped_X"), torch.ones(512, 7, 7))
    assert torch.equal(cache2.get(b, "ped_Y"), torch.full((512, 7, 7), 2.0))


def test_cache_parse_path():
    s, v, f = VideoShardCache.parse_path("/a/b/set03/video_0001/00042.png")
    assert (s, v, f) == ("set03", "video_0001", "00042")


# ---------------------------------------------------------------------------
# VGG forward (slow -- downloads weights on first run)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_extractor_forward_shape():
    from pie_pytorch.features.vgg16_features import (
        ExtractorConfig,
        VGG16FeatureExtractor,
    )

    ext = VGG16FeatureExtractor(ExtractorConfig(device="cpu"))
    x = torch.zeros(1, 3, 224, 224)
    out = ext(x)
    assert out.shape == (1, 512, 7, 7)
    assert ext.feature_shape == (512, 7, 7)
