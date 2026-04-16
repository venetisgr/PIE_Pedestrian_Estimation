"""Parity tests: pie_pytorch.data.transforms vs legacy_tf/utils.py.

Strategy: re-implement the legacy behavior *from the legacy source*
(not by importing it - keras is not installed) and assert byte-level
equivalence with the new port. The legacy source is the canonical
reference, so these "gold" helpers below are copy-pasted verbatim from
legacy_tf/utils.py with the only change being ``load_img`` swapped for
``PIL.Image.open`` (since keras is unavailable).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import PIL
import PIL.Image
import pytest

from pie_pytorch.data import transforms as tf_new


# ---------------------------------------------------------------------------
# Gold reference implementations (copy of legacy_tf/utils.py sans keras)
# ---------------------------------------------------------------------------
def _gold_img_pad(img, mode="warp", size=224):
    assert mode in ("same", "warp", "pad_same", "pad_resize", "pad_fit")
    image = img.copy()
    if mode == "warp":
        return image.resize((size, size), PIL.Image.NEAREST)
    elif mode == "same":
        return image
    elif mode in ("pad_same", "pad_resize", "pad_fit"):
        img_size = image.size
        ratio = float(size) / max(img_size)
        if mode == "pad_resize" or (
            mode == "pad_fit" and (img_size[0] > size or img_size[1] > size)
        ):
            img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size - img_size[0]) // 2, (size - img_size[1]) // 2))
        return padded_image


def _gold_squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[2] > img_width:
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox


def _gold_sanity(img, bbox):
    w, h = img.size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= w:
        bbox[2] = w - 1
    if bbox[3] >= h:
        bbox[3] = h - 1
    return bbox


def _gold_jitter(img_path, bbox, mode, ratio):
    """Deterministic-modes-only gold (no random_*; random parity covered separately)."""
    assert mode in ("same", "enlarge", "move")
    if mode == "same":
        return bbox
    img = PIL.Image.open(img_path).convert("RGB")
    jitter_ratio = abs(ratio) if mode == "enlarge" else ratio
    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]
        wc = bbox_width * jitter_ratio
        hc = bbox_height * jitter_ratio
        if wc < hc:
            hc = wc
        else:
            wc = hc
        if mode == "enlarge":
            b[0] = b[0] - wc // 2
            b[1] = b[1] - hc // 2
        else:
            b[0] = b[0] + wc // 2
            b[1] = b[1] + hc // 2
        b[2] = b[2] + wc // 2
        b[3] = b[3] + hc // 2
        b = _gold_sanity(img, b)
        jit_boxes.append(b)
    return jit_boxes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rgb_image():
    arr = (np.arange(300 * 500 * 3, dtype=np.uint8) % 255).reshape(300, 500, 3)
    return PIL.Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="module")
def small_image():
    arr = (np.arange(80 * 120 * 3, dtype=np.uint8) % 255).reshape(80, 120, 3)
    return PIL.Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="module")
def tmp_img_path(rgb_image):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.png")
        rgb_image.save(p)
        yield p


# ---------------------------------------------------------------------------
# img_pad parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["warp", "same", "pad_same", "pad_resize", "pad_fit"])
@pytest.mark.parametrize("size", [224, 300])
def test_img_pad_parity(rgb_image, small_image, mode, size):
    for src in (rgb_image, small_image):
        gold = _gold_img_pad(src, mode=mode, size=size)
        got = tf_new.img_pad(src, mode=mode, size=size)
        assert gold.size == got.size, f"size mismatch mode={mode}"
        assert np.array_equal(np.array(gold), np.array(got)), f"pixel mismatch mode={mode}"


def test_img_pad_invalid_mode(rgb_image):
    with pytest.raises(AssertionError):
        tf_new.img_pad(rgb_image, mode="bogus", size=224)


# ---------------------------------------------------------------------------
# squarify parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bbox",
    [
        [100.0, 50.0, 200.0, 300.0],  # typical
        [0.0, 0.0, 10.0, 20.0],  # flush left
        [480.0, 10.0, 500.0, 100.0],  # flush right
    ],
)
def test_squarify_parity(bbox):
    gold = _gold_squarify(list(bbox), 0.5, 500)
    got = tf_new.squarify(list(bbox), 0.5, 500)
    assert gold == got


# ---------------------------------------------------------------------------
# bbox_sanity_check parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bbox",
    [
        [-5.0, -1.0, 700.0, 500.0],  # all four clamped
        [10.0, 10.0, 100.0, 200.0],  # clean
        [-0.5, 0.0, 499.9999, 299.0],  # edge values
    ],
)
def test_sanity_parity(rgb_image, bbox):
    gold = _gold_sanity(rgb_image, list(bbox))
    got = tf_new.bbox_sanity_check(rgb_image, list(bbox))
    assert gold == got


# ---------------------------------------------------------------------------
# jitter_bbox parity (deterministic modes only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["same", "enlarge", "move"])
@pytest.mark.parametrize("ratio", [-0.3, 0.5, 1.0])
def test_jitter_parity_deterministic(tmp_img_path, mode, ratio):
    src = [[100.0, 80.0, 200.0, 180.0], [10.0, 10.0, 50.0, 60.0]]
    gold = _gold_jitter(tmp_img_path, [list(b) for b in src], mode, ratio)
    got = tf_new.jitter_bbox(tmp_img_path, [list(b) for b in src], mode, ratio)
    assert gold == got


def test_jitter_random_reproducible(tmp_img_path):
    src = [[100.0, 80.0, 200.0, 180.0]]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1 = tf_new.jitter_bbox(tmp_img_path, [list(src[0])], "random_enlarge", 0.5, rng=rng1)
    out2 = tf_new.jitter_bbox(tmp_img_path, [list(src[0])], "random_enlarge", 0.5, rng=rng2)
    assert out1 == out2


def test_jitter_invalid_mode(tmp_img_path):
    with pytest.raises(AssertionError):
        tf_new.jitter_bbox(tmp_img_path, [[0.0, 0.0, 10.0, 10.0]], "bogus", 0.1)


# ---------------------------------------------------------------------------
# Smoke test: vendored PIE dataset API is importable
# ---------------------------------------------------------------------------
def test_pie_data_importable():
    from pie_pytorch.data.pie_data import PIE

    assert callable(PIE)
