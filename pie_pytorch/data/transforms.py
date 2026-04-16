"""Image and bbox transforms ported from legacy_tf/utils.py.

These helpers are pure PIL + numpy - no deep-learning framework is
involved. The port mirrors the original behavior function-for-function
so the PyTorch pipeline produces byte-identical crops to the Keras
pipeline (except where the original depended on keras image loaders,
which we replace with PIL).

Parity is enforced by tests/test_transforms_parity.py.

See notepad.md -> D6 (integer-division quirk in jitter_bbox) and
D7 (PIL.Image.NEAREST vs bilinear) for gotchas that motivated the
exact implementation below.
"""

from __future__ import annotations

import sys

import numpy as np
import PIL
import PIL.Image


__all__ = [
    "update_progress",
    "img_pad",
    "squarify",
    "bbox_sanity_check",
    "jitter_bbox",
    "load_img",
]


# ---------------------------------------------------------------------------
# Image loading (framework-agnostic replacement for keras load_img)
# ---------------------------------------------------------------------------
def load_img(path: str) -> PIL.Image.Image:
    """Load an image as PIL.Image in RGB mode. Mirrors keras.preprocessing."""
    img = PIL.Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Progress bar (stdout) - kept for drop-in compatibility with legacy calls
# ---------------------------------------------------------------------------
def update_progress(progress: float) -> None:
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    block = int(round(bar_length * progress))
    text = "\r[{}] {:0.2f}% {}".format(
        "#" * block + "-" * (bar_length - block), progress * 100, ""
    )
    sys.stdout.write(text)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Image padding / resizing
# ---------------------------------------------------------------------------
def img_pad(
    img: PIL.Image.Image,
    mode: str = "warp",
    size: int = 224,
) -> PIL.Image.Image:
    """Pad/resize ``img`` per ``mode``. Matches legacy_tf/utils.py::img_pad.

    Modes:
        warp       - resize directly to (size, size) using NEAREST.
        same       - no-op, return a copy.
        pad_same   - keep original size, centered on black (size, size) canvas.
        pad_resize - scale the long edge to ``size`` preserving aspect ratio,
                     pad the rest with black.
        pad_fit    - like pad_same unless the image exceeds ``size``, in which
                     case behave like pad_resize.
    """
    assert mode in ("same", "warp", "pad_same", "pad_resize", "pad_fit"), (
        f"Pad mode {mode!r} is invalid"
    )
    image = img.copy()
    if mode == "warp":
        return image.resize((size, size), PIL.Image.NEAREST)
    if mode == "same":
        return image

    img_size = image.size  # (width, height)
    ratio = float(size) / max(img_size)
    if mode == "pad_resize" or (
        mode == "pad_fit" and (img_size[0] > size or img_size[1] > size)
    ):
        img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))
        image = image.resize(img_size, PIL.Image.NEAREST)
    padded = PIL.Image.new("RGB", (size, size))
    padded.paste(image, ((size - img_size[0]) // 2, (size - img_size[1]) // 2))
    return padded


# ---------------------------------------------------------------------------
# Bounding box geometry
# ---------------------------------------------------------------------------
def squarify(bbox: list[float], squarify_ratio: float, img_width: int) -> list[float]:
    """Reshape bbox horizontally toward a target aspect ratio.

    Port of legacy_tf/utils.py::squarify. Note: mutates ``bbox`` in place
    (we keep this to preserve byte-level parity with the original).
    """
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


def bbox_sanity_check(img: PIL.Image.Image, bbox: list[float]) -> list[float]:
    """Clamp bbox to image boundaries. Mirrors legacy_tf/utils.py."""
    img_width, img_height = img.size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_height:
        bbox[3] = img_height - 1
    return bbox


def jitter_bbox(
    img_path: str,
    bbox: list[list[float]],
    mode: str,
    ratio: float,
    rng: np.random.Generator | None = None,
) -> list[list[float]]:
    """Jitter or enlarge a list of bboxes.

    Port of legacy_tf/utils.py::jitter_bbox. Two behavioral improvements
    over the original, both additive:

      1. Accepts an optional np.random.Generator ``rng`` for reproducibility
         (the TF original used global np.random state, which made unit
         tests flaky).
      2. Uses our PIL-based load_img, not keras's.

    Output matches the original byte-for-byte when ``mode in {'same',
    'enlarge', 'move'}`` (deterministic modes). For 'random_enlarge' and
    'random_move' parity requires seeding the same Generator.

    See notepad.md D6 for the "//" (integer division) quirk in the
    original that we preserve intentionally.
    """
    assert mode in ("same", "enlarge", "move", "random_enlarge", "random_move"), (
        f"mode {mode!r} is invalid."
    )

    if mode == "same":
        return bbox

    img = load_img(img_path)
    img_width, img_height = img.size  # kept for parity (width used below)
    del img_height  # not used; keep variable to match original readability

    if mode in ("random_enlarge", "enlarge"):
        jitter_ratio = abs(ratio)
    else:
        jitter_ratio = ratio

    if rng is None:
        rng = np.random.default_rng()

    if mode == "random_enlarge":
        jitter_ratio = rng.random() * jitter_ratio
    elif mode == "random_move":
        jitter_ratio = rng.random() * jitter_ratio * 2 - jitter_ratio

    jit_boxes: list[list[float]] = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]
        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio
        # Preserve the original's "take the smaller of the two" behavior.
        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change
        if mode in ("enlarge", "random_enlarge"):
            b[0] = b[0] - width_change // 2
            b[1] = b[1] - height_change // 2
        else:
            b[0] = b[0] + width_change // 2
            b[1] = b[1] + height_change // 2
        b[2] = b[2] + width_change // 2
        b[3] = b[3] + height_change // 2
        b = bbox_sanity_check(img, b)
        jit_boxes.append(b)
    return jit_boxes
