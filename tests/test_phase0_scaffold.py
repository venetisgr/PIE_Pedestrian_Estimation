"""Phase 0 validation tests.

These run on any machine with torch installed - no dataset or GPU
required. They prove the package skeleton is wired correctly and
that the environment has the expected deps.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest


def test_pie_pytorch_imports():
    mod = importlib.import_module("pie_pytorch")
    assert hasattr(mod, "__version__")


@pytest.mark.parametrize(
    "subpkg",
    [
        "pie_pytorch.data",
        "pie_pytorch.data.transforms",
        "pie_pytorch.data.subset",
        "pie_pytorch.data.pie_dataset",
        "pie_pytorch.features",
        "pie_pytorch.features.vgg16_features",
        "pie_pytorch.models",
        "pie_pytorch.models.layers",
        "pie_pytorch.models.intent",
        "pie_pytorch.models.trajectory",
        "pie_pytorch.models.speed",
        "pie_pytorch.training",
        "pie_pytorch.training.trainer",
        "pie_pytorch.training.checkpoint",
        "pie_pytorch.training.metrics",
        "pie_pytorch.training.callbacks",
        "pie_pytorch.cli",
        "pie_pytorch.cli.train",
        "pie_pytorch.cli.eval",
    ],
)
def test_subpackage_imports(subpkg):
    importlib.import_module(subpkg)


def test_no_tensorflow_or_keras_in_pie_pytorch():
    """PyTorch package must not *import* TF/Keras anywhere.

    Looks at the actual module objects bound in each submodule's globals
    (not docstrings - mentioning Keras in prose is fine for context).
    """
    import types

    import pie_pytorch

    forbidden_tops = {"tensorflow", "keras"}
    for _, name, _ in pkgutil.walk_packages(pie_pytorch.__path__, prefix="pie_pytorch."):
        mod = importlib.import_module(name)
        for attr_name, attr_val in vars(mod).items():
            if isinstance(attr_val, types.ModuleType):
                top = attr_val.__name__.split(".")[0]
                assert top not in forbidden_tops, (
                    f"{name} imports forbidden module {attr_val.__name__} "
                    f"as {attr_name!r}"
                )


def test_torch_available():
    torch = importlib.import_module("torch")
    assert torch.__version__
    # Do NOT require cuda here - Phase 0 must pass on CPU-only boxes
    # (Colab free tier sometimes gives no GPU).


def test_subset_config_defaults():
    from pie_pytorch.data.subset import SubsetConfig

    cfg = SubsetConfig()
    assert cfg.is_noop() is True

    cfg2 = SubsetConfig(fraction=0.1)
    assert cfg2.is_noop() is False
