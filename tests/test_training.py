"""Tests for pie_pytorch.training: metrics, callbacks, checkpoint, trainer."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pie_pytorch.training.callbacks import EarlyStopping, ReduceLROnPlateau
from pie_pytorch.training.checkpoint import TopKCheckpointManager
from pie_pytorch.training.metrics import (
    MetricsAccumulator,
    accuracy,
    center_mse,
    f1,
    mse,
)
from pie_pytorch.training.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------
def test_accuracy_exact():
    y_pred = torch.tensor([0.9, 0.6, 0.2, 0.1])
    y_true = torch.tensor([1.0, 0.0, 0.0, 1.0])
    assert accuracy(y_pred, y_true) == 0.5


def test_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(torch.zeros(3), torch.zeros(4))


def test_f1_perfect():
    y_pred = torch.tensor([0.9, 0.8, 0.1, 0.2])
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert f1(y_pred, y_true) == pytest.approx(1.0, abs=1e-5)


def test_f1_all_wrong():
    y_pred = torch.tensor([0.9, 0.9])
    y_true = torch.tensor([0.0, 0.0])
    # No true positives, no false negatives -> F1 = 0
    assert f1(y_pred, y_true) == 0.0


def test_mse_matches_torch_functional():
    a = torch.randn(3, 5)
    b = torch.randn(3, 5)
    assert mse(a, b) == pytest.approx(torch.nn.functional.mse_loss(a, b).item())


def test_center_mse_zero_when_equal():
    bbox = torch.tensor([[100.0, 50.0, 200.0, 150.0]])
    assert center_mse(bbox, bbox) == 0.0


def test_center_mse_last_dim_must_be_4():
    with pytest.raises(ValueError):
        center_mse(torch.zeros(2, 5), torch.zeros(2, 5))


def test_metrics_accumulator_aggregates_across_batches():
    acc = MetricsAccumulator()
    # Two batches that together have 2/4 correct -> acc=0.5
    acc.update(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 1.0]))
    acc.update(torch.tensor([0.8, 0.3]), torch.tensor([0.0, 0.0]))
    out = acc.compute({"acc": accuracy})
    assert out["acc"] == 0.5
    acc.reset()
    assert math.isnan(acc.compute({"acc": accuracy})["acc"])


# ---------------------------------------------------------------------------
# callbacks.py
# ---------------------------------------------------------------------------
def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(monitor="val/loss", patience=2, mode="min")
    # First value is new best; then two non-improving -> stop.
    for loss in (1.0, 1.0, 1.0):
        es.update({"val/loss": loss})
    assert es.should_stop


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(monitor="val/loss", patience=2, mode="min")
    es.update({"val/loss": 1.0})
    es.update({"val/loss": 1.0})  # bad 1
    es.update({"val/loss": 0.5})  # improve -> reset
    es.update({"val/loss": 0.6})  # bad 1
    assert not es.should_stop


def test_early_stopping_missing_key_raises():
    es = EarlyStopping()
    with pytest.raises(KeyError):
        es.update({"other": 1.0})


def test_reduce_lr_on_plateau_halves_after_patience():
    model = nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, monitor="val/loss", factor=0.5, patience=2, min_lr=1e-8)

    sched.update({"val/loss": 1.0})           # new best
    sched.update({"val/loss": 1.0})           # bad 1
    new = sched.update({"val/loss": 1.0})     # bad 2 -> trigger
    assert new == pytest.approx(5e-4)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_reduce_lr_respects_min_lr():
    model = nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=2e-7)
    sched = ReduceLROnPlateau(opt, monitor="val/loss", factor=0.5, patience=1, min_lr=1e-7)
    sched.update({"val/loss": 1.0})
    sched.update({"val/loss": 1.0})
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-7)


def test_callback_state_roundtrip():
    es = EarlyStopping(monitor="val/loss", patience=2)
    es.update({"val/loss": 1.0})
    es.update({"val/loss": 1.0})
    state = es.state_dict()
    es2 = EarlyStopping(monitor="val/loss", patience=2)
    es2.load_state_dict(state)
    assert es2.bad_epochs == es.bad_epochs
    assert es2.best == es.best


# ---------------------------------------------------------------------------
# checkpoint.py
# ---------------------------------------------------------------------------
def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))


def test_topk_save_prunes_worst(tmp_path):
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    mgr = TopKCheckpointManager(tmp_path, monitor="val/loss", mode="min", k=2)

    # Three epochs, losses 0.5, 0.4, 0.6. Keep best 2 -> 0.4 and 0.5.
    for ep, loss in enumerate([0.5, 0.4, 0.6], start=1):
        mgr.save(epoch=ep, metrics={"val/loss": loss}, model=model, optimizer=opt)

    kept_epochs = sorted(r.epoch for r in mgr.records)
    assert kept_epochs == [1, 2]
    # Epoch 3 directory deleted.
    assert not (tmp_path / "epoch_0003").exists()


def test_topk_load_restores_weights(tmp_path):
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    mgr = TopKCheckpointManager(tmp_path, monitor="val/loss", k=1)

    # Save, then perturb, then reload.
    mgr.save(epoch=1, metrics={"val/loss": 0.1}, model=model, optimizer=opt)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    # Reload from the best checkpoint.
    ep_dir = mgr.records[0].path
    mgr.load(ep_dir, model=model)
    restored = model.state_dict()
    for k in original:
        assert torch.allclose(restored[k], original[k])


def test_topk_best_latest_pointers(tmp_path):
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    mgr = TopKCheckpointManager(tmp_path, monitor="val/loss", mode="min", k=3)

    for ep, loss in enumerate([0.8, 0.3, 0.5], start=1):
        mgr.save(epoch=ep, metrics={"val/loss": loss}, model=model, optimizer=opt)

    # Pointers may be symlinks or .txt fallback (Drive-safe). Either way,
    # best should resolve/pointer to epoch_0002, latest to epoch_0003.
    best_path = _read_pointer(tmp_path, "best")
    latest_path = _read_pointer(tmp_path, "latest")
    assert best_path == "epoch_0002"
    assert latest_path == "epoch_0003"


def _read_pointer(root: Path, name: str) -> str:
    p = root / name
    if p.is_symlink():
        return p.readlink().name
    if p.exists() and p.is_dir():
        return p.name
    txt = p.with_suffix(".txt")
    if txt.exists():
        return txt.read_text().strip()
    raise FileNotFoundError(f"no pointer for {name}")


# ---------------------------------------------------------------------------
# trainer.py end-to-end on a tiny regression task
# ---------------------------------------------------------------------------
class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _toy_loaders(batch: int = 8, n_train: int = 64, n_val: int = 16):
    torch.manual_seed(0)
    xs = torch.randn(n_train, 4)
    w = torch.tensor([[1.0, -2.0, 0.5, 0.1]])
    ys = xs @ w.t() + 0.01 * torch.randn(n_train, 1)
    xv = torch.randn(n_val, 4)
    yv = xv @ w.t() + 0.01 * torch.randn(n_val, 1)

    class D(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return {"x": self.x[i], "y": self.y[i]}

    return (
        DataLoader(D(xs, ys), batch_size=batch, shuffle=True),
        DataLoader(D(xv, yv), batch_size=batch),
    )


def test_trainer_overfits_tiny_regression(tmp_path):
    torch.manual_seed(0)  # seed BEFORE model init so weights are deterministic
    model = _ToyNet()

    def forward_fn(m, batch):
        return m(batch["x"]), batch["y"]

    def loss_fn(pred, tgt, batch):
        return torch.nn.functional.mse_loss(pred, tgt)

    cfg = TrainerConfig(
        epochs=60,
        lr=1e-2,
        clip_grad_norm=None,
        use_amp=False,
        device="cpu",
        monitor="val_loss",
        mode="min",
        plateau_patience=100,    # disable for this test
        early_stop_patience=100,
        log_every_n_steps=1000,
        checkpoint_dir=str(tmp_path / "ckpt"),
        keep_top_k=2,
        wandb=False,
    )
    trainer = Trainer(
        model,
        loss_fn=loss_fn,
        metrics={"mse": mse},
        forward_fn=forward_fn,
        cfg=cfg,
    )
    train_loader, val_loader = _toy_loaders()
    out = trainer.fit(train_loader, val_loader)
    # After 60 epochs of linear regression we should be near-zero MSE.
    assert out["best"] < 0.05, f"expected tiny val loss after 60 epochs, got {out['best']}"
    # Top-K checkpointing wrote exactly keep_top_k dirs.
    ckpt_dirs = sorted((tmp_path / "ckpt").glob("epoch_*"))
    assert len(ckpt_dirs) == 2
    assert all((d / "model.safetensors").exists() for d in ckpt_dirs)


def test_trainer_early_stopping_actually_halts():
    model = _ToyNet()

    def forward_fn(m, batch):
        return m(batch["x"]), batch["y"]

    def loss_fn(pred, tgt, batch):
        return torch.nn.functional.mse_loss(pred, tgt)

    cfg = TrainerConfig(
        epochs=100,
        lr=0.0,                     # no learning -> loss never improves
        clip_grad_norm=None,
        use_amp=False,
        device="cpu",
        early_stop_patience=3,
        plateau_patience=1000,
        wandb=False,
    )
    trainer = Trainer(
        model,
        loss_fn=loss_fn,
        metrics={"mse": mse},
        forward_fn=forward_fn,
        cfg=cfg,
    )
    train_loader, val_loader = _toy_loaders()
    out = trainer.fit(train_loader, val_loader)
    # Stopped well before 100 epochs.
    assert len(out["history"]) < 10
