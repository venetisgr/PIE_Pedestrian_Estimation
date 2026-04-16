"""Device-agnostic training loop (Phase 3.1).

Wires together: RMSprop (Keras-matched alpha=0.9, eps=1e-7), gradient
clipping, torch.amp GradScaler, ReduceLROnPlateau, early stopping, W&B.
"""

from __future__ import annotations
