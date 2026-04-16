"""torch.utils.data.Dataset wrappers around the PIE data API.

Stub. Phase 1.3 implements:
    - PIEIntentDataset (context crop + bbox seq -> binary intent)
    - PIETrajectoryDataset (bbox obs -> future bbox deltas)
    - PIESpeedDataset (obd_speed obs -> future obd_speed)
See plan.md and notepad.md D3 (first-frame normalization quirk).
"""

from __future__ import annotations
