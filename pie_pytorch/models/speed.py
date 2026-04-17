"""Speed model: thin re-export of the shared AttnEncDec.

The speed network shares its architecture with the trajectory network
(see ``trajectory.py``); only the input/output dimensions differ. The
``speed_model`` factory wires up the paper defaults.
"""

from __future__ import annotations

from .trajectory import AttnEncDec, AttnEncDecConfig, speed_model

__all__ = ["AttnEncDec", "AttnEncDecConfig", "speed_model"]
