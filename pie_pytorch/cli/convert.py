"""Convert a Keras .h5 paper checkpoint into a PyTorch safetensors file.

Usage:
    python -m pie_pytorch.cli.convert \
        --task intent \
        --h5 data/pie/intention/context_loc_pretrained/model.h5 \
        --out data/pie/intention/context_loc_pretrained/model.safetensors

Output is a single safetensors file next to (or replacing) the .h5 that
downstream code can load with ``safetensors.torch.load_file``.

The converter also writes a tiny ``config.json`` beside the safetensors
so ``eval.py`` can rebuild the correctly-sized model without re-inspecting
the .h5.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import save_file

from ..io.keras_to_torch import load_attn_encdec, load_intent


TASKS = ("intent", "trajectory", "speed")


def convert(h5_path: str, out_path: str, task: str) -> Path:
    if task == "intent":
        model, cfg = load_intent(h5_path)
    elif task in ("trajectory", "speed"):
        model, cfg = load_attn_encdec(h5_path, task=task)
    else:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(out))

    cfg_path = out.with_suffix(".config.json")
    payload = {"task": task, "config": asdict(cfg)}
    cfg_path.write_text(json.dumps(payload, indent=2))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pie-convert")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--h5", required=True, help="Path to the Keras .h5 checkpoint.")
    p.add_argument("--out", required=True, help="Output .safetensors path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        out = convert(args.h5, args.out, args.task)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {out} (+ {out.with_suffix('.config.json').name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
