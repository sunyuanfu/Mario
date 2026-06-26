"""Checkpoint I/O helpers for Stage-2 training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from .model import MarioLLM


def save_checkpoint(model: MarioLLM, output_dir: Path, epoch: int, val_loss: float) -> None:
    """Persist best Stage-2 state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "router": model.router.state_dict(),
        "projector": model.projector.state_dict(),
        "llm_lora": get_peft_model_state_dict(model.llm),
        "args": vars(model.args),
    }
    torch.save(state, output_dir / "best_stage2_llm.pt")


def load_checkpoint(model: MarioLLM, checkpoint_path: str) -> Tuple[int, float]:
    """Load Stage-2 checkpoint state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.router.load_state_dict(checkpoint["router"])
    model.projector.load_state_dict(checkpoint["projector"])
    set_peft_model_state_dict(model.llm, checkpoint["llm_lora"])
    return int(checkpoint.get("epoch", -1)), float(checkpoint.get("val_loss", float("inf")))
