"""Distributed and reproducibility utilities for Stage-2 training."""

from __future__ import annotations

import argparse
import os
import random
from datetime import timedelta
from typing import Tuple

import dgl
import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int) -> None:
    """Seed random sources used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Initialize DDP runtime and device mapping."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA devices")
        torch.cuda.set_device(local_rank)
        args.device = "cuda:{}".format(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    elif args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    return world_size, rank, local_rank


def is_main_process(rank: int) -> bool:
    """Whether the current process is rank 0."""
    return rank == 0


def cleanup_distributed(world_size: int) -> None:
    """Close DDP process groups when used."""
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def distributed_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce a scalar tensor by mean across all ranks."""
    if world_size > 1:
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(world_size)
    return value
