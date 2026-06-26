"""Training pipeline orchestration for Stage-2 Mario LLM training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .data import Stage2Data, load_movies
from .distributed import (
    cleanup_distributed,
    distributed_mean,
    is_main_process,
    seed_everything,
    setup_distributed,
)
from .evaluation import evaluate_generation, evaluate_loss, iter_nodes, shard_nodes_for_rank
from .model import MarioLLM


@dataclass
class TrainingState:
    """Mutable state tracked through the epoch loop."""

    start_epoch: int = 0
    best_val: float = float("inf")
    best_epoch: int = -1
    stale: int = 0
    best_checkpoint_path: Optional[Path] = None


def _initialize_runtime(args: argparse.Namespace) -> Tuple[int, int, int, Path]:
    world_size, rank, local_rank = setup_distributed(args)
    seed_everything(args.seed + rank)
    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    return world_size, rank, local_rank, output_dir


def _build_model(data: Stage2Data, args: argparse.Namespace) -> MarioLLM:
    return MarioLLM(
        args=args,
        graph=data.graph,
        texts=data.texts,
        labels=data.labels,
        label_names=data.label_names,
        train_idx=data.train_idx,
        features=data.features,
    )


def _create_optimizer(model: MarioLLM, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )


def _maybe_resume(model: MarioLLM, args: argparse.Namespace, rank: int) -> TrainingState:
    state = TrainingState()
    if args.resume_checkpoint:
        resume_epoch, resume_val = load_checkpoint(model, args.resume_checkpoint)
        state.start_epoch = resume_epoch + 1
        state.best_epoch = resume_epoch
        state.best_val = resume_val
        state.best_checkpoint_path = Path(args.resume_checkpoint)
        if is_main_process(rank):
            print(
                "resumed checkpoint {} at epoch {} with val_loss {:.6f}".format(
                    args.resume_checkpoint, resume_epoch, resume_val
                )
            )
    return state


def _train_one_epoch(
    epoch: int,
    model: MarioLLM,
    train_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Stage2Data,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> float:
    model.train()
    nodes = iter_nodes(data.train_idx, shuffle=True, seed=args.seed + epoch)
    if args.limit_train_steps > 0:
        nodes = nodes[: args.limit_train_steps * args.batch_size * max(world_size, 1)]
    nodes = shard_nodes_for_rank(nodes, rank, world_size, args.batch_size)

    running = []
    step_range = range(0, len(nodes), args.batch_size)
    if is_main_process(rank):
        step_range = tqdm(step_range, desc="epoch {}/{}".format(epoch + 1, args.epochs))
    for step, start in enumerate(step_range, start=1):
        batch_nodes = [int(node) for node in nodes[start : start + args.batch_size]]
        center_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=model.device)
        optimizer.zero_grad(set_to_none=True)
        loss = train_model(center_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 0.1
        )
        optimizer.step()
        reduced_loss = distributed_mean(loss.detach().float(), world_size)
        if is_main_process(rank):
            info = model.last_info
            running.append(float(reduced_loss.cpu()))
            if step % args.log_every == 0 and hasattr(step_range, "set_postfix"):
                step_range.set_postfix(
                    loss="{:.4f}".format(float(np.mean(running[-args.log_every :]))),
                    rt="{:.2f}".format(info.get("router_text", 0.0)),
                    ri="{:.2f}".format(info.get("router_image", 0.0)),
                    rm="{:.2f}".format(info.get("router_mm", 0.0)),
                )
    return float(np.mean(running)) if running else float("inf")


def _validate_and_checkpoint(
    epoch: int,
    model: MarioLLM,
    data: Stage2Data,
    args: argparse.Namespace,
    output_dir: Path,
    state: TrainingState,
    train_loss: float,
) -> Tuple[float, Dict[str, float], bool]:
    val_loss = float("inf")
    if (epoch + 1) % args.val_every == 0:
        val_loss = evaluate_loss(model, data.val_idx, args.limit_val_steps)
    record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
    history_record = record.copy()
    with open(output_dir / "history.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_record, sort_keys=True) + "\n")
    print(json.dumps(history_record, sort_keys=True))
    should_stop = False
    if val_loss < state.best_val:
        state.best_val = val_loss
        state.best_epoch = epoch
        state.stale = 0
        save_checkpoint(model, output_dir, epoch, val_loss)
        state.best_checkpoint_path = output_dir / "best_stage2_llm.pt"
    else:
        state.stale += 1
        if state.stale >= args.patience:
            print("early stop at epoch {}".format(epoch))
            should_stop = True
    return val_loss, history_record, should_stop


def _write_metrics_and_test(
    model: MarioLLM,
    data: Stage2Data,
    args: argparse.Namespace,
    output_dir: Path,
    state: TrainingState,
    world_size: int,
) -> Dict[str, float]:
    metrics = {
        "dataset": args.dataset,
        "best_epoch": state.best_epoch,
        "best_val_loss": state.best_val,
        "inference_policy": args.inference_policy,
        "train_nodes": int(data.train_idx.numel()),
        "val_nodes": int(data.val_idx.numel()),
        "test_nodes": int(data.test_idx.numel()),
        "world_size": world_size,
    }
    if state.best_checkpoint_path is not None and state.best_checkpoint_path.exists():
        load_checkpoint(model, str(state.best_checkpoint_path))
        print("loaded best checkpoint for evaluation: {}".format(state.best_checkpoint_path))
    if not args.skip_test_generation:
        test_acc, records = evaluate_generation(
            model,
            data.test_idx,
            args.limit_test_steps,
            args.inference_policy,
        )
        metrics["test_acc"] = test_acc
        with open(output_dir / "test_predictions.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def run_stage2(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    """Execute full Stage-2 training/evaluation with original behavior."""
    world_size, rank, local_rank, output_dir = _initialize_runtime(args)
    data = load_movies(args)
    model = _build_model(data, args)
    trainable, total = model.trainable_parameters()
    if is_main_process(rank):
        print("distributed world size: {} local rank: {}".format(world_size, local_rank))
        print(
            "trainable params: {} || all params: {} || trainable%: {:.6f}".format(
                trainable, total, 100.0 * trainable / total
            )
        )

    state = _maybe_resume(model, args, rank)
    train_model: torch.nn.Module = model
    if world_size > 1:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = _create_optimizer(model, args)

    for epoch in range(state.start_epoch, args.epochs):
        train_loss = _train_one_epoch(
            epoch=epoch,
            model=model,
            train_model=train_model,
            optimizer=optimizer,
            data=data,
            args=args,
            rank=rank,
            world_size=world_size,
        )

        stop_flag = torch.tensor([0], dtype=torch.long, device=model.device)
        if is_main_process(rank):
            _, _, should_stop = _validate_and_checkpoint(
                epoch=epoch,
                model=model,
                data=data,
                args=args,
                output_dir=output_dir,
                state=state,
                train_loss=train_loss,
            )
            if should_stop:
                stop_flag.fill_(1)

        if world_size > 1:
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            break

    if world_size > 1:
        dist.barrier()
    if not is_main_process(rank):
        cleanup_distributed(world_size)
        return None

    metrics = _write_metrics_and_test(
        model=model,
        data=data,
        args=args,
        output_dir=output_dir,
        state=state,
        world_size=world_size,
    )
    cleanup_distributed(world_size)
    return metrics
