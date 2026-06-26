"""CLI definitions for Stage-2 Mario LLM training."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Stage-2 training."""
    parser = argparse.ArgumentParser(
        description="Mario Modality-Adaptive LLaMA LoRA Stage 2 training"
    )
    parser.add_argument("--dataset", type=str, default="Movies")
    parser.add_argument("--data-root", type=str, default="../data/MAGB")
    parser.add_argument(
        "--stage1-feature", type=str, default="../stage1/Movies_stage1_mlp.pth"
    )
    parser.add_argument(
        "--llm-model-path",
        type=str,
        default="/storage/zhangx_data/model_weights/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--output-dir", type=str, default="runs/mario_llm_movies")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--limit-train-steps", type=int, default=0)
    parser.add_argument("--limit-val-steps", type=int, default=0)
    parser.add_argument("--limit-test-steps", type=int, default=0)
    parser.add_argument(
        "--inference-policy",
        type=str,
        default="router",
        choices=["router", "text", "image", "mm", "vote3"],
        help="Policy used during test-time generation.",
    )
    parser.add_argument("--skip-test-generation", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()
