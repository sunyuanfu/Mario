"""Evaluation and matching utilities for Stage-2 Mario LLM training."""

from __future__ import annotations

import difflib
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .constants import TEMPLATES
from .model import MarioLLM


def _best_edit_label(text: str, label_names: Sequence[str]) -> Tuple[str, float, float]:
    text_lower = text.lower().strip()
    scored = []
    for name in label_names:
        ratio = difflib.SequenceMatcher(None, text_lower, name.lower()).ratio()
        scored.append((ratio, name))
    if not scored:
        return "", 0.0, 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_label = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    return best_label, best_ratio, second_ratio


def _match_label(pred_text: str, label_names: Sequence[str]) -> str:
    """Map free-form LLM output to the closest dataset label."""
    text = (pred_text or "").strip()
    if not text:
        return ""
    lowered = text.lower().strip()
    for prefix in ("answer:", "category:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            lowered = text.lower().strip()
            break

    first_line = text.splitlines()[0].strip()
    if not first_line:
        return ""

    for name in label_names:
        if first_line.lower() == name.lower():
            return name

    best_label = ""
    best_len = -1
    for name in label_names:
        name_lower = name.lower()
        if name_lower and name_lower in lowered and len(name_lower) > best_len:
            best_label = name
            best_len = len(name_lower)
    if best_label:
        return best_label

    edit_label, edit_score, edit_second = _best_edit_label(first_line, label_names)
    if edit_label and edit_score >= 0.65 and (edit_score - edit_second) >= 0.05:
        return edit_label

    return first_line


def iter_nodes(indices: torch.Tensor, shuffle: bool, seed: int) -> List[int]:
    nodes = indices.cpu().tolist()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(nodes)
    return nodes


def shard_nodes_for_rank(
    nodes: List[int], rank: int, world_size: int, batch_size: int
) -> List[int]:
    if world_size <= 1:
        return nodes
    if not nodes:
        return []
    global_batch = batch_size * world_size
    remainder = len(nodes) % global_batch
    if remainder:
        pad_count = global_batch - remainder
        repeats = (pad_count + len(nodes) - 1) // len(nodes)
        nodes = nodes + (nodes * repeats)[:pad_count]
    return nodes[rank::world_size]


def evaluate_loss(model: MarioLLM, indices: torch.Tensor, limit: int) -> float:
    nodes = indices.cpu().tolist()
    if limit > 0:
        nodes = nodes[:limit]
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for start in tqdm(range(0, len(nodes), model.args.batch_size), desc="val", leave=False):
            batch = [int(node) for node in nodes[start : start + model.args.batch_size]]
            loss, _ = model.forward_nodes(batch)
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return float(np.mean(losses)) if losses else float("inf")


def _select_vote3_prediction(
    template_texts: Dict[str, str],
    router_probs: Dict[str, float],
    label_names: Sequence[str],
) -> Tuple[str, str, str]:
    template_labels = {
        template: _match_label(text, label_names)
        for template, text in template_texts.items()
    }
    votes: Dict[str, int] = {}
    for template, label in template_labels.items():
        if not label:
            continue
        votes[label] = votes.get(label, 0) + 1

    if votes:
        best_vote = max(votes.values())
        tied_labels = [label for label, count in votes.items() if count == best_vote]
        if len(tied_labels) == 1:
            chosen_label = tied_labels[0]
        else:
            chosen_label = max(
                tied_labels,
                key=lambda label: max(
                    router_probs.get(template, 0.0)
                    for template, value in template_labels.items()
                    if value == label
                ),
            )
        chosen_template = max(
            (template for template, value in template_labels.items() if value == chosen_label),
            key=lambda template: router_probs.get(template, 0.0),
        )
        return template_texts[chosen_template], chosen_label, chosen_template

    fallback_template = max(router_probs.items(), key=lambda item: item[1])[0]
    fallback_text = template_texts[fallback_template]
    fallback_label = _match_label(fallback_text, label_names)
    return fallback_text, fallback_label, fallback_template


def evaluate_generation(
    model: MarioLLM,
    indices: torch.Tensor,
    limit: int,
    inference_policy: str,
) -> Tuple[float, List[Dict[str, str]]]:
    nodes = indices.cpu().tolist()
    if limit > 0:
        nodes = nodes[:limit]
    records = []
    correct = 0
    for node in tqdm(nodes, desc="test", leave=False):
        if inference_policy == "vote3":
            template_texts, router_probs = model.predict_node_all_templates(int(node))
            pred_text, pred_label, selected_template = _select_vote3_prediction(
                template_texts, router_probs, model.label_names
            )
        else:
            pred_text = model.predict_node(int(node), policy=inference_policy)
            pred_label = _match_label(pred_text, model.label_names)
            selected_template = inference_policy
        label = model.label_names[int(model.labels[int(node)].item())]
        if pred_label.lower() == label.lower():
            correct += 1
        records.append(
            {
                "node": str(int(node)),
                "pred_text": pred_text,
                "pred_label": pred_label,
                "label": label,
                "inference_policy": inference_policy,
                "selected_template": selected_template,
            }
        )
    accuracy = correct / max(len(nodes), 1)
    return accuracy, records
