"""Data loading and graph preprocessing for Stage-2 training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Stage2Data:
    """Typed container for Stage-2 dataset artifacts."""

    graph: dgl.DGLGraph
    texts: List[str]
    labels: torch.Tensor
    label_names: List[str]
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    features: torch.Tensor

    def as_dict(self) -> Dict[str, object]:
        """Compatibility view used by existing training loops."""
        return {
            "graph": self.graph,
            "texts": self.texts,
            "labels": self.labels,
            "label_names": self.label_names,
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "features": self.features,
        }


def to_bidirected_with_self_loops(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Normalize graph edges to bidirected with explicit self loops."""
    graph = dgl.remove_self_loop(graph.cpu())
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


def split_indices(
    num_nodes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split nodes into train/val/test with 60/20/20 ratios."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_nodes)
    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    train = torch.tensor(indices[:train_size], dtype=torch.long)
    val = torch.tensor(indices[train_size : train_size + val_size], dtype=torch.long)
    test = torch.tensor(indices[train_size + val_size :], dtype=torch.long)
    return train, val, test


def load_movies(args: argparse.Namespace) -> Stage2Data:
    """Load Movies-style inputs used by the original Stage-2 script."""
    dataset_dir = Path(args.data_root) / args.dataset
    csv_path = dataset_dir / "{}.csv".format(args.dataset)
    graph_path = dataset_dir / "{}Graph.pt".format(args.dataset)
    graphs, _ = dgl.load_graphs(str(graph_path))
    graph = to_bidirected_with_self_loops(graphs[0])
    frame = pd.read_csv(csv_path)
    labels = graph.ndata["label"].long()
    label_names: List[str] = []
    for label_id in range(int(labels.max().item()) + 1):
        rows = frame.loc[frame["label"] == label_id, "second_category"]
        label_names.append(str(rows.iloc[0]))
    train_idx, val_idx, test_idx = split_indices(graph.num_nodes(), args.seed)
    features = torch.load(args.stage1_feature, map_location="cpu").float()
    if features.dim() != 3 or features.size(1) != 2:
        raise ValueError("Stage 1 feature must have shape [num_nodes, 2, hidden_dim]")
    if features.size(0) != graph.num_nodes():
        raise ValueError("Stage 1 feature node count does not match graph")

    return Stage2Data(
        graph=graph,
        texts=frame["text"].fillna("").astype(str).tolist(),
        labels=labels,
        label_names=label_names,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        features=features,
    )


def build_adjacency(graph: dgl.DGLGraph) -> List[List[int]]:
    """Build sorted adjacency lists without self loops."""
    src, dst = graph.edges()
    adjacency = [set() for _ in range(graph.num_nodes())]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adjacency[u].add(v)
    return [sorted(nodes) for nodes in adjacency]
