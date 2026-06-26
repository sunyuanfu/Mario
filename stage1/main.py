import argparse
import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)


PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Mario: Stage 1 - Graph-Conditioned Vision-Language Model")
    parser.add_argument("--mode", type=str, default="gpu", help="device mode")
    parser.add_argument("--n_epochs", type=int, default=32, help="number of training epochs")
    parser.add_argument("--dataset", type=str, default="Movies", help="dataset name")
    parser.add_argument("--checkpoint_folder", type=str, default="check", help="checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=500, help="batch size for training")
    parser.add_argument("--wd", type=float, default=5e-3, help="weight decay")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--full_neighbor", action="store_true", help="use full neighbor sampling")
    parser.add_argument("--num_of_neighbors", type=int, default=7, help="number of neighbors to sample")
    parser.add_argument("--num_layers", type=int, default=1, help="number of GNN layers")
    parser.add_argument("--lr_scheduler_step_size", type=int, default=10, help="learning rate scheduler step size")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.99, help="learning rate scheduler gamma")
    parser.add_argument("--patch_size", type=int, default=5, help="patch size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden dimension")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold for filtering")
    parser.add_argument("--filter", action="store_true", help="enable threshold filtering")
    parser.add_argument("--eval_gnn_layers", type=int, default=1, help="evaluation GNN layers")
    parser.add_argument("--save", action="store_true", help="save model checkpoints")
    parser.add_argument("--id", type=int, default=0, help="experiment id")
    parser.add_argument("--num", type=int, default=0, help="experiment number")
    parser.add_argument("--use_large_features", action="store_true", help="use large 4096D features from vision-language models")
    args = parser.parse_args()
    return args


def to_bidirected_with_reverse_mapping(g):
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True
    )
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


def load_graph_model(cfg):
    from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
    from src.models.modeling_graphformers import GraphFormersForNeighborPredict
    config = TuringNLRv3Config.from_pretrained(
        "config.json", output_hidden_states=True
    )
    config.hidden_size = cfg.hidden_dim
    config.num_hidden_layers = 1
    model = GraphFormersForNeighborPredict(config, cfg)
    return model


@torch.no_grad()
def infer_all_node_embeddings(cfg, g, sampler, model, device):
    """Compute Stage 1 text/image embeddings for every node in the graph."""
    model.eval()
    all_idx = torch.arange(g.num_nodes(), device=g.device)
    use_uva = cfg.mode == "mixed"
    dataloader = DataLoader(
        g,
        all_idx,
        sampler,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    embeddings = torch.zeros(g.num_nodes(), 2, cfg.hidden_dim, device=device)
    for _, (_, _, blocks) in enumerate(dataloader):
        blocks = [b.to(device) for b in blocks]
        x = blocks[0].srcdata
        text_feat = x["text_feat"].to(device)
        image_feat = x["image_feat"].to(device)
        attention_mask = x["attention_mask"].to(device)

        text_embedding, image_embedding = model(blocks, text_feat, attention_mask, image_feat)
        num_node = blocks[-1].num_dst_nodes()
        text_embedding = text_embedding[:num_node,]
        image_embedding = image_embedding[:num_node,]
        embeddings[blocks[-1].dstdata["index"].to(device)] = torch.stack(
            [text_embedding.detach(), image_embedding.detach()], dim=1
        )

    model.train()
    return embeddings


def train(cfg, g, splits, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)

    train_idx = splits["train_idx"].long().squeeze(-1)
    if cfg.full_neighbor:
        sampler = MultiLayerFullNeighborSampler(
            num_layers=cfg.num_layers, prefetch_node_feats=["text_feat", "image_feat"]
        )
    else:
        sampler = NeighborSampler(
            [cfg.num_of_neighbors] * cfg.num_layers, prefetch_node_feats=["text_feat", "image_feat"]
        )

    use_uva = cfg.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    feature = torch.zeros(g.num_nodes(), 2, cfg.hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma
    )

    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0

        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata
            blocks = [b.to(device) for b in blocks]
            num_node = blocks[-1].num_dst_nodes()

            text_feat = x["text_feat"].to(device)
            image_feat = x["image_feat"].to(device)
            attention_mask = x["attention_mask"].to(device)

            text_embedding, image_embedding = model(blocks, text_feat, attention_mask, image_feat)
            text_embedding, image_embedding = (text_embedding[:num_node, ], image_embedding[:num_node, ])

            feature[blocks[-1].dstdata["index"]] = torch.stack(
                [text_embedding.detach(), image_embedding.detach()], dim=1
            )

            epsilon = 1e-8
            image_embedding = image_embedding / (
                image_embedding.norm(dim=1, keepdim=True) + epsilon
            )
            text_embedding = text_embedding / (
                text_embedding.norm(dim=1, keepdim=True) + epsilon
            )

            logit_scale = model.module.logit_scale if hasattr(model, "module") else model.logit_scale
            logits_per_image = logit_scale.exp() * image_embedding @ text_embedding.t()

            if cfg.filter:
                diagonal_values = torch.diag(logits_per_image)
                keep_indices = torch.nonzero(diagonal_values >= cfg.threshold).squeeze()
                logits_per_image = logits_per_image[keep_indices][:, keep_indices]

            logits_per_text = logits_per_image.t()
            batch_size = logits_per_image.shape[0]
            labels = torch.arange(batch_size, device=device).long()

            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        lr_scheduler.step()
        torch.save(feature, f"{cfg.dataset}_stage1_mlp_trainonly.pth")

    # Export embeddings for all nodes for Stage 2 (paper uses the same dataset across stages).
    print("Exporting Stage 1 embeddings for all nodes...")
    full_feature = infer_all_node_embeddings(cfg, g, sampler, model, device)
    torch.save(full_feature, f"{cfg.dataset}_stage1_mlp.pth")
    return None


def main():
    cfg = parse_arguments()
    set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Main device: {device}")

    dataset_name = cfg.dataset
    data_root = os.environ.get("DATA_ROOT", "./data")
    dataset_root = os.path.join(data_root, dataset_name)
    verbose = True

    if cfg.dataset in ["Movies", "Toys", "Grocery", "RedditS"]:
        from dataset_amazon_small import NodeClassificationDataset
        dataset = NodeClassificationDataset(
            root=dataset_root,
            data_path=data_root,
            verbose=verbose,
            device=device,
            save=cfg.save,
            use_large_features=cfg.use_large_features
        )
    else:
        from dataset_amazon_large import NodeClassificationDataset
        dataset = NodeClassificationDataset(
            root=dataset_root,
            data_path=data_root,
            verbose=verbose,
            device=device,
            save=cfg.save,
            use_large_features=cfg.use_large_features
        )

    g = dataset.graph
    is_cuda = g.device.type == "cuda"

    if is_cuda:
        g = g.cpu()

    g = dgl.remove_self_loop(g)
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = dgl.add_self_loop(g)

    if is_cuda:
        src, dst = g.edges()
        src = src.cuda()
        dst = dst.cuda()
        new_g = dgl.graph((src, dst), num_nodes=g.num_nodes())
        for key in g.ndata:
            new_g.ndata[key] = g.ndata[key].cuda()
        g = new_g

    splits = {}
    splits["train_idx"] = g.ndata["train_mask"].nonzero()
    splits["val_idx"] = g.ndata["val_mask"].nonzero()
    splits["test_idx"] = g.ndata["test_mask"].nonzero()

    model = load_graph_model(cfg)

    print(f"Training on {cfg.dataset}")
    train(cfg, g, splits, model)


if __name__ == "__main__":
    main()
