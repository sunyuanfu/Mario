import torch
import torch.nn.functional as F
import dgl.nn as dglnn
import tqdm
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.nn import GATv2Conv
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Linear


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "mean"))
        elif num_layers == 2:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        elif num_layers == 3:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(
                dglnn.GraphConv(
                    in_size, out_size, norm="both", weight=True, bias=True, allow_zero_in_degree=True
                )
            )
        elif num_layers == 2:
            self.layers.append(
                dglnn.GraphConv(
                    in_size, hid_size, norm="both", weight=True, bias=True, allow_zero_in_degree=True
                )
            )
            self.layers.append(
                dglnn.GraphConv(
                    hid_size, out_size, norm="both", weight=True, bias=True, allow_zero_in_degree=True
                )
            )
        elif num_layers == 3:
            self.layers.append(
                dglnn.GraphConv(
                    in_size, hid_size, norm="both", weight=True, bias=True, allow_zero_in_degree=True
                )
            )
            self.layers.append(
                dglnn.GraphConv(
                    hid_size, hid_size, norm="both", weight=True, bias=True, allow_zero_in_degree=True
                )
            )
            self.layers.append(
                dglnn.GraphConv(
                    hid_size, out_size, norm="both", weight=True, bias=True, allow_zero_in_degree=True
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class MLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_size, out_size))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        elif num_layers == 3:
            self.layers.append(nn.Linear(in_size, hid_size))
            self.layers.append(nn.Linear(hid_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[output_nodes]
                h = layer(x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class GATv2(nn.Module):
    def __init__(self, in_size, hid_size, num_layers, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        self.heads = heads
        self.hid_size = hid_size
        self.layer_norms = torch.nn.ModuleList()
        self.gatv2_layers.append(
            GATv2Conv(
                in_size, hid_size, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation,
                bias=False, share_weights=True
            )
        )
        for l in range(num_layers - 1):
            self.layer_norms.append(nn.LayerNorm(hid_size * heads[l]))
            self.gatv2_layers.append(
                GATv2Conv(
                    hid_size * heads[l], hid_size, heads[l + 1], feat_drop, attn_drop, negative_slope, residual,
                    self.activation, bias=False, share_weights=True
                )
            )
        self.predictor = nn.Sequential(
            nn.Linear(hid_size * heads[-1], hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1)
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gatv2_layers, blocks)):
            h = layer(block, h).flatten(1)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0
        )
        buffer_device = torch.device("cpu")
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.gatv2_layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size * self.heads[l], device=buffer_device,
                pin_memory=pin_memory
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc="Inference"):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class GINMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, num_heads=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        if num_layers == 1:
            self.layers.append(
                dglnn.GATConv(
                    in_size, out_size, num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True
                )
            )
        elif num_layers == 2:
            self.layers.append(
                dglnn.GATConv(
                    in_size, hid_size // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True
                )
            )
            self.layers.append(
                dglnn.GATConv(
                    hid_size, out_size, 1, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True
                )
            )
        elif num_layers == 3:
            self.layers.append(
                dglnn.GATConv(
                    in_size, hid_size // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True
                )
            )
            self.layers.append(
                dglnn.GATConv(
                    hid_size, hid_size // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True
                )
            )
            self.layers.append(
                dglnn.GATConv(
                    hid_size, out_size, 1, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True
                )
            )
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                h = h.mean(1)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                    h = F.relu(h)
                    h = self.dropout(h)
                else:
                    h = h.mean(1)
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
