"""
Deep Graph Learning Architectures for Anti-Money Laundering (AML):
- Directed-Aware GraphSAGE (`DirSAGEConv`) separating incoming funding & outgoing payout flows
- Edge-Attention Graph Network (`GATConvLayer`) providing interpretable edge attention weights
- Focal Loss (`BinaryFocalLoss`) for extreme illicit class imbalance
- 64-D Penultimate Node Embedding Extraction for Hybrid GNN + LightGBM Ensembling
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return torch.mean(focal_weight * bce)


class DirectedSAGEConv(nn.Module):
    """
    Directed GraphSAGE Convolution Layer.
    Separately aggregates incoming (sender -> node) and outgoing (node -> receiver)
    neighborhood representations using sparse index_add message passing, combined
    with ego-node features and residual skip projection.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_combine = nn.Linear(out_channels * 3, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    @staticmethod
    def _mean_aggregate(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)
        ones = torch.ones((src.size(0), 1), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, ones)
        return out / torch.clamp(deg, min=1.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Incoming neighbor messages (src -> dst)
        agg_in = self._mean_aggregate(x, src, dst, num_nodes)
        # Outgoing neighbor messages (dst -> src)
        agg_out = self._mean_aggregate(x, dst, src, num_nodes)

        h_self = self.lin_self(x)
        h_in = self.lin_in(agg_in)
        h_out = self.lin_out(agg_out)

        combined = torch.cat([h_self, h_in, h_out], dim=-1)
        out = self.lin_combine(F.gelu(combined))
        out = self.norm(out + self.res_proj(x))
        out = self.dropout(F.gelu(out))
        return out


class EdgeAttentionConv(nn.Module):
    """
    Graph Attention (GAT) Layer over sparse edges that also returns per-edge
    attention importance weights for explainable subgraph forensics.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Add bidirectional edges for message passing while tracking original directed edges
        bi_src = torch.cat([src, dst], dim=0)
        bi_dst = torch.cat([dst, src], dim=0)

        h = self.lin(x)
        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        raw_e = F.leaky_relu(alpha_src[bi_src] + alpha_dst[bi_dst], negative_slope=0.2)
        # Numeric stability per destination node
        max_e = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        max_e.scatter_reduce_(0, bi_dst, raw_e, reduce="amax", include_self=False)
        exp_e = torch.exp(torch.clamp(raw_e - max_e[bi_dst], min=-10.0, max=10.0))

        sum_exp = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        sum_exp.index_add_(0, bi_dst, exp_e)
        attn_weights = exp_e / torch.clamp(sum_exp[bi_dst], min=1e-8)

        weighted_msg = h[bi_src] * attn_weights.unsqueeze(-1)
        agg = torch.zeros((num_nodes, h.size(1)), dtype=x.dtype, device=x.device)
        agg.index_add_(0, bi_dst, weighted_msg)

        out = self.norm(agg + self.res_proj(x))
        out = self.dropout(F.gelu(out))

        # Return node embeddings + attention weights on original directed edges (first half)
        orig_edge_attn = attn_weights[: src.size(0)]
        return out, orig_edge_attn


class CryptoAMLGraphNet(nn.Module):
    """
    Hybrid Directed-GraphSAGE + Edge-Attention GAT Architecture for AML Transaction Classification.
    Produces both:
    - Logit predictions for Illicit vs. Licit classification
    - 64-dimensional structural node embeddings for Hybrid GNN + LightGBM Ensembling
    - Per-edge attention scores for GNNExplainer Subgraph Forensics
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        embed_dim: int = 64,
        dropout: float = 0.22,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.sage1 = DirectedSAGEConv(hidden_channels, hidden_channels, dropout=dropout)
        self.sage2 = DirectedSAGEConv(hidden_channels, embed_dim, dropout=dropout)
        self.gat = EdgeAttentionConv(embed_dim, embed_dim, dropout=dropout)

        # Skip connection combining raw projected features + 2-hop SAGE + GAT attention representation
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_channels + embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.input_proj(x)
        h1 = self.sage1(h0, edge_index)
        h2 = self.sage2(h1, edge_index)
        h_gat, edge_attn = self.gat(h2, edge_index)

        concat = torch.cat([h0, h2, h_gat], dim=-1)
        node_embeddings = self.embedding_head(concat)
        logits = self.classifier(node_embeddings).squeeze(-1)
        return logits, node_embeddings, edge_attn


def train_gnn_model(
    features: np.ndarray,
    edge_index_np: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    epochs: int = 85,
    lr: float = 0.0045,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[CryptoAMLGraphNet, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """
    Train the Directed GraphSAGE + GAT network using semi-supervised full-graph message passing
    (where unlabeled/unknown nodes participate in message passing, while loss is computed strictly
    on labeled training nodes in timesteps <= 30).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.float32, device=device)
    train_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long, device=device)
    val_idx = torch.tensor(np.where(val_mask)[0], dtype=torch.long, device=device)

    model = CryptoAMLGraphNet(
        in_channels=features.shape[1],
        hidden_channels=96,
        embed_dim=64,
        dropout=0.20,
    ).to(device)

    criterion = BinaryFocalLoss(alpha=0.78, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(x_tensor, edge_index)
        loss = criterion(logits[train_idx], y_tensor[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(x_tensor, edge_index)
            val_loss = criterion(val_logits[val_idx], y_tensor[val_idx]).item()

        history["train_loss"].append(round(float(loss.item()), 5))
        history["val_loss"].append(round(float(val_loss), 5))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_logits, final_embeddings, final_edge_attn = model(x_tensor, edge_index)
        gnn_probs = torch.sigmoid(final_logits).cpu().numpy()
        embeddings_np = final_embeddings.cpu().numpy()
        edge_attn_np = final_edge_attn.cpu().numpy()

    return model.cpu(), gnn_probs, embeddings_np, edge_attn_np, history
