"""
Topological Graph Feature Engineering Engine for Anti-Money Laundering (AML).

Extracts structural & neighborhood features from directed transaction graphs:
- Degree & Flow Asymmetry (Fan-In / Fan-Out detection)
- Directed PageRank & Undirected Triangle Clustering Coefficient
- K-Core Decomposition & Short Directed Cycle / Wash-Ring Participation
- Vectorized 1-Hop & 2-Hop Sparse Neighborhood Aggregations
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


TOPOLOGICAL_FEATURE_NAMES: List[str] = [
    "graph_in_degree",
    "graph_out_degree",
    "graph_total_degree",
    "graph_in_out_ratio",
    "graph_flow_asymmetry",
    "graph_pagerank",
    "graph_clustering_coeff",
    "graph_core_number",
    "graph_in_cycle_indicator",
    "nbr_1hop_mean_amount",
    "nbr_1hop_max_velocity",
    "nbr_1hop_mean_fee_ratio",
    "nbr_1hop_mean_coinjoin",
    "nbr_1hop_mean_burst_zscore",
    "nbr_1hop_min_utxo_age",
    "nbr_2hop_mean_velocity",
    "nbr_2hop_mean_coinjoin",
    "nbr_2hop_mean_burst_zscore",
    "nbr_2hop_max_degree",
]


def extract_topological_features(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    local_feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute structural graph features and 1-hop/2-hop neighborhood aggregations
    independently within each transaction window and without target labels.
    """
    df = nodes_df.copy().sort_values("node_idx").reset_index(drop=True)
    num_nodes = len(df)

    src = edges_df["src_idx"].values.astype(np.int64)
    dst = edges_df["dst_idx"].values.astype(np.int64)

    # 1. Fast degree statistics
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    tot_deg = in_deg + out_deg
    in_out_ratio = (in_deg + 0.5) / (out_deg + 0.5)
    flow_asym = np.abs(in_deg - out_deg) / (tot_deg + 1.0)

    # 2. NetworkX structural metrics (PageRank, Clustering, K-Core, Directed Cycle check)
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(range(num_nodes))
    G_dir.add_edges_from(zip(src.tolist(), dst.tolist()))

    G_undir = G_dir.to_undirected()
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))

    pagerank = np.zeros(num_nodes, dtype=np.float32)
    for ids in df.groupby("timestep").indices.values():
        subgraph = G_dir.subgraph(ids)
        scores = nx.pagerank(subgraph, alpha=0.85, max_iter=100, tol=1e-6)
        pagerank[ids] = [scores[i] * len(ids) for i in ids]

    clustering_dict = nx.clustering(G_undir)
    clustering = np.array([clustering_dict.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)

    core_dict = nx.core_number(G_undir)
    core_num = np.array([core_dict.get(i, 0) for i in range(num_nodes)], dtype=np.float32)

    # Detect nodes participating in non-trivial Strongly Connected Components (directed cycles / wash rings!)
    in_cycle = np.zeros(num_nodes, dtype=np.float32)
    for scc in nx.strongly_connected_components(G_dir):
        if len(scc) > 1:
            for nid in scc:
                in_cycle[nid] = 1.0

    # 3. Sparse Adjacency Matrix for fast 1-hop and 2-hop neighborhood aggregations
    # Bidirectional adjacency so each node aggregates both incoming funding sources and outgoing destinations
    row_idx = np.concatenate([src, dst])
    col_idx = np.concatenate([dst, src])
    data = np.ones(len(row_idx), dtype=np.float32)
    adj = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(num_nodes, num_nodes))
    # Binarize duplicate edges
    adj.data[:] = 1.0

    deg_vec = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_deg = np.zeros_like(deg_vec, dtype=np.float32)
    np.divide(1.0, deg_vec, out=inv_deg, where=deg_vec > 0)
    norm_adj = sparse.diags(inv_deg) @ adj

    # Helper to extract named column or fallback to first columns for raw Elliptic
    def _col(name: str, fallback_idx: int) -> np.ndarray:
        if name in df.columns:
            return df[name].values.astype(np.float32)
        col_name = local_feature_cols[min(fallback_idx, len(local_feature_cols) - 1)]
        return df[col_name].values.astype(np.float32)

    amt = _col("tx_amount_btc", 0)
    vel = _col("hourly_tx_velocity", 1)
    fee_r = _col("fee_to_amount_ratio", 2)
    cj = _col("coinjoin_similarity_score", 3)
    burst = _col("burst_activity_zscore", 4)
    utxo = _col("utxo_age_hours", 5)

    nbr_1hop_mean_amt = norm_adj @ amt
    nbr_1hop_mean_fee = norm_adj @ fee_r
    nbr_1hop_mean_cj = norm_adj @ cj
    nbr_1hop_mean_burst = norm_adj @ burst

    # 1-hop max velocity & min utxo age using edge-wise scatter
    nbr_1hop_max_vel = vel.copy()
    nbr_1hop_min_utxo = utxo.copy()
    np.maximum.at(nbr_1hop_max_vel, row_idx, vel[col_idx])
    np.minimum.at(nbr_1hop_min_utxo, row_idx, utxo[col_idx])

    # 2-hop sparse propagation
    nbr_2hop_mean_vel = norm_adj @ (norm_adj @ vel)
    nbr_2hop_mean_cj = norm_adj @ (norm_adj @ cj)
    nbr_2hop_mean_burst = norm_adj @ (norm_adj @ burst)

    nbr_1hop_max_deg = tot_deg.copy()
    np.maximum.at(nbr_1hop_max_deg, row_idx, tot_deg[col_idx])
    nbr_2hop_max_deg = nbr_1hop_max_deg.copy()
    np.maximum.at(nbr_2hop_max_deg, row_idx, nbr_1hop_max_deg[col_idx])

    topo_dict = {
        "graph_in_degree": in_deg,
        "graph_out_degree": out_deg,
        "graph_total_degree": tot_deg,
        "graph_in_out_ratio": in_out_ratio,
        "graph_flow_asymmetry": flow_asym,
        "graph_pagerank": pagerank,
        "graph_clustering_coeff": clustering,
        "graph_core_number": core_num,
        "graph_in_cycle_indicator": in_cycle,
        "nbr_1hop_mean_amount": nbr_1hop_mean_amt,
        "nbr_1hop_max_velocity": nbr_1hop_max_vel,
        "nbr_1hop_mean_fee_ratio": nbr_1hop_mean_fee,
        "nbr_1hop_mean_coinjoin": nbr_1hop_mean_cj,
        "nbr_1hop_mean_burst_zscore": nbr_1hop_mean_burst,
        "nbr_1hop_min_utxo_age": nbr_1hop_min_utxo,
        "nbr_2hop_mean_velocity": nbr_2hop_mean_vel,
        "nbr_2hop_mean_coinjoin": nbr_2hop_mean_cj,
        "nbr_2hop_mean_burst_zscore": nbr_2hop_mean_burst,
        "nbr_2hop_max_degree": nbr_2hop_max_deg,
    }

    for k, v in topo_dict.items():
        df[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    return df, list(TOPOLOGICAL_FEATURE_NAMES)
