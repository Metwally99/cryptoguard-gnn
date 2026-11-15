"""
Subgraph Forensics, Exact TreeSHAP Feature Attribution, & GNN Edge Attention Explainer.

Provides:
1. Exact TreeSHAP feature attributions via LightGBM `pred_contrib=True`
2. 2-Hop Directed Ego-Subgraph extraction with learned GAT edge attention weights
3. Automated FinCEN Suspicious Activity Report (SAR) compliance narrative generation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.data.graph_features import TOPOLOGICAL_FEATURE_NAMES
from src.models.pipeline import PipelineArtifacts


FEATURE_HUMAN_LABELS: Dict[str, str] = {
    "tx_amount_btc": "Transaction Value (BTC)",
    "tx_fee_btc": "Network Fee (BTC)",
    "fee_to_amount_ratio": "Fee-to-Volume Ratio",
    "num_inputs": "UTXO Input Count",
    "num_outputs": "UTXO Output Fan-Out",
    "input_output_ratio": "Input/Output Symmetry Ratio",
    "utxo_age_hours": "Holding Duration / UTXO Age (hrs)",
    "time_since_last_tx_min": "Inter-Transaction Delay (min)",
    "output_entropy": "Output Value Distribution Entropy",
    "round_amount_indicator": "Structured Round Denomination",
    "cross_exchange_flag": "Cross-Border / High-Risk VASP Hop",
    "hourly_tx_velocity": "Wallet Hourly Velocity",
    "address_reuse_score": "Address Reuse Index",
    "coinjoin_similarity_score": "CoinJoin / Mixer Fingerprint Score",
    "dust_output_count": "Micro-Dust Output Count",
    "script_complexity_score": "Multi-Sig / Script Obfuscation",
    "timezone_anomaly_score": "Circadian Timing Anomaly",
    "burst_activity_zscore": "Temporal Burst Z-Score",
    "graph_in_degree": "Graph In-Degree (Funding Sources)",
    "graph_out_degree": "Graph Out-Degree (Payout Sinks)",
    "graph_total_degree": "Total Ego-Network Degree",
    "graph_in_out_ratio": "Fan-In / Fan-Out Topology Ratio",
    "graph_flow_asymmetry": "Directed Flow Asymmetry",
    "graph_pagerank": "Network PageRank Centrality",
    "graph_clustering_coeff": "Local Clique / Triangle Clustering",
    "graph_core_number": "Syndicate K-Core Depth",
    "graph_in_cycle_indicator": "Directed Wash-Cycle Participation",
    "nbr_1hop_mean_amount": "1-Hop Neighbor Mean Volume (BTC)",
    "nbr_1hop_max_velocity": "1-Hop Neighbor Max Velocity",
    "nbr_1hop_mean_fee_ratio": "1-Hop Neighbor Fee Ratio",
    "nbr_1hop_mean_coinjoin": "1-Hop Neighbor Mixer Exposure",
    "nbr_1hop_mean_burst_zscore": "1-Hop Neighbor Burst Activity",
    "nbr_1hop_min_utxo_age": "1-Hop Rapid Peel UTXO Age (hrs)",
    "nbr_1hop_unknown_ratio": "Unverified / Dark Wallet Neighbor Ratio",
    "nbr_2hop_mean_velocity": "2-Hop Contagion Velocity",
    "nbr_2hop_mean_coinjoin": "2-Hop Syndicate Mixer Exposure",
    "nbr_2hop_mean_burst_zscore": "2-Hop Coordinated Burst Score",
    "nbr_2hop_max_degree": "2-Hop Hub Connectivity",
}


def resolve_node_idx(artifacts: PipelineArtifacts, identifier: str | int) -> int:
    """Resolve either a `tx_id` string (e.g. 'tx_38_06712') or integer `node_idx`."""
    df = artifacts.nodes_df
    if isinstance(identifier, int) or str(identifier).isdigit():
        idx = int(identifier)
        if 0 <= idx < len(df):
            return idx
    matches = df.index[df["tx_id"] == str(identifier)].tolist()
    if matches:
        return int(df.loc[matches[0], "node_idx"])
    raise KeyError(f"Transaction identifier '{identifier}' not found in graph.")


def extract_ego_subgraph(
    artifacts: PipelineArtifacts,
    identifier: str | int,
    hops: int = 2,
    max_nodes: int = 36,
) -> Dict[str, object]:
    """
    Extract a k-hop directed ego-network around the target transaction,
    complete with node risk metadata and GNN learned edge attention weights.
    """
    center_idx = resolve_node_idx(artifacts, identifier)
    df = artifacts.nodes_df
    edges_df = artifacts.edges_df

    visited: Set[int] = {center_idx}
    frontier: Set[int] = {center_idx}
    hop_distance: Dict[int, int] = {center_idx: 0}

    for hop in range(1, hops + 1):
        if not frontier or len(visited) >= max_nodes:
            break
        sub_edges = edges_df[edges_df["src_idx"].isin(frontier) | edges_df["dst_idx"].isin(frontier)]
        candidates = set(sub_edges["src_idx"].tolist()) | set(sub_edges["dst_idx"].tolist())
        new_nodes = list(candidates - visited)
        # Prioritize high-risk neighbors if we exceed max_nodes
        if len(visited) + len(new_nodes) > max_nodes:
            new_nodes.sort(key=lambda nid: float(df.loc[nid, "prob_hybrid"]), reverse=True)
            new_nodes = new_nodes[: max(0, max_nodes - len(visited))]
        for nid in new_nodes:
            visited.add(int(nid))
            hop_distance[int(nid)] = hop
        frontier = set(new_nodes)

    sub_edges_df = edges_df[edges_df["src_idx"].isin(visited) & edges_df["dst_idx"].isin(visited)].copy()

    nodes_payload: List[Dict[str, object]] = []
    for nid in sorted(visited):
        row = df.iloc[nid]
        lbl = int(row["label"])
        nodes_payload.append(
            {
                "node_idx": int(nid),
                "tx_id": str(row["tx_id"]),
                "wallet_id": str(row.get("wallet_id", f"bc1q{nid:08d}")),
                "timestep": int(row["timestep"]),
                "hop_distance": int(hop_distance.get(nid, 1)),
                "is_target": bool(nid == center_idx),
                "label": lbl,
                "label_name": "Illicit" if lbl == 1 else ("Licit" if lbl == 0 else "Unknown"),
                "typology": str(row.get("typology", "Unknown")),
                "is_camouflaged": bool(row.get("is_camouflaged", False)),
                "prob_hybrid": round(float(row["prob_hybrid"]), 4),
                "prob_gnn": round(float(row["prob_gnn"]), 4),
                "prob_tabular": round(float(row["prob_tabular"]), 4),
                "risk_tier": str(row["risk_tier"]),
                "tx_amount_btc": round(float(row.get("tx_amount_btc", 0.0)), 4),
                "in_degree": int(row.get("graph_in_degree", 0)),
                "out_degree": int(row.get("graph_out_degree", 0)),
            }
        )

    edges_payload: List[Dict[str, object]] = []
    for _, erow in sub_edges_df.iterrows():
        edges_payload.append(
            {
                "src_idx": int(erow["src_idx"]),
                "dst_idx": int(erow["dst_idx"]),
                "src_tx_id": str(erow["src_tx_id"]),
                "dst_tx_id": str(erow["dst_tx_id"]),
                "edge_type": str(erow.get("edge_type", "transfer")),
                "attention_weight": round(float(erow.get("attention_weight", 0.5)), 4),
            }
        )

    illicit_nbrs = sum(1 for n in nodes_payload if not n["is_target"] and n["label"] == 1)
    high_risk_nbrs = sum(1 for n in nodes_payload if not n["is_target"] and float(n["prob_hybrid"]) >= 0.60)

    return {
        "target_node_idx": int(center_idx),
        "target_tx_id": str(df.iloc[center_idx]["tx_id"]),
        "hops": hops,
        "num_subgraph_nodes": len(nodes_payload),
        "num_subgraph_edges": len(edges_payload),
        "illicit_neighbors_count": illicit_nbrs,
        "high_risk_neighbors_count": high_risk_nbrs,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }


def explain_transaction_node(
    artifacts: PipelineArtifacts,
    identifier: str | int,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Generate a full forensic explanation for a single transaction:
    - TreeSHAP log-odds contributions from the hybrid LightGBM component
    - GNN vs. Tabular risk uplift analysis
    - Subgraph neighbor contagion summary
    - Automated FinCEN SAR (Suspicious Activity Report) narrative
    """
    node_idx = resolve_node_idx(artifacts, identifier)
    df = artifacts.nodes_df
    row = df.iloc[node_idx]

    feature_cols = artifacts.all_feature_cols
    x_topo_row = df.loc[[node_idx], feature_cols].values.astype(np.float32)
    emb_row = artifacts.gnn_embeddings[[node_idx]].astype(np.float32)
    x_hybrid_row = np.hstack([x_topo_row, emb_row]).astype(np.float32)

    # Exact TreeSHAP contributions via Hybrid LightGBM pred_contrib=True (last column is expected value bias)
    shap_contribs = artifacts.lgb_hybrid_model.predict(x_hybrid_row, pred_contrib=True)[0]
    feature_shap = shap_contribs[: len(feature_cols)]
    gnn_emb_shap_sum = float(np.sum(shap_contribs[len(feature_cols) : -1]))
    base_value = float(shap_contribs[-1])

    attributions: List[Dict[str, object]] = []
    for col, val, sval in zip(feature_cols, x_topo_row[0], feature_shap):
        is_topo = col in TOPOLOGICAL_FEATURE_NAMES
        attributions.append(
            {
                "feature": col,
                "display_name": FEATURE_HUMAN_LABELS.get(col, col),
                "category": "Graph Topology & Neighborhood" if is_topo else "Local Transaction Attribute",
                "value": round(float(val), 4),
                "shap_log_odds": round(float(sval), 4),
                "abs_impact": round(abs(float(sval)), 4),
                "direction": "Increases Illicit Risk" if sval > 0 else "Decreases Illicit Risk",
            }
        )

    attributions.append(
        {
            "feature": "gnn_64d_structural_embedding",
            "display_name": "64-D Directed GraphSAGE + GAT Embedding",
            "category": "Deep Graph Neural Network Representation",
            "value": round(float(row["prob_gnn"]), 4),
            "shap_log_odds": round(gnn_emb_shap_sum, 4),
            "abs_impact": round(abs(gnn_emb_shap_sum), 4),
            "direction": "Increases Illicit Risk" if gnn_emb_shap_sum > 0 else "Decreases Illicit Risk",
        }
    )

    attributions_sorted = sorted(attributions, key=lambda d: float(d["abs_impact"]), reverse=True)
    top_positive_drivers = [a for a in attributions_sorted if float(a["shap_log_odds"]) > 0][:6]
    top_all_drivers = attributions_sorted[:top_k]

    subgraph = extract_ego_subgraph(artifacts, node_idx, hops=2, max_nodes=32)

    prob_hybrid = float(row["prob_hybrid"])
    prob_gnn = float(row["prob_gnn"])
    prob_tabular = float(row["prob_tabular"])
    gnn_uplift = round(prob_hybrid - prob_tabular, 4)

    # Build automated FinCEN / AML Compliance Investigator Narrative
    tx_id = str(row["tx_id"])
    wallet_id = str(row.get("wallet_id", f"bc1q{node_idx:08d}"))
    risk_tier = str(row["risk_tier"])
    typology = str(row.get("typology", "Unknown"))
    is_camo = bool(row.get("is_camouflaged", False))

    driver_phrases = [
        f"{d['display_name']} ({d['value']}, SHAP +{d['shap_log_odds']:.2f})"
        for d in top_positive_drivers[:3]
    ]
    drivers_str = ", ".join(driver_phrases) if driver_phrases else "elevated multi-hop structural anomalies"

    camo_clause = (
        f" Notably, this wallet exhibits **camouflaged local behavior** (Tabular-only risk score: {prob_tabular:.1%}), "
        f"but **Directed GraphSAGE + GAT message passing** detected severe 2-hop syndicate contagion, elevating the final hybrid risk score to **{prob_hybrid:.1%} (+{gnn_uplift:.1%} uplift)**."
        if is_camo or gnn_uplift > 0.15
        else f" Both local velocity indicators ({prob_tabular:.1%}) and graph message passing ({prob_gnn:.1%}) corroborate a **{prob_hybrid:.1%}** composite laundering probability."
    )

    sar_narrative = (
        f"**[SYNTHETIC CASE SUMMARY — {risk_tier} PRIORITY]** Transaction `{tx_id}` (Wallet `{wallet_id}`, "
        f"Timestep {int(row['timestep'])}) was scored at **{prob_hybrid:.1%} illicit probability** "
        f"(Predicted Pattern: `{typology}`). Primary forensic triggers include {drivers_str}. "
        f"Within its 2-hop ego-network ({subgraph['num_subgraph_nodes']} nodes, {subgraph['num_subgraph_edges']} directed edges), "
        f"the node connects to **{subgraph['high_risk_neighbors_count']} high-risk counterparties**."
        f"{camo_clause}"
    )

    return {
        "node_idx": int(node_idx),
        "tx_id": tx_id,
        "wallet_id": wallet_id,
        "timestep": int(row["timestep"]),
        "true_label": int(row["label"]),
        "true_label_name": "Illicit" if int(row["label"]) == 1 else ("Licit" if int(row["label"]) == 0 else "Unknown"),
        "typology": typology,
        "is_camouflaged": is_camo,
        "risk_tier": risk_tier,
        "scores": {
            "hybrid_ensemble_prob": round(prob_hybrid, 4),
            "deep_gnn_prob": round(prob_gnn, 4),
            "topology_lgbm_prob": round(float(row["prob_topology_lgbm"]), 4),
            "tabular_baseline_prob": round(prob_tabular, 4),
            "gnn_over_tabular_uplift": gnn_uplift,
        },
        "shap_base_value": round(base_value, 4),
        "top_feature_attributions": top_all_drivers,
        "sar_compliance_narrative": sar_narrative,
        "subgraph": subgraph,
    }
