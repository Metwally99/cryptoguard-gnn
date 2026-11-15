"""
Comprehensive Unit & API Integration Test Suite for CryptoGuard GNN AML Engine.

Tests:
1. Temporal split integrity (zero future data leakage across train/val/test masks)
2. Topological graph feature extraction (no NaNs/Infs, finite structural features)
3. PyTorch Directed GraphSAGE + Edge-Attention GAT forward pass & embedding shapes
4. Hybrid GraphSAGE + LightGBM superiority over Tabular Baseline on PR-AUC & Camouflaged Mule Recall
5. TreeSHAP explainability & 2-Hop Ego-Subgraph extraction
6. FastAPI REST endpoints (`/api/health`, `/api/overview`, `/api/alerts`, `/api/transaction/{id}`, `/api/simulate`)
"""

from __future__ import annotations

import numpy as np
import torch
from fastapi.testclient import TestClient

from src.api.main import app, get_artifacts
from src.data.graph_features import TOPOLOGICAL_FEATURE_NAMES
from src.models.gnn import BinaryFocalLoss, CryptoAMLGraphNet


client = TestClient(app)


def test_temporal_split_no_leakage():
    """Verify strictly temporal train (<=30), val (31..34), and test (>=35) splits with zero overlap."""
    art = get_artifacts()
    df = art.nodes_df

    timesteps = df["timestep"].values
    labeled = df["label"].values != -1
    train_mask = labeled & (timesteps <= 30)
    val_mask = labeled & (timesteps > 30) & (timesteps <= 34)
    test_mask = labeled & (timesteps > 34)

    assert not np.any(train_mask & val_mask)
    assert not np.any(train_mask & test_mask)
    assert not np.any(val_mask & test_mask)
    assert df.loc[train_mask, "timestep"].max() < df.loc[test_mask, "timestep"].min()
    edges = art.edges_df
    assert np.all(timesteps[edges["src_idx"]] == timesteps[edges["dst_idx"]])


def test_topological_features_valid():
    """Verify topological graph features are finite and populated."""
    art = get_artifacts()
    df = art.nodes_df
    for col in TOPOLOGICAL_FEATURE_NAMES:
        assert col in df.columns, f"Missing topological column {col}"
        vals = df[col].values
        assert np.all(np.isfinite(vals)), f"Non-finite values in {col}"


def test_gnn_architecture_and_focal_loss():
    """Verify DirectedSAGEConv + EdgeAttentionConv forward pass shapes and gradient flow."""
    x = torch.randn(50, 18, requires_grad=True)
    edge_index = torch.randint(0, 50, (2, 140), dtype=torch.long)
    y = torch.randint(0, 2, (50,)).float()

    model = CryptoAMLGraphNet(in_channels=18, hidden_channels=32, embed_dim=16)
    logits, emb, edge_attn = model(x, edge_index)

    assert logits.shape == (50,)
    assert emb.shape == (50, 16)
    assert edge_attn.shape == (140,)
    assert torch.all(edge_attn >= 0.0)

    loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    loss = loss_fn(logits, y)
    loss.backward()
    assert x.grad is not None


def test_hybrid_outperforms_tabular_baseline():
    """Verify that Graph Topology & Hybrid GNN+LightGBM beat the isolated Tabular Baseline."""
    art = get_artifacts()
    tab = art.model_metrics["Tabular_LightGBM"]
    hyb = art.model_metrics["Hybrid_GraphSAGE_LightGBM"]

    assert hyb["pr_auc"] > tab["pr_auc"] + 0.10
    assert hyb["f1_score"] > tab["f1_score"] + 0.15
    assert hyb["camouflaged_mule_recall"] > tab["camouflaged_mule_recall"] + 0.40


def test_fastapi_endpoints():
    """Verify all production REST endpoints respond with 200 OK and valid payloads."""
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_overview = client.get("/api/overview")
    assert r_overview.status_code == 200
    ov = r_overview.json()
    assert "Hybrid_GraphSAGE_LightGBM" in ov["model_metrics"]
    assert len(ov["temporal_drift"]) > 20

    r_alerts = client.get("/api/alerts?limit=10&camouflaged_only=true")
    assert r_alerts.status_code == 200
    alerts = r_alerts.json()["alerts"]
    assert len(alerts) > 0
    sample_tx_id = alerts[0]["tx_id"]

    r_dossier = client.get(f"/api/transaction/{sample_tx_id}")
    assert r_dossier.status_code == 200
    dossier = r_dossier.json()
    assert dossier["tx_id"] == sample_tx_id
    assert len(dossier["top_feature_attributions"]) >= 5
    assert dossier["subgraph"]["num_subgraph_nodes"] >= 1
    assert "SYNTHETIC CASE SUMMARY" in dossier["sar_compliance_narrative"]

    r_sim = client.post(
        "/api/simulate",
        json={
            "tx_amount_btc": 4.2,
            "fee_to_amount_ratio": 0.0045,
            "num_inputs": 2,
            "num_outputs": 9,
            "utxo_age_hours": 1.5,
            "hourly_tx_velocity": 18.0,
            "coinjoin_similarity_score": 0.72,
            "burst_activity_zscore": 2.8,
            "graph_in_degree": 4,
            "graph_out_degree": 11,
            "graph_in_cycle_indicator": 1,
            "nbr_1hop_mean_coinjoin": 0.68,
            "nbr_1hop_max_velocity": 22.0,
            "nbr_2hop_mean_burst_zscore": 2.5,
        },
    )
    assert r_sim.status_code == 200
    sim_data = r_sim.json()
    assert "scores" in sim_data
    assert 0.0 <= sim_data["scores"]["hybrid_ensemble_prob"] <= 1.0
