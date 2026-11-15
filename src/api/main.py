"""
FastAPI Microservice for Real-Time Crypto Anti-Money Laundering (AML) Scoring,
Subgraph Forensics, TreeSHAP Explainability, and Temporal Drift Monitoring.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import HealthResponse, SimulationRequest
from src.explain.explainer import (
    FEATURE_HUMAN_LABELS,
    explain_transaction_node,
    extract_ego_subgraph,
)
from src.models.pipeline import PipelineArtifacts, run_training_pipeline


app = FastAPI(
    title="CryptoGuard GNN — Graph AML Research Demo",
    description=(
        "Synthetic transaction risk experiments with Directed GraphSAGE, edge attention, and LightGBM."
    ),
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_ARTIFACTS: Optional[PipelineArtifacts] = None
_ARTIFACTS_LOCK = Lock()


def get_artifacts() -> PipelineArtifacts:
    """Lazy-load or initialize trained pipeline artifacts."""
    global _ARTIFACTS
    with _ARTIFACTS_LOCK:
        if _ARTIFACTS is None:
            _ARTIFACTS = run_training_pipeline()
        return _ARTIFACTS


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_command_center():
    """Serve the interactive CryptoGuard GNN AML Investigator Command Center."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>CryptoGuard GNN API is running. Visit <a href='/docs'>/docs</a></h1>")


@app.get("/book", include_in_schema=False)
def download_pdf_book():
    """Serve the complete illustrated CryptoGuard GNN PDF Handbook."""
    book_file = Path(__file__).resolve().parents[2] / "docs" / "CryptoGuard_GNN_Complete_Book.pdf"
    if book_file.exists():
        return FileResponse(
            book_file,
            media_type="application/pdf",
            filename="CryptoGuard_GNN_Complete_Book.pdf",
        )
    raise HTTPException(status_code=404, detail="PDF book not generated yet.")


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return service status and loaded graph/model metadata."""
    art = get_artifacts()
    return HealthResponse(
        status="healthy",
        service="CryptoGuard GNN AML Engine",
        version="1.0.0",
        dataset_source=str(art.dataset_metadata.get("dataset_source", "Elliptic AML Graph")),
        num_nodes=int(len(art.nodes_df)),
        num_edges=int(len(art.edges_df)),
        models_loaded=list(art.model_metrics.keys()),
    )


@app.get("/api/overview", tags=["Analytics & Benchmarks"])
def get_overview() -> Dict[str, object]:
    """
    Return the complete 4-model benchmark comparison, typology recall analysis,
    top feature importances, GNN training curve, and 49-timestep temporal drift metrics.
    """
    art = get_artifacts()
    return {
        "dataset_metadata": art.dataset_metadata,
        "model_metrics": art.model_metrics,
        "typology_recall_comparison": art.typology_recall_comparison,
        "feature_importances": art.feature_importances[:16],
        "temporal_drift": art.temporal_drift_records,
        "gnn_training_history": art.gnn_history,
    }


@app.get("/api/alerts", tags=["AML Investigator Queue"])
def get_alert_queue(
    limit: int = Query(default=40, ge=5, le=200),
    typology: Optional[str] = Query(default=None, description="Filter by laundering typology"),
    risk_tier: Optional[str] = Query(default=None, description="CRITICAL, HIGH, ELEVATED, LOW"),
    camouflaged_only: bool = Query(default=False, description="Show only locally camouflaged mules caught by GNN"),
) -> Dict[str, object]:
    """
    Return prioritized AML transaction alerts from the out-of-time test split (Timesteps 35..49).
    """
    art = get_artifacts()
    df = art.nodes_df
    # Focus on test window (timesteps 35..49)
    sub = df[df["timestep"] >= 35].copy()

    if typology and typology != "ALL":
        sub = sub[sub["typology"] == typology]
    if risk_tier and risk_tier != "ALL":
        sub = sub[sub["risk_tier"] == risk_tier]
    if camouflaged_only:
        sub = sub[sub["is_camouflaged"] == True]  # noqa: E712

    sub["gnn_uplift"] = np.round(sub["prob_hybrid"] - sub["prob_tabular"], 4)
    sub = sub.sort_values(["prob_hybrid", "gnn_uplift"], ascending=False).head(limit)

    records: List[Dict[str, object]] = []
    for _, row in sub.iterrows():
        lbl = int(row["label"])
        records.append(
            {
                "node_idx": int(row["node_idx"]),
                "tx_id": str(row["tx_id"]),
                "wallet_id": str(row.get("wallet_id", f"bc1q{int(row['node_idx']):08d}")),
                "timestep": int(row["timestep"]),
                "typology": str(row["typology"]),
                "true_label": lbl,
                "true_label_name": "Illicit" if lbl == 1 else ("Licit" if lbl == 0 else "Unknown"),
                "is_camouflaged": bool(row.get("is_camouflaged", False)),
                "risk_tier": str(row["risk_tier"]),
                "prob_hybrid": round(float(row["prob_hybrid"]), 4),
                "prob_gnn": round(float(row["prob_gnn"]), 4),
                "prob_topology_lgbm": round(float(row["prob_topology_lgbm"]), 4),
                "prob_tabular": round(float(row["prob_tabular"]), 4),
                "gnn_uplift": round(float(row["gnn_uplift"]), 4),
                "tx_amount_btc": round(float(row["tx_amount_btc"]), 4),
                "hourly_tx_velocity": round(float(row["hourly_tx_velocity"]), 2),
                "in_degree": int(row["graph_in_degree"]),
                "out_degree": int(row["graph_out_degree"]),
            }
        )

    available_typologies = sorted(df[df["timestep"] >= 35]["typology"].unique().tolist())
    return {
        "count": len(records),
        "available_typologies": available_typologies,
        "alerts": records,
    }


@app.get("/api/transaction/{identifier}", tags=["Forensics & Explainability"])
def get_transaction_dossier(identifier: str) -> Dict[str, object]:
    """
    Generate a full XAI Forensic Dossier for a transaction (`tx_id` or `node_idx`),
    including TreeSHAP feature attributions, 2-hop ego-subgraph, and FinCEN SAR narrative.
    """
    art = get_artifacts()
    try:
        return explain_transaction_node(art, identifier, top_k=12)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/subgraph/{identifier}", tags=["Forensics & Explainability"])
def get_transaction_subgraph(
    identifier: str,
    hops: int = Query(default=2, ge=1, le=3),
    max_nodes: int = Query(default=36, ge=5, le=80),
) -> Dict[str, object]:
    """Return the k-hop directed ego-network around `identifier` with GNN edge attention weights."""
    art = get_artifacts()
    try:
        return extract_ego_subgraph(art, identifier, hops=hops, max_nodes=max_nodes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/simulate", tags=["Live Interactive Simulator"])
def simulate_transaction(payload: SimulationRequest) -> Dict[str, object]:
    """
    Score a custom 'What-If' transaction & ego-network configuration in real time
    comparing Tabular Baseline vs. Topology-Augmented vs. Hybrid GraphSAGE+LightGBM.
    """
    art = get_artifacts()
    df = art.nodes_df

    # Use only earlier training windows for the scenario reference profiles.
    reference = df[df["timestep"] <= 30].reset_index(drop=True)
    base_medians = reference[art.all_feature_cols].median().to_dict()

    io_ratio = float(payload.num_inputs) / max(1.0, float(payload.num_outputs))
    tot_deg = float(payload.graph_in_degree + payload.graph_out_degree)
    in_out_r = (float(payload.graph_in_degree) + 0.5) / (float(payload.graph_out_degree) + 0.5)
    flow_asym = abs(float(payload.graph_in_degree - payload.graph_out_degree)) / (tot_deg + 1.0)

    overrides = {
        "tx_amount_btc": payload.tx_amount_btc,
        "tx_fee_btc": round(payload.tx_amount_btc * payload.fee_to_amount_ratio, 6),
        "fee_to_amount_ratio": payload.fee_to_amount_ratio,
        "num_inputs": float(payload.num_inputs),
        "num_outputs": float(payload.num_outputs),
        "input_output_ratio": io_ratio,
        "utxo_age_hours": payload.utxo_age_hours,
        "time_since_last_tx_min": max(1.0, payload.utxo_age_hours * 1.4),
        "output_entropy": 0.25 if payload.num_outputs >= 6 else 0.65,
        "round_amount_indicator": 1.0 if abs(payload.tx_amount_btc - round(payload.tx_amount_btc)) < 0.05 else 0.0,
        "cross_exchange_flag": 1.0 if payload.nbr_1hop_mean_coinjoin > 0.45 else 0.0,
        "hourly_tx_velocity": payload.hourly_tx_velocity,
        "coinjoin_similarity_score": payload.coinjoin_similarity_score,
        "burst_activity_zscore": payload.burst_activity_zscore,
        "graph_in_degree": float(payload.graph_in_degree),
        "graph_out_degree": float(payload.graph_out_degree),
        "graph_total_degree": tot_deg,
        "graph_in_out_ratio": in_out_r,
        "graph_flow_asymmetry": flow_asym,
        "graph_pagerank": min(6.0, 0.5 + tot_deg * 0.22),
        "graph_clustering_coeff": 0.48 if payload.graph_in_cycle_indicator else 0.08,
        "graph_core_number": min(6.0, max(1.0, tot_deg * 0.45)),
        "graph_in_cycle_indicator": float(payload.graph_in_cycle_indicator),
        "nbr_1hop_mean_amount": payload.tx_amount_btc * 0.9,
        "nbr_1hop_max_velocity": payload.nbr_1hop_max_velocity,
        "nbr_1hop_mean_fee_ratio": payload.fee_to_amount_ratio,
        "nbr_1hop_mean_coinjoin": payload.nbr_1hop_mean_coinjoin,
        "nbr_1hop_mean_burst_zscore": (payload.burst_activity_zscore + payload.nbr_2hop_mean_burst_zscore) * 0.5,
        "nbr_1hop_min_utxo_age": min(payload.utxo_age_hours, 4.0 if payload.nbr_1hop_max_velocity > 14 else 24.0),
        "nbr_1hop_unknown_ratio": min(0.85, payload.nbr_1hop_mean_coinjoin * 0.8),
        "nbr_2hop_mean_velocity": payload.nbr_1hop_max_velocity * 0.78,
        "nbr_2hop_mean_coinjoin": payload.nbr_1hop_mean_coinjoin * 0.85,
        "nbr_2hop_mean_burst_zscore": payload.nbr_2hop_mean_burst_zscore,
        "nbr_2hop_max_degree": max(tot_deg + 3.0, 8.0),
    }
    base_medians.update(overrides)

    x_local = np.array([[base_medians[c] for c in art.local_cols]], dtype=np.float32)
    x_full = np.array([[base_medians[c] for c in art.all_feature_cols]], dtype=np.float32)

    topo_1hop_cols = art.local_cols + [
        "graph_in_degree",
        "graph_out_degree",
        "graph_total_degree",
        "graph_in_out_ratio",
        "graph_flow_asymmetry",
        "graph_pagerank",
        "graph_clustering_coeff",
        "nbr_1hop_mean_amount",
        "nbr_1hop_max_velocity",
    ]
    x_topo_1hop = np.array([[base_medians[c] for c in topo_1hop_cols]], dtype=np.float32)

    prob_tab = float(art.lgb_tabular_model.predict_proba(x_local)[0, 1])
    prob_topo = float(art.lgb_topo_model.predict_proba(x_topo_1hop)[0, 1])

    # Find nearest structural archetype in the graph to approximate GNN message-passing embedding
    topo_slice_cols = [
        "graph_in_degree",
        "graph_out_degree",
        "graph_in_cycle_indicator",
        "nbr_1hop_mean_coinjoin",
        "nbr_1hop_max_velocity",
        "nbr_2hop_mean_burst_zscore",
    ]
    query_vec = np.array([base_medians[c] for c in topo_slice_cols], dtype=np.float32)
    graph_vecs = reference[topo_slice_cols].values.astype(np.float32)
    std_vecs = np.maximum(graph_vecs.std(axis=0), 1e-4)
    dists = np.linalg.norm((graph_vecs - query_vec) / std_vecs, axis=1)
    nearest_idx = int(np.argmin(dists))

    nearest_node = reference.iloc[nearest_idx]
    nearest_emb = art.gnn_embeddings[[int(nearest_node["node_idx"])]].astype(np.float32)
    prob_gnn = float(nearest_node["prob_gnn"])

    x_hybrid = np.hstack([x_full, nearest_emb]).astype(np.float32)
    lgb_hyb_raw = float(art.lgb_hybrid_model.predict_proba(x_hybrid)[0, 1])
    prob_hybrid = 0.55 * lgb_hyb_raw + 0.25 * prob_gnn + 0.20 * prob_topo

    # TreeSHAP attributions for the simulated transaction
    shap_contribs = art.lgb_hybrid_model.predict(x_hybrid, pred_contrib=True)[0]
    feature_shap = shap_contribs[: len(art.all_feature_cols)]
    top_drivers = sorted(
        [
            {
                "feature": col,
                "display_name": FEATURE_HUMAN_LABELS.get(col, col),
                "value": round(float(base_medians[col]), 4),
                "shap_log_odds": round(float(sval), 4),
            }
            for col, sval in zip(art.all_feature_cols, feature_shap)
        ],
        key=lambda d: abs(float(d["shap_log_odds"])),
        reverse=True,
    )[:8]

    risk_tier = (
        "CRITICAL"
        if prob_hybrid >= 0.82
        else ("HIGH" if prob_hybrid >= 0.60 else ("ELEVATED" if prob_hybrid >= 0.30 else "LOW"))
    )

    return {
        "risk_tier": risk_tier,
        "scores": {
            "tabular_baseline_prob": round(prob_tab, 4),
            "topology_lgbm_prob": round(prob_topo, 4),
            "deep_gnn_prob": round(prob_gnn, 4),
            "hybrid_ensemble_prob": round(prob_hybrid, 4),
            "gnn_uplift_over_tabular": round(prob_hybrid - prob_tab, 4),
        },
        "matched_graph_archetype": str(nearest_node["typology"]),
        "top_shap_drivers": top_drivers,
    }
