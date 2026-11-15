"""
End-to-End Multi-Model Training, Evaluation, & Temporal Drift Pipeline.

Trains and benchmarks 4 architectures on strictly temporal splits (Timesteps 1..30 Train, 31..34 Val, 35..49 Test):
1. Tabular Baseline (LightGBM on isolated local transaction features)
2. Topology-Augmented GBDT (LightGBM on local + nine graph features)
3. Deep Graph Neural Network (Directed GraphSAGE + Edge-Attention GAT with Focal Loss)
4. Hybrid GraphSAGE + LightGBM Ensemble (64-D GNN embeddings + Topological + Local features)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data.graph_features import TOPOLOGICAL_FEATURE_NAMES, extract_topological_features
from src.data.graph_generator import GraphDatasetBundle, load_or_generate_dataset
from src.models.gnn import CryptoAMLGraphNet, train_gnn_model


@dataclass
class PipelineArtifacts:
    """In-memory and serializable artifacts from the trained AML pipeline."""

    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    local_cols: List[str]
    topo_cols: List[str]
    all_feature_cols: List[str]
    gnn_embeddings: np.ndarray
    edge_attention: np.ndarray
    model_metrics: Dict[str, Dict[str, float]]
    temporal_drift_records: List[Dict[str, object]]
    feature_importances: List[Dict[str, object]]
    typology_recall_comparison: List[Dict[str, object]]
    gnn_history: Dict[str, list]
    lgb_tabular_model: object
    lgb_topo_model: object
    lgb_hybrid_model: object
    gnn_model: CryptoAMLGraphNet
    scaler: StandardScaler
    dataset_metadata: Dict[str, object]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    is_camouflaged: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute comprehensive AML fraud metrics on imbalanced test data."""
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
        best_idx = int(np.argmax(f1s))
        threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Precision in the first 100 alerts; recall in a 10% review budget.
    top_100_idx = np.argsort(y_prob)[::-1][: min(100, len(y_true))]
    precision_at_100 = float(y_true[top_100_idx].mean())
    k_budget = max(1, int(np.ceil(0.10 * len(y_true))))
    top_k_idx = np.argsort(y_prob)[::-1][:k_budget]
    recall_at_k = float(y_true[top_k_idx].sum() / max(1, y_true.sum()))

    # Camouflaged Mule Recall (how many locally disguised illicit nodes were caught)
    camo_recall = 0.0
    if is_camouflaged is not None and is_camouflaged.sum() > 0:
        camo_mask = (y_true == 1) & is_camouflaged
        if camo_mask.sum() > 0:
            camo_recall = float(y_pred[camo_mask].mean())

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "precision_at_100": round(precision_at_100, 4),
        "recall_at_top10pct": round(recall_at_k, 4),
        "camouflaged_mule_recall": round(camo_recall, 4),
        "optimal_threshold": round(threshold, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _find_optimal_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    if len(thresholds) == 0:
        return 0.5
    f1s = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
    return float(thresholds[int(np.argmax(f1s))])


def run_training_pipeline(
    artifacts_dir: Optional[Path] = None,
    force_retrain: bool = False,
) -> PipelineArtifacts:
    """
    Execute or load the complete 4-model AML training and evaluation pipeline.
    """
    base_dir = Path(__file__).resolve().parents[2]
    artifacts_dir = artifacts_dir or (base_dir / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle_cache_path = artifacts_dir / "pipeline_v2.pkl"

    if not force_retrain and bundle_cache_path.exists():
        try:
            with open(bundle_cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    t0 = time.perf_counter()
    # The public benchmark uses the seeded synthetic graph. Keep optional external CSVs separate.
    dataset: GraphDatasetBundle = load_or_generate_dataset(
        raw_dir=base_dir / "data" / "synthetic-input", force_regenerate=force_retrain
    )
    edges = dataset.edges_df
    timesteps = dataset.nodes_df["timestep"].to_numpy()
    if np.any(timesteps[edges["src_idx"]] != timesteps[edges["dst_idx"]]):
        raise ValueError("The snapshot benchmark requires edges within a single timestep.")

    # Step 1: Extract topological graph features
    df, topo_cols = extract_topological_features(
        dataset.nodes_df,
        dataset.edges_df,
        dataset.local_feature_cols,
    )
    local_cols = dataset.local_feature_cols
    all_feature_cols = local_cols + topo_cols

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    y_all = df["label"].values
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]
    camo_test = df["is_camouflaged"].values[test_mask] if "is_camouflaged" in df.columns else None

    # Step 2: Model 1 — Tabular Baseline (LightGBM on local transaction features ONLY)
    X_local = df[local_cols].values.astype(np.float32)
    lgb_tabular = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=6,
        subsample_for_bin=10000,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgb_tabular.fit(
        X_local[train_mask],
        y_train,
        eval_set=[(X_local[val_mask], y_val)],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    tab_probs_all = lgb_tabular.predict_proba(X_local)[:, 1]
    tab_thresh = _find_optimal_threshold(y_val, tab_probs_all[val_mask])
    tab_metrics = compute_classification_metrics(y_test, tab_probs_all[test_mask], camo_test, threshold=tab_thresh)

    # Step 3: Model 2 — Topology-Augmented LightGBM (Local + 1-Hop Structural Graph Features)
    topo_1hop_cols = local_cols + [
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
    X_topo_1hop = df[topo_1hop_cols].values.astype(np.float32)
    X_topo = df[all_feature_cols].values.astype(np.float32)
    lgb_topo = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=23,
        max_depth=5,
        colsample_bytree=0.80,
        subsample=0.80,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgb_topo.fit(
        X_topo_1hop[train_mask],
        y_train,
        eval_set=[(X_topo_1hop[val_mask], y_val)],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    topo_probs_all = lgb_topo.predict_proba(X_topo_1hop)[:, 1]
    topo_thresh = _find_optimal_threshold(y_val, topo_probs_all[val_mask])
    topo_metrics = compute_classification_metrics(y_test, topo_probs_all[test_mask], camo_test, threshold=topo_thresh)

    # Step 4: Model 3 — Deep GraphSAGE + Edge-Attention GAT (PyTorch end-to-end over local node attributes + graph edges)
    scaler = StandardScaler()
    scaler.fit(X_local[train_mask])
    X_local_scaled = np.clip(scaler.transform(X_local), -6.0, 6.0).astype(np.float32)

    edge_index_np = np.vstack(
        [
            dataset.edges_df["src_idx"].values.astype(np.int64),
            dataset.edges_df["dst_idx"].values.astype(np.int64),
        ]
    )

    gnn_model, gnn_probs_all, gnn_embeddings, edge_attn_np, gnn_history = train_gnn_model(
        features=X_local_scaled,
        edge_index_np=edge_index_np,
        labels=y_all,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=90,
        lr=0.0045,
    )
    gnn_thresh = _find_optimal_threshold(y_val, gnn_probs_all[val_mask])
    gnn_metrics = compute_classification_metrics(y_test, gnn_probs_all[test_mask], camo_test, threshold=gnn_thresh)

    # Step 5: Model 4 — Flagship Hybrid GraphSAGE Embeddings + Topological + Local LightGBM Ensemble
    embed_cols = [f"gnn_emb_{i:02d}" for i in range(gnn_embeddings.shape[1])]
    X_hybrid = np.hstack([X_topo, gnn_embeddings]).astype(np.float32)

    lgb_hybrid = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=7,
        colsample_bytree=0.80,
        subsample=0.85,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgb_hybrid.fit(
        X_hybrid[train_mask],
        y_train,
        eval_set=[(X_hybrid[val_mask], y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    lgb_hybrid_raw = lgb_hybrid.predict_proba(X_hybrid)[:, 1]
    # Fixed weighted blend of the three models; its threshold is picked on validation.
    hybrid_probs_all = 0.55 * lgb_hybrid_raw + 0.25 * gnn_probs_all + 0.20 * topo_probs_all
    hybrid_thresh = _find_optimal_threshold(y_val, hybrid_probs_all[val_mask])
    hybrid_metrics = compute_classification_metrics(
        y_test, hybrid_probs_all[test_mask], camo_test, threshold=hybrid_thresh
    )

    # Store predictions & risk tiers on nodes_df
    df["prob_tabular"] = np.round(tab_probs_all, 4)
    df["prob_topology_lgbm"] = np.round(topo_probs_all, 4)
    df["prob_gnn"] = np.round(gnn_probs_all, 4)
    df["prob_hybrid"] = np.round(hybrid_probs_all, 4)
    df["pred_hybrid"] = (hybrid_probs_all >= hybrid_thresh).astype(int)

    df["risk_tier"] = pd.cut(
        df["prob_hybrid"],
        bins=[-0.01, 0.30, 0.60, 0.82, 1.01],
        labels=["LOW", "ELEVATED", "HIGH", "CRITICAL"],
    ).astype(str)

    # Store edge attention weights on edges_df
    edges_with_attn = dataset.edges_df.copy()
    edges_with_attn["attention_weight"] = np.round(edge_attn_np, 4)

    # Step 6: Track per-window performance and the measured feature shift.
    coinjoin_scores = df["coinjoin_similarity_score"].to_numpy()
    train_mean = coinjoin_scores[timesteps <= 30].mean()
    train_std = max(float(coinjoin_scores[timesteps <= 30].std()), 1e-8)
    temporal_drift_records: List[Dict[str, object]] = []
    for ts in sorted(df["timestep"].unique()):
        ts_mask = (df["timestep"].values == ts) & (df["label"].values != -1)
        if ts_mask.sum() < 5 or (df["label"].values[ts_mask] == 1).sum() == 0:
            continue
        y_ts = df["label"].values[ts_mask]
        pr_tab = float(average_precision_score(y_ts, tab_probs_all[ts_mask]))
        pr_gnn = float(average_precision_score(y_ts, gnn_probs_all[ts_mask]))
        pr_hyb = float(average_precision_score(y_ts, hybrid_probs_all[ts_mask]))
        rec_hyb = float(recall_score(y_ts, (hybrid_probs_all[ts_mask] >= hybrid_thresh).astype(int), zero_division=0))
        rec_tab = float(recall_score(y_ts, (tab_probs_all[ts_mask] >= tab_thresh).astype(int), zero_division=0))

        mean_shift = abs(float(coinjoin_scores[timesteps == ts].mean()) - train_mean) / train_std

        temporal_drift_records.append(
            {
                "timestep": int(ts),
                "split": "Train" if ts <= 30 else ("Val" if ts <= 34 else "Test"),
                "is_drift_regime": bool(ts >= 43),
                "num_labeled": int(ts_mask.sum()),
                "num_illicit": int((y_ts == 1).sum()),
                "tabular_pr_auc": round(pr_tab, 4),
                "gnn_pr_auc": round(pr_gnn, 4),
                "hybrid_pr_auc": round(pr_hyb, 4),
                "tabular_recall": round(rec_tab, 4),
                "hybrid_recall": round(rec_hyb, 4),
                "coinjoin_mean_shift_std": round(float(mean_shift), 4),
            }
        )

    # Step 7: Compute Recall by Money Laundering Typology on Test Set
    typology_recall_comparison: List[Dict[str, object]] = []
    test_illicit_df = df[test_mask & (df["label"] == 1)]
    for typo, grp in test_illicit_df.groupby("typology"):
        idx_t = grp.index.values
        typology_recall_comparison.append(
            {
                "typology": str(typo),
                "count": int(len(idx_t)),
                "tabular_recall": round(float((tab_probs_all[idx_t] >= tab_thresh).mean()), 4),
                "topology_lgbm_recall": round(float((topo_probs_all[idx_t] >= topo_thresh).mean()), 4),
                "gnn_recall": round(float((gnn_probs_all[idx_t] >= gnn_thresh).mean()), 4),
                "hybrid_recall": round(float((hybrid_probs_all[idx_t] >= hybrid_thresh).mean()), 4),
            }
        )

    # Step 8: Extract Top Feature Importances (from Hybrid LightGBM over named features)
    named_importances = lgb_hybrid.feature_importances_[: len(all_feature_cols)].astype(np.float64)
    named_importances = named_importances / max(1.0, named_importances.sum())
    feature_importances = [
        {
            "feature": col,
            "category": "Topological Graph Feature" if col in TOPOLOGICAL_FEATURE_NAMES else "Local Transaction Feature",
            "importance": round(float(imp), 4),
        }
        for col, imp in sorted(zip(all_feature_cols, named_importances), key=lambda x: x[1], reverse=True)
    ]

    elapsed = round(time.perf_counter() - t0, 2)
    model_metrics = {
        "Tabular_LightGBM": tab_metrics,
        "Topology_LightGBM": topo_metrics,
        "Deep_GraphSAGE_GAT": gnn_metrics,
        "Hybrid_GraphSAGE_LightGBM": hybrid_metrics,
    }

    dataset_metadata = {
        **dataset.metadata,
        "dataset_source": dataset.dataset_source,
        "synthetic": True,
        "training_time_sec": elapsed,
        "train_nodes": int(train_mask.sum()),
        "val_nodes": int(val_mask.sum()),
        "test_nodes": int(test_mask.sum()),
    }

    artifacts = PipelineArtifacts(
        nodes_df=df,
        edges_df=edges_with_attn,
        local_cols=local_cols,
        topo_cols=topo_cols,
        all_feature_cols=all_feature_cols,
        gnn_embeddings=gnn_embeddings,
        edge_attention=edge_attn_np,
        model_metrics=model_metrics,
        temporal_drift_records=temporal_drift_records,
        feature_importances=feature_importances,
        typology_recall_comparison=typology_recall_comparison,
        gnn_history=gnn_history,
        lgb_tabular_model=lgb_tabular,
        lgb_topo_model=lgb_topo,
        lgb_hybrid_model=lgb_hybrid,
        gnn_model=gnn_model,
        scaler=scaler,
        dataset_metadata=dataset_metadata,
    )

    temporary_cache = bundle_cache_path.with_name(f".{bundle_cache_path.name}.{os.getpid()}.tmp")
    with temporary_cache.open("wb") as f:
        pickle.dump(artifacts, f)
    temporary_cache.replace(bundle_cache_path)

    # Also save a human-readable JSON summary of metrics in artifacts/benchmark_summary.json
    summary_payload = {
        "dataset_metadata": dataset_metadata,
        "model_metrics": model_metrics,
        "typology_recall_comparison": typology_recall_comparison,
        "top_15_features": feature_importances[:15],
    }
    with open(artifacts_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return artifacts


if __name__ == "__main__":
    from src.models.pipeline import run_training_pipeline as train

    art = train(force_retrain=True)
    print(json.dumps(art.model_metrics, indent=2))
