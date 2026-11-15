"""Pydantic schemas for the FastAPI AML Graph Forensics endpoints."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Payload for real-time 'What-If' transaction & ego-network scoring."""

    tx_amount_btc: float = Field(default=2.45, ge=0.001, le=5000.0, description="Transaction amount in BTC")
    fee_to_amount_ratio: float = Field(default=0.0038, ge=0.00001, le=0.10, description="Fee to transaction volume ratio")
    num_inputs: int = Field(default=2, ge=1, le=100, description="Number of input UTXOs")
    num_outputs: int = Field(default=8, ge=1, le=100, description="Number of output UTXOs")
    utxo_age_hours: float = Field(default=2.1, ge=0.05, le=2000.0, description="Holding time of input UTXOs in hours")
    hourly_tx_velocity: float = Field(default=16.5, ge=0.1, le=100.0, description="Transactions per hour from originating wallet")
    coinjoin_similarity_score: float = Field(default=0.68, ge=0.0, le=1.0, description="CoinJoin / mixer structural similarity")
    burst_activity_zscore: float = Field(default=2.6, ge=-3.0, le=10.0, description="Temporal burst Z-score")
    graph_in_degree: int = Field(default=3, ge=0, le=200, description="Incoming payment edges in ego-graph")
    graph_out_degree: int = Field(default=9, ge=0, le=200, description="Outgoing payment edges in ego-graph")
    graph_in_cycle_indicator: int = Field(default=1, ge=0, le=1, description="1 if node participates in a directed cycle")
    nbr_1hop_mean_coinjoin: float = Field(default=0.62, ge=0.0, le=1.0, description="1-hop neighbor average mixer score")
    nbr_1hop_max_velocity: float = Field(default=19.4, ge=0.1, le=100.0, description="Max velocity among 1-hop neighbors")
    nbr_2hop_mean_burst_zscore: float = Field(default=2.3, ge=-3.0, le=10.0, description="2-hop syndicate burst Z-score")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    dataset_source: str
    num_nodes: int
    num_edges: int
    models_loaded: List[str]
