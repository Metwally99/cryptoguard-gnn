# CryptoGuard GNN

[![Build and tests](https://github.com/Metwally99/cryptoguard-gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/Metwally99/cryptoguard-gnn/actions/workflows/ci.yml)

Graph machine learning project that tests whether transaction relationships improve the ranking of suspicious activity. It compares a tabular baseline, engineered graph features, a PyTorch graph neural network, and a LightGBM ensemble on a controlled synthetic benchmark.

**[Open the public demo](https://metwally99.github.io/cryptoguard-gnn/)** · [Benchmark JSON](artifacts/benchmark_summary.json) · [Synthetic data and split](data/README.md)

In the demo, select a held-out transaction to compare all four model scores, inspect its two-hop neighborhood, and see which features drove the LightGBM component. The browser replays saved predictions. The FastAPI deployment supports the same investigation workflow and a what-if simulator.

## Results

The test set contains 2,210 labeled transactions, including 679 positives, from windows 35–49. Thresholds were chosen separately for each model to maximize F1 on windows 31–34.

| Model | Average precision ↑ | F1 ↑ | Precision ↑ | Recall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Tabular LightGBM | 0.7796 | 0.6567 | 0.8194 | 0.5479 |
| LightGBM with graph features | 0.9725 | 0.9112 | 0.9079 | 0.9146 |
| Directed GraphSAGE with edge attention | 0.9882 | 0.9474 | 0.9663 | 0.9293 |
| **Hybrid GNN and LightGBM** | **0.9953** | **0.9631** | **0.9645** | **0.9617** |

Average precision improved by **21.57 percentage points** over the tabular baseline. The nine-feature graph baseline explains most of that gain. At their validation thresholds, recall on the synthetic subgroup with ordinary-looking local features rose from **13.18% to 93.12%**. The hybrid model flagged 677 test transactions, including 653 true positives and 24 false positives. These values describe this simulation and seed.

## Methodology

1. **Data.** A seeded generator creates 8,838 synthetic transactions and 18,939 directed edges across 49 independent graph snapshots. The modeled distributions change at window 43. The repository includes the generated data, and the training command can regenerate it.
2. **Split.** Train on windows 1–30, tune on 31–34, and evaluate on 35–49. Graph edges stay inside a window and PageRank normalizes inside each snapshot. The scaler fits on training samples. Unlabeled nodes contribute graph structure but have no supervised loss or evaluation score.
3. **Models.** The tabular model uses 18 local attributes. The topology model adds nine graph features. Directed GraphSAGE combines incoming and outgoing neighbors through sparse mean aggregation, then edge attention weights neighboring messages. Focal loss downweights easy examples during training. The hybrid uses a 64-dimensional embedding and 19 structural features alongside local inputs, then combines the hybrid tree, standalone GNN, and topology model scores with fixed weights.
4. **Evaluation.** Average precision summarizes how well the model ranks positives across thresholds. F1 combines precision and recall at the validation-selected threshold. The four-model comparison is an ablation of feature representations and model families. See the [training pipeline](src/models/pipeline.py) for fitting and thresholds and the [graph model](src/models/gnn.py) for message passing.

### Scope

The generator deliberately embeds patterns that the graph models can recover. Results measure detection of those simulated patterns. A real transaction evaluation would need point-in-time graph construction, independent labels, and a new out-of-time benchmark. The experiment scores a complete snapshot at the end of its window. TreeSHAP explains the hybrid LightGBM component's log-odds. Edge attention shows the model's learned weights. The what-if simulator estimates embeddings from similar training profiles.

## Run locally

**Docker** trains on CPU when building the image.

```bash
docker compose up --build
```

Open [dashboard](http://127.0.0.1:8000/) or [OpenAPI](http://127.0.0.1:8000/docs). The API exposes `/api/health`, `/api/overview`, `/api/alerts`, `/api/transaction/{id}`, and `/api/simulate`.

**Python 3.13** runs the same experiment locally from the repo root.

```bash
python -m pip install -r requirements.txt
python -m src.models.pipeline
python -m pytest -q
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Training regenerates the synthetic data and benchmark report. The app reads a model cache generated locally by training. To regenerate the static replay from that model, run `python -m scripts.export_replay`.
