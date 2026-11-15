# Synthetic transaction graph

The files in [processed/](processed/) were generated with the fixed seed in [graph_generator.py](../src/data/graph_generator.py). All values, transaction IDs, wallet IDs, and labels are simulated by the generator. The public benchmark uses these files. The code also contains an optional CSV loader intended for separate experiments.

| File | Contents |
| --- | --- |
| `aml_nodes.parquet` | 8,838 transactions, the window number, 18 local features, and generated labels and typologies |
| `aml_edges.parquet` | 18,939 directed edges linking node indices within individual windows |
| `aml_metadata.json` | Graph size, class counts, and synthetic regime-shift window |

Label `1` denotes a simulated illicit transaction, `0` a simulated licit transaction, and `-1` an unlabeled node. The masks use windows 1–30 for training (4,432 labeled), 31–34 for validation (602), and 35–49 for evaluation (2,210). Unlabeled nodes remain in the graph but are excluded from loss and metrics.

To regenerate the files and fit the four models, run `python -m src.models.pipeline` from the repository root.
