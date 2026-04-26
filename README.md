# HTL Pairwise Ranking

A Graph Transformer-based model for predicting the ranking of Hole Transport Layer (HTL) materials in perovskite solar cells.

## Project Overview

This project uses deep learning methods to predict the performance ranking of hole transport layer materials in perovskite solar cells. It employs a siamese network architecture, combining molecular graph features and additional molecular descriptors to learn material performance ranking through pairwise comparisons.

### Model Architecture

```
SMILES → Graph Transformer → mol_emb [H] ─┐
extra_features [E] ───────────────────────┼─ concat → FFN → score
global_features [G] ──────────────────────┘
```

- **Graph Transformer**: Graph Transformer network for encoding molecular structure
- **extra_features**: Additional molecular-level features (e.g., HOMO, TPSA, MolLogP, etc.)
- **global_features**: Global features (e.g., MO_ITO), shared across each sample pair

## Feature Description

### Molecular Features (EXTRA_COLS)

Additional features for each molecule (with `_1` or `_2` suffix):

| Feature Name      | Description                              |
| ----------------- | ---------------------------------------- |
| Alkyl             | Length of connected alkyl chain          |
| TailSym           | Tail symmetry                            |
| TailPlanarity     | Tail planarity                           |
| NumHAcceptors     | Number of hydrogen bond acceptors        |
| NumHDonors        | Number of hydrogen bond donors           |
| TPSA              | Topological Polar Surface Area           |
| MolLogP           | Octanol-water partition coefficient      |
| HOMO              | Highest Occupied Molecular Orbital energy (eV) |
| dipole            | Dipole moment (Debye)                    |
| MPI               | Molecular Polarity Index                 |
| surface_min/max   | Minimum/maximum of molecular surface ESP |
| PSA               | Polar Surface Area                       |

### Global Features (GLOBAL_COLS)

| Feature Name | Description                         |
| ------------- | ----------------------------------- |
| MO_ITO       | ITO substrate treatment identifier (0/1) |

## Installation

```bash
pip install torch rdkit pandas numpy scikit-learn scipy chemprop>=2.0.0
```

## Data Format

### Training Data Format (htl-data-combinations.csv)

```csv
doi,MO_ITO,mol_1,SMILES_1,Alkyl_1,...,PCE_1,mol_2,SMILES_2,Alkyl_2,...,PCE_2
```

- Each row contains a pair of materials and their features
- `PCE_1` and `PCE_2` are target values (power conversion efficiency)

### Prediction Data Format

**Pairwise Prediction**: Same format as training data, no PCE columns needed

**List Ranking**: Single material list

```csv
Materials,SMILES,Alkyl,TailSym,...,MO_ITO
```

## Usage

### Training the Model

```bash
python htl_ranking.py --mode train --csv htl-data-combinations.csv
```

Optional parameters:

- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 5e-4)
- `--hidden_size`: Hidden layer dimension (default: 300)
- `--depth`: MPNN depth (default: 6)
- `--dropout`: Dropout rate (default: 0.1)
- `--patience`: Early stopping patience (default: 50)

### Fine-tuning the Model

```bash
python htl_ranking.py --mode finetune \
    --csv htl-data-combinations.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### Pairwise Prediction

```bash
python htl_ranking.py --mode predict \
    --predict_csv htl-new.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

### List Ranking Prediction

```bash
python htl_ranking.py --mode list_rank \
    --predict_csv ranking-new.csv \
    --checkpoint checkpoints/final_model.pt \
    --output ranked_results.csv
```

## Model Configuration

### ModelConfig

| Parameter    | Default | Description           |
| ------------ | ------- | --------------------- |
| hidden_size  | 300     | Hidden layer dimension |
| depth        | 6       | Transformer depth     |
| dropout      | 0.1     | Dropout rate          |
| ffn_hidden   | 256     | FFN hidden layer dimension |
| extra_dim    | 13      | Extra features dimension |
| global_dim   | 1       | Global features dimension |
| num_tasks    | 1       | Number of tasks       |
| aggregation  | mean    | Graph aggregation method |

### TrainingConfig

| Parameter      | Default | Description                     |
| ------------- | ------- | ------------------------------ |
| epochs        | 1000    | Number of training epochs       |
| batch_size    | 32      | Batch size                      |
| lr            | 5e-4    | Learning rate                   |
| weight_decay  | 1e-5    | Weight decay                    |
| patience      | 50      | Early stopping patience         |
| val_ratio     | 0.1     | Validation set ratio            |
| test_ratio    | 0.1     | Test set ratio                  |
| margin        | 0.2     | Margin Ranking Loss boundary    |

## Loss Function

The model uses a combined loss:

1. **Margin Ranking Loss**: `L_rank = ReLU(margin - sign(y1-y2) * (s1-s2))`
2. **Delta Regression Loss**: `L_reg = MSE(s1-s2, y1-y2)`

Total loss: `L = α * L_rank + β * L_reg` (default α=0.6, β=0.4)

## Evaluation Metrics

- **Pairwise Accuracy**: Pairwise ranking accuracy
- **Spearman ρ**: Spearman correlation coefficient

## Output Files

- `checkpoints/best_model.pt`: Best model on validation set
- `checkpoints/final_model.pt`: Final model
- `predictions.csv`: Prediction results
- `ranked_results.csv`: Ranking results

## Project Structure

```
htl_pairwise_ranking/
├── htl_ranking.py          # Main program
├── htl-data-combinations.csv  # Training data
├── htl-new.csv             # Prediction data
├── ranking-new.csv         # Ranking data
├── checkpoints/            # Model save directory
│   ├── best_model.pt
│   └── final_model.pt
├── predictions.csv         # Prediction output
└── ranked_results.csv      # Ranking output
```

## Python API Usage

```python
from htl_ranking import predict_pair, predict_batch, predict_list

# Single pair prediction
result = predict_pair(
    smiles_1="COc1ccc2c(c1)c1cc(OC)ccc1n2CCP(=O)(O)O",
    smiles_2="COc1ccc2c(c1)sc1c3ccc(CCCCP(=O)(O)O)cc3sc21",
    extra_raw_1=extra_features_1,
    extra_raw_2=extra_features_2,
    checkpoint_path="checkpoints/final_model.pt",
    global_feat=np.array([[1.0]]),  # MO_ITO=1
)

# Batch prediction
df_result = predict_batch(
    df_new=df,
    checkpoint_path="checkpoints/final_model.pt",
    output_path="predictions.csv",
)

# List ranking
df_ranked = predict_list(
    df_list=df,
    checkpoint_path="checkpoints/final_model.pt",
    output_path="ranked.csv",
    global_feat=np.array([[1.0]]),
)
```
