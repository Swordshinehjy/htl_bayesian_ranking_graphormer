# HTL Bayesian Personalized Ranking

Perovskite Hole Transport Layer (HTL) ranking prediction model based on Graphormer and Bayesian Personalized Ranking.

## Project Overview

This project uses deep learning methods to predict the performance ranking of hole transport layer materials in perovskite solar cells. It adopts a Graphormer architecture combined with Bayesian Personalized Ranking (Bradley-Terry model), encoding molecular structures through Graph Transformer and learning material performance ranking via probabilistic pairwise comparison.

### Model Architecture

```
SMILES → GraphormerEncoder → mol_emb [H] ─┐
extra_features [E] ───────────────────────┼─ concat → FFN → score
global_features [G] ──────────────────────┘
```

- **GraphormerEncoder**: Graph Transformer network with spatial encoding, degree encoding, and edge encoding for molecular structure representation
- **extra_features**: Molecule-level additional features (e.g., HOMO, TPSA, MolLogP, etc.)
- **global_features**: Global features (e.g., MO_ITO), shared per sample pair

### Graphormer Encoding Features

- **Centrality Encoding**: Degree-based atom feature projection
- **Spatial Encoding**: Shortest path distance bias for each attention head
- **Edge Encoding**: Mean bond features along shortest paths
- **Virtual Node Readout**: [CLS] node or mean/sum aggregation for graph-level representation

## Feature Description

### Molecular Features (EXTRA_COLS)

Additional features for each molecule (with `_1` or `_2` suffix):

| Feature Name     | Description                  |
| ---------------- | ---------------------------- |
| Alkyl            | Alkyl chain length           |
| TailSym          | Tail symmetry                |
| TailPlanarity    | Tail planarity               |
| NumHAcceptors    | Number of hydrogen bond acceptors |
| NumHDonors       | Number of hydrogen bond donors    |
| TPSA             | Topological polar surface area    |
| MolLogP          | Lipid-water partition coefficient |
| HOMO             | Highest occupied molecular orbital energy (eV) |
| dipole           | Dipole moment (Debye)        |
| MPI              | Molecular polarity index     |
| surface\_min/max | Molecular surface ESP minimum/maximum |
| PSA              | Polar surface area           |

### Global Features (GLOBAL_COLS)

| Feature Name | Description                        |
| ------------ | ---------------------------------- |
| MO\_ITO      | ITO substrate treatment identifier (0/1) |

## Install Dependencies

```bash
pip install torch rdkit pandas numpy scikit-learn scipy
```

## Data Format

### Training Data Format (htl-data-combinations.csv)

```csv
doi,MO_ITO,mol_1,SMILES_1,Alkyl_1,...,PCE_1,mol_2,SMILES_2,Alkyl_2,...,PCE_2
```

- Each row contains a pair of materials and their features
- `PCE_1` and `PCE_2` are target values (power conversion efficiency)

### Prediction Data Format

**Pairwise Prediction**: Same as training data format, no PCE column needed

**List Ranking**: Single material list

```csv
Materials,SMILES,Alkyl,TailSym,...,MO_ITO
```

## Usage

### Training the Model

```bash
python htl_ranking_graphormer.py --mode train --csv htl-data-combinations.csv
```

Optional parameters:

- `--epochs`: Number of training epochs (default 1000)
- `--batch_size`: Batch size (default 32)
- `--lr`: Learning rate (default 5e-4)
- `--hidden_size`: Hidden layer dimension (default 300)
- `--num_heads`: Number of attention heads (default 6)
- `--depth`: Transformer depth (default 3)
- `--dropout`: Dropout rate (default 0.1)
- `--patience`: Early stopping patience (default 50)
- `--max_degree`: Maximum degree for centrality encoding (default 20)
- `--max_dist`: Maximum distance for spatial encoding (default 10)
- `--split`: Data split method (random/group, default random)
- `--n_cv_folds`: Number of cross-validation folds (for group split)

### Fine-tuning the Model

```bash
python htl_ranking_graphormer.py --mode finetune \
    --csv htl-data-combinations.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### Pairwise Prediction

```bash
python htl_ranking_graphormer.py --mode predict \
    --predict_csv htl-new.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

### List Ranking Prediction

```bash
python htl_ranking_graphormer.py --mode list_rank \
    --predict_csv ranking-new.csv \
    --checkpoint checkpoints/final_model.pt \
    --output ranked_results.csv
```

### Explanation Analysis

```bash
python htl_ranking_graphormer.py --mode explain \
    --explain_csv ranking-new.csv \
    --checkpoint checkpoints/final_model.pt \
    --explain_dir explain_output \
    --n_steps 100
```

### Differential Attribution Analysis

```bash
python htl_ranking_graphormer.py --mode diff_attr \
    --diff_csv htl-new.csv \
    --checkpoint checkpoints/final_model.pt \
    --diff_dir diff_output \
    --n_steps 100
```

## Model Configuration

### ModelConfig

| Parameter      | Default | Description              |
| -------------- | ------- | ------------------------ |
| hidden\_size   | 300     | Hidden layer dimension   |
| depth          | 3       | Transformer depth        |
| num\_heads     | 6       | Number of attention heads |
| dropout        | 0.1     | Dropout rate             |
| ffn\_hidden    | 256     | FFN hidden layer dimension |
| max\_degree    | 20      | Maximum degree for centrality encoding |
| max\_dist      | 10      | Maximum distance for spatial encoding |
| extra\_dim     | 13      | Additional feature dimension |
| global\_dim    | 1       | Global feature dimension |
| num\_tasks     | 1       | Number of tasks          |
| aggregation    | mean    | Graph aggregation method |

### TrainingConfig

| Parameter       | Default | Description                   |
| --------------- | ------- | ----------------------------- |
| epochs          | 1000    | Number of training epochs     |
| batch\_size     | 32      | Batch size                    |
| lr              | 5e-4    | Learning rate                 |
| weight\_decay   | 1e-5    | Weight decay                  |
| patience        | 50      | Early stopping patience       |
| early\_stop\_warmup | 100 | Warmup epochs before early stopping |
| val\_ratio      | 0.1     | Validation set ratio          |
| test\_ratio     | 0.1     | Test set ratio                |
| split           | random  | Data split method (random/group) |
| n\_cv\_folds    | 5       | Number of CV folds (for group split) |

## Loss Function

The model uses a Bayesian Personalized Ranking Loss (Bradley-Terry sigmoid model) combined with regression loss:

1. **Bayesian Personalized Ranking Loss**: `L_bayes = -log sigma(sign(y1-y2) * (s1-s2))`
   - Based on Bradley-Terry model: `P(i > j) = sigma(s1 - s2)`
   - Provides natural probability calibration
   - No margin hyperparameter needed
   - Smooth gradients for stable training

2. **Delta Regression Loss**: `L_reg = MSE(s1-s2, y1-y2)`

Total loss: `L = L_bayes + L_reg`

### Advantages of Bayesian Personalized Ranking

- **Probabilistic interpretation**: Output represents ranking probability
- **Natural calibration**: Sigmoid function provides well-calibrated probabilities
- **Smooth gradients**: No zero-gradient issues compared to ReLU-based margin loss
- **No margin tuning**: Eliminates need for margin hyperparameter selection

## Evaluation Metrics

- **Pairwise Accuracy**: Pairwise ranking accuracy
- **Spearman ρ**: Spearman correlation coefficient
- **Test Metrics**: Comprehensive metrics logged during training

## Output Files

- `checkpoints/best_model.pt`: Best model on validation set
- `checkpoints/final_model.pt`: Final model
- `predictions.csv`: Prediction results
- `ranked_results.csv`: Ranking results
- `explain_output/`: Explanation analysis results
- `diff_output/`: Differential attribution results

## Project Structure

```
htl_bayesian_ranking_graphormer/
├── htl_ranking_graphormer.py    # Main program
├── htl_package/                 # Core modules
│   ├── configs.py               # Configuration classes
│   ├── training.py              # Training and fine-tuning
│   ├── prediction.py            # Prediction and ranking
│   └── explainer.py             # IGExplainer for attribution
├── htl-data-combinations.csv    # Training data
├── htl-new.csv                  # Prediction data
├── ranking-new.csv              # Ranking data
├── checkpoints/                 # Model checkpoints
│   ├── best_model.pt
│   └── final_model.pt
├── predictions.csv              # Prediction output
├── ranked_results.csv           # Ranking output
├── explain_output/              # Explanation results
└── diff_output/                 # Differential attribution results
```

## Python API Usage

```python
from htl_package.prediction import predict_batch, predict_list

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
)
```
