# HTL Pairwise Ranking

Perovskite Hole Transport Layer (HTL) ranking prediction model based on Graph Transformer.

## Project Overview

This project uses deep learning methods to predict the performance ranking of hole transport layer materials in perovskite solar cells. It adopts a siamese network architecture, combining molecular graph features and additional molecular descriptors, and learns material performance ranking through pairwise comparison.

### Model Architecture

```
SMILES → Graph Transformer → mol_emb [H] ─┐
extra_features [E] ───────────┼─ concat → FFN → score
global_features [G] ──────────┘
```

- **Graph Transformer**: Graph Transformer network for encoding molecular structures
- **extra_features**: Molecule-level additional features (e.g., HOMO, TPSA, MolLogP, etc.)
- **global_features**: Global features (e.g., MO_ITO), shared per sample pair

## Feature Description

### Molecular Features (EXTRA\_COLS)

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

### Global Features (GLOBAL\_COLS)

| Feature Name | Description                        |
| ------------ | ---------------------------------- |
| MO\_ITO      | ITO substrate treatment identifier (0/1) |

## Install Dependencies

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

**Pairwise Prediction**: Same as training data format, no PCE column needed

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

- `--epochs`: Number of training epochs (default 1000)
- `--batch_size`: Batch size (default 32)
- `--lr`: Learning rate (default 5e-4)
- `--hidden_size`: Hidden layer dimension (default 300)
- `--depth`: MPNN depth (default 6)
- `--dropout`: Dropout rate (default 0.1)
- `--patience`: Early stopping patience (default 50)

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

| Parameter      | Default | Description              |
| -------------- | ------- | ------------------------ |
| hidden\_size   | 300     | Hidden layer dimension   |
| depth          | 6       | Transformer depth        |
| dropout        | 0.1     | Dropout rate             |
| ffn\_hidden    | 256     | FFN hidden layer dimension |
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
| val\_ratio      | 0.1     | Validation set ratio          |
| test\_ratio     | 0.1     | Test set ratio                |
| margin          | 0.2     | Margin Ranking Loss margin    |

## Loss Function

The model uses a combined loss:

1. **Margin Ranking Loss**: `L_rank = ReLU(margin - sign(y1-y2) * (s1-s2))`
2. **Delta Regression Loss**: `L_reg = MSE(s1-s2, y1-y2)`

总损失: `L = α * L_rank + β * L_reg` (默认 α=0.6, β=0.4)

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
├── htl_ranking.py          # 主程序
├── htl-data-combinations.csv  # 训练数据
├── htl-new.csv             # 预测数据
├── ranking-new.csv         # 排序数据
├── checkpoints/            # 模型保存目录
│   ├── best_model.pt
│   └── final_model.pt
├── predictions.csv         # 预测输出
└── ranked_results.csv      # 排序输出
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

