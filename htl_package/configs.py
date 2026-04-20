from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from .constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS


@dataclass
class ModelConfig:
    hidden_size: int = 300
    depth: int = 3
    num_heads: int = 6
    dropout: float = 0.0
    ffn_hidden: int = 256
    extra_dim: int = EXTRA_DIM
    global_dim: int = GLOBAL_DIM
    num_tasks: int = NUM_TASKS
    aggregation: str = "cls"
    max_degree: int = 15
    max_dist: int = 15
    auto_compute_stats: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{
            k: v
            for k, v in d.items() if k in cls.__dataclass_fields__
        })


@dataclass
class TrainingConfig:
    csv_path: str = "htl-data-combinations.csv"
    save_dir: str = "checkpoints"
    epochs: int = 1000
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 50
    early_stop_warmup: int = 20
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    cache_val_test: bool = True
    split: str = "random"
    n_cv_folds: Optional[int] = None


@dataclass
class FinetuneConfig:
    csv_path: str = "htl-data-combinations.csv"
    checkpoint_path: str = "checkpoints/best_model.pt"
    save_dir: str = "checkpoints"
    finetune_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-6
    seed: int = 42


@dataclass
class PredictConfig:
    predict_csv: str = ""
    checkpoint_path: str = "checkpoints/best_model.pt"
    output_path: str = "predictions.csv"
