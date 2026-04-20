"""htl_package — Graphormer-based HTL prediction pipeline.

Convenience import hub: import all public symbols from submodules.
"""

from .constants import (
    EXTRA_COLS,
    EXTRA_DIM,
    GLOBAL_COLS,
    GLOBAL_DIM,
    TASK_NAMES,
    NUM_TASKS,
    logger,
    DEVICE,
)

from .configs import (
    ModelConfig,
    TrainingConfig,
    FinetuneConfig,
    PredictConfig,
)

from .features import (
    _one_hot,
    _atom_features,
    _bond_features,
    _extra_feat,
    _extra_feat_single,
    _global_feat,
    MolGraphData,
    featurize_mol,
    load_and_preprocess,
    compute_dataset_stats,
    ATOM_FDIM,
    BOND_FDIM,
)

from .datasets import (
    MolBatch,
    PairDataset,
    ListDataset,
    CachedPairDataset,
    DynamicPairDataset,
    collate_mol_graphs,
    collate_fn,
    collate_list_fn,
    collate_cached_batch,
)

from .models import (
    EdgeEncoding,
    GraphormerLayer,
    GraphormerEncoder,
    HTLRankingModel,
    BayesianRankingLoss,
    EarlyStopping,
)

from .checkpoint import (
    save_checkpoint,
    load_model_for_inference,
)

from .training import (
    train,
    finetune,
)

from .prediction import (
    predict_pair,
    predict_batch,
    predict_list,
)

from .visualization import (
    attr_to_rgb,
    mol_to_image,
    draw_mol_attribution,
    draw_feature_attribution,
    draw_score_ranking,
    draw_pair_comparison,
    draw_diff_features,
    merge_csvs,
)

from .explainer import (
    _compute_extra_atom_contribs,
    IGExplainer,
    DiffAttrExplainer,
    explain,
    diff_attr,
)

__all__ = [
    "EXTRA_COLS", "EXTRA_DIM", "GLOBAL_COLS", "GLOBAL_DIM",
    "TASK_NAMES", "NUM_TASKS", "logger", "DEVICE",
    "ModelConfig", "TrainingConfig", "FinetuneConfig", "PredictConfig",
    "_one_hot", "_atom_features", "_bond_features",
    "_extra_feat", "_extra_feat_single", "_global_feat",
    "MolGraphData", "featurize_mol", "load_and_preprocess",
    "compute_dataset_stats", "ATOM_FDIM", "BOND_FDIM",
    "MolBatch",
    "PairDataset", "ListDataset", "CachedPairDataset", "DynamicPairDataset",
    "collate_mol_graphs", "collate_fn", "collate_list_fn", "collate_cached_batch",
    "EdgeEncoding", "GraphormerLayer", "GraphormerEncoder",
    "HTLRankingModel", "BayesianRankingLoss", "EarlyStopping",
    "save_checkpoint", "load_model_for_inference",
    "train", "finetune",
    "predict_pair", "predict_batch", "predict_list",
    "attr_to_rgb", "mol_to_image",
    "draw_mol_attribution", "draw_feature_attribution",
    "draw_score_ranking", "draw_pair_comparison", "draw_diff_features",
    "merge_csvs",
    "_compute_extra_atom_contribs",
    "IGExplainer", "DiffAttrExplainer", "explain", "diff_attr",
]
