from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .configs import ModelConfig, TrainingConfig, FinetuneConfig
from .constants import DEVICE, TASK_NAMES, logger
from .datasets import (
    CachedPairDataset,
    DynamicPairDataset,
    PairDataset,
    collate_cached_batch,
    collate_fn,
)
from .features import load_and_preprocess, compute_dataset_stats
from .models import BayesianRankingLoss, EarlyStopping, HTLRankingModel


def _run_epoch(
    model: HTLRankingModel,
    loader: DataLoader,
    criterion: BayesianRankingLoss,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_s1, all_s2, all_y1, all_y2 = [], [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for mb1, mb2, ef1, ef2, gf, y1, y2 in loader:
            ef1, ef2 = ef1.to(DEVICE), ef2.to(DEVICE)
            gf = gf.to(DEVICE)
            y1, y2 = y1.to(DEVICE), y2.to(DEVICE)

            s1, s2 = model(mb1, ef1, mb2, ef2, gf)
            loss, _ = criterion(s1, s2, y1, y2)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_s1.append(s1.detach().cpu())
            all_s2.append(s2.detach().cpu())
            all_y1.append(y1.cpu())
            all_y2.append(y2.cpu())

    def _cat(lst):
        return torch.cat(lst).numpy()

    return (
        total_loss / len(loader),
        _cat(all_s1),
        _cat(all_s2),
        _cat(all_y1),
        _cat(all_y2),
    )


def _metrics(s1, s2, y1, y2) -> Dict[str, float]:
    out = {}
    for t, name in enumerate(TASK_NAMES):
        dp = s1[:, t] - s2[:, t]
        dy = y1[:, t] - y2[:, t]
        mask = dy != 0
        acc = ((np.sign(dp[mask]) == np.sign(
            dy[mask]))).mean() if mask.any() else 0.0
        prob = float(
            1.0 / (1.0 + np.exp(-np.sign(dy[mask]) * dp[mask])).mean()
        ) if mask.any() else 0.0
        scores = np.concatenate([s1[:, t], s2[:, t]])
        targets = np.concatenate([y1[:, t], y2[:, t]])
        rho, _ = spearmanr(scores, targets)
        rho = 0.0 if np.isnan(rho) else rho
        out[f"{name}_rank_acc"] = float(acc)
        out[f"{name}_bayes_prob"] = prob
        out[f"{name}_spearman"] = float(rho)
    return out


def _train_single_split(
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    fold_tag: str = "",
) -> Dict[str, Any]:
    cfg = train_config
    mcfg = model_config

    max_dist = mcfg.max_dist
    tr_ds = PairDataset(df.iloc[tr_idx], fit_scaler=True, max_dist=max_dist)

    if cfg.cache_val_test:
        va_ds = CachedPairDataset(df.iloc[va_idx],
                                  cfg.batch_size,
                                  scaler=tr_ds.scaler,
                                  max_dist=max_dist)
        te_ds = CachedPairDataset(df.iloc[te_idx],
                                  cfg.batch_size,
                                  scaler=tr_ds.scaler,
                                  max_dist=max_dist)
    else:
        va_ds = DynamicPairDataset(df.iloc[va_idx],
                                   cfg.batch_size,
                                   scaler=tr_ds.scaler,
                                   max_dist=max_dist)
        te_ds = DynamicPairDataset(df.iloc[te_idx],
                                   cfg.batch_size,
                                   scaler=tr_ds.scaler,
                                   max_dist=max_dist)

    tr_loader = DataLoader(tr_ds,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           collate_fn=collate_fn,
                           num_workers=0)
    va_loader = DataLoader(va_ds,
                           batch_size=1,
                           shuffle=False,
                           collate_fn=collate_cached_batch)
    te_loader = DataLoader(te_ds,
                           batch_size=1,
                           shuffle=False,
                           collate_fn=collate_cached_batch)

    logger.info(
        f"{fold_tag}Train/Val/Test: {len(tr_ds)}/{len(va_ds.df)}/{len(te_ds.df)}")

    model_kwargs = mcfg.to_dict()
    model_kwargs.pop("auto_compute_stats", None)
    model = HTLRankingModel(**model_kwargs).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{fold_tag}Model parameters: {n_params:,}")

    criterion = BayesianRankingLoss()
    optimizer = AdamW(model.parameters(),
                      lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=cfg.epochs,
                                  eta_min=cfg.lr * 1e-2)
    stopper = EarlyStopping(patience=cfg.patience, warmup=cfg.early_stop_warmup)

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, *_ = _run_epoch(model, tr_loader, criterion, optimizer)
        va_loss, s1, s2, y1, y2 = _run_epoch(model, va_loader, criterion)
        va_met = _metrics(s1, s2, y1, y2)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_metrics"].append(va_met)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"{fold_tag}Ep {epoch:4d} | "
                f"tr={tr_loss:.4f}  va={va_loss:.4f} | "
                f"PCE acc={va_met['PCE_rank_acc']:.3f} prob={va_met['PCE_bayes_prob']:.3f} ρ={va_met['PCE_spearman']:.3f}"
            )

        if stopper.step(va_loss, model):
            logger.info(f"{fold_tag}Early stopping at epoch {epoch}")
            break

    if stopper.best_state:
        model.load_state_dict({
            k: v.to(DEVICE)
            for k, v in stopper.best_state.items()
        })

    te_loss, s1, s2, y1, y2 = _run_epoch(model, te_loader, criterion)
    te_met = _metrics(s1, s2, y1, y2)
    logger.info(f"\n{fold_tag}========== Test Results ==========")
    for k, v in te_met.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    return {
        "test_metrics": te_met,
        "history": history,
        "model": model,
        "scaler": tr_ds.scaler,
    }


def _make_group_split(
    df: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_dois = df["doi"].unique()
    n = len(unique_dois)
    rng = np.random.RandomState(seed)

    perm = rng.permutation(n)
    n_te = max(1, int(round(n * test_ratio)))
    n_va = max(1, int(round(n * val_ratio)))
    n_tr = n - n_te - n_va

    te_groups = set(unique_dois[perm[:n_te]])
    va_groups = set(unique_dois[perm[n_te:n_te + n_va]])
    tr_groups = set(unique_dois[perm[n_te + n_va:]])

    tr_idx = np.array([i for i, d in enumerate(df["doi"]) if d in tr_groups])
    va_idx = np.array([i for i, d in enumerate(df["doi"]) if d in va_groups])
    te_idx = np.array([i for i, d in enumerate(df["doi"]) if d in te_groups])

    logger.info(
        f"Group split: {len(tr_groups)}/{len(va_groups)}/{len(te_groups)} "
        f"doi groups → {len(tr_idx)}/{len(va_idx)}/{len(te_idx)} rows"
    )
    return tr_idx, va_idx, te_idx


def train(model_config: ModelConfig,
          train_config: TrainingConfig) -> Dict[str, Any]:
    cfg = train_config
    mcfg = model_config

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = load_and_preprocess(cfg.csv_path)

    if mcfg.auto_compute_stats:
        max_degree, max_dist = compute_dataset_stats(df, mcfg.max_dist,
                                                     mcfg.max_degree)
        mcfg.max_degree = max_degree
        mcfg.max_dist = max_dist

    if cfg.split == "group" and cfg.n_cv_folds is not None and cfg.n_cv_folds > 0:
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        groups = df["doi"].values

        doi_counts = pd.Series(groups).value_counts()
        top_dois = doi_counts.head(cfg.n_cv_folds).index.tolist()
        top_set = set(top_dois)

        fold_metrics: List[Dict[str, float]] = []
        fold_results: List[Dict[str, Any]] = []

        for fold_i, (trn_idx, tst_idx) in enumerate(logo.split(df, groups=groups)):
            test_doi = groups[tst_idx[0]]
            if test_doi not in top_set:
                continue
            fold_num = len(fold_metrics) + 1
            tag = f"[Fold {fold_num}/{cfg.n_cv_folds} doi={test_doi[:30]}] "

            logger.info(f"\n{'='*60}")
            logger.info(f"{tag}Test group: {test_doi} ({len(tst_idx)} rows)")
            logger.info(f"{'='*60}")

            trn_df = df.iloc[trn_idx]
            unique_tr_dois = trn_df["doi"].unique()
            n_tr_dois = len(unique_tr_dois)
            rng = np.random.RandomState(cfg.seed)
            perm = rng.permutation(n_tr_dois)
            n_va_dois = max(1, int(round(n_tr_dois * cfg.val_ratio)))
            va_groups_set = set(unique_tr_dois[perm[:n_va_dois]])
            tr_groups_set = set(unique_tr_dois[perm[n_va_dois:]])

            va_idx = np.array([i for i in trn_idx if groups[i] in va_groups_set])
            tr_idx = np.array([i for i in trn_idx if groups[i] in tr_groups_set])

            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

            result = _train_single_split(df, tr_idx, va_idx, tst_idx,
                                         mcfg, cfg, fold_tag=tag)
            fold_metrics.append(result["test_metrics"])
            fold_results.append(result)

        if fold_metrics:
            agg_metrics: Dict[str, float] = {}
            for k in fold_metrics[0]:
                vals = [m[k] for m in fold_metrics]
                agg_metrics[f"{k}_mean"] = float(np.mean(vals))
                agg_metrics[f"{k}_std"] = float(np.std(vals))
            logger.info("\n========== LOGO CV Summary ==========")
            for k, v in agg_metrics.items():
                logger.info(f"  {k:35s}: {v:.4f}")

            ckpt_path = str(Path(cfg.save_dir) / "best_model.pt")
            save_checkpoint(fold_results[-1]["model"],
                           fold_results[-1]["scaler"],
                           mcfg, ckpt_path)

        return {
            "test_metrics": agg_metrics if fold_metrics else {},
            "cv_fold_metrics": fold_metrics,
            "history": fold_results[-1]["history"] if fold_results else {},
            "model": fold_results[-1]["model"] if fold_results else None,
            "scaler": fold_results[-1]["scaler"] if fold_results else None,
            "checkpoint": str(ckpt_path) if fold_metrics else None,
        }

    if cfg.split == "group":
        tr_idx, va_idx, te_idx = _make_group_split(
            df, cfg.test_ratio, cfg.val_ratio, cfg.seed)
    else:
        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(idx,
                                          test_size=cfg.test_ratio,
                                          random_state=cfg.seed)
        tr_idx, va_idx = train_test_split(tr_idx,
                                          test_size=cfg.val_ratio /
                                          (1 - cfg.test_ratio),
                                          random_state=cfg.seed)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    result = _train_single_split(df, tr_idx, va_idx, te_idx, mcfg, cfg)

    ckpt_path = str(Path(cfg.save_dir) / "best_model.pt")
    save_checkpoint(result["model"], result["scaler"], mcfg, ckpt_path)

    return {
        "test_metrics": result["test_metrics"],
        "history": result["history"],
        "model": result["model"],
        "scaler": result["scaler"],
        "checkpoint": str(ckpt_path),
    }


def finetune(config: FinetuneConfig) -> Dict[str, Any]:
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    ckpt = torch.load(config.checkpoint_path,
                      map_location=DEVICE,
                      weights_only=False)
    mcfg = ModelConfig.from_dict(ckpt["config"])
    scaler = ckpt["scaler"]

    df = load_and_preprocess(config.csv_path)
    logger.info(f"Full dataset size: {len(df)} pairs")

    full_ds = PairDataset(df,
                          scaler=scaler,
                          fit_scaler=False,
                          max_dist=mcfg.max_dist)
    full_loader = DataLoader(full_ds,
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn,
                             num_workers=0)

    model_kwargs = mcfg.to_dict()
    model_kwargs.pop("auto_compute_stats", None)
    model = HTLRankingModel(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Loaded best model weights as initialization")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    criterion = BayesianRankingLoss()
    optimizer = AdamW(model.parameters(),
                      lr=config.lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.finetune_epochs,
                                  eta_min=config.lr * 0.1)

    history = {"train_loss": [], "train_metrics": []}

    logger.info(
        f"Starting fine-tuning for {config.finetune_epochs} epochs with lr={config.lr}"
    )
    for epoch in range(1, config.finetune_epochs + 1):
        tr_loss, s1, s2, y1, y2 = _run_epoch(
            model, full_loader, criterion, optimizer)
        tr_met = _metrics(s1, s2, y1, y2)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_metrics"].append(tr_met)

        logger.info(
            f"Ep {epoch:4d} | loss={tr_loss:.4f} | "
            f"PCE acc={tr_met['PCE_rank_acc']:.3f} prob={tr_met['PCE_bayes_prob']:.3f} ρ={tr_met['PCE_spearman']:.3f}"
        )

    final_path = str(Path(config.save_dir) / "final_model.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "config": mcfg.to_dict(),
            "finetune_history": history
        }, final_path)
    logger.info(f"Final model saved → {final_path}")
    return {"final_checkpoint": str(final_path), "history": history, "model": model}
