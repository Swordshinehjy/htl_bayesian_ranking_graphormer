from pathlib import Path
from typing import Tuple

import torch
from sklearn.preprocessing import StandardScaler

from .configs import ModelConfig
from .constants import DEVICE, logger
from .models import HTLRankingModel


def save_checkpoint(
    model: HTLRankingModel,
    scaler: StandardScaler,
    config: ModelConfig,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "config": config.to_dict(),
        }, path)
    logger.info(f"Checkpoint saved → {path}")


def load_model_for_inference(
    checkpoint_path: str,
) -> Tuple[HTLRankingModel, ModelConfig, StandardScaler]:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    mcfg = ModelConfig.from_dict(ckpt["config"])
    model_kwargs = mcfg.to_dict()
    model_kwargs.pop("auto_compute_stats", None)
    model = HTLRankingModel(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = ckpt["scaler"]
    return model, mcfg, scaler
