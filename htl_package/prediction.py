from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from .checkpoint import load_model_for_inference
from .constants import DEVICE, GLOBAL_DIM, TASK_NAMES, logger
from .datasets import ListDataset, collate_list_fn, collate_mol_graphs
from .features import (
    MolGraphData,
    _extra_feat,
    _extra_feat_single,
    _global_feat,
    featurize_mol,
)
from .models import HTLRankingModel


@torch.no_grad()
def predict_pair(
    smiles_1: str,
    smiles_2: str,
    extra_raw_1: np.ndarray,
    extra_raw_2: np.ndarray,
    checkpoint_path: str,
    global_feat: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    model, mcfg, scaler = load_model_for_inference(checkpoint_path)

    g1 = featurize_mol(Chem.MolFromSmiles(smiles_1), mcfg.max_dist)
    g2 = featurize_mol(Chem.MolFromSmiles(smiles_2), mcfg.max_dist)
    if g1 is None or g2 is None:
        raise ValueError("Invalid SMILES or empty molecule")

    mb1 = collate_mol_graphs([g1], mcfg.max_dist)
    mb2 = collate_mol_graphs([g2], mcfg.max_dist)
    ef1 = torch.tensor(scaler.transform(extra_raw_1.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    ef2 = torch.tensor(scaler.transform(extra_raw_2.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    gf = torch.tensor(
        global_feat.reshape(1, -1) if global_feat is not None else np.zeros(
            (1, GLOBAL_DIM), dtype=np.float32),
        dtype=torch.float32,
    ).to(DEVICE)

    s1, s2 = model(mb1, ef1, mb2, ef2, gf)
    s1, s2 = s1.cpu().numpy()[0], s2.cpu().numpy()[0]

    ranking = {}
    probs = {}
    for i, n in enumerate(TASK_NAMES):
        prob = float(1.0 / (1.0 + np.exp(-(s1[i] - s2[i]))))
        probs[n] = prob
        ranking[n] = "HTL_1" if s1[i] > s2[i] else "HTL_2"

    return {
        "scores_1": {
            n: float(s1[i])
            for i, n in enumerate(TASK_NAMES)
        },
        "scores_2": {
            n: float(s2[i])
            for i, n in enumerate(TASK_NAMES)
        },
        "ranking": ranking,
        "ranking_prob": probs,
    }


@torch.no_grad()
def predict_batch(
    df_new: pd.DataFrame,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    model, mcfg, scaler = load_model_for_inference(checkpoint_path)

    df = df_new.copy().reset_index(drop=True)
    for s in ["1", "2"]:
        df[f"mol_{s}"] = df[f"SMILES_{s}"].apply(Chem.MolFromSmiles)
    valid = df.dropna(subset=["mol_1", "mol_2"]).copy().reset_index(drop=True)
    logger.info(f"Valid pairs: {len(valid)}/{len(df)}")

    ef1 = scaler.transform(_extra_feat(valid, "1"))
    ef2 = scaler.transform(_extra_feat(valid, "2"))
    gf  = _global_feat(valid) if all(c in valid.columns for c in ["MO_ITO"]) \
          else np.zeros((len(valid), GLOBAL_DIM), dtype=np.float32)

    graphs1: List[MolGraphData] = []
    graphs2: List[MolGraphData] = []
    valid_feat_idx: List[int] = []
    for i in range(len(valid)):
        g1 = featurize_mol(valid["mol_1"].iloc[i], mcfg.max_dist)
        g2 = featurize_mol(valid["mol_2"].iloc[i], mcfg.max_dist)
        if g1 is not None and g2 is not None:
            graphs1.append(g1)
            graphs2.append(g2)
            valid_feat_idx.append(i)
        else:
            logger.warning(f"Skipping row {i}: featurization failed")

    if valid_feat_idx:
        valid = valid.iloc[valid_feat_idx].reset_index(drop=True)
        ef1 = scaler.transform(_extra_feat(valid, "1"))
        ef2 = scaler.transform(_extra_feat(valid, "2"))
        gf  = _global_feat(valid) if all(c in valid.columns for c in ["MO_ITO"]) \
              else np.zeros((len(valid), GLOBAL_DIM), dtype=np.float32)
    else:
        logger.warning("No valid pairs after featurization")
        return valid

    all_s1, all_s2 = [], []
    n_samples = len(valid)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        idxs = list(range(start, end))

        mb1 = collate_mol_graphs([graphs1[i] for i in idxs], mcfg.max_dist)
        mb2 = collate_mol_graphs([graphs2[i] for i in idxs], mcfg.max_dist)
        t1 = torch.tensor(ef1[idxs], dtype=torch.float32).to(DEVICE)
        t2 = torch.tensor(ef2[idxs], dtype=torch.float32).to(DEVICE)
        gt = torch.tensor(gf[idxs], dtype=torch.float32).to(DEVICE)

        s1, s2 = model(mb1, t1, mb2, t2, gt)
        all_s1.append(s1.cpu().numpy())
        all_s2.append(s2.cpu().numpy())

    S1, S2 = np.concatenate(all_s1, axis=0), np.concatenate(all_s2, axis=0)
    for i, n in enumerate(TASK_NAMES):
        valid[f"score_{n}_1"] = S1[:, i]
        valid[f"score_{n}_2"] = S2[:, i]
        prob = 1.0 / (1.0 + np.exp(-(S1[:, i] - S2[:, i])))
        valid[f"prob_{n}_1_gt_2"] = prob
        valid[f"preferred_{n}"] = np.where(S1[:, i] > S2[:, i],
                                           valid["Materials_1"],
                                           valid["Materials_2"])
    if output_path:
        valid.to_csv(output_path, index=False)
        logger.info(f"Predictions saved → {output_path}")
    return valid


@torch.no_grad()
def predict_list(
    df_list: pd.DataFrame,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 32,
    global_feat: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    model, mcfg, scaler = load_model_for_inference(checkpoint_path)

    list_ds = ListDataset(df_list,
                          scaler,
                          global_feat=global_feat,
                          max_dist=mcfg.max_dist)
    loader = torch.utils.data.DataLoader(list_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_list_fn,
                        num_workers=0)
    logger.info(f"Valid materials: {len(list_ds)}/{len(df_list)}")

    all_scores, all_mats = [], []
    for bmg, ef, gf, materials in loader:
        ef = ef.to(DEVICE)
        gf = gf.to(DEVICE)
        scores = model.encode(bmg, ef, gf)
        all_scores.append(scores.cpu().numpy())
        all_mats.extend(materials)

    sc = np.concatenate(all_scores, axis=0)
    result_df = list_ds.df.copy()
    result_df["Materials"] = list_ds.materials
    for i, n in enumerate(TASK_NAMES):
        result_df[f"score_{n}"] = sc[:, i]
    result_df = result_df.sort_values(f"score_{TASK_NAMES[0]}",
                                      ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)

    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"List ranking saved → {output_path}")
    return result_df
