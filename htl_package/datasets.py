from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .constants import GLOBAL_COLS, GLOBAL_DIM, logger
from .features import (
    MolGraphData,
    _extra_feat,
    _extra_feat_single,
    _global_feat,
    featurize_mol,
    ATOM_FDIM,
    BOND_FDIM,
)


class MolBatch:
    __slots__ = (
        "atom_feats",
        "dist_matrix",
        "edge_path_feats",
        "degree",
        "padding_mask",
        "n_atoms",
    )

    def __init__(
        self,
        atom_feats: torch.Tensor,
        dist_matrix: torch.Tensor,
        edge_path_feats: torch.Tensor,
        degree: torch.Tensor,
        padding_mask: torch.Tensor,
        n_atoms: List[int],
    ):
        self.atom_feats = atom_feats
        self.dist_matrix = dist_matrix
        self.edge_path_feats = edge_path_feats
        self.degree = degree
        self.padding_mask = padding_mask
        self.n_atoms = n_atoms

    def to(self, device: torch.device) -> "MolBatch":
        return MolBatch(
            self.atom_feats.to(device),
            self.dist_matrix.to(device),
            self.edge_path_feats.to(device),
            self.degree.to(device),
            self.padding_mask.to(device),
            self.n_atoms,
        )


def collate_mol_graphs(graphs: List[MolGraphData],
                       max_dist: Optional[int] = None) -> MolBatch:
    B = len(graphs)
    N = max(g.n_atoms for g in graphs)

    if max_dist is None:
        max_dist = max(g.edge_path_feats.shape[2] for g in graphs)

    atom_feats = torch.zeros(B, N, ATOM_FDIM, dtype=torch.float32)
    dist_matrix = torch.full((B, N, N), -1, dtype=torch.long)
    edge_path_feats = torch.zeros(B,
                                  N,
                                  N,
                                  max_dist,
                                  BOND_FDIM,
                                  dtype=torch.float32)
    degree = torch.zeros(B, N, dtype=torch.long)
    padding_mask = torch.ones(B, N, dtype=torch.bool)

    for idx, g in enumerate(graphs):
        n = g.n_atoms
        atom_feats[idx, :n] = torch.from_numpy(g.atom_feats)
        dist_matrix[idx, :n, :n] = torch.from_numpy(
            g.dist_matrix.astype(np.int64))
        g_max_dist = g.edge_path_feats.shape[2]
        if g_max_dist <= max_dist:
            edge_path_feats[idx, :n, :n, :g_max_dist] = torch.from_numpy(
                g.edge_path_feats)
        else:
            edge_path_feats[idx, :n, :n] = torch.from_numpy(
                g.edge_path_feats[:, :, :max_dist])
        degree[idx, :n] = torch.from_numpy(g.degree)
        padding_mask[idx, :n] = False

    return MolBatch(atom_feats, dist_matrix, edge_path_feats, degree,
                    padding_mask, [g.n_atoms for g in graphs])


class PairDataset(Dataset):

    def __init__(
        self,
        df,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        max_dist: int = 10,
    ):
        self.df = df.reset_index(drop=True)
        self.max_dist = max_dist

        self.graphs1: List[MolGraphData] = []
        self.graphs2: List[MolGraphData] = []
        valid_idx: List[int] = []

        for i in range(len(self.df)):
            g1 = featurize_mol(self.df.loc[i, "mol_1"], max_dist)
            g2 = featurize_mol(self.df.loc[i, "mol_2"], max_dist)
            if g1 is not None and g2 is not None:
                self.graphs1.append(g1)
                self.graphs2.append(g2)
                valid_idx.append(i)

        self.df = self.df.iloc[valid_idx].reset_index(drop=True)

        ef1 = _extra_feat(self.df, "1")
        ef2 = _extra_feat(self.df, "2")

        if fit_scaler:
            self.scaler = StandardScaler().fit(np.vstack([ef1, ef2]))
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2
        self.gf = _global_feat(self.df)
        self.y1 = self.df[["PCE_1"]].values.astype(np.float32)
        self.y2 = self.df[["PCE_2"]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.graphs1[idx],
            self.graphs2[idx],
            torch.tensor(self.ef1[idx]),
            torch.tensor(self.ef2[idx]),
            torch.tensor(self.gf[idx]),
            torch.tensor(self.y1[idx]),
            torch.tensor(self.y2[idx]),
        )


class ListDataset(Dataset):

    def __init__(
        self,
        df,
        scaler: StandardScaler,
        global_feat: Optional[np.ndarray] = None,
        max_dist: int = 10,
    ):
        self.df = df.reset_index(drop=True)

        smiles_col = next(
            (c
             for c in ["SMILES", "smiles", "Smiles"] if c in self.df.columns),
            None)
        if smiles_col is None:
            raise ValueError("CSV must contain a SMILES column")
        self.smiles_col = smiles_col

        mat_col = next((
            c for c in
            ["Materials", "materials", "Material", "material", "Name", "name"]
            if c in self.df.columns), None)
        self.mat_col = mat_col

        self.graphs: List[MolGraphData] = []
        self.materials: List[str] = []
        valid_idx: List[int] = []

        for i in range(len(self.df)):
            smiles = self.df.loc[i, smiles_col]
            mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None
            g = featurize_mol(mol, max_dist) if mol else None
            if g is not None:
                self.graphs.append(g)
                valid_idx.append(i)
                self.materials.append(
                    str(self.df.loc[i,
                                    mat_col]) if mat_col else f"Material_{i}")

        self.df = self.df.iloc[valid_idx].reset_index(drop=True)
        self.ef = scaler.transform(_extra_feat_single(self.df))

        if global_feat is not None:
            self.gf = np.tile(global_feat,
                              (len(self.df), 1)).astype(np.float32)
        elif all(col in self.df.columns for col in GLOBAL_COLS):
            self.gf = _global_feat(self.df)
        else:
            self.gf = np.zeros((len(self.df), GLOBAL_DIM), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            torch.tensor(self.ef[idx]),
            torch.tensor(self.gf[idx]),
            self.materials[idx],
        )


class CachedPairDataset:

    def __init__(
        self,
        df,
        batch_size: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        max_dist: int = 10,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size

        graphs1: List[MolGraphData] = []
        graphs2: List[MolGraphData] = []
        valid_idx: List[int] = []

        for i in range(len(self.df)):
            g1 = featurize_mol(self.df.loc[i, "mol_1"], max_dist)
            g2 = featurize_mol(self.df.loc[i, "mol_2"], max_dist)
            if g1 is not None and g2 is not None:
                graphs1.append(g1)
                graphs2.append(g2)
                valid_idx.append(i)

        self.df = self.df.iloc[valid_idx].reset_index(drop=True)

        ef1 = _extra_feat(self.df, "1")
        ef2 = _extra_feat(self.df, "2")

        if fit_scaler:
            self.scaler = StandardScaler().fit(np.vstack([ef1, ef2]))
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2
        self.gf = _global_feat(self.df)
        self.y1 = self.df[["PCE_1"]].values.astype(np.float32)
        self.y2 = self.df[["PCE_2"]].values.astype(np.float32)

        self._cached: list = []
        n = len(self.df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idxs = list(range(start, end))
            mb1 = collate_mol_graphs([graphs1[i] for i in idxs], max_dist)
            mb2 = collate_mol_graphs([graphs2[i] for i in idxs], max_dist)
            self._cached.append((
                mb1,
                mb2,
                torch.tensor(self.ef1[idxs]),
                torch.tensor(self.ef2[idxs]),
                torch.tensor(self.gf[idxs]),
                torch.tensor(self.y1[idxs]),
                torch.tensor(self.y2[idxs]),
            ))

    def __len__(self):
        return len(self._cached)

    def __getitem__(self, idx):
        return self._cached[idx]


class DynamicPairDataset:

    def __init__(
        self,
        df,
        batch_size: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        max_dist: int = 10,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_dist = max_dist

        self.graphs1: List[MolGraphData] = []
        self.graphs2: List[MolGraphData] = []
        valid_idx: List[int] = []

        for i in range(len(self.df)):
            g1 = featurize_mol(self.df.loc[i, "mol_1"], max_dist)
            g2 = featurize_mol(self.df.loc[i, "mol_2"], max_dist)
            if g1 is not None and g2 is not None:
                self.graphs1.append(g1)
                self.graphs2.append(g2)
                valid_idx.append(i)

        self.df = self.df.iloc[valid_idx].reset_index(drop=True)

        ef1 = _extra_feat(self.df, "1")
        ef2 = _extra_feat(self.df, "2")

        if fit_scaler:
            self.scaler = StandardScaler().fit(np.vstack([ef1, ef2]))
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2
        self.gf = _global_feat(self.df)
        self.y1 = self.df[["PCE_1"]].values.astype(np.float32)
        self.y2 = self.df[["PCE_2"]].values.astype(np.float32)

        self._n_batches = (len(self.df) + batch_size - 1) // batch_size

    def __len__(self):
        return self._n_batches

    def __getitem__(self, batch_idx: int):
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.df))
        idxs = list(range(start, end))

        mb1 = collate_mol_graphs([self.graphs1[i] for i in idxs],
                                 self.max_dist)
        mb2 = collate_mol_graphs([self.graphs2[i] for i in idxs],
                                 self.max_dist)

        return (
            mb1,
            mb2,
            torch.tensor(self.ef1[idxs]),
            torch.tensor(self.ef2[idxs]),
            torch.tensor(self.gf[idxs]),
            torch.tensor(self.y1[idxs]),
            torch.tensor(self.y2[idxs]),
        )


def collate_fn(batch):
    g1s, g2s, ef1s, ef2s, gfs, y1s, y2s = zip(*batch)
    return (
        collate_mol_graphs(list(g1s)),
        collate_mol_graphs(list(g2s)),
        torch.stack(ef1s),
        torch.stack(ef2s),
        torch.stack(gfs),
        torch.stack(y1s),
        torch.stack(y2s),
    )


def collate_cached_batch(batch):
    if len(batch) == 1:
        return batch[0]
    return batch[0]


def collate_list_fn(batch):
    graphs, efs, gfs, materials = zip(*batch)
    return (
        collate_mol_graphs(list(graphs)),
        torch.stack(efs),
        torch.stack(gfs),
        list(materials),
    )
