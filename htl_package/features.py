from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as sp_shortest_path

from .constants import (
    EXTRA_COLS,
    GLOBAL_COLS,
    _ATOM_SYMBOLS,
    _HYBRIDIZATIONS,
    _BOND_TYPES,
    _STEREO_TYPES,
    logger,
)


def _one_hot(value, choices: list) -> List[int]:
    return [int(value == c) for c in choices] + [int(value not in choices)]


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    feats = (
        _one_hot(atom.GetSymbol(), _ATOM_SYMBOLS)
        + _one_hot(atom.GetTotalDegree(), list(range(8)))
        + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + _one_hot(atom.GetTotalNumHs(), list(range(5)))
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
    )
    return np.array(feats, dtype=np.float32)


def _bond_features(bond: Chem.Bond) -> np.ndarray:
    feats = (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), _STEREO_TYPES)
    )
    return np.array(feats, dtype=np.float32)


_test_mol = Chem.MolFromSmiles("CC")
ATOM_FDIM = len(_atom_features(_test_mol.GetAtomWithIdx(0)))
BOND_FDIM = len(_bond_features(_test_mol.GetBondWithIdx(0)))
del _test_mol

logger.info(f"ATOM_FDIM={ATOM_FDIM}  BOND_FDIM={BOND_FDIM}")


class MolGraphData:
    __slots__ = ("atom_feats", "dist_matrix", "edge_path_feats", "degree",
                 "n_atoms")

    def __init__(
        self,
        atom_feats: np.ndarray,
        dist_matrix: np.ndarray,
        edge_path_feats: np.ndarray,
        degree: np.ndarray,
        n_atoms: int,
    ):
        self.atom_feats = atom_feats
        self.dist_matrix = dist_matrix
        self.edge_path_feats = edge_path_feats
        self.degree = degree
        self.n_atoms = n_atoms


def featurize_mol(mol: Chem.Mol, max_dist: int = 10) -> Optional[MolGraphData]:
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0:
        return None

    atom_feats = np.stack([
        _atom_features(mol.GetAtomWithIdx(i)) for i in range(n)
    ]).astype(np.float32)

    bond_feat_mat = np.zeros((n, n, BOND_FDIM), dtype=np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        bond_feat_mat[i, j] = bf
        bond_feat_mat[j, i] = bf
        adj[i, j] = adj[j, i] = 1.0

    dist_float, predecessors = sp_shortest_path(
        csr_matrix(adj),
        method='D',
        directed=False,
        unweighted=True,
        return_predecessors=True,
    )
    dist_int = np.where(np.isinf(dist_float), -1, dist_float).astype(np.int16)

    edge_path_feats = _compute_edge_path_feats(n, dist_int, predecessors,
                                               bond_feat_mat, max_dist)

    degree = np.array(
        [mol.GetAtomWithIdx(i).GetDegree() for i in range(n)],
        dtype=np.int64,
    )

    return MolGraphData(atom_feats, dist_int, edge_path_feats, degree, n)


def _compute_edge_path_feats(
    n: int,
    dist_sp: np.ndarray,
    predecessors: np.ndarray,
    bond_feat_mat: np.ndarray,
    max_dist: int,
) -> np.ndarray:
    F = bond_feat_mat.shape[-1]
    edge_path_feats = np.zeros((n, n, max_dist, F), dtype=np.float32)

    valid = (dist_sp > 0) & (dist_sp <= max_dist)
    if not np.any(valid):
        return edge_path_feats

    cur_nodes = np.broadcast_to(np.arange(n), (n, n)).copy()
    rows = np.arange(n)[:, None]

    for step in range(max_dist):
        prev_nodes = predecessors[rows, cur_nodes]
        valid_prev = (prev_nodes >= 0) & valid

        safe_prev = np.where(valid_prev, prev_nodes, 0)
        edge_feats = bond_feat_mat[safe_prev, cur_nodes]

        store_pos = dist_sp - 1 - step
        valid_store = valid_prev & (store_pos >= 0) & (store_pos < max_dist)

        i_idx, j_idx = np.where(valid_store)
        if len(i_idx) > 0:
            pos_idx = store_pos[i_idx, j_idx]
            edge_path_feats[i_idx, j_idx, pos_idx, :] = edge_feats[i_idx,
                                                                   j_idx, :]

        cur_nodes = np.where(valid_prev, prev_nodes, cur_nodes)
        valid = valid_prev & (cur_nodes != rows)

        if not np.any(valid):
            break

    return edge_path_feats


def _extra_feat(df: pd.DataFrame, suffix: str) -> np.ndarray:
    cols = [c.format(s=suffix) for c in EXTRA_COLS]
    sub = df[cols]
    return sub.fillna(sub.mean()).values.astype(np.float32)


def _extra_feat_single(df: pd.DataFrame) -> np.ndarray:
    cols = [c.replace("_{s}", "") for c in EXTRA_COLS]
    sub = df[cols]
    return sub.fillna(sub.mean()).values.astype(np.float32)


def _global_feat(df: pd.DataFrame) -> np.ndarray:
    sub = df[GLOBAL_COLS]
    return sub.fillna(sub.mean()).values.astype(np.float32)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for s in ["1", "2"]:
        df[f"mol_{s}"] = df[f"SMILES_{s}"].apply(Chem.MolFromSmiles)
    df = df[df["mol_1"].notna() & df["mol_2"].notna()].reset_index(drop=True)
    logger.info(f"Preprocessed dataset: {len(df)} valid pairs")
    return df


def compute_dataset_stats(
    df: pd.DataFrame,
    max_dist_default: int = 15,
    max_degree_default: int = 15,
) -> Tuple[int, int]:
    max_degree = 0
    max_dist = 0

    mol_cols = [c for c in df.columns if c.startswith("mol_")]

    for col in mol_cols:
        for mol in df[col]:
            if mol is None:
                continue

            for atom in mol.GetAtoms():
                degree = atom.GetDegree()
                max_degree = max(max_degree, degree)

            n = mol.GetNumAtoms()
            if n > 1:
                adj = np.zeros((n, n), dtype=np.float32)
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    adj[i, j] = adj[j, i] = 1.0

                try:
                    dist_matrix = sp_shortest_path(csr_matrix(adj),
                                                   method='D',
                                                   directed=False,
                                                   unweighted=True)
                    dist_max = int(
                        np.nanmax(dist_matrix[~np.isinf(dist_matrix)]))
                    max_dist = max(max_dist, dist_max)
                except Exception:
                    pass

    max_degree = max(max_degree, max_degree_default)
    max_dist = max(max_dist, max_dist_default)

    logger.info(f"Dataset stats: max_degree={max_degree}, max_dist={max_dist}")
    return max_degree, max_dist
