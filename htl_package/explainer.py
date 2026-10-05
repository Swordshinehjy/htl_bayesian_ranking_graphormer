from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors as _rdmd, rdFMCS
from sklearn.preprocessing import StandardScaler

from .checkpoint import load_model_for_inference
from .configs import ModelConfig
from .constants import EXTRA_COLS, EXTRA_DIM, GLOBAL_COLS, GLOBAL_DIM, TASK_NAMES, DEVICE, logger
from .datasets import collate_mol_graphs
from .features import _extra_feat, _extra_feat_single, _global_feat, featurize_mol
from .models import HTLRankingModel
from .visualization import (
    attr_to_rgb,
    draw_mol_attribution,
    draw_feature_attribution,
    draw_score_ranking,
    draw_pair_comparison,
    draw_diff_features,
    merge_csvs,
    mol_to_image,
)


def _compute_extra_atom_contribs(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    contribs = np.zeros((n, EXTRA_DIM), dtype=np.float32)
    extra_names = [c.replace("_{s}", "") for c in EXTRA_COLS]

    for k, name in enumerate(extra_names):
        if name == "MolLogP":
            atom_lp = Crippen._GetAtomContribs(mol)
            for i, (lp, _) in enumerate(atom_lp):
                contribs[i, k] = lp
        elif name in ("TPSA", "PSA"):
            for i, v in enumerate(_rdmd._CalcTPSAContribs(mol)):
                contribs[i, k] = v
        elif name == "NumHAcceptors":
            for i, atom in enumerate(mol.GetAtoms()):
                an = atom.GetAtomicNum()
                if an == 7:
                    contribs[i,
                             k] = 0.5 if (atom.IsInRing() and
                                          atom.GetTotalDegree() == 3) else 1.0
                elif an == 8:
                    contribs[i, k] = 1.0
                elif an == 16:
                    contribs[i, k] = 0.5
        elif name == "NumHDonors":
            for i, atom in enumerate(mol.GetAtoms()):
                an = atom.GetAtomicNum()
                nh = atom.GetTotalNumHs()
                if an in (7, 8) and nh > 0:
                    contribs[i, k] = 1.0
                elif an == 16 and nh > 0:
                    contribs[i, k] = 0.5
        else:
            contribs[:, k] = 1.0 / n

    col_sum = contribs.sum(axis=0, keepdims=True)
    mask = np.abs(col_sum) < 1e-8
    col_sum[mask] = 1.0
    contribs = np.where(mask, contribs, contribs / col_sum)
    return contribs


class IGExplainer:

    def __init__(
        self,
        model: HTLRankingModel,
        scaler: StandardScaler,
        n_steps: int = 50,
        target_task: int = 0,
    ):
        self.model = model
        self.scaler = scaler
        self.n_steps = n_steps
        self.target_task = target_task
        self.model.eval()

    @torch.enable_grad()
    def _joint_ig(
        self,
        mol_batch,
        ef: torch.Tensor,
        gf: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        from .datasets import MolBatch

        mol_batch = mol_batch.to(DEVICE)
        ef = ef.to(DEVICE)
        gf = gf.to(DEVICE)

        af_orig = mol_batch.atom_feats.detach().clone()
        ef_orig = ef.detach().clone()
        gf_orig = gf.detach().clone()

        af_base = torch.zeros_like(af_orig)
        ef_base = torch.zeros_like(ef_orig)
        gf_base = torch.zeros_like(gf_orig)

        af_diff = af_orig - af_base
        ef_diff = ef_orig - ef_base
        gf_diff = gf_orig - gf_base

        all_af_grads: List[torch.Tensor] = []
        all_ef_grads: List[torch.Tensor] = []
        all_gf_grads: List[torch.Tensor] = []

        for alpha in np.linspace(0.0, 1.0, self.n_steps):
            af_interp = (af_base + alpha * af_diff).clone()
            ef_interp = (ef_base + alpha * ef_diff).clone()
            gf_interp = (gf_base + alpha * gf_diff).clone()

            af_interp.requires_grad_(True)
            ef_interp.requires_grad_(True)
            gf_interp.requires_grad_(True)

            mb_copy = MolBatch(
                atom_feats=af_interp,
                dist_matrix=mol_batch.dist_matrix,
                edge_path_feats=mol_batch.edge_path_feats,
                degree=mol_batch.degree,
                padding_mask=mol_batch.padding_mask,
                n_atoms=mol_batch.n_atoms,
            )
            score = self.model.encode(mb_copy, ef_interp,
                                      gf_interp)[:, self.target_task]
            score.sum().backward()

            if af_interp.grad is not None:
                all_af_grads.append(af_interp.grad.detach().clone())
            if ef_interp.grad is not None:
                all_ef_grads.append(ef_interp.grad.detach().clone())
            if gf_interp.grad is not None:
                all_gf_grads.append(gf_interp.grad.detach().clone())

            del mb_copy, score, af_interp, ef_interp, gf_interp
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        if not all_af_grads:
            n_atoms = af_orig.shape[1]
            return dict(
                atom_attrs=np.zeros(n_atoms),
                extra_attrs=np.zeros(ef_orig.shape[-1]),
                global_attrs=np.zeros(gf_orig.shape[-1]),
            )

        avg_af_grad = torch.stack(all_af_grads).mean(0)
        avg_ef_grad = torch.stack(all_ef_grads).mean(0)
        avg_gf_grad = torch.stack(all_gf_grads).mean(0)

        atom_ig = (af_diff * avg_af_grad).sum(-1).squeeze(0).cpu().numpy()
        extra_ig = (ef_diff * avg_ef_grad).squeeze(0).cpu().numpy()
        global_ig = (gf_diff * avg_gf_grad).squeeze(0).cpu().numpy()

        return dict(atom_attrs=atom_ig,
                    extra_attrs=extra_ig,
                    global_attrs=global_ig)

    def explain_molecule(
        self,
        smiles: str,
        ef_scaled: np.ndarray,
        gf: np.ndarray,
        material_name: str = "material",
        save_dir: str = "explain_output",
        attrib_extra_to_atom: bool = False,
        max_dist: int = 10,
        **_kwargs,
    ) -> Dict[str, Any]:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        AllChem.Compute2DCoords(mol)

        g_data = featurize_mol(mol, max_dist)
        if g_data is None:
            raise ValueError(f"Cannot generate molecular graph: {smiles}")

        mol_batch = collate_mol_graphs([g_data], max_dist)
        ef = torch.tensor(ef_scaled.reshape(1, -1), dtype=torch.float32)
        gf_t = torch.tensor(gf.reshape(1, -1), dtype=torch.float32)

        with torch.no_grad():
            scores = self.model.encode(
                mol_batch.to(DEVICE), ef.to(DEVICE),
                gf_t.to(DEVICE))
            score = scores[0, self.target_task].item()

        logger.info(
            f"[{material_name}] Predicted score ({TASK_NAMES[self.target_task]}): "
            f"{score:.4f}"
        )

        logger.info(
            f"[{material_name}] Computing joint IG ({self.n_steps} steps)..."
        )
        ig_result = self._joint_ig(mol_batch, ef.clone(), gf_t.clone())
        n_heavy = mol.GetNumAtoms()
        atom_attrs = ig_result["atom_attrs"][:n_heavy]
        extra_attrs = ig_result["extra_attrs"]
        global_attrs = ig_result["global_attrs"]

        extra_names = [c.replace("_{s}", "") for c in EXTRA_COLS]
        global_names = [c.replace("_{s}", "") for c in GLOBAL_COLS]
        combined_atom_attrs = atom_attrs.copy()
        extra_to_atom = None
        if attrib_extra_to_atom:
            extra_atom_matrix = _compute_extra_atom_contribs(mol)
            extra_to_atom = (extra_atom_matrix *
                             extra_attrs[np.newaxis, :]).sum(axis=1)
            combined_atom_attrs = atom_attrs + extra_to_atom

        safe_name = material_name.replace("/", "_").replace(" ", "_")
        svg_path = str(Path(save_dir) / f"{safe_name}_mol.svg")
        png_path = str(Path(save_dir) / f"{safe_name}_mol.png")
        bar_path = str(Path(save_dir) / f"{safe_name}_features.png")
        csv_path = str(Path(save_dir) / f"{safe_name}_summary.csv")

        draw_mol_attribution(mol, combined_atom_attrs, svg_path,
                             png_path, material_name, score,
                             target_task=self.target_task)
        draw_feature_attribution(extra_attrs, extra_names, bar_path,
                                 material_name, score,
                                 target_task=self.target_task)
        self._save_summary_csv(material_name, smiles, score, atom_attrs,
                               extra_attrs, extra_names, csv_path,
                               extra_to_atom, combined_atom_attrs,
                               global_attrs, global_names)

        return {
            "score": score,
            "atom_attrs": atom_attrs,
            "extra_to_atom": extra_to_atom,
            "combined_atom_attrs": combined_atom_attrs,
            "extra_attrs": extra_attrs,
            "extra_names": extra_names,
            "global_attrs": global_attrs,
            "global_names": global_names,
            "pos_atoms":
            [i for i, v in enumerate(combined_atom_attrs) if v > 0],
            "mol_svg_path": svg_path,
            "mol_png_path": png_path,
            "bar_chart_path": bar_path,
            "summary_csv_path": csv_path,
        }

    @staticmethod
    def _save_summary_csv(
        material_name: str,
        smiles: str,
        score: float,
        atom_attrs: np.ndarray,
        extra_attrs: np.ndarray,
        extra_names: List[str],
        csv_path: str,
        extra_to_atom: Optional[np.ndarray] = None,
        combined_atom_attrs: Optional[np.ndarray] = None,
        global_attrs: Optional[np.ndarray] = None,
        global_names: Optional[List[str]] = None,
    ) -> None:
        rows = []
        has = extra_to_atom is not None and combined_atom_attrs is not None
        for i, v in enumerate(atom_attrs):
            row = {
                "material":
                material_name,
                "type":
                "atom",
                "index":
                i,
                "name":
                f"atom_{i}",
                "attribution":
                float(combined_atom_attrs[i]) if has else float(v),
                "sign":
                ("positive" if
                 (combined_atom_attrs[i] > 0 if has else v > 0) else
                 ("negative" if
                  (combined_atom_attrs[i] < 0 if has else v < 0) else "zero"))
            }
            if has:
                row["attribution_extra"] = float(extra_to_atom[i])
            rows.append(row)
        for i, (n, v) in enumerate(zip(extra_names, extra_attrs)):
            rows.append({
                "material":
                material_name,
                "type":
                "extra_feature",
                "index":
                i,
                "name":
                n,
                "attribution":
                float(v),
                "sign":
                "positive" if v > 0 else ("negative" if v < 0 else "zero")
            })
        if global_attrs is not None and global_names is not None:
            for i, (n, v) in enumerate(zip(global_names, global_attrs)):
                rows.append({
                    "material":
                    material_name,
                    "type":
                    "global_feature",
                    "index":
                    i,
                    "name":
                    n,
                    "attribution":
                    float(v),
                    "sign":
                    "positive" if v > 0 else ("negative" if v < 0 else "zero")
                })
        df_out = pd.DataFrame(rows)
        df_out["smiles"] = smiles
        df_out["predicted_score"] = score
        df_out["task"] = TASK_NAMES[0]
        df_out.to_csv(csv_path, index=False)
        logger.info(f"  Summary CSV → {csv_path}")


class DiffAttrExplainer:

    def __init__(
        self,
        model: HTLRankingModel,
        scaler: StandardScaler,
        n_steps: int = 50,
        target_task: int = 0,
    ):
        self.model = model
        self.scaler = scaler
        self.n_steps = n_steps
        self.target_task = target_task
        self.model.eval()

        self._ig = IGExplainer(model=model,
                               scaler=scaler,
                               n_steps=n_steps,
                               target_task=target_task)

    def _score_and_attrs(
        self,
        smiles: str,
        ef_scaled: np.ndarray,
        gf: np.ndarray,
        max_dist: int = 15,
    ) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        AllChem.Compute2DCoords(mol)

        ef = torch.tensor(ef_scaled.reshape(1, -1),
                          dtype=torch.float32).to(DEVICE)
        gf_t = torch.tensor(gf.reshape(1, -1), dtype=torch.float32).to(DEVICE)

        g_data = featurize_mol(mol, max_dist)
        if g_data is None:
            raise ValueError(f"Cannot generate molecular graph: {smiles}")

        mol_batch = collate_mol_graphs([g_data], max_dist)

        with torch.no_grad():
            scores = self.model.encode(mol_batch.to(DEVICE), ef,
                                        gf_t)
            score = scores[0, self.target_task].item()

        ig_result = self._ig._joint_ig(mol_batch.to(DEVICE), ef, gf_t)
        atom_attrs = ig_result["atom_attrs"][:mol.GetNumAtoms()]
        extra_attrs = ig_result["extra_attrs"]
        global_attrs = ig_result["global_attrs"]

        return dict(mol=mol,
                    score=score,
                    atom_attrs=atom_attrs,
                    extra_attrs=extra_attrs,
                    global_attrs=global_attrs)

    @staticmethod
    def _find_mcs(mol1, mol2) -> Dict[str, Any]:
        try:
            result = rdFMCS.FindMCS(
                [mol1, mol2],
                timeout=30,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder,
                completeRingsOnly=False,
                ringMatchesRingOnly=False,
            )
            if result.numAtoms == 0:
                raise ValueError("MCS is empty")

            mcs_mol = Chem.MolFromSmarts(result.smartsString)
            match1 = set(mol1.GetSubstructMatch(mcs_mol))
            match2 = set(mol2.GetSubstructMatch(mcs_mol))

            return dict(
                mcs_smarts=result.smartsString,
                mcs_num_atoms=result.numAtoms,
                mol1_mcs_atoms=match1,
                mol2_mcs_atoms=match2,
                mol1_unique_atoms=set(range(mol1.GetNumAtoms())) - match1,
                mol2_unique_atoms=set(range(mol2.GetNumAtoms())) - match2,
            )
        except Exception as exc:
            logger.warning(
                f"MCS search failed ({exc}); treating all atoms as unique.")
            return dict(
                mcs_smarts=None,
                mcs_num_atoms=0,
                mol1_mcs_atoms=set(),
                mol2_mcs_atoms=set(),
                mol1_unique_atoms=set(range(mol1.GetNumAtoms())),
                mol2_unique_atoms=set(range(mol2.GetNumAtoms())),
            )

    @staticmethod
    def _struct_summary(atom_attrs_1, atom_attrs_2,
                        mcs_info) -> Dict[str, float]:

        def _safe_sum(attrs, idx_set):
            idx = list(idx_set)
            return float(attrs[idx].sum()) if idx else 0.0

        return dict(
            mcs_attr_1=_safe_sum(atom_attrs_1, mcs_info["mol1_mcs_atoms"]),
            mcs_attr_2=_safe_sum(atom_attrs_2, mcs_info["mol2_mcs_atoms"]),
            unique_attr_1=_safe_sum(atom_attrs_1,
                                    mcs_info["mol1_unique_atoms"]),
            unique_attr_2=_safe_sum(atom_attrs_2,
                                    mcs_info["mol2_unique_atoms"]),
            mcs_num_atoms=mcs_info["mcs_num_atoms"],
            unique_num_atoms_1=len(mcs_info["mol1_unique_atoms"]),
            unique_num_atoms_2=len(mcs_info["mol2_unique_atoms"]),
        )

    def explain_pair(
        self,
        smiles_1: str,
        smiles_2: str,
        ef_scaled_1: np.ndarray,
        ef_scaled_2: np.ndarray,
        gf: np.ndarray,
        name_1: str = "HTL_1",
        name_2: str = "HTL_2",
        pair_id: str = "pair_0",
        save_dir: str = "diff_output",
        max_dist: int = 15,
    ) -> Dict[str, Any]:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"  [{pair_id}] Attributions for {name_1} ...")
        r1 = self._score_and_attrs(smiles_1, ef_scaled_1, gf, max_dist)

        logger.info(f"  [{pair_id}] Attributions for {name_2} ...")
        r2 = self._score_and_attrs(smiles_2, ef_scaled_2, gf, max_dist)

        delta_extra = r2["extra_attrs"] - r1["extra_attrs"]
        extra_names = [c.replace("_{s}", "") for c in EXTRA_COLS]

        logger.info(f"  [{pair_id}] Computing MCS ...")
        mcs_info = self._find_mcs(r1["mol"], r2["mol"])
        struct_info = self._struct_summary(r1["atom_attrs"], r2["atom_attrs"],
                                           mcs_info)

        safe = pair_id.replace("/", "_").replace(" ", "_")
        cmp_path = str(Path(save_dir) / f"{safe}_comparison.png")
        feat_path = str(Path(save_dir) / f"{safe}_diff_features.png")
        csv_path = str(Path(save_dir) / f"{safe}_diff_summary.csv")

        draw_pair_comparison(
            r1["mol"],
            r2["mol"],
            r1["atom_attrs"],
            r2["atom_attrs"],
            mcs_info,
            struct_info,
            name_1,
            name_2,
            r1["score"],
            r2["score"],
            cmp_path,
            target_task=self.target_task,
        )
        draw_diff_features(
            extra_names,
            r1["extra_attrs"],
            r2["extra_attrs"],
            delta_extra,
            name_1,
            name_2,
            r1["score"],
            r2["score"],
            feat_path,
        )
        self._save_diff_csv(
            pair_id,
            name_1,
            name_2,
            smiles_1,
            smiles_2,
            r1["score"],
            r2["score"],
            r1["atom_attrs"],
            r2["atom_attrs"],
            r1["extra_attrs"],
            r2["extra_attrs"],
            delta_extra,
            extra_names,
            mcs_info,
            struct_info,
            csv_path,
        )

        return dict(
            pair_id=pair_id,
            name_1=name_1,
            name_2=name_2,
            score_1=r1["score"],
            score_2=r2["score"],
            score_diff=r2["score"] - r1["score"],
            preferred=name_1 if r1["score"] >= r2["score"] else name_2,
            atom_attrs_1=r1["atom_attrs"],
            atom_attrs_2=r2["atom_attrs"],
            extra_attrs_1=r1["extra_attrs"],
            extra_attrs_2=r2["extra_attrs"],
            global_attrs_1=r1["global_attrs"],
            global_attrs_2=r2["global_attrs"],
            delta_extra=delta_extra,
            extra_names=extra_names,
            mcs_info=mcs_info,
            struct_info=struct_info,
            comparison_path=cmp_path,
            diff_feat_path=feat_path,
            csv_path=csv_path,
        )

    @staticmethod
    def _save_diff_csv(
        pair_id: str,
        name_1: str,
        name_2: str,
        smiles_1: str,
        smiles_2: str,
        score_1: float,
        score_2: float,
        atom_attrs_1: np.ndarray,
        atom_attrs_2: np.ndarray,
        extra_attrs_1: np.ndarray,
        extra_attrs_2: np.ndarray,
        delta_extra: np.ndarray,
        extra_names: List[str],
        mcs_info: Dict[str, Any],
        struct_info: Dict[str, float],
        csv_path: str,
    ) -> None:
        preferred = name_1 if score_1 >= score_2 else name_2
        score_diff = score_2 - score_1
        rows = []

        for i, fname in enumerate(extra_names):
            rows.append(
                dict(
                    pair_id=pair_id,
                    type="extra_feature",
                    name=fname,
                    molecule="both",
                    attr_1=float(extra_attrs_1[i]),
                    attr_2=float(extra_attrs_2[i]),
                    delta_attr=float(delta_extra[i]),
                    favors=(name_2 if delta_extra[i] > 0 else
                            (name_1 if delta_extra[i] < 0 else "neutral")),
                    in_mcs=None,
                    is_unique=None,
                ))

        for i, v in enumerate(atom_attrs_1):
            in_mcs = i in mcs_info["mol1_mcs_atoms"]
            rows.append(
                dict(
                    pair_id=pair_id,
                    type="atom",
                    name=f"atom_{i}",
                    molecule=name_1,
                    attr_1=float(v),
                    attr_2=None,
                    delta_attr=None,
                    favors=None,
                    in_mcs=in_mcs,
                    is_unique=not in_mcs,
                ))

        for i, v in enumerate(atom_attrs_2):
            in_mcs = i in mcs_info["mol2_mcs_atoms"]
            rows.append(
                dict(
                    pair_id=pair_id,
                    type="atom",
                    name=f"atom_{i}",
                    molecule=name_2,
                    attr_1=None,
                    attr_2=float(v),
                    delta_attr=None,
                    favors=None,
                    in_mcs=in_mcs,
                    is_unique=not in_mcs,
                ))

        df_out = pd.DataFrame(rows)
        df_out["smiles_1"] = smiles_1
        df_out["smiles_2"] = smiles_2
        df_out["score_1"] = score_1
        df_out["score_2"] = score_2
        df_out["score_diff"] = score_diff
        df_out["preferred"] = preferred
        df_out["mcs_num_atoms"] = mcs_info["mcs_num_atoms"]
        df_out["unique_num_atoms_1"] = struct_info["unique_num_atoms_1"]
        df_out["unique_num_atoms_2"] = struct_info["unique_num_atoms_2"]
        df_out["task"] = TASK_NAMES[0]

        df_out.to_csv(csv_path, index=False)
        logger.info(f"  Differential summary CSV → {csv_path}")


@torch.no_grad()
def explain(
    df_list: pd.DataFrame,
    checkpoint_path: str,
    save_dir: str = "explain_output",
    n_steps: int = 50,
    batch_size: int = 1,
    global_feat: Optional[np.ndarray] = None,
    attrib_extra_to_atom: bool = False,
) -> List[Dict[str, Any]]:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model, mcfg, scaler = load_model_for_inference(checkpoint_path)

    explainer = IGExplainer(model=model,
                            scaler=scaler,
                            n_steps=n_steps,
                            target_task=0)

    smiles_col = next(
        (c for c in ["SMILES", "smiles", "Smiles"] if c in df_list.columns),
        None)
    if smiles_col is None:
        raise ValueError("DataFrame must contain a SMILES column")
    mat_col = next(
        (c for c in
         ["Materials", "materials", "Material", "material", "Name", "name"]
         if c in df_list.columns), None)

    ef_all_scaled = scaler.transform(_extra_feat_single(df_list))

    if global_feat is not None:
        gf_all = np.tile(global_feat, (len(df_list), 1)).astype(np.float32)
    elif all(col in df_list.columns for col in GLOBAL_COLS):
        gf_all = _global_feat(df_list)
    else:
        gf_all = np.zeros((len(df_list), GLOBAL_DIM), dtype=np.float32)

    df_list = df_list.reset_index(drop=True)
    results, all_scores = [], []

    for idx, row in df_list.iterrows():
        smiles = row[smiles_col]
        material_name = str(row[mat_col]) if mat_col else f"material_{idx}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Explaining [{idx+1}/{len(df_list)}]: {material_name}")
        logger.info(f"{'='*60}")
        try:
            result = explainer.explain_molecule(
                smiles=smiles,
                ef_scaled=ef_all_scaled[idx],
                gf=gf_all[idx],
                material_name=material_name,
                save_dir=save_dir,
                attrib_extra_to_atom=attrib_extra_to_atom,
                max_dist=mcfg.max_dist,
            )
            result["material_name"] = material_name
            result["smiles"] = smiles
            results.append(result)
            all_scores.append((material_name, result["score"]))
        except Exception as e:
            logger.error(f"  Failed to explain {material_name}: {e}")

    if len(all_scores) > 1:
        draw_score_ranking(all_scores, save_dir)
    merge_csvs(save_dir, "_summary.csv", "all_attributions.csv")
    logger.info(f"\nExplain mode done. All outputs → {save_dir}/")
    return results


def diff_attr(
    df_pairs: pd.DataFrame,
    checkpoint_path: str,
    save_dir: str = "diff_output",
    n_steps: int = 50,
    global_feat: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model, mcfg, scaler = load_model_for_inference(checkpoint_path)

    explainer = DiffAttrExplainer(model=model, scaler=scaler, n_steps=n_steps)

    df = df_pairs.copy().reset_index(drop=True)
    for s in ["1", "2"]:
        df[f"mol_{s}"] = df[f"SMILES_{s}"].apply(Chem.MolFromSmiles)

    ef1_all = scaler.transform(_extra_feat(df, "1"))
    ef2_all = scaler.transform(_extra_feat(df, "2"))

    if global_feat is not None:
        gf_all = np.tile(global_feat, (len(df), 1)).astype(np.float32)
    elif all(col in df.columns for col in GLOBAL_COLS):
        gf_all = _global_feat(df)
    else:
        gf_all = np.zeros((len(df), GLOBAL_DIM), dtype=np.float32)

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        if row.get("mol_1") is None or row.get("mol_2") is None:
            logger.warning(f"Skipping row {idx}: invalid SMILES, skipping.")
            continue

        name_1 = str(row.get("Materials_1", f"HTL1_{idx}"))
        name_2 = str(row.get("Materials_2", f"HTL2_{idx}"))
        pair_id = f"pair{idx:03d}_{name_1[:20]}_vs_{name_2[:20]}"

        logger.info(f"\n{'='*64}")
        logger.info(f"Differential attribution [{idx+1}/{len(df)}]: {pair_id}")
        logger.info(f"{'='*64}")

        try:
            result = explainer.explain_pair(
                smiles_1=row["SMILES_1"],
                smiles_2=row["SMILES_2"],
                ef_scaled_1=ef1_all[idx],
                ef_scaled_2=ef2_all[idx],
                gf=gf_all[idx],
                name_1=name_1,
                name_2=name_2,
                pair_id=pair_id,
                save_dir=save_dir,
                max_dist=mcfg.max_dist,
            )
            results.append(result)
        except Exception as exc:
            import traceback
            logger.error(f"  Failed for {pair_id}: {exc}")
            traceback.print_exc()
            continue

    merge_csvs(save_dir, "_diff_summary.csv", "all_diff_attributions.csv")
    logger.info(
        f"\nDiff attribution complete. {len(results)} pairs → {save_dir}/")
    return results
