"""
Perovskite HTL Prediction — Graphormer Version
Pipeline:
  1. Custom 2D Graph Transformer encoding + additional feature concatenation
  2. Bayesian Ranking Loss (Bradley-Terry sigmoid model)

Architecture Changes (compared to D-MPNN version):
  - Completely removed chemprop dependency, using RDKit + scipy + pure PyTorch implementation
  - DMPNNEncoder  →  GraphormerEncoder
      · Atom feature projection + degree encoding  (Centrality Encoding)
      · Spatial encoding               (Spatial Encoding, per head bias based on shortest path distance)
      · Bond feature path encoding     (Edge Encoding, mean bond features on shortest path)
      · Multi-layer Graph Transformer  (Pre-LN Multi-Head Self-Attention + FFN + GELU)
      · [CLS] virtual node readout (or mean/sum aggregation)
  - MolGraph / BatchMolGraph  →  MolGraphData / MolBatch
      · Molecular graphs are precomputed once during Dataset initialization (shortest paths, mean bond path features, etc.)
      · collate_mol_graphs handles padding and concatenation into MolBatch tensors
  - Atom IG in IGExplainer changed to compute gradients on MolBatch.atom_feats
  - Bayesian Ranking Loss → Bradley-Terry sigmoid model
      · P(i > j) = sigma(s1 - s2)  (sigmoid)
      · L_bayes = -log sigma( sign(y1 - y2) * (s1 - s2) )
      · L_reg  = MSE(s1 - s2,  y1 - y2)
      · No longer requires margin hyperparameter, sigmoid provides natural calibration
      · Smooth gradients (compared to ReLU's zero gradient issue)
      · Probabilistic interpretation (output is ranking probability)
"""

from htl_package.configs import ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig
from htl_package.training import train, finetune
from htl_package.prediction import predict_batch, predict_list
from htl_package.explainer import explain, diff_attr

if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser(
        description="HTL Bayesian Ranking via Graphormer")
    p.add_argument("--mode",
                   choices=[
                       "train", "predict", "finetune", "list_rank", "explain",
                       "diff_attr"
                   ],
                   default="train")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--predict_csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--max_degree", type=int, default=None)
    p.add_argument("--max_dist", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--ffn_hidden", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--auto_compute_stats", action="store_true", default=True)
    p.add_argument("--no_auto_compute_stats",
                   action="store_false",
                   dest="auto_compute_stats")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--early_stop_warmup", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cache_val_test", action="store_true", default=True)
    p.add_argument("--no_cache_val_test",
                   action="store_false",
                   dest="cache_val_test")
    p.add_argument("--split", type=str, choices=["random", "group"],
                   default=None)
    p.add_argument("--n_cv_folds", type=int, default=None)
    p.add_argument("--finetune_epochs", type=int, default=None)
    p.add_argument("--finetune_lr", type=float, default=None)
    p.add_argument("--explain_csv", type=str, default=None)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--explain_dir", type=str, default="explain_output")
    p.add_argument("--attrib_extra_to_atom", action="store_true")
    p.add_argument("--diff_csv", type=str, default=None)
    p.add_argument("--diff_dir", type=str, default="diff_output")
    args = p.parse_args()

    def _merge(config_cls, args, mapping):
        kwargs = {}
        for cf, an in mapping.items():
            v = getattr(args, an)
            if v is not None:
                kwargs[cf] = v
        return config_cls(**kwargs)

    if args.mode == "train":
        mc = _merge(
            ModelConfig, args, {
                "hidden_size": "hidden_size",
                "depth": "depth",
                "num_heads": "num_heads",
                "dropout": "dropout",
                "ffn_hidden": "ffn_hidden",
                "max_dist": "max_dist",
                "max_degree": "max_degree",
                "auto_compute_stats": "auto_compute_stats"
            })
        tc = _merge(
            TrainingConfig, args, {
                "csv_path": "csv",
                "save_dir": "save_dir",
                "epochs": "epochs",
                "batch_size": "batch_size",
                "lr": "lr",
                "weight_decay": "weight_decay",
                "patience": "patience",
                "early_stop_warmup": "early_stop_warmup",
                "val_ratio": "val_ratio",
                "test_ratio": "test_ratio",
                "seed": "seed",
                "cache_val_test": "cache_val_test",
                "split": "split",
                "n_cv_folds": "n_cv_folds",
            })
        res = train(mc, tc)
        print("\n===== Final Test Metrics =====")
        for k, v in res["test_metrics"].items():
            print(f"  {k:30s}: {v:.4f}")

    elif args.mode == "finetune":
        fc = _merge(
            FinetuneConfig, args, {
                "csv_path": "csv",
                "checkpoint_path": "checkpoint",
                "save_dir": "save_dir",
                "finetune_epochs": "finetune_epochs",
                "batch_size": "batch_size",
                "lr": "finetune_lr",
                "seed": "seed"
            })
        res = finetune(fc)
        print(f"\n===== Final model → {res['final_checkpoint']} =====")

    elif args.mode == "predict":
        if not args.predict_csv:
            p.error("--predict_csv required")
        pc = _merge(
            PredictConfig, args, {
                "predict_csv": "predict_csv",
                "checkpoint_path": "checkpoint",
                "output_path": "output"
            })
        df_new = pd.read_csv(pc.predict_csv)
        res = predict_batch(df_new,
                            pc.checkpoint_path,
                            pc.output_path,
                            batch_size=args.batch_size or 32)
        print(res[["Materials_1", "Materials_2", "preferred_PCE"]])

    elif args.mode == "list_rank":
        if not args.predict_csv:
            p.error("--predict_csv required")
        df_list = pd.read_csv(args.predict_csv)
        res = predict_list(df_list,
                           checkpoint_path=args.checkpoint
                           or "checkpoints/final_model.pt",
                           output_path=args.output,
                           batch_size=args.batch_size or 32)
        print(res[["rank", "Materials", "score_PCE"]])

    elif args.mode == "explain":
        if not args.explain_csv:
            p.error("--explain_csv required")
        if not args.checkpoint:
            p.error("--checkpoint required")
        df_exp = pd.read_csv(args.explain_csv)
        results = explain(df_exp,
                          checkpoint_path=args.checkpoint,
                          save_dir=args.explain_dir or "explain_output",
                          n_steps=args.n_steps or 50,
                          attrib_extra_to_atom=args.attrib_extra_to_atom)
        print(f"\n===== Explain Done: {len(results)} materials =====")
        for r in results:
            print(
                f"  {r['material_name']:30s} score={r['score']:.4f}  "
                f"pos_atoms={r['pos_atoms']}  "
                f"top_feature={r['extra_names'][int(np.argmax(r['extra_attrs']))]}"
            )

    elif args.mode == "diff_attr":
        if not args.diff_csv:
            p.error("--diff_csv required")
        if not args.checkpoint:
            p.error("--checkpoint required")
        df_diff = pd.read_csv(args.diff_csv)
        results = diff_attr(
            df_pairs=df_diff,
            checkpoint_path=args.checkpoint,
            save_dir=args.diff_dir or "diff_output",
            n_steps=args.n_steps or 50,
        )
        print(f"\n===== Diff Attribution Done: {len(results)} pairs =====")
        print(
            f"{'Pair':<50} {'Δscore':>9}  {'Preferred':<20}  {'Top Δfeature (val)'}"
        )
        print("-" * 105)
        for r in results:
            best_feat_idx = int(np.argmax(np.abs(r["delta_extra"])))
            best_feat = r["extra_names"][best_feat_idx]
            best_val = r["delta_extra"][best_feat_idx]
            print(f"  {r['pair_id'][:48]:<50} "
                  f"{r['score_diff']:+9.4f}  "
                  f"{r['preferred']:<20}  "
                  f"{best_feat} ({best_val:+.4f})")

"""
# Usage Examples (Command Line)

# Training (Bayesian Ranking, Bradley-Terry sigmoid model)
python htl_ranking_graphormer.py --mode train --csv htl-data-combinations.csv --hidden_size 300 --num_heads 6 --depth 3

# Training (group split + LOGO CV)
python htl_ranking_graphormer.py --mode train --csv htl-data-combinations.csv --split group --n_cv_folds 5

# Fine-tuning
python htl_ranking_graphormer.py --mode finetune --csv htl-data-combinations.csv --checkpoint checkpoints/best_model.pt --finetune_epochs 10 --finetune_lr 1e-5

# Pairwise prediction (output includes score, ranking probability)
python htl_ranking_graphormer.py --mode predict --predict_csv htl-new.csv --checkpoint checkpoints/final_model.pt

# List ranking
python htl_ranking_graphormer.py --mode list_rank --predict_csv ranking-new.csv --checkpoint checkpoints/final_model.pt --output ranked_results.csv

# Explanation
python htl_ranking_graphormer.py --mode explain --explain_csv ranking-new.csv --checkpoint checkpoints/final_model.pt --explain_dir explain_output --n_steps 100

# Differential attribution analysis
python htl_ranking_graphormer.py --mode diff_attr --diff_csv htl-new.csv --checkpoint checkpoints/final_model.pt --diff_dir diff_output --n_steps 100
"""
