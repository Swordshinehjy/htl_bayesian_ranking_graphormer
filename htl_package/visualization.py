import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from cairosvg import svg2png as _svg2png
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from .constants import TASK_NAMES, logger


def attr_to_rgb(v: float) -> tuple:
    v = float(np.clip(v, -1.0, 1.0))
    blue = (0.122, 0.467, 0.706)
    red = (0.839, 0.153, 0.157)
    if v < 0:
        t = v + 1
        return (blue[0] + t * (1 - blue[0]), blue[1] + t * (1 - blue[1]),
                blue[2] + t * (1 - blue[2]))
    else:
        return (1 - v * (1 - red[0]), 1 - v * (1 - red[1]),
                1 - v * (1 - red[2]))


def mol_to_image(mol: Chem.Mol, atom_attrs: np.ndarray) -> "Image.Image":
    max_abs = np.abs(atom_attrs).max() + 1e-8
    normed = atom_attrs / max_abs
    n_atoms = len(normed)

    atom_colors = {i: attr_to_rgb(normed[i]) for i in range(n_atoms)}
    h_atoms = list(range(n_atoms))
    h_bonds, b_colors = [], {}

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 < n_atoms and a2 < n_atoms:
            avg = (normed[a1] + normed[a2]) / 2.0
            h_bonds.append(bond.GetIdx())
            b_colors[bond.GetIdx()] = attr_to_rgb(avg)

    drawer = rdMolDraw2D.MolDraw2DSVG(520, 400)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=h_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=h_bonds,
        highlightBondColors=b_colors,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    try:
        png_data = _svg2png(bytestring=svg_text.encode())
        return Image.open(io.BytesIO(png_data))
    except Exception:
        return Draw.MolToImage(mol,
                               size=(520, 400),
                               highlightAtoms=h_atoms,
                               highlightAtomColors=atom_colors)


def draw_mol_attribution(
    mol: Chem.Mol,
    atom_attrs: np.ndarray,
    svg_path: str,
    png_path: str,
    title: str,
    score: float,
    target_task: int = 0,
) -> None:
    max_abs = np.abs(atom_attrs).max()
    normed = atom_attrs / (max_abs + 1e-8)
    n = len(normed)

    atom_colors = {i: attr_to_rgb(normed[i]) for i in range(n)}
    highlight_bonds, bond_colors = [], {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 < n and a2 < n:
            bi = bond.GetIdx()
            highlight_bonds.append(bi)
            bond_colors[bi] = attr_to_rgb((normed[a1] + normed[a2]) / 2)

    drawer = rdMolDraw2D.MolDraw2DSVG(600, 450)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(n)),
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()
    with open(svg_path, "w") as f:
        f.write(svg_text)

    fig, axes = plt.subplots(1,
                             2,
                             figsize=(13, 5),
                             gridspec_kw={"width_ratios": [4, 1]})
    mol_img = None
    try:
        mol_img = Image.open(
            io.BytesIO(_svg2png(bytestring=svg_text.encode())))
    except Exception:
        pass
    if mol_img is None:
        mol_img = Draw.MolToImage(mol,
                                  size=(600, 450),
                                  highlightAtoms=list(range(n)),
                                  highlightAtomColors=atom_colors)
    axes[0].imshow(mol_img)
    axes[0].axis("off")
    axes[0].set_title(
        f"{title}\nPredicted {TASK_NAMES[target_task]}: {score:.4f}",
        fontsize=12,
        fontweight="bold")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "attr", ["#1f77b4", "white", "#d62728"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm,
                        ax=axes[1],
                        orientation="vertical",
                        fraction=0.8)
    cbar.set_label("Normalized Attribution", fontsize=10)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(
        ["−1\n(negative)", "−0.5", "0", "+0.5", "+1\n(positive)"])
    axes[1].axis("off")

    top5 = np.argsort(np.abs(atom_attrs))[-5:]
    axes[0].text(0.02,
                 0.02,
                 "\n".join(f"Atom {i}: {atom_attrs[i]:+.3f}"
                           for i in top5),
                 color="black",
                 fontsize=8,
                 transform=axes[0].transAxes,
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="white",
                           alpha=0.8))
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  PNG → {png_path}")


def draw_feature_attribution(
    attrs: np.ndarray,
    names: List[str],
    save_path: str,
    title: str,
    score: float,
    target_task: int = 0,
) -> None:
    order = np.argsort(attrs)
    sa, sn = attrs[order], [names[i] for i in order]
    colors = ["#1f77b4" if v < 0 else "#d62728" for v in sa]
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45)))
    bars = ax.barh(range(len(sa)),
                   sa,
                   color=colors,
                   edgecolor="white",
                   linewidth=0.5)
    ax.set_yticks(range(len(sn)))
    ax.set_yticklabels(sn, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Integrated Gradient Attribution", fontsize=11)
    ax.set_title(
        f"{title} — Extra Feature Attribution\n"
        f"Task: {TASK_NAMES[target_task]} | Score: {score:.4f}",
        fontsize=12,
        fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xm = np.abs(sa).max() if len(sa) else 1.0
    ax.set_xlim(-xm * 1.25, xm * 1.25)
    for bar, val in zip(bars, sa):
        if abs(val) < 1e-8: continue
        ax.text(val + (xm * 0.005 if val >= 0 else -xm * 0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center",
                ha=("left" if val >= 0 else "right"),
                fontsize=8,
                color="black")
    ax.legend(handles=[
        mpatches.Patch(color="#d62728", label="Positive"),
        mpatches.Patch(color="#1f77b4", label="Negative")
    ],
              loc="lower right",
              fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Feature bar chart → {save_path}")


def draw_score_ranking(
    name_score_pairs: List[Tuple[str, float]],
    save_dir: str,
) -> None:
    names, scores = zip(*sorted(name_score_pairs, key=lambda x: x[1]))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(scores)))
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(f"Predicted {TASK_NAMES[0]} Score", fontsize=11)
    ax.set_title("Material Score Ranking", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    rng = abs(max(scores) - min(scores))
    for bar, v in zip(bars, scores):
        ax.text(v + rng * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}",
                va="center",
                fontsize=9)
    plt.tight_layout()
    path = str(Path(save_dir) / "score_ranking.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Score ranking chart → {path}")


def draw_pair_comparison(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    atom_attrs_1: np.ndarray,
    atom_attrs_2: np.ndarray,
    mcs_info: Dict[str, Any],
    struct_info: Dict[str, float],
    name_1: str,
    name_2: str,
    score_1: float,
    score_2: float,
    save_path: str,
    target_task: int = 0,
) -> None:
    import matplotlib.gridspec as gridspec

    winner = name_1 if score_1 >= score_2 else name_2
    score_diff = abs(score_1 - score_2)

    fig = plt.figure(figsize=(17, 13))
    gs = gridspec.GridSpec(2,
                           2,
                           figure=fig,
                           hspace=0.40,
                           wspace=0.28,
                           left=0.05,
                           right=0.88,
                           top=0.90,
                           bottom=0.06)

    ax_m1 = fig.add_subplot(gs[0, 0])
    ax_m2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_str = fig.add_subplot(gs[1, 1])

    for ax, mol, attrs, name, score, mcs_idx, uniq_idx in [
        (ax_m1, mol1, atom_attrs_1, name_1, score_1,
         mcs_info["mol1_mcs_atoms"], mcs_info["mol1_unique_atoms"]),
        (ax_m2, mol2, atom_attrs_2, name_2, score_2,
         mcs_info["mol2_mcs_atoms"], mcs_info["mol2_unique_atoms"]),
    ]:
        img = mol_to_image(mol, attrs)
        ax.imshow(img)
        ax.axis("off")

        uniq_sorted = sorted(uniq_idx)
        uniq_str = (f"★ unique: {uniq_sorted}"
                    if uniq_sorted else "no unique atoms vs MCS")
        title_color = "#d62728" if name == winner else "#333333"
        title_prefix = "▶ " if name == winner else "  "
        ax.set_title(
            f"{title_prefix}{name}   Score: {score:.4f}\n{uniq_str}",
            fontsize=10,
            fontweight="bold",
            color=title_color,
        )

        top5 = np.argsort(np.abs(attrs))[-5:][::-1]
        lines = [
            f"atom {i}: {attrs[i]:+.3f}{'★' if i in uniq_idx else ''}"
            for i in top5
        ]
        ax.text(0.02,
                0.02,
                "\n".join(lines),
                fontsize=7.5,
                transform=ax.transAxes,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          alpha=0.80))

    bar_colors = [
        "#d62728" if name_1 == winner else "#aaaaaa",
        "#d62728" if name_2 == winner else "#aaaaaa"
    ]
    bars = ax_bar.bar([name_1, name_2], [score_1, score_2],
                      color=bar_colors,
                      edgecolor="white",
                      linewidth=0.8,
                      width=0.45)

    score_min = min(score_1, score_2)
    score_max = max(score_1, score_2)
    score_range = score_max - score_min if score_max != score_min else 1.0

    for bar, v in zip(bars, [score_1, score_2]):
        offset = score_range * 0.05
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset),
                    f"{v:.4f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=11,
                    fontweight="bold")

    ax_bar.set_ylabel(f"Predicted {TASK_NAMES[target_task]} Score",
                      fontsize=10)
    ax_bar.set_title("Score Comparison", fontsize=11, fontweight="bold")

    if score_min >= 0:
        ax_bar.set_ylim(0, score_max * 1.18)
    else:
        margin = score_range * 0.18
        ax_bar.set_ylim(score_min - margin, score_max + margin)

    ax_bar.axhline(0,
                   color="black",
                   linewidth=0.5,
                   linestyle="-",
                   alpha=0.3)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="x", labelsize=10)

    mcs_n = struct_info["mcs_num_atoms"]
    un1 = struct_info["unique_num_atoms_1"]
    un2 = struct_info["unique_num_atoms_2"]
    labels = [
        f"MCS scaffold\n({mcs_n} atoms)",
        f"Unique fragment\n({un1} / {un2} atoms)",
    ]
    vals_1 = [struct_info["mcs_attr_1"], struct_info["unique_attr_1"]]
    vals_2 = [struct_info["mcs_attr_2"], struct_info["unique_attr_2"]]

    x = np.arange(len(labels))
    width = 0.30
    b1 = ax_str.bar(x - width / 2,
                    vals_1,
                    width,
                    label=name_1,
                    color="#d62728",
                    alpha=0.82)
    b2 = ax_str.bar(x + width / 2,
                    vals_2,
                    width,
                    label=name_2,
                    color="#1f77b4",
                    alpha=0.82)
    ax_str.set_xticks(x)
    ax_str.set_xticklabels(labels, fontsize=9)
    ax_str.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_str.set_ylabel("Σ Atom Attribution", fontsize=10)
    ax_str.set_title(
        f"Structural Attribution Breakdown\n"
        f"MCS: {mcs_n} atoms  |  ★ unique diff visualised above",
        fontsize=10,
        fontweight="bold",
    )
    ax_str.legend(fontsize=8)
    ax_str.spines["top"].set_visible(False)
    ax_str.spines["right"].set_visible(False)

    for bar, v in [(bb, vv) for bset, vvals in [(b1, vals_1), (b2, vals_2)]
                   for bb, vv in zip(bset, vvals)]:
        if abs(v) > 1e-5:
            ax_str.text(bar.get_x() + bar.get_width() / 2,
                        v + np.sign(v) * 0.005 *
                        (abs(vals_1[0]) + abs(vals_2[0]) + 1e-6),
                        f"{v:+.3f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=8)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "attr", ["#1f77b4", "white", "#d62728"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.55, 0.015, 0.32])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalised Atom Attribution", fontsize=9)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["−1\n(neg)", "0", "+1\n(pos)"])

    fig.suptitle(
        f"Differential Attribution: {name_1}  vs  {name_2}\n"
        f"Preferred: ▶ {winner}    |    Δscore = {score_diff:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Comparison figure → {save_path}")


def draw_diff_features(
    extra_names: List[str],
    attrs_1: np.ndarray,
    attrs_2: np.ndarray,
    delta: np.ndarray,
    name_1: str,
    name_2: str,
    score_1: float,
    score_2: float,
    save_path: str,
) -> None:
    n = len(extra_names)
    winner = name_1 if score_1 >= score_2 else name_2
    fig, axes = plt.subplots(1,
                             3,
                             figsize=(19, max(5, n * 0.52)),
                             constrained_layout=True)

    def _panel(ax, attrs, name, score, is_winner):
        order = np.argsort(attrs)
        sa, sn = attrs[order], [extra_names[i] for i in order]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in sa]
        ax.barh(range(n),
                sa,
                color=colors,
                edgecolor="white",
                linewidth=0.4)
        ax.set_yticks(range(n))
        ax.set_yticklabels(sn, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("IG Attribution", fontsize=10)
        title_color = "#d62728" if is_winner else "#444444"
        prefix = "▶ " if is_winner else "   "
        ax.set_title(f"{prefix}{name}\nScore: {score:.4f}",
                     fontsize=10,
                     fontweight="bold",
                     color=title_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _panel(axes[0], attrs_1, name_1, score_1, name_1 == winner)
    _panel(axes[1], attrs_2, name_2, score_2, name_2 == winner)

    order_d = np.argsort(delta)
    sd, snd = delta[order_d], [extra_names[i] for i in order_d]
    colors_d = ["#d62728" if v > 0 else "#1f77b4" for v in sd]
    bars = axes[2].barh(range(n),
                        sd,
                        color=colors_d,
                        edgecolor="white",
                        linewidth=0.4)
    axes[2].set_yticks(range(n))
    axes[2].set_yticklabels(snd, fontsize=9)
    axes[2].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Δ IG  (IG₂ − IG₁)", fontsize=10)
    axes[2].set_title(
        f"Differential Attribution\n"
        f"← {name_1}  |  → {name_2}",
        fontsize=10,
        fontweight="bold",
    )
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    x_max = np.abs(sd).max() if len(sd) > 0 else 1.0
    offset = x_max * 0.015
    axes[2].set_xlim(-x_max * 1.30, x_max * 1.30)
    for bar, val in zip(bars, sd):
        if abs(val) < 1e-8:
            continue
        axes[2].text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7.5,
            color="black",
        )

    pos_p = mpatches.Patch(color="#d62728", label=f"Favors {name_2}")
    neg_p = mpatches.Patch(color="#1f77b4", label=f"Favors {name_1}")
    axes[2].legend(handles=[pos_p, neg_p], loc="lower right", fontsize=8)

    fig.suptitle(
        f"Extra-Feature Attribution: {name_1} vs {name_2}",
        fontsize=12,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Differential feature chart → {save_path}")


def merge_csvs(save_dir: str, suffix: str, output_name: str) -> None:
    csv_files = [
        str(f) for f in Path(save_dir).iterdir()
        if f.name.endswith(suffix)
    ]
    if not csv_files:
        return
    merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    out = str(Path(save_dir) / output_name)
    merged.to_csv(out, index=False)
    logger.info(f"Merged → {out}")
