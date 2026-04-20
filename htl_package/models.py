from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS, TASK_NAMES, DEVICE
from .datasets import MolBatch
from .features import ATOM_FDIM, BOND_FDIM


class EdgeEncoding(nn.Module):

    def __init__(self, edge_dim: int, num_heads: int, max_path_len: int):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, num_heads, bias=False)
        self.path_weights = nn.Parameter(torch.randn(max_path_len, num_heads))
        nn.init.normal_(self.path_weights, std=0.02)

    def forward(self, edge_path_feats: torch.Tensor) -> torch.Tensor:
        projected = self.edge_proj(edge_path_feats)
        weights = F.softmax(self.path_weights, dim=0)
        edge_bias = (projected * weights[None, None, None, :, :]).sum(dim=-2)
        return edge_bias


class GraphormerLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            attn_bias: torch.Tensor,
            key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_bias,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GraphormerEncoder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_degree: int = 10,
        max_dist: int = 10,
        aggregation: str = "cls",
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.max_dist = max_dist

        self.atom_proj = nn.Linear(ATOM_FDIM, hidden_size)
        self.degree_emb = nn.Embedding(max_degree + 2, hidden_size)

        self.spatial_bias = nn.Embedding(max_dist + 2, num_heads)
        self.edge_encoding = EdgeEncoding(BOND_FDIM, num_heads, max_dist)

        if aggregation == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.atom_proj.weight)
        nn.init.zeros_(self.atom_proj.bias)
        nn.init.normal_(self.spatial_bias.weight, std=0.02)

    def forward(self, batch: MolBatch) -> torch.Tensor:
        B, N = batch.atom_feats.shape[:2]
        device = batch.atom_feats.device

        x = self.atom_proj(batch.atom_feats)
        deg_clamped = batch.degree.clamp(max=self.degree_emb.num_embeddings -
                                         1)
        x = x + self.degree_emb(deg_clamped)

        dist = batch.dist_matrix
        unreachable_idx = self.spatial_bias.num_embeddings - 1
        dist_idx = torch.where(
            dist < 0,
            torch.full_like(dist, unreachable_idx),
            dist.clamp(max=self.max_dist),
        )
        sp_bias = self.spatial_bias(dist_idx)

        edge_bias = self.edge_encoding(batch.edge_path_feats)

        attn_bias = sp_bias + edge_bias

        if self.aggregation == "cls":
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
            N_eff = N + 1

            zero_col = attn_bias.new_zeros(B, N, 1, self.num_heads)
            attn_bias = torch.cat([zero_col, attn_bias], dim=2)
            zero_row = attn_bias.new_zeros(B, 1, N_eff, self.num_heads)
            attn_bias = torch.cat([zero_row, attn_bias], dim=1)

            cls_false = torch.zeros(B, 1, dtype=torch.bool, device=device)
            key_pad = torch.cat([cls_false, batch.padding_mask], dim=1)
        else:
            N_eff = N
            key_pad = batch.padding_mask

        attn_bias_mha = (attn_bias.permute(0, 3, 1,
                                           2).reshape(B * self.num_heads,
                                                      N_eff, N_eff))

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_bias_mha, key_pad)
        x = self.norm(x)

        if self.aggregation == "cls":
            return x[:, 0, :]
        elif self.aggregation == "mean":
            mask = (~batch.padding_mask).float().unsqueeze(-1)
            return (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            mask = (~batch.padding_mask).float().unsqueeze(-1)
            return (x * mask).sum(1)


class HTLRankingModel(nn.Module):

    def __init__(
        self,
        hidden_size: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden: int = 256,
        extra_dim: int = EXTRA_DIM,
        global_dim: int = GLOBAL_DIM,
        num_tasks: int = NUM_TASKS,
        aggregation: str = "cls",
        max_degree: int = 10,
        max_dist: int = 10,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder = GraphormerEncoder(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            max_degree=max_degree,
            max_dist=max_dist,
            aggregation=aggregation,
        )

        ffn_in = hidden_size + extra_dim + global_dim
        self.ffn = nn.Sequential(
            nn.Linear(ffn_in, ffn_hidden),
            nn.LayerNorm(ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, ffn_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(ffn_hidden // 2, num_tasks),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self,
        mol_batch: MolBatch,
        extra: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        mol_batch = mol_batch.to(extra.device)
        emb = self.encoder(mol_batch)
        x = torch.cat([emb, extra, global_feat], dim=-1)
        return self.ffn(x)

    def forward(
        self,
        mb1: MolBatch,
        ef1: torch.Tensor,
        mb2: MolBatch,
        ef2: torch.Tensor,
        gf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode(mb1, ef1, gf), self.encode(mb2, ef2, gf)


class BayesianRankingLoss(nn.Module):

    def __init__(
        self,
        rank_weight: float = 0.6,
        reg_weight: float = 0.4,
        task_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.alpha = rank_weight
        self.beta = reg_weight
        self.task_w = task_weights or [1.0] * NUM_TASKS

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=s1.device)
        log: Dict[str, float] = {}

        for t, (name, lam) in enumerate(zip(TASK_NAMES, self.task_w)):
            diff_s = s1[:, t] - s2[:, t]
            diff_y = y1[:, t] - y2[:, t]

            sign = diff_y.sign()

            nonzero_mask = sign != 0
            if nonzero_mask.any():
                sign_nonzero = sign[nonzero_mask]
                diff_s_nonzero = diff_s[nonzero_mask]
                l_rank = -F.logsigmoid(sign_nonzero * diff_s_nonzero).mean()
            else:
                l_rank = torch.tensor(0.0, device=s1.device)

            l_reg = F.mse_loss(diff_s, diff_y)

            task_loss = lam * (self.alpha * l_rank + self.beta * l_reg)
            total += task_loss

            log[f"{name}_rank"] = l_rank.item()
            log[f"{name}_reg"] = l_reg.item()

        log["total"] = total.item()
        return total, log


class EarlyStopping:

    def __init__(self, patience: int = 50, delta: float = 1e-4, warmup: int = 0):
        self.patience = patience
        self.delta = delta
        self.warmup = warmup
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[Dict] = None
        self._epoch = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        self._epoch += 1

        if self._epoch <= self.warmup:
            return False

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience
