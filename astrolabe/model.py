"""Small decoder-only transformer, nanoGPT-flavored.

Uses pre-norm blocks, learned positional embeddings, and weight-tied
input/output embeddings. Four auxiliary CFG heads attach to the final hidden
states for multi-task pretraining; they are dropped after pretraining.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        # CFG auxiliary heads (dropped after pretraining)
        self.bb_head      = nn.Linear(cfg.n_embd, 1, bias=False)
        self.du_use_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.du_def_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.edge_head    = nn.Linear(cfg.n_embd, 4, bias=False)
        self.dom_head     = nn.Linear(cfg.n_embd * 2, 1, bias=False)

        # Risk heads (nil-deref + bounds-check)
        self.risk_nil_head    = nn.Linear(cfg.n_embd, 1, bias=False)
        self.risk_bounds_head = nn.Linear(cfg.n_embd, 1, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        bb_labels: torch.Tensor | None = None,
        du_pairs: torch.Tensor | None = None,
        edge_labels: torch.Tensor | None = None,
        dom_pairs: torch.Tensor | None = None,
        nil_labels: torch.Tensor | None = None,
        bounds_labels: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block_size"
        pos = torch.arange(T, device=idx.device)
        h = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        logits = self.head(h)
        lm_loss = bb_loss = du_loss = edge_loss = dom_loss = nil_loss = bounds_loss = None

        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if bb_labels is not None:
            bb_logits = self.bb_head(h).squeeze(-1)  # (B, T)
            bb_loss = F.binary_cross_entropy_with_logits(bb_logits, bb_labels.float())

        if du_pairs is not None:
            # du_pairs: (B, MAX_DU_PAIRS, 2) — use_pos, def_pos; -1 = padding
            valid = du_pairs[:, :, 0] >= 0  # (B, P)
            if valid.any():
                use_pos = du_pairs[:, :, 0].clamp(min=0)  # (B, P)
                def_tgt = du_pairs[:, :, 1].clamp(min=0)  # (B, P)
                h_use = self.du_use_proj(h)  # (B, T, C)
                h_def = self.du_def_proj(h)  # (B, T, C)
                bidx = torch.arange(B, device=h.device).unsqueeze(1).expand_as(use_pos)
                h_at_use = h_use[bidx, use_pos]              # (B, P, C)
                scores = torch.bmm(h_at_use, h_def.transpose(1, 2))  # (B, P, T)
                du_loss = F.cross_entropy(scores[valid], def_tgt[valid])

        if edge_labels is not None and (edge_labels >= 0).any():
            edge_logits = self.edge_head(h)  # (B, T, 4)
            edge_loss = F.cross_entropy(
                edge_logits.view(-1, 4),
                edge_labels.view(-1),
                ignore_index=-1,
            )

        if dom_pairs is not None:
            # dom_pairs: (B, MAX_DOM_PAIRS, 3) — A_pos, B_pos, label; -1 = padding
            valid = dom_pairs[:, :, 0] >= 0  # (B, P)
            if valid.any():
                A_pos  = dom_pairs[:, :, 0].clamp(min=0)  # (B, P)
                B_pos  = dom_pairs[:, :, 1].clamp(min=0)
                labels = dom_pairs[:, :, 2].float()
                bidx   = torch.arange(B, device=h.device).unsqueeze(1).expand_as(A_pos)
                h_A = h[bidx, A_pos]                            # (B, P, C)
                h_B = h[bidx, B_pos]
                pair_emb  = torch.cat([h_A, h_B], dim=-1)       # (B, P, 2C)
                dom_logits = self.dom_head(pair_emb).squeeze(-1) # (B, P)
                dom_loss = F.binary_cross_entropy_with_logits(
                    dom_logits[valid], labels[valid]
                )

        if nil_labels is not None:
            nil_logits = self.risk_nil_head(h).squeeze(-1)  # (B, T)
            nil_loss = F.binary_cross_entropy_with_logits(nil_logits, nil_labels.float())

        if bounds_labels is not None:
            bounds_logits = self.risk_bounds_head(h).squeeze(-1)  # (B, T)
            bounds_loss = F.binary_cross_entropy_with_logits(bounds_logits, bounds_labels.float())

        return logits, lm_loss, bb_loss, du_loss, edge_loss, dom_loss, nil_loss, bounds_loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, *_ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -math.inf
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def detect_risks(
        self,
        idx: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Run forward on a single sequence and return risky positions.

        Returns:
            nil_positions:    list of (token_pos, confidence)
            bounds_positions:   list of (token_pos, confidence)
        """
        self.eval()
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        nil_logits = self.risk_nil_head(h).squeeze(-1)       # (B, T)
        bounds_logits = self.risk_bounds_head(h).squeeze(-1)  # (B, T)

        nil_probs = torch.sigmoid(nil_logits).squeeze(0)      # (T,)
        bounds_probs = torch.sigmoid(bounds_logits).squeeze(0)  # (T,)

        nil_risks = [
            (i, p.item())
            for i, p in enumerate(nil_probs)
            if p.item() >= threshold
        ]
        bounds_risks = [
            (i, p.item())
            for i, p in enumerate(bounds_probs)
            if p.item() >= threshold
        ]
        return nil_risks, bounds_risks
