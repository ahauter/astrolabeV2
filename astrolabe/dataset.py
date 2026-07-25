"""Datasets for AST-token pretraining.

TokenWindowDataset: legacy random-window sampler over a flat uint16 stream.

CFGUnitMixDataset: loads declaration units saved by prepare.py, assembles
random permutations on-the-fly, and builds per-position CFG label tensors
(block-boundary, def-use, edge-type, dominance) aligned to each sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from astrolabe.vocab import BOS_ID, EOF_ID, PAD_ID


class TokenWindowDataset(Dataset):
    """Memory-mapped sampler over a flat uint16 token stream (legacy format)."""

    def __init__(self, bin_path: Path, block_size: int, epoch_windows: int | None = None):
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        if self.data.size <= block_size + 1:
            raise ValueError(
                f"{bin_path} only has {self.data.size} tokens, need > {block_size + 1}"
            )
        self.epoch_windows = epoch_windows or max(1, self.data.size // block_size)

    def __len__(self) -> int:
        return self.epoch_windows

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        hi = self.data.size - self.block_size - 1
        start = int(np.random.randint(0, hi))
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class CFGUnitMixDataset(Dataset):
    """Live-mix dataset with CFG supervision labels.

    Loads declaration units and their CFG annotations produced by prepare.py.
    Each sample is BOS + randomly packed units padded to block_size, plus four
    label tensors aligned to the assembled sequence:

    - bb_labels   (block_size,) int64: 1 at block-boundary tokens, 0 elsewhere
    - du_pairs    (MAX_DU_PAIRS,  2) int64: (use_pos, def_pos); -1 rows = padding
    - edge_labels (block_size,) int64: edge class at block-exit tokens, -1 elsewhere
    - dom_pairs   (MAX_DOM_PAIRS, 3) int64: (A_pos, B_pos, label); -1 rows = padding

    Edge classes: 0=conditional(T/F), 2=unconditional, 3=back edge.
    """

    MAX_DU_PAIRS  = 64
    MAX_DOM_PAIRS = 32

    def __init__(
        self,
        units_bin: Path,
        units_idx: Path,
        ann_path: Path,
        block_size: int,
        epoch_samples: int | None = None,
    ):
        self.data    = np.memmap(Path(units_bin), dtype=np.uint16, mode="r")
        self.offsets = np.load(Path(units_idx))
        self.n_units = len(self.offsets) - 1
        if self.n_units == 0:
            raise ValueError(f"no units found in {units_bin}")
        with open(Path(ann_path)) as f:
            self.anns = [json.loads(line) for line in f]
        if len(self.anns) != self.n_units:
            raise ValueError(
                f"ann count {len(self.anns)} != unit count {self.n_units} in {units_bin}"
            )
        # json.loads("null") → None, so None entries round-trip correctly.
        self.block_size   = block_size
        self.epoch_samples = epoch_samples or self.n_units

    def __len__(self) -> int:
        return self.epoch_samples

    def __getitem__(self, _: int) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        T = self.block_size
        target = T + 1

        tokens = [BOS_ID]
        bb_labels   = np.zeros(T,                          dtype=np.int64)
        du_pairs    = np.full((self.MAX_DU_PAIRS,  2), -1, dtype=np.int64)
        edge_labels = np.full(T,                      -1, dtype=np.int64)
        dom_pairs   = np.full((self.MAX_DOM_PAIRS, 3), -1, dtype=np.int64)
        du_count  = 0
        dom_count = 0

        while len(tokens) < target:
            i = int(np.random.randint(self.n_units))
            start = int(self.offsets[i])
            end   = int(self.offsets[i + 1])
            unit  = self.data[start:end]
            if len(tokens) + len(unit) + 1 > target:
                break
            unit_offset = len(tokens)
            tokens.extend(unit.tolist())
            tokens.append(EOF_ID)

            ann = self.anns[i]
            if ann is None:
                continue

            bb_list = ann.get("bb", [])

            # Block-boundary labels
            for p in bb_list:
                pos = unit_offset + p
                if 0 <= pos < T:
                    bb_labels[pos] = 1

            # Edge labels — B > U > conditional(T/F)
            for exit_str, succs in ann.get("edges", {}).items():
                exit_pos = unit_offset + int(exit_str)
                if not (0 <= exit_pos < T):
                    continue
                types = list(succs.values())
                if "B" in types:
                    etype = 3
                elif "U" in types:
                    etype = 2
                else:
                    etype = 0
                edge_labels[exit_pos] = etype

            # Def-use pairs
            for use_str, def_idx in ann.get("du", {}).items():
                if du_count >= self.MAX_DU_PAIRS:
                    break
                use_pos = unit_offset + int(use_str)
                def_pos = unit_offset + int(def_idx)
                if 0 <= use_pos < T and 0 <= def_pos < T:
                    du_pairs[du_count] = [use_pos, def_pos]
                    du_count += 1

            # Dominance pairs — positive from idom, balanced negatives
            n_blocks = len(bb_list)
            idom = ann.get("idom", {})
            if n_blocks >= 2 and idom:
                pos_set: set[tuple[int, int]] = set()
                for b_str, a_idx in idom.items():
                    if dom_count >= self.MAX_DOM_PAIRS:
                        break
                    if a_idx < 0:
                        continue
                    b_idx = int(b_str)
                    if a_idx >= n_blocks or b_idx >= n_blocks:
                        continue
                    a_pos = unit_offset + bb_list[a_idx]
                    b_pos = unit_offset + bb_list[b_idx]
                    if 0 <= a_pos < T and 0 <= b_pos < T:
                        dom_pairs[dom_count] = [a_pos, b_pos, 1]
                        dom_count += 1
                        pos_set.add((b_idx, a_idx))

                n_neg     = min(len(pos_set), self.MAX_DOM_PAIRS - dom_count)
                attempts  = 0
                while n_neg > 0 and attempts < 32:
                    attempts += 1
                    a_idx = int(np.random.randint(n_blocks))
                    b_idx = int(np.random.randint(n_blocks))
                    if a_idx == b_idx or (b_idx, a_idx) in pos_set:
                        continue
                    a_pos = unit_offset + bb_list[a_idx]
                    b_pos = unit_offset + bb_list[b_idx]
                    if 0 <= a_pos < T and 0 <= b_pos < T:
                        dom_pairs[dom_count] = [a_pos, b_pos, 0]
                        dom_count += 1
                        n_neg -= 1

        tokens.extend([PAD_ID] * (target - len(tokens)))
        chunk = np.array(tokens[:target], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return (
            x,
            y,
            torch.from_numpy(bb_labels),
            torch.from_numpy(du_pairs),
            torch.from_numpy(edge_labels),
            torch.from_numpy(dom_pairs),
        )
