"""Risk dataset: loads declaration units + pre-computed risk labels.

Extends CFGUnitMixDataset to also read nil_risks / bounds_risks JSON lines
produced by risk_miner.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from astrolabe.dataset import CFGUnitMixDataset
from astrolabe.vocab import BOS_ID, EOF_ID, PAD_ID


class RiskUnitDataset(Dataset):
    """Loads units with per-position binary risk labels.

    Each sample is BOS + randomly packed units padded to block_size, plus
    two float label tensors aligned to the assembled sequence:
        nil_labels     (block_size,) — 1.0 at nil-risk positions, 0.0 elsewhere
        bounds_labels  (block_size,) — 1.0 at bounds-risk positions, 0.0 elsewhere
    """

    def __init__(
        self,
        units_bin: Path,
        units_idx: Path,
        ann_path: Path | None,
        risk_path: Path,
        block_size: int,
        epoch_samples: int | None = None,
        max_units: int | None = None,
    ):
        self.data    = np.memmap(Path(units_bin), dtype=np.uint16, mode="r")
        self.offsets = np.load(Path(units_idx))
        self.n_units = len(self.offsets) - 1
        if self.n_units == 0:
            raise ValueError(f"no units found in {units_bin}")

        # Only load risk labels (ann_path kept for API compat but unused).
        with open(Path(risk_path)) as f:
            self.risks = [json.loads(line) for line in f]

        if max_units is not None:
            self.n_units = min(self.n_units, max_units)
            self.risks = self.risks[:self.n_units]

        if len(self.risks) != self.n_units:
            raise ValueError(
                f"risk count mismatch: risk={len(self.risks)} units={self.n_units}"
            )

        self.block_size    = block_size
        self.epoch_samples = epoch_samples or self.n_units

    def __len__(self) -> int:
        return self.epoch_samples

    def __getitem__(self, _: int) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        T = self.block_size
        target = T + 1

        tokens = [BOS_ID]
        nil_labels    = np.zeros(T, dtype=np.float32)
        bounds_labels = np.zeros(T, dtype=np.float32)

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

            risk = self.risks[i]
            for p in risk.get("nil_risks", []):
                pos = unit_offset + p
                if 0 <= pos < T:
                    nil_labels[pos] = 1.0
            for p in risk.get("bounds_risks", []):
                pos = unit_offset + p
                if 0 <= pos < T:
                    bounds_labels[pos] = 1.0

        tokens.extend([PAD_ID] * (target - len(tokens)))
        chunk = np.array(tokens[:target], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return (
            x,
            y,
            torch.from_numpy(nil_labels),
            torch.from_numpy(bounds_labels),
        )
