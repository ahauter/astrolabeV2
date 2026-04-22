"""Memory-mapped sampler over a flat uint16 token stream.

Each `__getitem__` samples a random window of `block_size + 1` tokens and
returns (x, y) where y is x shifted by one position — standard decoder-only
LM training. No padding: the stream is continuous and `EOF` naturally
delimits declarations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    def __init__(self, bin_path: Path, block_size: int, epoch_windows: int | None = None):
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        # `r` (read-only) memmap; contents live on disk, process reads lazily.
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        if self.data.size <= block_size + 1:
            raise ValueError(
                f"{bin_path} only has {self.data.size} tokens, need > {block_size + 1}"
            )
        # Each epoch draws this many random windows. Default: roughly one
        # window per block_size tokens of data.
        self.epoch_windows = epoch_windows or max(1, self.data.size // block_size)

    def __len__(self) -> int:
        return self.epoch_windows

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Uniform random start; torch autograd doesn't care about the index
        # argument here — DataLoader treats it as "give me one more sample".
        hi = self.data.size - self.block_size - 1
        start = int(np.random.randint(0, hi))
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
