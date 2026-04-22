"""Training and model hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # paths
    data_dir: Path = Path("data")
    out_dir: Path = Path("checkpoints")

    # model
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

    # optim
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 20000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # logging / eval
    eval_interval: int = 200
    eval_iters: int = 20
    sample_tokens: int = 256
    log_interval: int = 10

    # misc
    seed: int = 0
    compile: bool = False
