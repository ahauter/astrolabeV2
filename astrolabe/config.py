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
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    # optim
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 40000
    warmup_steps: int = 400
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

    # CFG auxiliary head loss weights (0 = disable)
    cfg_bb_weight:   float = 0.5
    cfg_du_weight:   float = 0.5
    cfg_edge_weight: float = 0.3
    cfg_dom_weight:  float = 0.3

    # misc
    seed: int = 0
    compile: bool = False


@dataclass
class RiskTrainConfig:
    # paths
    data_dir: Path = Path("data")
    out_dir: Path = Path("checkpoints_risk")

    # model (must match pretrained checkpoint)
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    # risk head weights
    risk_nil_weight: float = 2.0
    risk_bounds_weight: float = 2.0

    # positive-class weighting for BCE (None = no weighting)
    risk_nil_pos_weight: float | None = None
    risk_bounds_pos_weight: float | None = None

    # freeze schedule
    freeze_backbone_steps: int = 1000

    # optim
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 5000
    warmup_steps: int = 200
    lr: float = 1e-4
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # logging / eval
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 10

    # misc
    seed: int = 0
    compile: bool = False


@dataclass
class RaceTrainConfig:
    # paths
    data_dir: Path = Path("data_race")
    out_dir: Path = Path("checkpoints_race")
    resume: Path | None = None

    # model (must match pretrained checkpoint)
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    # hierarchical context
    func_len: int = 256
    caller_len: int = 128
    max_callers: int = 8
    n_ctx_layers: int = 2

    # risk heads
    risk_race_weight: float = 1.0
    risk_race_pos_weight: float | None = 10.0
    risk_nil_weight: float = 2.0
    risk_nil_pos_weight: float | None = None
    risk_bounds_weight: float = 2.0
    risk_bounds_pos_weight: float | None = None

    # sync hard-negative weighting
    sync_neg_weight: float = 5.0

    # online mutation fractions (must sum to <= 1.0)
    mutate_frac: float = 0.2
    unlock_mutate_frac: float = 0.0

    # freeze schedule
    freeze_backbone_steps: int = 1000

    # optim
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 5000
    warmup_steps: int = 200
    lr: float = 1e-4
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # logging / eval
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 10

    # misc
    seed: int = 0
    compile: bool = False
