"""Fine-tune pretrained GPT with risk classification heads.

Loads a pretrained checkpoint, freezes backbone for N steps, then low-LR
fine-tunes the full model on nil-deref + bounds-check binary labels.

Usage:
    python -m astrolabe.finetune_risk \
        --data-dir data \
        --out-dir checkpoints_risk_v3 \
        --resume checkpoints/ckpt_100000.pt
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from astrolabe.config import RiskTrainConfig
from astrolabe.risk_dataset import RiskUnitDataset
from astrolabe.model import GPT, GPTConfig
from astrolabe.vocab import BOS_ID, VOCAB_SIZE, bracket_balance_rate


def get_lr(step: int, cfg: RiskTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def _to(batch: tuple, device: str) -> tuple:
    return tuple(t.to(device, non_blocking=True) for t in batch)


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str, iters: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {k: 0.0 for k in ("total", "nil", "bounds")}
    counts: dict[str, int] = {k: 0 for k in totals}
    it = iter(loader)
    for _ in range(iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        x, y, nil_l, bounds_l = _to(batch, device)
        _, lm, bb, du, edge, dom, nil_loss, bounds_loss = model(
            x, y, None, None, None, None, nil_l, bounds_l
        )
        loss = 0.0
        if nil_loss is not None:
            loss += nil_loss.item()
            totals["nil"] += nil_loss.item()
            counts["nil"] += 1
        if bounds_loss is not None:
            loss += bounds_loss.item()
            totals["bounds"] += bounds_loss.item()
            counts["bounds"] += 1
        totals["total"] += loss
        counts["total"] += 1
    model.train()
    return {k: totals[k] / max(1, counts[k]) for k in totals}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    cfg = RiskTrainConfig()
    for k, v in asdict(cfg).items():
        kind = type(v)
        flag = "--" + k.replace("_", "-")
        if kind is bool:
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=v)
        elif kind is Path:
            p.add_argument(flag, type=Path, default=v)
        else:
            p.add_argument(flag, type=kind, default=v)
    p.add_argument("--resume", type=Path, required=True,
                   help="Pretrained checkpoint to resume from")
    p.add_argument("--max-units", type=int, default=None,
                   help="Limit dataset to first N units (for POC)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume = args.resume
    max_units = args.max_units
    del args.resume
    del args.max_units
    cfg = RiskTrainConfig(**vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() \
        else torch.float32

    torch.manual_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RiskUnitDataset(
        cfg.data_dir / "train_units.bin",
        cfg.data_dir / "train_units.idx.npy",
        None,
        cfg.data_dir / "risk_train_v3.jsonl",
        cfg.block_size,
        max_units=max_units,
    )
    val_ds = RiskUnitDataset(
        cfg.data_dir / "val_units.bin",
        cfg.data_dir / "val_units.idx.npy",
        None,
        cfg.data_dir / "risk_val_v3.jsonl",
        cfg.block_size,
        max_units=max_units,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, num_workers=0)

    gpt_cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(gpt_cfg).to(device)
    print(f"model params: {model.num_params() / 1e6:.2f}M  "
          f"vocab: {VOCAB_SIZE}  device: {device}  dtype: {dtype}")

    if cfg.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    step = 0
    ckpt = torch.load(resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    # opt state from pretraining is not useful for new heads; skip it.
    print(f"loaded pretrained checkpoint from {resume}")

    # Freeze backbone parameters for first N steps.
    risk_params = list(model.risk_nil_head.parameters()) + list(model.risk_bounds_head.parameters())
    for p in model.parameters():
        p.requires_grad = False
    for p in risk_params:
        p.requires_grad = True

    running: dict[str, float] = {k: 0.0 for k in ("total", "nil", "bounds")}
    running_n = 0
    t0 = time.time()
    train_iter = iter(train_dl)

    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        # Unfreeze backbone after freeze period.
        if step == cfg.freeze_backbone_steps:
            for p in model.parameters():
                p.requires_grad = True
            print(f"[step {step}] backbone unfrozen")

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

            x, y, nil_l, bounds_l = _to(batch, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, lm_loss, bb_loss, du_loss, edge_loss, dom_loss, nil_loss, bounds_loss = model(
                    x, y, None, None, None, None, nil_l, bounds_l
                )
                loss = 0.0
                if nil_loss is not None and cfg.risk_nil_weight > 0:
                    loss = loss + cfg.risk_nil_weight * nil_loss
                if bounds_loss is not None and cfg.risk_bounds_weight > 0:
                    loss = loss + cfg.risk_bounds_weight * bounds_loss
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            for name, val in (("nil", nil_loss), ("bounds", bounds_loss)):
                if val is not None:
                    running[name] += val.item()
            running["total"] += loss.item() * cfg.grad_accum_steps
            running_n += 1

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1

        if step % cfg.log_interval == 0:
            scale = max(1, running_n)
            parts = "  ".join(
                f"{k} {running[k] / scale:.4f}"
                for k in ("total", "nil", "bounds")
            )
            dt = (time.time() - t0) / cfg.log_interval
            print(f"step {step:6d}  {parts}  lr {lr:.2e}  {dt*1000:.0f}ms/step", flush=True)
            for k in running:
                running[k] = 0.0
            running_n = 0
            t0 = time.time()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            eval_losses = evaluate(model, val_dl, device, cfg.eval_iters)
            parts = "  ".join(f"val_{k} {v:.4f}" for k, v in eval_losses.items())
            print(f"[eval] step {step}  {parts}", flush=True)
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": asdict(cfg),
                "gpt_cfg": asdict(gpt_cfg),
            }
            torch.save(ckpt, cfg.out_dir / f"ckpt_risk_{step}.pt")


if __name__ == "__main__":
    main()
