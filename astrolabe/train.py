"""AST-token pretraining loop.

Next-token prediction over the packed token stream produced by
`astrolabe.prepare`. Logs train/val loss plus a bracket-balance rate on
sampled generations — a cheap structural-validity metric unique to this
tokenization.

Usage:
    python -m astrolabe.train --data-dir data --out-dir checkpoints
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from astrolabe.config import TrainConfig
from astrolabe.dataset import TokenWindowDataset
from astrolabe.model import GPT, GPTConfig
from astrolabe.vocab import BOS_ID, VOCAB_SIZE, bracket_balance_rate


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str, iters: int) -> float:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample_balance(model: GPT, device: str, n_tokens: int) -> float:
    model.eval()
    start = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=n_tokens, temperature=1.0, top_k=40)
    model.train()
    return bracket_balance_rate(out[0].tolist())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    cfg = TrainConfig()
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(**vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() \
        else torch.float32

    torch.manual_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TokenWindowDataset(cfg.data_dir / "train.bin", cfg.block_size)
    val_ds = TokenWindowDataset(cfg.data_dir / "val.bin", cfg.block_size)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)

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
    running_loss = 0.0
    running_n = 0
    t0 = time.time()
    train_iter = iter(train_dl)

    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps
            running_n += 1

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1

        if step % cfg.log_interval == 0:
            avg = running_loss / max(1, running_n)
            dt = (time.time() - t0) / cfg.log_interval
            print(f"step {step:6d}  loss {avg:.4f}  lr {lr:.2e}  {dt*1000:.0f}ms/step",
                  flush=True)
            running_loss = 0.0
            running_n = 0
            t0 = time.time()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val_loss = evaluate(model, val_dl, device, cfg.eval_iters)
            balance = sample_balance(model, device, cfg.sample_tokens)
            print(f"[eval] step {step}  val_loss {val_loss:.4f}  "
                  f"bracket_balance {balance:.3f}", flush=True)
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": asdict(cfg),
                "gpt_cfg": asdict(gpt_cfg),
            }
            torch.save(ckpt, cfg.out_dir / f"ckpt_{step}.pt")


if __name__ == "__main__":
    main()
