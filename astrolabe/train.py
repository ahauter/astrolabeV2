"""AST-token pretraining loop with CFG auxiliary heads.

Five simultaneous objectives: LM (next-token), block-boundary (BB), def-use
(DU), edge-type (Edge), and dominance (DOM).  CFG losses are weighted by the
cfg_*_weight fields in TrainConfig; set any weight to 0 to disable that head.

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
from astrolabe.dataset import CFGUnitMixDataset
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


def _to(batch: tuple, device: str) -> tuple:
    return tuple(t.to(device, non_blocking=True) for t in batch)


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str, iters: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {k: 0.0 for k in ("lm", "bb", "du", "edge", "dom")}
    counts: dict[str, int] = {k: 0 for k in totals}
    it = iter(loader)
    for _ in range(iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        x, y, bb_l, du_p, edge_l, dom_p = _to(batch, device)
        _, lm, bb, du, edge, dom, _, _ = model(x, y, bb_l, du_p, edge_l, dom_p)
        for name, val in (("lm", lm), ("bb", bb), ("du", du), ("edge", edge), ("dom", dom)):
            if val is not None:
                totals[name] += val.item()
                counts[name] += 1
    model.train()
    return {k: totals[k] / max(1, counts[k]) for k in totals}


@torch.no_grad()
def sample_balance(model: GPT, device: str, n_tokens: int, n_samples: int = 8) -> float:
    model.eval()
    rates = []
    for _ in range(n_samples):
        start = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
        out = model.generate(start, max_new_tokens=n_tokens, temperature=1.0, top_k=40)
        rates.append(bracket_balance_rate(out[0].tolist()))
    model.train()
    return sum(rates) / len(rates)


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
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint to resume from (model + optimizer state)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume = args.resume
    del args.resume
    cfg = TrainConfig(**vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() \
        else torch.float32

    torch.manual_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CFGUnitMixDataset(
        cfg.data_dir / "train_units.bin",
        cfg.data_dir / "train_units.idx.npy",
        cfg.data_dir / "train_ann.jsonl",
        cfg.block_size,
    )
    val_ds = CFGUnitMixDataset(
        cfg.data_dir / "val_units.bin",
        cfg.data_dir / "val_units.idx.npy",
        cfg.data_dir / "val_ann.jsonl",
        cfg.block_size,
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
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"resumed from {resume} at step {step}")

    running: dict[str, float] = {k: 0.0 for k in ("lm", "bb", "du", "edge", "dom", "total")}
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
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

            x, y, bb_l, du_p, edge_l, dom_p = _to(batch, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, lm_loss, bb_loss, du_loss, edge_loss, dom_loss, _, _ = model(
                    x, y, bb_l, du_p, edge_l, dom_p
                )
                loss = lm_loss
                if bb_loss   is not None and cfg.cfg_bb_weight   > 0:
                    loss = loss + cfg.cfg_bb_weight   * bb_loss
                if du_loss   is not None and cfg.cfg_du_weight   > 0:
                    loss = loss + cfg.cfg_du_weight   * du_loss
                if edge_loss is not None and cfg.cfg_edge_weight > 0:
                    loss = loss + cfg.cfg_edge_weight * edge_loss
                if dom_loss  is not None and cfg.cfg_dom_weight  > 0:
                    loss = loss + cfg.cfg_dom_weight  * dom_loss
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            for name, val in (
                ("lm", lm_loss), ("bb", bb_loss),
                ("du", du_loss), ("edge", edge_loss), ("dom", dom_loss),
            ):
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
                for k in ("total", "lm", "bb", "du", "edge", "dom")
            )
            dt = (time.time() - t0) / cfg.log_interval
            print(f"step {step:6d}  {parts}  lr {lr:.2e}  {dt*1000:.0f}ms/step", flush=True)
            for k in running:
                running[k] = 0.0
            running_n = 0
            t0 = time.time()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            eval_losses = evaluate(model, val_dl, device, cfg.eval_iters)
            balance = sample_balance(model, device, cfg.sample_tokens)
            parts = "  ".join(f"val_{k} {v:.4f}" for k, v in eval_losses.items())
            print(f"[eval] step {step}  {parts}  bracket_balance {balance:.3f}", flush=True)
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
