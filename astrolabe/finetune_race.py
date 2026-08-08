"""Fine-tune HierarchicalRiskGPT for race-condition detection.

Loads a pretrained GPT backbone, attaches the L2 context aggregator and race
head, and fine-tunes on the mutated race corpus.

Usage:
    python -m astrolabe.finetune_race \
        --data-dir data_race \
        --out-dir checkpoints_race \
        --resume /path/to/pretrained.pt
"""
from __future__ import annotations

import argparse
import math
import time
import typing
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from astrolabe.config import RaceTrainConfig
from astrolabe.model import GPTConfig, HierarchicalRiskGPT
from astrolabe.race_dataset import RaceContextDataset
from astrolabe.vocab import VOCAB_SIZE


def get_lr(step: int, cfg: RaceTrainConfig) -> float:
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
def evaluate(model: HierarchicalRiskGPT, loader: DataLoader, device: str, iters: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {k: 0.0 for k in ("total", "race", "nil", "bounds")}
    counts: dict[str, int] = {k: 0 for k in totals}
    it = iter(loader)
    for _ in range(iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        t_ids, c_ids, t_mask, c_mask, c_pres, race_l, nil_l, bounds_l, t_sync, race_w = _to(batch, device)
        _, race_loss, _, nil_loss, _, bounds_loss = model(
            t_ids, c_ids, t_mask, c_mask, c_pres,
            race_labels=race_l, nil_labels=nil_l, bounds_labels=bounds_l,
            target_sync_mask=t_sync, race_weight_mask=race_w,
        )
        loss = 0.0
        for name, val in (("race", race_loss), ("nil", nil_loss), ("bounds", bounds_loss)):
            if val is not None:
                loss += val.item()
                totals[name] += val.item()
                counts[name] += 1
        totals["total"] += loss
        counts["total"] += 1
    model.train()
    return {k: totals[k] / max(1, counts[k]) for k in totals}


def _arg_type(annotation: type, default):
    """Return an argparse type for a config field annotation."""
    import types
    kind = annotation
    if kind is bool:
        return lambda s: s.lower() in ("1", "true", "yes")
    if kind is Path:
        return Path
    if isinstance(kind, types.UnionType):
        non_none = [a for a in kind.__args__ if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        return str
    return kind


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    cfg = RaceTrainConfig()
    annotations = typing.get_type_hints(RaceTrainConfig)
    for k, v in asdict(cfg).items():
        if k == "resume":
            continue
        ann = annotations.get(k, type(v))
        flag = "--" + k.replace("_", "-")
        p.add_argument(flag, type=_arg_type(ann, v), default=v)
    p.add_argument("--resume", type=Path, default=None,
                   help="Pretrained checkpoint to load L1 from")
    p.add_argument("--resume-ckpt", type=Path, default=None,
                   help="Race checkpoint to resume training from (loads model + optimizer + step)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume = args.resume
    cfg = RaceTrainConfig(**{k: v for k, v in vars(args).items() if k not in ("resume", "resume_ckpt")})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    torch.manual_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RaceContextDataset(
        cfg.data_dir / "train_units.bin",
        cfg.data_dir / "train_units.idx.npy",
        cfg.data_dir / "train_ann.jsonl",
        cfg.data_dir / "race_train_meta.jsonl",
        cfg.data_dir / "risk_train.jsonl",
        static_race_path=cfg.data_dir / "race_risk_train.jsonl",
        func_len=cfg.func_len,
        caller_len=cfg.caller_len,
        max_callers=cfg.max_callers,
        mutate_frac=cfg.mutate_frac,
        unlock_mutate_frac=cfg.unlock_mutate_frac,
        sync_neg_weight=cfg.sync_neg_weight,
        seed=cfg.seed,
        deterministic=False,
    )
    val_ds = RaceContextDataset(
        cfg.data_dir / "val_units.bin",
        cfg.data_dir / "val_units.idx.npy",
        cfg.data_dir / "val_ann.jsonl",
        cfg.data_dir / "race_val_meta.jsonl",
        cfg.data_dir / "risk_val.jsonl",
        static_race_path=cfg.data_dir / "race_risk_val.jsonl",
        func_len=cfg.func_len,
        caller_len=cfg.caller_len,
        max_callers=cfg.max_callers,
        mutate_frac=cfg.mutate_frac,
        unlock_mutate_frac=cfg.unlock_mutate_frac,
        sync_neg_weight=cfg.sync_neg_weight,
        seed=cfg.seed,
        deterministic=True,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)

    gpt_cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = HierarchicalRiskGPT(
        gpt_cfg, n_ctx_layers=cfg.n_ctx_layers, max_callers=cfg.max_callers
    ).to(device)
    print(f"model params: {model.num_params() / 1e6:.2f}M  "
          f"vocab: {VOCAB_SIZE}  device: {device}  dtype: {dtype}")

    if cfg.compile:
        model = torch.compile(model)

    step = 0
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        print(f"resumed race checkpoint from {args.resume_ckpt} at step {step}")
        if missing:
            print(f"  missing keys (will be freshly initialized): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    elif resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.l1.load_state_dict(ckpt["model"], strict=False)
        print(f"loaded pretrained L1 from {resume}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Freeze backbone for the initial warmup period.
    risk_params = []
    for name, p in model.named_parameters():
        if not name.startswith("l1."):
            risk_params.append(p)
    for p in model.parameters():
        p.requires_grad = False
    for p in risk_params:
        p.requires_grad = True

    # If resuming past the freeze period, unfreeze immediately.
    if step >= cfg.freeze_backbone_steps:
        for p in model.parameters():
            p.requires_grad = True

    running: dict[str, float] = {"total": 0.0, "race": 0.0, "nil": 0.0, "bounds": 0.0}
    running_n = 0
    t0 = time.time()
    train_iter = iter(train_dl)

    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

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

            t_ids, c_ids, t_mask, c_mask, c_pres, race_l, nil_l, bounds_l, t_sync, race_w = _to(batch, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, race_loss, _, nil_loss, _, bounds_loss = model(
                    t_ids, c_ids, t_mask, c_mask, c_pres,
                    race_labels=race_l,
                    nil_labels=nil_l,
                    bounds_labels=bounds_l,
                    target_sync_mask=t_sync,
                    race_weight_mask=race_w,
                    race_pos_weight=cfg.risk_race_pos_weight,
                    nil_pos_weight=cfg.risk_nil_pos_weight,
                    bounds_pos_weight=cfg.risk_bounds_pos_weight,
                )
                loss = 0.0
                if race_loss is not None and cfg.risk_race_weight > 0:
                    loss = loss + cfg.risk_race_weight * race_loss
                if nil_loss is not None and cfg.risk_nil_weight > 0:
                    loss = loss + cfg.risk_nil_weight * nil_loss
                if bounds_loss is not None and cfg.risk_bounds_weight > 0:
                    loss = loss + cfg.risk_bounds_weight * bounds_loss
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            for name, val in (("race", race_loss), ("nil", nil_loss), ("bounds", bounds_loss)):
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
            parts = "  ".join(f"{k} {running[k] / scale:.4f}" for k in ("total", "race", "nil", "bounds"))
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
            torch.save(ckpt, cfg.out_dir / f"ckpt_race_{step}.pt")


if __name__ == "__main__":
    main()
