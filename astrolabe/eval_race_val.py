"""Evaluate a HierarchicalRiskGPT race checkpoint on the reserved validation corpus.

Usage:
    python -m astrolabe.eval_race_val \
        --checkpoint checkpoints_race/ckpt_race_2000.pt \
        --data-dir data_race
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from astrolabe.config import RaceTrainConfig
from astrolabe.model import GPTConfig, HierarchicalRiskGPT
from astrolabe.race_dataset import RaceContextDataset


def load_model(checkpoint_path: Path, device: str) -> HierarchicalRiskGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_cfg"])
    model = HierarchicalRiskGPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate(model: HierarchicalRiskGPT, ds: RaceContextDataset, device: str, threshold: float, indices: list[int] | None = None) -> dict:
    tp = fp = fn = 0
    sample_results = []
    model.eval()
    if indices is None:
        indices = range(len(ds))
    with torch.no_grad():
        for idx in indices:
            t_ids, c_ids, t_mask, c_mask, c_pres, race_l, _, _, t_sync, _ = ds[idx]
            t_ids = t_ids.unsqueeze(0).to(device)
            c_ids = c_ids.unsqueeze(0).to(device)
            t_mask = t_mask.unsqueeze(0).to(device)
            c_mask = c_mask.unsqueeze(0).to(device)
            c_pres = c_pres.unsqueeze(0).to(device)
            t_sync = t_sync.unsqueeze(0).to(device)

            preds = model.detect_race(t_ids, c_ids, t_mask, c_mask, c_pres, target_sync_mask=t_sync, threshold=threshold)
            pred_positions = {int(p) for p, _ in preds}
            gt_positions = {int(i) for i, v in enumerate(race_l) if v.item() > 0}

            sample_tp = len(pred_positions & gt_positions)
            sample_fp = len(pred_positions - gt_positions)
            sample_fn = len(gt_positions - pred_positions)
            tp += sample_tp
            fp += sample_fp
            fn += sample_fn

            sample_results.append({
                "idx": idx,
                "gt": sorted(gt_positions),
                "pred": sorted(pred_positions),
                "tp": sample_tp,
                "fp": sample_fp,
                "fn": sample_fn,
            })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": threshold,
        "n_samples": len(indices),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": sample_results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data_race"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--samples", type=int, default=2000,
                   help="Number of validation samples to evaluate (default 2000)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    cfg = RaceTrainConfig()

    val_ds = RaceContextDataset(
        args.data_dir / "val_units.bin",
        args.data_dir / "val_units.idx.npy",
        args.data_dir / "val_ann.jsonl",
        args.data_dir / "race_val_meta.jsonl",
        args.data_dir / "risk_val.jsonl",
        static_race_path=args.data_dir / "race_risk_val.jsonl",
        func_len=cfg.func_len,
        caller_len=cfg.caller_len,
        max_callers=cfg.max_callers,
        mutate_frac=0.2,
        sync_neg_weight=cfg.sync_neg_weight,
        seed=cfg.seed,
        deterministic=True,
    )

    rng = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(val_ds), generator=rng)[:args.samples].tolist()
    n_positive = sum(1 for i in indices if val_ds[i][5].sum() > 0)
    print(f"Validation subset: {len(indices)} samples ({n_positive} positive)")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        res = evaluate(model, val_ds, device, thr, indices)
        print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
