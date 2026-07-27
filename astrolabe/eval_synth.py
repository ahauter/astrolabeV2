"""Evaluate a risk model against the synthetic eval set.

For each generated buggy function, run model inference and compare predictions
with the runtime-verified ground-truth risk position.

Usage:
    python -m astrolabe.eval_synth \
        --checkpoint checkpoints_risk_v5/ckpt_risk_10000.pt \
        --eval-dir synth_eval \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from astrolabe.detect import load_model, tokenize_go_file
from astrolabe.vocab import BOS_ID, TOKEN_TO_ID, VOCAB_SIZE


def evaluate_file(
    file_path: Path,
    model: torch.nn.Module,
    helper: Path,
    block_size: int,
    threshold: float,
    device: str,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    tokens, pos_map, pkg_names, name_pos_map = tokenize_go_file(file_path, helper)
    ids: list[int] = []
    for t in tokens:
        tid = TOKEN_TO_ID.get(t, TOKEN_TO_ID["PAD"])
        ids.append(min(tid, VOCAB_SIZE - 1))

    max_len = block_size - 1
    if len(ids) > max_len:
        ids = ids[:max_len]
        tokens = tokens[:max_len]
        pos_map = pos_map[:max_len] if pos_map else pos_map
        name_pos_map = {k: v for k, v in name_pos_map.items() if k < max_len}
    ids = [BOS_ID] + ids
    while len(ids) < block_size:
        ids.append(TOKEN_TO_ID["PAD"])

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        nil_risks, bounds_risks = model.detect_risks(x, threshold=threshold)

    # Filter out package-import false positives.
    nil_risks = [(pos, conf) for pos, conf in nil_risks if name_pos_map.get(pos - 1) not in pkg_names]
    bounds_risks = [(pos, conf) for pos, conf in bounds_risks if name_pos_map.get(pos - 1) not in pkg_names]
    return nil_risks, bounds_risks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, default=Path("synth_eval"))
    p.add_argument("--helper", type=Path, default=Path("ast-tokenize"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    block_size = model.cfg.block_size

    labels_path = args.eval_dir / "labels.jsonl"
    if not labels_path.exists():
        sys.exit(f"labels not found at {labels_path}")

    labels: list[dict] = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]

    kind_stats: dict[str, dict[str, int]] = {
        "nil": {"tp": 0, "fp": 0, "fn": 0},
        "bounds": {"tp": 0, "fp": 0, "fn": 0},
    }
    per_sample: list[dict] = []

    for label in labels:
        kind = label["kind"]
        gt_pos = label["token_pos"]  # 0-based index in token list (after BOS)
        model_pos = gt_pos + 1  # model uses 1-based position with BOS at 0
        source = Path(label["source"])

        nil_risks, bounds_risks = evaluate_file(
            source, model, args.helper, block_size, args.threshold, device
        )
        preds = nil_risks if kind == "nil" else bounds_risks
        pred_positions = {pos for pos, _ in preds}

        tp = 1 if model_pos in pred_positions else 0
        fp = len([p for p in pred_positions if p != model_pos])
        fn = 1 if tp == 0 else 0

        kind_stats[kind]["tp"] += tp
        kind_stats[kind]["fp"] += fp
        kind_stats[kind]["fn"] += fn

        per_sample.append({
            "id": label["id"],
            "kind": kind,
            "gt_token_pos": gt_pos,
            "predicted_positions": sorted(pred_positions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    def metrics(stats: dict[str, int]) -> dict[str, float]:
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    nil_metrics = metrics(kind_stats["nil"])
    bounds_metrics = metrics(kind_stats["bounds"])
    total_tp = kind_stats["nil"]["tp"] + kind_stats["bounds"]["tp"]
    total_fp = kind_stats["nil"]["fp"] + kind_stats["bounds"]["fp"]
    total_fn = kind_stats["nil"]["fn"] + kind_stats["bounds"]["fn"]
    overall = metrics({"tp": total_tp, "fp": total_fp, "fn": total_fn})

    result = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "n_samples": len(labels),
        "nil": nil_metrics,
        "bounds": bounds_metrics,
        "overall": overall,
    }

    print(json.dumps(result, indent=2))

    if args.output:
        args.output.write_text(json.dumps({"summary": result, "per_sample": per_sample}, indent=2))
        print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
