"""Evaluate a HierarchicalRiskGPT checkpoint against the synthetic race eval set.

Usage:
    python -m astrolabe.eval_synth_race \
        --checkpoint checkpoints_race/ckpt_race_2000.pt \
        --eval-dir synth_eval_race_manual
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from astrolabe.detect import extract_function_units
from astrolabe.model import GPTConfig, HierarchicalRiskGPT
from astrolabe.race_dataset import _pad_sequence as pad_sequence
from astrolabe.vocab import PAD_ID, TOKEN_TO_ID, VOCAB_SIZE


def load_model(checkpoint_path: Path, device: str) -> HierarchicalRiskGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_cfg"])
    model = HierarchicalRiskGPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def tokens_to_ids(tokens: list[str]) -> list[int]:
    return [min(TOKEN_TO_ID.get(t, PAD_ID), VOCAB_SIZE - 1) for t in tokens]


def build_context(
    source: Path,
    func_name: str,
    helper: Path,
    func_len: int,
    caller_len: int,
    max_callers: int,
    include_callers: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int | None,
]:
    """Return target/caller tensors plus the relative target start offset."""
    units = extract_function_units(source, helper)
    if not units:
        raise ValueError(f"no function units found in {source}")

    target_unit = None
    caller_units: list[tuple[str, list[str]]] = []
    for name, toks, start, _ in units:
        if name == func_name:
            target_unit = (name, toks, start)
        elif include_callers and name == "main":
            caller_units.append((name, toks))

    if target_unit is None:
        # Fall back to the first unit if the name does not match.
        target_unit = units[0]

    target_name, target_tokens, target_start = target_unit
    target_ids, target_mask = pad_sequence(tokens_to_ids(target_tokens), func_len)
    target_ids = torch.from_numpy(target_ids).unsqueeze(0)
    target_mask = torch.from_numpy(target_mask).unsqueeze(0)

    caller_ids = np.full((max_callers, caller_len), PAD_ID, dtype=np.int64)
    caller_mask = np.zeros((max_callers, caller_len), dtype=np.bool_)
    caller_present = np.zeros((max_callers,), dtype=np.bool_)
    for i, (_, ctoks) in enumerate(caller_units[:max_callers]):
        cids, cmask = pad_sequence(tokens_to_ids(ctoks), caller_len)
        caller_ids[i] = cids
        caller_mask[i] = cmask
        caller_present[i] = True

    caller_ids_t = torch.from_numpy(caller_ids).unsqueeze(0)
    caller_mask_t = torch.from_numpy(caller_mask).unsqueeze(0)
    caller_present_t = torch.from_numpy(caller_present).unsqueeze(0)

    return (
        target_ids,
        target_mask,
        caller_ids_t,
        caller_mask_t,
        caller_present_t,
        target_start,
    )


def evaluate_file(
    source: Path,
    func_name: str,
    model: HierarchicalRiskGPT,
    helper: Path,
    func_len: int,
    caller_len: int,
    max_callers: int,
    threshold: float,
    device: str,
    include_callers: bool,
) -> tuple[list[tuple[int, float]], int | None]:
    target_ids, target_mask, caller_ids, caller_mask, caller_present, target_start = build_context(
        source, func_name, helper, func_len, caller_len, max_callers, include_callers
    )
    target_ids = target_ids.to(device)
    target_mask = target_mask.to(device)
    caller_ids = caller_ids.to(device)
    caller_mask = caller_mask.to(device)
    caller_present = caller_present.to(device)

    preds = model.detect_race(
        target_ids, caller_ids, target_mask, caller_mask, caller_present, threshold=threshold
    )
    return preds, target_start


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, default=Path("synth_eval_race_manual"))
    p.add_argument("--helper", type=Path, default=Path("ast-tokenize"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--func-len", type=int, default=256)
    p.add_argument("--caller-len", type=int, default=128)
    p.add_argument("--max-callers", type=int, default=8)
    p.add_argument("--include-callers", action="store_true")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    labels_path = args.eval_dir / "labels.jsonl"
    if not labels_path.exists():
        sys.exit(f"labels not found at {labels_path}")

    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]

    tp = fp = fn = 0
    per_sample: list[dict] = []

    for label in labels:
        gt_pos = label["token_pos"]
        source = Path(label["source"])
        preds, target_start = evaluate_file(
            source,
            label["func_name"],
            model,
            args.helper,
            args.func_len,
            args.caller_len,
            args.max_callers,
            args.threshold,
            device,
            args.include_callers,
        )
        # Map the global ground-truth token position to the target-unit index.
        if target_start is not None:
            rel_gt = gt_pos - target_start
        else:
            rel_gt = gt_pos

        pred_positions = {pos for pos, _ in preds}

        sample_tp = 1 if rel_gt in pred_positions else 0
        sample_fp = len([p for p in pred_positions if p != rel_gt])
        sample_fn = 1 if sample_tp == 0 else 0
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn

        per_sample.append({
            "id": label["id"],
            "gt_token_pos": gt_pos,
            "rel_gt": rel_gt,
            "predicted_positions": sorted(pred_positions),
            "tp": sample_tp,
            "fp": sample_fp,
            "fn": sample_fn,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "include_callers": args.include_callers,
        "n_samples": len(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
