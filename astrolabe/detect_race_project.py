"""Run the HierarchicalRiskGPT race head over a live Go project.

Usage:
    python -m astrolabe.detect_race_project \
        --project ../../PolyScam/main \
        --checkpoint checkpoints_race/ckpt_race_5000.pt \
        --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from astrolabe.config import RaceTrainConfig
from astrolabe.detect import extract_function_units, tokenize_go_file
from astrolabe.model import GPTConfig, HierarchicalRiskGPT
from astrolabe.race_dataset import _pad_sequence as pad_sequence
from astrolabe.vocab import PAD_ID, TOKEN_TO_ID, VOCAB_SIZE


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = REPO_ROOT / "ast-tokenize"
DEFAULT_CALLGRAPH = REPO_ROOT / "ast-callgraph"


def load_model(checkpoint_path: Path, device: str) -> HierarchicalRiskGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_cfg"])
    model = HierarchicalRiskGPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def build_callgraph(helper: Path, project: Path) -> dict[str, list[str]]:
    proc = subprocess.run(
        [str(helper), str(project)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(f"warning: ast-callgraph failed: {proc.stderr[:500]}", file=sys.stderr)
        return {}
    callers: dict[str, list[str]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        callers[rec["func"]] = rec["callers"]
    return callers


def find_go_files(project: Path) -> list[Path]:
    return list(project.rglob("*.go"))


def tokenize_project(project: Path, tokenizer: Path) -> tuple[
    dict[str, tuple[Path, list[str], int, dict]],
    dict[Path, tuple[list[str], list[int] | None, dict[int, str]]],
]:
    """Return (func_name -> unit info, file -> tokenization metadata)."""
    func_index: dict[str, tuple[Path, list[str], int, dict]] = {}
    file_meta: dict[Path, tuple[list[str], list[int] | None, dict[int, str]]] = {}
    for go_file in find_go_files(project):
        try:
            tokens, pos_map, _, name_pos_map = tokenize_go_file(go_file, tokenizer)
            file_meta[go_file] = (tokens, pos_map, name_pos_map)
            units = extract_function_units(go_file, tokenizer)
            for func_name, unit_tokens, start, ann in units:
                if not func_name:
                    continue
                # Prefer the first occurrence if a name appears in multiple files.
                if func_name not in func_index:
                    func_index[func_name] = (go_file, unit_tokens, start, ann)
        except Exception as exc:
            print(f"warning: tokenization failed for {go_file}: {exc}", file=sys.stderr)
    return func_index, file_meta


def tokens_to_ids(tokens: list[str]) -> list[int]:
    return [min(TOKEN_TO_ID.get(t, PAD_ID), VOCAB_SIZE - 1) for t in tokens]


def detect_function(
    func_name: str,
    func_index: dict[str, tuple[Path, list[str], int, dict]],
    callers_map: dict[str, list[str]],
    model: HierarchicalRiskGPT,
    cfg: RaceTrainConfig,
    device: str,
    threshold: float,
) -> list[dict] | None:
    target = func_index.get(func_name)
    if target is None:
        return None
    go_file, target_tokens, _, _ = target
    target_ids, target_mask = pad_sequence(tokens_to_ids(target_tokens), cfg.func_len)
    target_ids_t = torch.from_numpy(target_ids).unsqueeze(0).to(device)
    target_mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)

    caller_ids = torch.full((1, cfg.max_callers, cfg.caller_len), PAD_ID, dtype=torch.long, device=device)
    caller_mask = torch.zeros((1, cfg.max_callers, cfg.caller_len), dtype=torch.bool, device=device)
    caller_present = torch.zeros((1, cfg.max_callers), dtype=torch.bool, device=device)

    callers = callers_map.get(func_name, [])[:cfg.max_callers]
    for k, caller_name in enumerate(callers):
        caller = func_index.get(caller_name)
        if caller is None:
            continue
        _, caller_tokens, _, _ = caller
        cids, cmask = pad_sequence(tokens_to_ids(caller_tokens), cfg.caller_len)
        caller_ids[0, k] = torch.from_numpy(cids)
        caller_mask[0, k] = torch.from_numpy(cmask)
        caller_present[0, k] = True

    with torch.no_grad():
        preds = model.detect_race(
            target_ids_t, caller_ids, target_mask_t, caller_mask, caller_present, threshold=threshold
        )

    return [
        {
            "func": func_name,
            "file": str(go_file),
            "token_pos": int(pos),
            "token": target_tokens[pos] if 0 <= pos < len(target_tokens) else "PAD",
            "confidence": float(conf),
        }
        for pos, conf in preds
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints_race/ckpt_race_5000.pt"))
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--callgraph", type=Path, default=DEFAULT_CALLGRAPH)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    cfg = RaceTrainConfig()

    print(f"Building call graph for {args.project} ...")
    callers_map = build_callgraph(args.callgraph, args.project)

    print("Tokenizing project ...")
    func_index, file_meta = tokenize_project(args.project, args.tokenizer)
    print(f"Indexed {len(func_index)} functions across {len(file_meta)} files")

    target_funcs = sorted(func_index.keys())
    if args.max_files:
        target_funcs = target_funcs[:args.max_files]

    all_findings: list[dict] = []
    for func_name in target_funcs:
        findings = detect_function(
            func_name, func_index, callers_map, model, cfg, device, args.threshold
        )
        if findings:
            all_findings.extend(findings)

    # Sort by confidence descending.
    all_findings.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"\n=== Race findings (threshold {args.threshold}) ===")
    for f in all_findings:
        file_path = Path(f["file"])
        _, pos_map, _ = file_meta.get(file_path, ([], None, {}))
        line_num = None
        if pos_map is not None:
            # token_pos is relative to the target unit; map back to global token
            # index using the unit's start offset in the full token stream.
            target_file, target_tokens, target_start, _ = func_index[f["func"]]
            global_idx = target_start + f["token_pos"]
            if 0 <= global_idx < len(pos_map):
                line_num = pos_map[global_idx]
        line_str = f"line {line_num:5d}" if line_num else f"pos  {f['token_pos']:4d}"
        print(
            f"{f['confidence']:.3f}  {f['func']:60s}  {line_str}  {f['token']:20s}  {f['file']}"
        )

    print(f"\nTotal findings: {len(all_findings)}")


if __name__ == "__main__":
    main()
