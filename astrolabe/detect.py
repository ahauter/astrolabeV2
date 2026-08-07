"""Inference: detect nil-deref, bounds-check, and race-condition risks.

Usage:
    python -m astrolabe.detect --kind nil --checkpoint checkpoints_risk/ckpt_risk_5000.pt --file some_file.go
    python -m astrolabe.detect --kind race --checkpoint checkpoints_race/ckpt_race_5000.pt --file some_file.go --project ../../PolyScam/main
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from astrolabe.config import RaceTrainConfig
from astrolabe.model import GPT, GPTConfig, HierarchicalRiskGPT
from astrolabe.race_dataset import _pad_sequence as pad_sequence
from astrolabe.vocab import BOS_ID, PAD_ID, TOKEN_TO_ID, VOCAB_SIZE


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_tokenizer(file_path: Path, helper: Path) -> list[str]:
    helper = helper.resolve()
    return subprocess.check_output([str(helper), str(file_path)], text=True).splitlines()


def tokenize_go_file(file_path: Path, helper: Path) -> tuple[
    list[str], list[int] | None, set[str], dict[int, str]
]:
    """Run ast-tokenize and return (token_list, pos_map, pkg_names, name_pos_map).

    pos_map       : list where pos_map[i] = line number of token i.
    pkg_names     : set of package import short names in the file.
    name_pos_map  : dict mapping global token index → original identifier.
    """
    tokens: list[str] = []
    pos_map: list[int] | None = None
    pkg_names: set[str] = set()
    name_pos_map: dict[int, str] = {}
    for line in _run_tokenizer(file_path, helper):
        line = line.strip()
        if not line:
            continue
        if line.startswith("POSMAP "):
            pos_map = json.loads(line[len("POSMAP "):])
            continue
        if line.startswith("PKGS "):
            pkg_names = set(json.loads(line[len("PKGS "):]))
            continue
        if line.startswith("NAMEPOSMAP "):
            # NAMEPOSMAP is {"token_index": "original_name", ...}
            raw = json.loads(line[len("NAMEPOSMAP "):])
            name_pos_map = {int(k): v for k, v in raw.items()}
            continue
        if line.startswith("ANN "):
            continue
        tokens.append(line)
    return tokens, pos_map, pkg_names, name_pos_map


def extract_function_units(
    file_path: Path, helper: Path
) -> list[tuple[str, list[str], int, dict]]:
    """Extract individual function/tokenization units from a Go file.

    Returns a list of (func_name, tokens, global_start_index, ann_dict) tuples,
    where global_start_index is the position of OPEN_FUNC_DECL in the full file
    token stream, and ann_dict is the JSON object emitted by ast-tokenize.
    """
    lines = _run_tokenizer(file_path, helper)
    units: list[tuple[str, list[str], int, dict]] = []
    current_tokens: list[str] = []
    current_start: int | None = None
    global_idx = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(("POSMAP ", "PKGS ", "NAMEPOSMAP ")):
            continue
        if line.startswith("ANN "):
            ann = json.loads(line[len("ANN "):])
            if current_start is not None and current_tokens:
                func_name = ann.get("func", "").split(".")[-1]
                units.append((func_name, current_tokens, current_start, ann))
            current_tokens = []
            current_start = None
            continue
        if line == "OPEN_FUNC_DECL":
            current_start = global_idx
            current_tokens = []
        if current_start is not None:
            current_tokens.append(line)
        global_idx += 1
    return units


def load_model(checkpoint_path: Path, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_cfg"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def print_risks(
    risks: list[tuple[int, float]],
    tokens: list[str],
    pos_map: list[int] | None,
    pkg_names: set[str],
    name_pos_map: dict[int, str],
    source_lines: list[str],
    label: str,
    threshold: float,
) -> None:
    filtered: list[tuple[int, float]] = []
    skipped = 0
    for pos, conf in risks:
        tok_idx = pos - 1  # BOS is at model position 0
        orig_name = name_pos_map.get(tok_idx)
        if orig_name is not None and orig_name in pkg_names:
            skipped += 1
            continue
        filtered.append((pos, conf))

    if skipped:
        print(f"  (skipped {skipped} package-import false positives)")

    if not filtered:
        print(f"\nNo {label} risks detected above threshold {threshold}.")
        return

    print(f"\n=== {label} RISKS ({len(filtered)} detected) ===")
    for pos, conf in filtered:
        tok_idx = pos - 1
        tok_name = tokens[tok_idx] if 0 <= tok_idx < len(tokens) else "PAD"
        line_num = pos_map[tok_idx] if pos_map and 0 <= tok_idx < len(pos_map) else None

        if line_num is not None and 1 <= line_num <= len(source_lines):
            snippet = source_lines[line_num - 1].strip()
            print(f"  line {line_num:4d}  {tok_name:20s}  conf {conf:.3f}  |  {snippet}")
        else:
            print(f"  pos  {pos:4d}  {tok_name:20s}  conf {conf:.3f}")


def _build_callgraph(project: Path, helper: Path) -> dict[str, list[str]]:
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


def _project_function_index(
    project: Path, tokenizer: Path
) -> dict[str, tuple[Path, list[str]]]:
    index: dict[str, tuple[Path, list[str]]] = {}
    for go_file in project.rglob("*.go"):
        try:
            for func_name, unit_tokens, _, _ in extract_function_units(go_file, tokenizer):
                if func_name and func_name not in index:
                    index[func_name] = (go_file, unit_tokens)
        except Exception:
            continue
    return index


def _detect_race_file(
    file_path: Path,
    checkpoint_path: Path,
    project: Path | None,
    tokenizer: Path,
    callgraph_helper: Path,
    threshold: float,
    device: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_cfg"])
    model = HierarchicalRiskGPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = RaceTrainConfig()

    source_lines = file_path.read_text().splitlines()
    _, pos_map, _, _ = tokenize_go_file(file_path, tokenizer)

    callers_map: dict[str, list[str]] = {}
    project_index: dict[str, tuple[Path, list[str]]] = {}
    if project is not None:
        callers_map = _build_callgraph(project, callgraph_helper)
        project_index = _project_function_index(project, tokenizer)

    findings: list[tuple[str, int, str, float, int | None]] = []
    for func_name, target_tokens, start, _ in extract_function_units(file_path, tokenizer):
        target_ids, target_mask = pad_sequence(
            [min(TOKEN_TO_ID.get(t, PAD_ID), VOCAB_SIZE - 1) for t in target_tokens],
            cfg.func_len,
        )
        target_ids_t = torch.from_numpy(target_ids).unsqueeze(0).to(device)
        target_mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)

        caller_ids = torch.full((1, cfg.max_callers, cfg.caller_len), PAD_ID, dtype=torch.long, device=device)
        caller_mask = torch.zeros((1, cfg.max_callers, cfg.caller_len), dtype=torch.bool, device=device)
        caller_present = torch.zeros((1, cfg.max_callers), dtype=torch.bool, device=device)

        if project is not None:
            for k, caller_name in enumerate(callers_map.get(func_name, [])[:cfg.max_callers]):
                caller = project_index.get(caller_name)
                if caller is None:
                    continue
                _, caller_tokens = caller
                cids, cmask = pad_sequence(
                    [min(TOKEN_TO_ID.get(t, PAD_ID), VOCAB_SIZE - 1) for t in caller_tokens],
                    cfg.caller_len,
                )
                caller_ids[0, k] = torch.from_numpy(cids)
                caller_mask[0, k] = torch.from_numpy(cmask)
                caller_present[0, k] = True

        with torch.no_grad():
            preds = model.detect_race(
                target_ids_t, caller_ids, target_mask_t, caller_mask, caller_present, threshold=threshold
            )

        for pos, conf in preds:
            line_num = None
            if pos_map is not None:
                global_idx = start + int(pos)
                if 0 <= global_idx < len(pos_map):
                    line_num = pos_map[global_idx]
            findings.append((func_name, int(pos), target_tokens[int(pos)], float(conf), line_num))

    if not findings:
        print(f"\nNo race risks detected above threshold {threshold}.")
        return

    print(f"\n=== RACE RISKS ({len(findings)} detected) ===")
    for func_name, pos, tok, conf, line_num in findings:
        snippet = ""
        if line_num is not None and 1 <= line_num <= len(source_lines):
            snippet = f"  |  {source_lines[line_num - 1].strip()}"
        line_str = f"line {line_num:4d}" if line_num else f"pos  {pos:4d}"
        print(f"  {func_name:40s}  {line_str}  {tok:20s}  conf {conf:.3f}{snippet}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", type=str, choices=["nil", "bounds", "race"], default="nil",
                   help="Risk kind to detect (nil/bounds use GPT, race uses HierarchicalRiskGPT)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--helper", type=Path, default=REPO_ROOT / "ast-tokenize",
                   help="Path to ast-tokenize binary")
    p.add_argument("--project", type=Path, default=None,
                   help="Project root for race caller-context (required for meaningful race results)")
    p.add_argument("--callgraph", type=Path, default=REPO_ROOT / "ast-callgraph",
                   help="Path to ast-callgraph binary")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.kind == "race":
        _detect_race_file(
            args.file, args.checkpoint, args.project, args.helper, args.callgraph, args.threshold, device
        )
        return

    model = load_model(args.checkpoint, device)
    block_size = model.cfg.block_size

    # Load source lines for snippet display.
    source_lines = args.file.read_text().splitlines()

    tokens, pos_map, pkg_names, name_pos_map = tokenize_go_file(args.file, args.helper)
    print(f"Tokenized {args.file} → {len(tokens)} tokens")
    if pos_map is None:
        print("WARNING: ast-tokenize did not emit POSMAP; line numbers unavailable")
        print("         Rebuild ast-tokenize from latest source.")
    if not pkg_names:
        print("WARNING: ast-tokenize did not emit PKGS; package false positives won't be filtered")
    if not name_pos_map:
        print("WARNING: ast-tokenize did not emit NAMEPOSMAP; package false positives won't be filtered")

    # Convert tokens to IDs.
    ids: list[int] = []
    for t in tokens:
        tid = TOKEN_TO_ID.get(t)
        if tid is None:
            tid = TOKEN_TO_ID["PAD"]
        ids.append(min(tid, VOCAB_SIZE - 1))

    # Reserve 1 slot for BOS.
    max_len = block_size - 1
    if len(ids) > max_len:
        print(f"WARNING: sequence length {len(ids)} > {max_len}; truncating")
        ids = ids[:max_len]
        tokens = tokens[:max_len]
        pos_map = pos_map[:max_len] if pos_map else pos_map
        # name_pos_map keys are global token indices; truncation removes the tail,
        # so keys >= max_len are irrelevant now.
        name_pos_map = {k: v for k, v in name_pos_map.items() if k < max_len}
    ids = [BOS_ID] + ids
    while len(ids) < block_size:
        ids.append(TOKEN_TO_ID["PAD"])

    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        nil_risks, bounds_risks = model.detect_risks(x, threshold=args.threshold)

    if args.kind in ("nil", "bounds"):
        print_risks(nil_risks, tokens, pos_map, pkg_names, name_pos_map, source_lines, "NIL DEREFERENCE", args.threshold)
        print_risks(bounds_risks, tokens, pos_map, pkg_names, name_pos_map, source_lines, "BOUNDS CHECK", args.threshold)


if __name__ == "__main__":
    main()
