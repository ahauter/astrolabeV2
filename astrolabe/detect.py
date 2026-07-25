"""Inference: detect nil-deref and bounds risks in a Go source file.

Usage:
    python -m astrolabe.detect \
        --checkpoint checkpoints_risk/ckpt_risk_5000.pt \
        --file some_file.go
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from astrolabe.model import GPT, GPTConfig
from astrolabe.vocab import BOS_ID, TOKEN_TO_ID, VOCAB_SIZE


def tokenize_go_file(file_path: Path, helper: Path) -> tuple[
    list[str], list[int] | None, set[str], dict[int, str]
]:
    """Run ast-tokenize and return (token_list, pos_map, pkg_names, name_pos_map).

    pos_map       : list where pos_map[i] = line number of token i.
    pkg_names     : set of package import short names in the file.
    name_pos_map  : dict mapping global token index → original identifier.
    """
    helper = helper.resolve()
    out = subprocess.check_output([str(helper), str(file_path)], text=True)
    tokens: list[str] = []
    pos_map: list[int] | None = None
    pkg_names: set[str] = set()
    name_pos_map: dict[int, str] = {}
    for line in out.splitlines():
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--helper", type=Path, default=Path("ast-tokenize"),
                   help="Path to ast-tokenize binary (default: ./ast-tokenize)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
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

    print_risks(nil_risks, tokens, pos_map, pkg_names, name_pos_map, source_lines, "NIL DEREFERENCE", args.threshold)
    print_risks(bounds_risks, tokens, pos_map, pkg_names, name_pos_map, source_lines, "BOUNDS CHECK", args.threshold)


if __name__ == "__main__":
    main()
