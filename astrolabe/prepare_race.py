"""Build a race-condition training corpus from Go source.

1. Tokenize a subset of Go files with the race-aware ast-tokenize helper.
2. Build a static call graph with ast-callgraph.
3. Generate mutated variants by stripping synchronization calls.
4. Label previously-protected shared variable accesses as races.

Usage:
    python -m astrolabe.prepare_race \
        --src scraped_code_remote/scraped_code/some_repo \
        --dst data_race \
        --val-frac 0.05 \
        --max-files 2000
"""
from __future__ import annotations

import argparse
import array
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from astrolabe.prepare import find_go_files, stream_units
from astrolabe.risk_miner import find_bounds_risks, find_nil_risks
from astrolabe.vocab import ID_TO_TOKEN, TOKEN_TO_ID, VOCAB_SIZE, BOS_ID, EOF_ID


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = REPO_ROOT / "ast-tokenize"
DEFAULT_CALLGRAPH = REPO_ROOT / "ast-callgraph"

RISKY_TYPES = {"ptr", "slice", "map", "chan", "interface", "struct", "func"}


def tokenize_raw(
    helper: Path,
    files: list[Path],
    dst: Path,
    val_frac: float,
    batch: int,
    rng: random.Random,
) -> None:
    """Run ast-tokenize and write train/val units, offsets, and annotations."""
    dst.mkdir(parents=True, exist_ok=True)

    train_offsets: array.array = array.array("Q", [0])
    val_offsets: array.array = array.array("Q", [0])

    with open(dst / "train_units.bin", "wb") as tf, \
         open(dst / "val_units.bin", "wb") as vf, \
         open(dst / "train_ann.jsonl", "w") as ta, \
         open(dst / "val_ann.jsonl", "w") as va:
        for i in range(0, len(files), batch):
            batch_files = files[i:i + batch]
            for unit, ann in stream_units(helper, batch_files):
                if not unit:
                    continue
                arr = np.array(unit, dtype=np.uint16)
                if rng.random() < val_frac:
                    arr.tofile(vf)
                    val_offsets.append(val_offsets[-1] + len(unit))
                    va.write(json.dumps(ann) + "\n")
                else:
                    arr.tofile(tf)
                    train_offsets.append(train_offsets[-1] + len(unit))
                    ta.write(json.dumps(ann) + "\n")

    np.save(dst / "train_units.idx.npy", np.frombuffer(train_offsets, dtype=np.uint64))
    np.save(dst / "val_units.idx.npy", np.frombuffer(val_offsets, dtype=np.uint64))


def build_callgraph(helper: Path, src: Path, dst: Path) -> dict[str, list[str]]:
    """Run ast-callgraph and return callee -> callers mapping."""
    out_path = dst / "callgraph.jsonl"
    proc = subprocess.run(
        [str(helper), str(src)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(f"warning: ast-callgraph failed: {proc.stderr[:500]}", file=sys.stderr)
        return {}

    callers: dict[str, list[str]] = {}
    with open(out_path, "w") as f:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            f.write(line + "\n")
            rec = json.loads(line)
            callers[rec["func"]] = rec["callers"]
    return callers


def build_meta(dst: Path, split: str, callers_map: dict[str, list[str]]) -> list[dict]:
    """Build per-unit metadata: func name and resolved caller unit indices."""
    ann_path = dst / f"{split}_ann.jsonl"
    meta_path = dst / f"{split}_meta.jsonl"

    func_to_idx: dict[str, int] = {}
    anns: list[dict] = []
    with open(ann_path) as f:
        for idx, line in enumerate(f):
            ann = json.loads(line) if line.strip() and line.strip() != "null" else {}
            anns.append(ann)
            name = ann.get("func") if ann else None
            if name and name not in func_to_idx:
                func_to_idx[name] = idx

    meta: list[dict] = []
    with open(meta_path, "w") as f:
        for idx, ann in enumerate(anns):
            name = ann.get("func") if ann else None
            caller_funcs = callers_map.get(name, []) if name else []
            caller_idxs: list[int] = []
            for cf in caller_funcs[:8]:
                if cf in func_to_idx:
                    caller_idxs.append(func_to_idx[cf])
            record = {"func": name, "callers": caller_idxs}
            meta.append(record)
            f.write(json.dumps(record) + "\n")
    return meta


def mutate_split(
    dst: Path,
    split: str,
    meta: list[dict],
    max_callers: int = 8,
) -> None:
    """Create mutated race corpus from raw tokenized units."""
    data = np.memmap(dst / f"{split}_units.bin", dtype=np.uint16, mode="r")
    offsets = np.load(dst / f"{split}_units.idx.npy")

    with open(dst / f"{split}_ann.jsonl") as f:
        anns = [json.loads(line) if line.strip() and line.strip() != "null" else {}
                for line in f]

    n_units = len(offsets) - 1
    assert len(anns) == n_units

    has_go = [bool(ann.get("go")) for ann in anns]

    out_units = dst / f"race_{split}_units.bin"
    out_idx = dst / f"race_{split}_units.idx.npy"
    out_labels = dst / f"race_{split}_labels.jsonl"
    out_meta = dst / f"race_{split}_meta.jsonl"

    offsets_arr: array.array = array.array("Q", [0])

    with open(out_units, "wb") as uf, \
         open(out_labels, "w") as lf, \
         open(out_meta, "w") as mf:
        for i in range(n_units):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            unit = data[start:end].tolist()
            ann = anns[i]
            m = meta[i]
            token_names = [ID_TO_TOKEN[t] for t in unit]

            # Original synchronized unit: nil/bounds labels from the miner.
            nil_labels = find_nil_risks(token_names, ann)
            bounds_labels = find_bounds_risks(token_names, ann)
            _write_unit(unit, [], nil_labels, bounds_labels, m, offsets_arr, uf, lf, mf)

            # Mutated unit: strip synchronization calls and label race positions.
            sync_positions = _sync_positions(ann)
            if sync_positions:
                mutated, pos_map = _remove_positions(unit, sync_positions)
                concurrent = _is_concurrent(i, m, has_go)
                race_labels = _race_labels(unit, mutated, pos_map, ann, concurrent)
                _write_unit(mutated, race_labels, [], [], m, offsets_arr, uf, lf, mf)

    np.save(out_idx, np.frombuffer(offsets_arr, dtype=np.uint64))


def _write_unit(
    unit: list[int],
    race_labels: list[int],
    nil_labels: list[int],
    bounds_labels: list[int],
    meta: dict,
    offsets_arr: array.array,
    uf,
    lf,
    mf,
) -> None:
    arr = np.array(unit, dtype=np.uint16)
    arr.tofile(uf)
    offsets_arr.append(offsets_arr[-1] + len(unit))
    lf.write(json.dumps({
        "race_risks": sorted(set(race_labels)),
        "nil_risks": sorted(set(nil_labels)),
        "bounds_risks": sorted(set(bounds_labels)),
    }) + "\n")
    mf.write(json.dumps(meta) + "\n")


def _sync_positions(ann: dict) -> set[int]:
    """Return token positions occupied by synchronization calls."""
    positions: set[int] = set()
    for ev in ann.get("sync", []):
        for p in range(ev["start"], ev["end"] + 1):
            positions.add(p)
    return positions


def _remove_positions(unit: list[int], remove: set[int]) -> tuple[list[int], dict[int, int]]:
    """Return unit with positions removed and a map old->new positions."""
    mutated: list[int] = []
    pos_map: dict[int, int] = {}
    for old_pos, tok in enumerate(unit):
        if old_pos not in remove:
            pos_map[old_pos] = len(mutated)
            mutated.append(tok)
    return mutated, pos_map


def _is_concurrent(unit_idx: int, meta: dict, has_go: list[bool]) -> bool:
    if has_go[unit_idx]:
        return True
    for cidx in meta.get("callers", []):
        if has_go[cidx]:
            return True
    return False


def _race_labels(
    original: list[int],
    mutated: list[int],
    pos_map: dict[int, int],
    ann: dict,
    concurrent: bool,
) -> list[int]:
    # We label stripped shared accesses as potential races regardless of a
    # static goroutine signal. The caller-context aggregator can learn to
    # modulate the score based on whether callers spawn goroutines.
    types_map = ann.get("types", {})
    use_set = set(ann.get("use", []))
    labels: list[int] = []

    for old_pos in use_set:
        if old_pos < 0 or old_pos >= len(original):
            continue
        if old_pos not in pos_map:
            continue
        tok_id = int(original[old_pos])
        tok_name = ID_TO_TOKEN[tok_id]
        if not tok_name.startswith("NAME_"):
            continue
        suffix = tok_name[len("NAME_"):]
        if not suffix.isdigit():
            continue
        slot = int(suffix)
        cat = types_map.get(str(slot), "unknown")
        if cat in RISKY_TYPES:
            labels.append(pos_map[old_pos])

    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Source directory of .go files")
    p.add_argument("--dst", type=Path, required=True, help="Output directory")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--callgraph-helper", type=Path, default=DEFAULT_CALLGRAPH)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-files", type=int, default=None,
                   help="Limit number of .go files (for quick POC)")
    args = p.parse_args()

    if not args.tokenizer.exists():
        sys.exit(f"ast-tokenize not found at {args.tokenizer}; build it first")
    if not args.callgraph_helper.exists():
        sys.exit(f"ast-callgraph not found at {args.callgraph_helper}; build it first")

    assert VOCAB_SIZE < 2**16, f"vocab {VOCAB_SIZE} too large for uint16"
    args.dst.mkdir(parents=True, exist_ok=True)

    files = find_go_files(args.src)
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        sys.exit(f"no .go files under {args.src}")

    rng = random.Random(args.seed)
    rng.shuffle(files)

    print(f"Tokenizing {len(files)} files...")
    tokenize_raw(args.tokenizer, files, args.dst, args.val_frac, args.batch, rng)

    print("Building call graph...")
    callers_map = build_callgraph(args.callgraph_helper, args.src, args.dst)

    print("Building metadata and mutations...")
    for split in ("train", "val"):
        meta = build_meta(args.dst, split, callers_map)
        mutate_split(args.dst, split, meta)
        n = len(meta)
        print(f"  {split}: {n} original units, up to {n} mutated units")

    print(f"Done. Race corpus in {args.dst}")


if __name__ == "__main__":
    main()
