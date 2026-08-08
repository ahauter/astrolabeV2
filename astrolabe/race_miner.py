"""Static race-risk miner for the race-condition detection pipeline.

Reads the base corpus annotations and caller metadata to produce a JSONL file
with per-unit static race-risk positions.  Only functions WITHOUT internal sync
primitives are considered; functions that do contain sync primitives are left
to the online mutation dataloader.

A no-sync function is labeled with its *risky external* use positions when it
is in a concurrent context:
  - the function itself contains a `go` statement, OR
  - any depth-1 caller contains a `go` statement.

A use is considered risky and external when:
  - it is a NAME_N token whose coarse type is in RISKY_TYPES, and
  - it is not defined locally inside the function (package var, parameter,
    receiver, or closure capture).

Sequential no-sync functions produce empty labels, giving the model hard
negatives for shared accesses that are never reached from a goroutine.

Usage:
    python -m astrolabe.race_miner \
        --units data_v1/train_units.bin \
        --idx   data_v1/train_units.idx.npy \
        --ann   data_v1/train_ann.jsonl \
        --meta  data_v1/race_train_meta.jsonl \
        --out   data_v1/race_risk_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astrolabe.vocab import ID_TO_TOKEN


RISKY_TYPES = {"ptr", "slice", "map", "chan", "interface", "struct", "func"}


def load_units(units_bin: Path, idx_path: Path) -> tuple[np.memmap, np.ndarray]:
    data = np.memmap(Path(units_bin), dtype=np.uint16, mode="r")
    offsets = np.load(Path(idx_path))
    return data, offsets


def _load_ann_or_empty(line: str) -> dict:
    line = line.strip()
    if not line or line == "null":
        return {}
    return json.loads(line)


def _build_line_offsets(path: Path) -> np.ndarray:
    """Build a uint64 offset array for each newline in *path*.

    Saves the result as ``<path>.idx.npy`` and reuses it on subsequent runs.
    """
    idx_path = Path(str(path) + ".idx.npy")
    if idx_path.exists():
        return np.load(idx_path)

    offsets = [0]
    pos = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            start = 0
            while True:
                nl = chunk.find(b"\n", start)
                if nl == -1:
                    break
                offsets.append(pos + nl + 1)
                start = nl + 1
            pos += len(chunk)

    arr = np.array(offsets, dtype=np.uint64)
    np.save(idx_path, arr)
    return arr


def _name_slot(name_tok: str) -> int | None:
    """Return the integer slot of a NAME_N token, or None for special NAME_* tokens."""
    if not name_tok.startswith("NAME_"):
        return None
    suffix = name_tok[len("NAME_"):]
    if suffix.isdigit():
        return int(suffix)
    return None


def _slot_type(name_tok: str, types_map: dict[str, str]) -> str:
    """Return the coarse type category for a NAME_N token using the ANN types map."""
    slot = _name_slot(name_tok)
    if slot is None:
        return "unknown"
    return types_map.get(str(slot), "unknown")


def _is_external_use(pos: int, use_set: set[int], du: dict[int, int], local_defs: set[int]) -> bool:
    """Return True if the token at position `pos` is a use of a name
    that is defined *outside* the current function (package import, global,
    parameter, receiver, or closure capture).
    """
    if pos not in use_set:
        return False
    if pos in du:
        return du[pos] not in local_defs
    return True


def find_static_race_risks(tokens: list[str], ann: dict | None) -> list[int]:
    """Find risky external uses in a function with no internal sync primitives."""
    risks: list[int] = []
    if ann is None:
        return risks

    use_set = set(ann.get("use", []))
    du = {int(k): v for k, v in ann.get("du", {}).items()}
    local_defs = set(ann.get("def", []))
    types_map = ann.get("types", {})

    for i, tok in enumerate(tokens):
        if _name_slot(tok) is None:
            continue
        if _slot_type(tok, types_map) not in RISKY_TYPES:
            continue
        if not _is_external_use(i, use_set, du, local_defs):
            continue
        risks.append(i)

    return risks


def _has_go_spawns(ann: dict) -> bool:
    return bool(ann.get("go"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=Path, required=True)
    p.add_argument("--idx", type=Path, required=True)
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-units", type=int, default=None,
                   help="Process only first N units (for quick POC iteration)")
    args = p.parse_args()

    data, offsets = load_units(args.units, args.idx)
    n_units = len(offsets) - 1
    if args.max_units is not None:
        n_units = min(n_units, args.max_units)

    # Pass 1: determine which units contain go statements.
    print("Pass 1: indexing goroutine spawns ...", flush=True)
    has_go = np.zeros(n_units, dtype=np.bool_)
    with open(args.ann, "r", encoding="utf-8") as f:
        for idx in range(n_units):
            line = f.readline()
            ann = _load_ann_or_empty(line)
            has_go[idx] = _has_go_spawns(ann)
            if (idx + 1) % 1_000_000 == 0:
                print(f"  processed {idx + 1:,} units", flush=True)

    # Pass 2: write static race risks.
    print("Pass 2: writing static race risks ...", flush=True)
    total_risks = 0
    units_with_risks = 0
    concurrent_units = 0
    with open(args.ann, "r", encoding="utf-8") as f_ann, \
         open(args.meta, "r", encoding="utf-8") as f_meta, \
         open(args.out, "w", encoding="utf-8") as f_out:
        for idx in range(n_units):
            ann_line = f_ann.readline()
            meta_line = f_meta.readline()
            ann = _load_ann_or_empty(ann_line)
            meta = _load_ann_or_empty(meta_line)

            race_risks: list[int] = []
            if not ann.get("sync"):
                # No internal sync: check concurrent context.
                concurrent = bool(has_go[idx])
                if not concurrent:
                    for cidx in meta.get("callers", []):
                        if 0 <= cidx < n_units and has_go[cidx]:
                            concurrent = True
                            break
                if concurrent:
                    start = int(offsets[idx])
                    end = int(offsets[idx + 1])
                    tokens = [ID_TO_TOKEN[t] for t in data[start:end].tolist()]
                    race_risks = sorted(set(find_static_race_risks(tokens, ann)))
                    if race_risks:
                        total_risks += len(race_risks)
                        units_with_risks += 1
                    concurrent_units += 1

            f_out.write(json.dumps({"race_risks": race_risks}) + "\n")

            if (idx + 1) % 1_000_000 == 0:
                print(f"  wrote {idx + 1:,} units", flush=True)

    # Build line-offset index for fast random access by the dataloader.
    print("Building line-offset index ...", flush=True)
    idx_path = Path(str(args.out) + ".idx.npy")
    if idx_path.exists():
        idx_path.unlink()
    _build_line_offsets(args.out)

    print(f"Wrote {args.out}")
    print(f"  Concurrent no-sync units: {concurrent_units:,} / {n_units:,}")
    print(f"  Units with static race risks: {units_with_risks:,}")
    print(f"  Total static race risk positions: {total_risks:,}")


if __name__ == "__main__":
    main()
