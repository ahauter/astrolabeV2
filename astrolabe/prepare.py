"""Extract Go AST declaration units and persist them for live-mix training.

Walks --src for `.go` files, hands them to the `ast-tokenize` Go helper in
batches, splits each file's token stream into individual top-level declaration
units (functions, structs, type/var/const decls), and saves them as a flat
uint16 binary + CSR offset index under --dst.

ANN lines emitted by the helper after each function declaration are parsed and
stored as a parallel JSONL file (train_ann.jsonl / val_ann.jsonl). Non-function
units get a null line.

The data iterator (CFGUnitMixDataset) assembles random permutations of units
on-the-fly during training, giving effectively unlimited sample diversity.

Usage:
    python -m astrolabe.prepare --src scraped_code --dst data --val-frac 0.05
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
from typing import Iterator

import numpy as np

from astrolabe.vocab import TOKEN_TO_ID, VOCAB_SIZE, BOS_ID, EOF_ID


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HELPER = REPO_ROOT / "ast-tokenize"

_LOW_QUALITY_PARTS = frozenset({"vendor", "_workspace"})


def _is_quality(path: Path) -> bool:
    parts = set(path.parts)
    if parts & _LOW_QUALITY_PARTS:
        return False
    if path.name.endswith(".pb.go"):
        return False
    return True


def find_go_files(src: Path, quality_filter: bool = True) -> list[Path]:
    # Use os.walk so directory symlinks are followed (useful for corpus
    # directories that are assembled from multiple external trees).
    files: list[Path] = []
    for root, _, filenames in os.walk(src, followlinks=True):
        for name in filenames:
            if not name.endswith(".go"):
                continue
            p = Path(root) / name
            if not p.is_file():
                continue
            if quality_filter and not _is_quality(p):
                continue
            files.append(p)
    return sorted(files)


def stream_units(
    helper: Path, paths: list[Path]
) -> Iterator[tuple[list[int], dict | None]]:
    """Stream individual declaration units from the Go helper one at a time.

    Spawns the helper once for the batch and reads its stdout line-by-line to
    avoid buffering the entire output.  Yields (tokens, annotation) for each
    EOF-delimited unit; annotation is None for non-function declarations.
    """
    if not paths:
        return
    proc = subprocess.Popen(
        [str(helper), *[str(p) for p in paths]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None

    current: list[int] = []
    pending_ann: dict | None = None

    for line in proc.stdout:
        raw = line.strip()
        if not raw:
            continue

        if raw.startswith("ANN "):
            try:
                pending_ann = json.loads(raw[4:])
            except json.JSONDecodeError as exc:
                print(f"warn: bad ANN JSON: {exc}", file=sys.stderr)
                pending_ann = None
            continue

        # File-level metadata lines are not part of the token stream.
        if raw.startswith(("POSMAP ", "PKGS ", "NAMEPOSMAP ")):
            continue

        tok_id = TOKEN_TO_ID.get(raw)
        if tok_id is None:
            print(f"warn: unknown token {raw!r} from helper", file=sys.stderr)
            continue

        if raw == "BOS":
            # Flush any incomplete unit at a file boundary
            if current:
                yield (current, None)
                current = []
                pending_ann = None
        elif raw == "EOF":
            if current:
                yield (current, pending_ann)
            current = []
            pending_ann = None
        else:
            current.append(tok_id)

    if current:
        yield (current, None)

    proc.wait()
    if proc.returncode != 0:
        stderr_out = proc.stderr.read(200)
        print(f"warn: helper exited {proc.returncode}: {stderr_out}",
              file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Directory of .go files")
    p.add_argument("--dst", type=Path, required=True, help="Output directory")
    p.add_argument("--helper", type=Path, default=DEFAULT_HELPER,
                   help="Path to ast-tokenize binary")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=32,
                   help="Files per helper invocation")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.helper.exists():
        sys.exit(f"ast-tokenize helper not found at {args.helper}; "
                 f"run `go build ./cmd/ast-tokenize` from the repo root")

    assert VOCAB_SIZE < 2**16, f"vocab {VOCAB_SIZE} too large for uint16"
    args.dst.mkdir(parents=True, exist_ok=True)

    files = find_go_files(args.src)
    if not files:
        sys.exit(f"no .go files under {args.src}")

    rng = random.Random(args.seed)
    rng.shuffle(files)

    # array.array uses 8 bytes/entry vs ~28 for a Python list[int], saving
    # ~440 MB at 23M units compared to the previous list approach.
    train_offsets: array.array = array.array('Q', [0])
    val_offsets:   array.array = array.array('Q', [0])

    with open(args.dst / "train_units.bin",  "wb") as tf, \
         open(args.dst / "val_units.bin",    "wb") as vf, \
         open(args.dst / "train_ann.jsonl",  "w")  as ta, \
         open(args.dst / "val_ann.jsonl",    "w")  as va:
        for i in range(0, len(files), args.batch):
            batch = files[i:i + args.batch]
            for unit, ann in stream_units(args.helper, batch):
                if not unit:
                    continue
                arr = np.array(unit, dtype=np.uint16)
                if rng.random() < args.val_frac:
                    arr.tofile(vf)
                    val_offsets.append(val_offsets[-1] + len(unit))
                    va.write(json.dumps(ann) + "\n")
                else:
                    arr.tofile(tf)
                    train_offsets.append(train_offsets[-1] + len(unit))
                    ta.write(json.dumps(ann) + "\n")
            done = min(i + args.batch, len(files))
            print(f"  processed {done}/{len(files)} files  "
                  f"train_units={len(train_offsets)-1} val_units={len(val_offsets)-1}",
                  flush=True)

    np.save(args.dst / "train_units.idx.npy",
            np.frombuffer(train_offsets, dtype=np.uint64))
    np.save(args.dst / "val_units.idx.npy",
            np.frombuffer(val_offsets,   dtype=np.uint64))

    n_train = len(train_offsets) - 1
    n_val   = len(val_offsets)   - 1
    print(f"\nwrote {args.dst}/train_units.bin  "
          f"({n_train} units, {train_offsets[-1]} tokens)")
    print(f"wrote {args.dst}/val_units.bin    "
          f"({n_val} units, {val_offsets[-1]} tokens)")
    print(f"wrote {args.dst}/train_ann.jsonl  ({n_train} annotations)")
    print(f"wrote {args.dst}/val_ann.jsonl    ({n_val} annotations)")


if __name__ == "__main__":
    main()
