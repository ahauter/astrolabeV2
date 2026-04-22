"""Tokenize a directory of Go source files into a flat uint16 stream.

Walks --src for `.go` files, hands them to the `ast-tokenize` Go helper in
batches, maps each emitted token string through `vocab`, and appends the
resulting IDs to `train.bin` / `val.bin` under --dst.

Usage:
    python -m astrolabe.prepare --src scraped_code --dst data --val-frac 0.05
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from astrolabe.vocab import TOKEN_TO_ID, VOCAB_SIZE


# Path to the compiled helper, resolved relative to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HELPER = REPO_ROOT / "ast-tokenize"


def find_go_files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*.go") if p.is_file())


def tokenize_batch(helper: Path, paths: list[Path]) -> list[list[int]]:
    """Spawn the Go helper once for a batch of files; partition the flat
    output stream by the BOS token so we get one id-list per file."""
    if not paths:
        return []
    proc = subprocess.run(
        [str(helper), *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        check=False,
    )
    streams: list[list[int]] = []
    current: list[int] | None = None
    for line in proc.stdout.splitlines():
        tok = line.strip()
        if not tok:
            continue
        tok_id = TOKEN_TO_ID.get(tok)
        if tok_id is None:
            print(f"warn: unknown token {tok!r} from helper", file=sys.stderr)
            continue
        if tok == "BOS":
            if current is not None:
                streams.append(current)
            current = [tok_id]
        else:
            if current is None:
                # helper emitted something before BOS — skip
                continue
            current.append(tok_id)
    if current is not None:
        streams.append(current)
    if proc.returncode != 0 and not streams:
        print(f"warn: helper exited {proc.returncode}: {proc.stderr[:200]}",
              file=sys.stderr)
    return streams


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

    args.dst.mkdir(parents=True, exist_ok=True)
    files = find_go_files(args.src)
    if not files:
        sys.exit(f"no .go files under {args.src}")

    rng = random.Random(args.seed)
    rng.shuffle(files)

    train_path = args.dst / "train.bin"
    val_path = args.dst / "val.bin"
    # uint16 since vocab is small; assert that assumption holds.
    assert VOCAB_SIZE < 2**16, f"vocab {VOCAB_SIZE} too large for uint16"

    total = {"train": 0, "val": 0}
    file_count = {"train": 0, "val": 0}
    with open(train_path, "wb") as tf, open(val_path, "wb") as vf:
        for i in range(0, len(files), args.batch):
            batch = files[i:i + args.batch]
            streams = tokenize_batch(args.helper, batch)
            for stream in streams:
                if not stream:
                    continue
                split = "val" if rng.random() < args.val_frac else "train"
                arr = np.asarray(stream, dtype=np.uint16)
                sink = vf if split == "val" else tf
                arr.tofile(sink)
                total[split] += arr.size
                file_count[split] += 1
            done = min(i + args.batch, len(files))
            print(f"  processed {done}/{len(files)} files  "
                  f"train={total['train']} val={total['val']} tokens",
                  flush=True)

    print(f"\nwrote {train_path} ({total['train']} tokens, "
          f"{file_count['train']} files)")
    print(f"wrote {val_path} ({total['val']} tokens, "
          f"{file_count['val']} files)")


if __name__ == "__main__":
    main()
