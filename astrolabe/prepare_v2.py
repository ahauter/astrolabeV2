"""Parallel tokenizer for the unified Go corpus.

Produces the base unit streams used by pretraining, nil/bounds fine-tuning,
and the online-mutation race dataloader.
"""
from __future__ import annotations

import argparse
import array
import json
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from astrolabe.prepare import find_go_files, stream_units


def _worker(args):
    """Tokenize one chunk and write per-split temp files."""
    idx, files, tokenizer, batch, val_frac, seed, tmpdir, quality_filter = args
    rng = random.Random(seed ^ (idx + 1))
    tmpdir = Path(tmpdir)
    train_bin = tmpdir / f"train_{idx}.bin"
    val_bin = tmpdir / f"val_{idx}.bin"
    train_ann = tmpdir / f"train_{idx}.ann.jsonl"
    val_ann = tmpdir / f"val_{idx}.ann.jsonl"
    train_idx = tmpdir / f"train_{idx}.idx.npy"
    val_idx = tmpdir / f"val_{idx}.idx.npy"

    t_off = array.array("Q", [0])
    v_off = array.array("Q", [0])

    with open(train_bin, "wb") as tf, open(val_bin, "wb") as vf, \
         open(train_ann, "w") as ta, open(val_ann, "w") as va:
        for i in range(0, len(files), batch):
            batch_files = files[i:i + batch]
            for unit, ann in stream_units(tokenizer, batch_files):
                if not unit:
                    continue
                arr = np.array(unit, dtype=np.uint16)
                if rng.random() < val_frac:
                    arr.tofile(vf)
                    v_off.append(v_off[-1] + len(unit))
                    va.write(json.dumps(ann) + "\n")
                else:
                    arr.tofile(tf)
                    t_off.append(t_off[-1] + len(unit))
                    ta.write(json.dumps(ann) + "\n")

    np.save(train_idx, np.frombuffer(t_off, dtype=np.uint64))
    np.save(val_idx, np.frombuffer(v_off, dtype=np.uint64))

    return {
        "train_count": len(t_off) - 1,
        "val_count": len(v_off) - 1,
        "train_tokens": int(t_off[-1]),
        "val_tokens": int(v_off[-1]),
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "train_ann": str(train_ann),
        "val_ann": str(val_ann),
        "train_idx": str(train_idx),
        "val_idx": str(val_idx),
    }


def _merge_split(dst: Path, split: str, results: list[dict]) -> None:
    """Concatenate per-worker temp files into final split files and offsets."""
    bin_out = dst / f"{split}_units.bin"
    ann_out = dst / f"{split}_ann.jsonl"
    idx_out = dst / f"{split}_units.idx.npy"

    offsets = array.array("Q", [0])

    with open(bin_out, "wb") as bf, open(ann_out, "w") as af:
        for res in results:
            with open(res[f"{split}_bin"], "rb") as rf:
                shutil.copyfileobj(rf, bf)
            with open(res[f"{split}_ann"], "r") as rf:
                shutil.copyfileobj(rf, af)
            woff = np.load(res[f"{split}_idx"])
            base = offsets[-1]
            for k in range(1, len(woff)):
                offsets.append(base + int(woff[k]))

    np.save(idx_out, np.frombuffer(offsets, dtype=np.uint64))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Root directory of .go files")
    p.add_argument("--dst", type=Path, required=True, help="Output directory")
    p.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "ast-tokenize")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=64, help="Files per helper invocation")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quality-filter", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=True, help="Filter vendor/_workspace/.pb.go files")
    p.add_argument("--tmpdir", type=Path, default=None,
                   help="Directory for intermediate worker files (default: dst/.tmp_tok)")
    args = p.parse_args()

    if not args.tokenizer.exists():
        sys.exit(f"ast-tokenize not found at {args.tokenizer}")

    args.dst.mkdir(parents=True, exist_ok=True)
    tmpdir = args.tmpdir or args.dst / ".tmp_tok"
    tmpdir.mkdir(parents=True, exist_ok=True)

    files = find_go_files(args.src, quality_filter=args.quality_filter)
    if not files:
        sys.exit(f"no .go files under {args.src}")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    print(f"Tokenizing {len(files)} files with {args.workers} workers "
          f"(batch={args.batch}, val_frac={args.val_frac})...")

    n = len(files)
    workers = min(args.workers, n)
    chunk_size = (n + workers - 1) // workers
    chunks = [files[i:i + chunk_size] for i in range(0, n, chunk_size)]

    try:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            work = [(i, chunk, args.tokenizer, args.batch, args.val_frac, args.seed,
                     tmpdir, args.quality_filter)
                    for i, chunk in enumerate(chunks)]
            results = list(exe.map(_worker, work))
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    print(f"Worker totals: train={sum(r['train_count'] for r in results)}, "
          f"val={sum(r['val_count'] for r in results)}")

    for split in ("train", "val"):
        _merge_split(args.dst, split, results)
        print(f"Merged {split} split")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"Done. Base corpus in {args.dst}")


if __name__ == "__main__":
    main()
