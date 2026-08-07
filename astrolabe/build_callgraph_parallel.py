"""Parallel wrapper around ast-callgraph.

Partitions the source tree into chunks and runs ast-callgraph on each chunk in
parallel, then merges the callee->callers mappings into one sorted JSONL file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _find_top_dirs(src: Path) -> list[Path]:
    """Return immediate subdirectories that contain at least one .go file."""
    tops: list[Path] = []
    for entry in os.scandir(src):
        if not entry.is_dir(follow_symlinks=True):
            continue
        has_go = False
        for root, _, files in os.walk(entry.path, followlinks=True):
            if any(f.endswith(".go") for f in files):
                has_go = True
                break
        if has_go:
            tops.append(Path(entry.path))
    return sorted(tops)


def _run_chunk(args: tuple[int, list[Path], Path, Path]) -> Path:
    idx, dirs, helper, tmpdir = args
    out = tmpdir / f"cg_{idx}.jsonl"
    proc = subprocess.run(
        [str(helper), *[str(d) for d in dirs]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(f"warning: chunk {idx} failed: {proc.stderr[:500]}", file=sys.stderr)
    with open(out, "w") as f:
        f.write(proc.stdout)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Root directory to scan")
    p.add_argument("--out", type=Path, required=True, help="Output callgraph.jsonl")
    p.add_argument("--helper", type=Path, default=REPO_ROOT / "ast-callgraph")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk-dirs", type=int, default=50,
                   help="Number of top-level directories per ast-callgraph invocation")
    args = p.parse_args()

    if not args.helper.exists():
        sys.exit(f"ast-callgraph not found at {args.helper}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    top_dirs = _find_top_dirs(args.src)
    if not top_dirs:
        sys.exit(f"no Go directories under {args.src}")
    print(f"Found {len(top_dirs)} top-level directories; chunk size {args.chunk_dirs}")

    chunks = [top_dirs[i:i + args.chunk_dirs] for i in range(0, len(top_dirs), args.chunk_dirs)]
    tmpdir = Path(tempfile.mkdtemp(prefix="cg_"))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            work = [(i, chunk, args.helper, tmpdir) for i, chunk in enumerate(chunks)]
            chunk_files = list(exe.map(_run_chunk, work))

        callers: dict[str, set[str]] = {}
        for cf in chunk_files:
            with open(cf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    callee = rec["func"]
                    s = callers.setdefault(callee, set())
                    s.update(rec.get("callers", []))

        with open(args.out, "w") as f:
            for callee in sorted(callers):
                f.write(json.dumps({
                    "func": callee,
                    "callers": sorted(callers[callee]),
                }) + "\n")
        print(f"Wrote {args.out} ({len(callers)} callees)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
