"""Build race caller-meta files from a base corpus + callgraph.

Keeps memory low by storing the callgraph in an SQLite database instead of
loading it into a Python dict.

Usage:
    python -m astrolabe.build_race_meta --data-dir data_v1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _load_ann_or_empty(line: str) -> dict:
    line = line.strip()
    if not line or line == "null":
        return {}
    return json.loads(line)


def _ensure_callgraph_db(cg_path: Path, db_path: Path, max_callers: int) -> None:
    """Build a SQLite lookup table from the callgraph JSONL file."""
    if db_path.exists():
        print(f"Using existing callgraph db: {db_path}")
        return

    print(f"Building callgraph db from {cg_path} ...")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE callers (func TEXT PRIMARY KEY, callers TEXT)")

    batch: list[tuple[str, str]] = []
    total = 0
    with open(cg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            func = rec["func"]
            callers = rec.get("callers", [])[:max_callers]
            batch.append((func, json.dumps(callers)))
            if len(batch) >= 50_000:
                conn.executemany("INSERT OR IGNORE INTO callers VALUES (?, ?)", batch)
                total += len(batch)
                batch.clear()
                if total % 500_000 == 0:
                    print(f"  inserted {total} entries")
    if batch:
        conn.executemany("INSERT OR IGNORE INTO callers VALUES (?, ?)", batch)
        total += len(batch)

    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_callers_func ON callers(func)")
    conn.commit()
    conn.close()
    print(f"  done: {total} entries -> {db_path}")


def _build_meta_streaming(
    ann_path: Path,
    db_path: Path,
    out_path: Path,
    max_callers: int = 8,
) -> int:
    """Two-pass streaming builder. Returns number of units written."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Pass 1: map function name -> first unit index.
    print(f"  pass 1: indexing function names ...", flush=True)
    func_to_idx: dict[str, int] = {}
    with open(ann_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            ann = _load_ann_or_empty(line)
            name = ann.get("func")
            if name and name not in func_to_idx:
                func_to_idx[name] = idx
            if (idx + 1) % 1_000_000 == 0:
                print(f"    indexed {idx + 1} units, {len(func_to_idx)} names", flush=True)

    # Pass 2: resolve caller names to indices and write meta.
    print(f"  pass 2: resolving callers and writing {out_path.name} ...", flush=True)
    written = 0
    with open(ann_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            ann = _load_ann_or_empty(line)
            name = ann.get("func")
            caller_funcs: list[str] = []
            if name:
                row = cur.execute(
                    "SELECT callers FROM callers WHERE func = ?", (name,)
                ).fetchone()
                if row:
                    caller_funcs = json.loads(row[0])[:max_callers]

            caller_idxs: list[int] = []
            for cf in caller_funcs:
                ci = func_to_idx.get(cf)
                if ci is not None:
                    caller_idxs.append(ci)

            record = {"func": name, "callers": caller_idxs}
            f_out.write(json.dumps(record) + "\n")
            written += 1
            if written % 1_000_000 == 0:
                print(f"    wrote {written} units", flush=True)

    conn.close()
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--callgraph", type=Path, default=None)
    p.add_argument("--callgraph-db", type=Path, default=None)
    p.add_argument("--max-callers", type=int, default=8)
    p.add_argument("--rebuild-db", action="store_true")
    args = p.parse_args()

    cg_path = args.callgraph or args.data_dir / "callgraph.jsonl"
    if not cg_path.exists():
        sys.exit(f"callgraph not found: {cg_path}")

    db_path = args.callgraph_db or args.data_dir / "callgraph.db"
    if args.rebuild_db and db_path.exists():
        db_path.unlink()

    _ensure_callgraph_db(cg_path, db_path, args.max_callers)

    for split in ("train", "val"):
        ann_path = args.data_dir / f"{split}_ann.jsonl"
        if not ann_path.exists():
            sys.exit(f"annotations not found: {ann_path}")
        out_path = args.data_dir / f"race_{split}_meta.jsonl"
        print(f"Building {split} race meta -> {out_path.name} ...")
        n = _build_meta_streaming(ann_path, db_path, out_path, args.max_callers)
        print(f"  wrote {n} units")

    print(f"Done. Race meta files in {args.data_dir}")


if __name__ == "__main__":
    main()
