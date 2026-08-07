"""Generate a synthetic race-condition corpus for HierarchicalRiskGPT.

Each sample is a self-contained Go function that either contains a real data
race (positive) or protects shared state with a mutex (negative).  The race
samples are verified with `go run -race`.  The output matches the layout
expected by `RaceContextDataset`.

Usage:
    python -m astrolabe.prepare_race_synth \
        --out-dir data_race_synth \
        --n-train 600 \
        --n-val 100
"""
from __future__ import annotations

import argparse
import array
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from astrolabe.detect import extract_function_units, tokenize_go_file
from astrolabe.vocab import TOKEN_TO_ID, VOCAB_SIZE, PAD_ID


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = REPO_ROOT / "ast-tokenize"


TEMPLATES: list[dict] = [
    {
        "name": "slice_append",
        "shared": "s",
        "positive": """package main
import "sync"
func F() int {{
    var s []int
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); s = append(s, 1) }}()
    s = append(s, 2)
    wg.Wait()
    return len(s)
}}
func main() {{ F() }}
""",
        "negative": """package main
import "sync"
func F() int {{
    var s []int
    var mu sync.Mutex
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); mu.Lock(); s = append(s, 1); mu.Unlock() }}()
    mu.Lock(); s = append(s, 2); mu.Unlock()
    wg.Wait()
    return len(s)
}}
func main() {{ F() }}
""",
    },
    {
        "name": "map_increment",
        "shared": "m",
        "positive": """package main
import "sync"
func F() int {{
    m := make(map[string]int)
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); m["x"]++ }}()
    m["x"]++
    wg.Wait()
    return m["x"]
}}
func main() {{ F() }}
""",
        "negative": """package main
import "sync"
func F() int {{
    m := make(map[string]int)
    var mu sync.Mutex
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); mu.Lock(); m["x"]++; mu.Unlock() }}()
    mu.Lock(); m["x"]++; mu.Unlock()
    wg.Wait()
    return m["x"]
}}
func main() {{ F() }}
""",
    },
    {
        "name": "pointer_increment",
        "shared": "p",
        "positive": """package main
import "sync"
func F() int {{
    p := new(int)
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); *p = *p + 1 }}()
    *p = *p + 2
    wg.Wait()
    return *p
}}
func main() {{ F() }}
""",
        "negative": """package main
import "sync"
func F() int {{
    p := new(int)
    var mu sync.Mutex
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); mu.Lock(); *p = *p + 1; mu.Unlock() }}()
    mu.Lock(); *p = *p + 2; mu.Unlock()
    wg.Wait()
    return *p
}}
func main() {{ F() }}
""",
    },
    {
        "name": "struct_field",
        "shared": "c",
        "positive": """package main
import "sync"
type Counter struct {{ n int }}
func F() int {{
    c := &Counter{{}}
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); c.n++ }}()
    c.n++
    wg.Wait()
    return c.n
}}
func main() {{ F() }}
""",
        "negative": """package main
import "sync"
type Counter struct {{ n int }}
func F() int {{
    c := &Counter{{}}
    var mu sync.Mutex
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {{ defer wg.Done(); mu.Lock(); c.n++; mu.Unlock() }}()
    mu.Lock(); c.n++; mu.Unlock()
    wg.Wait()
    return c.n
}}
func main() {{ F() }}
""",
    },
]


def run_race_check(src: Path, timeout: int = 15) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["go", "run", "-race", src.name],
        cwd=src.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def build_corpus(
    n_train: int,
    n_val: int,
    out_dir: Path,
    tokenizer: Path,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    splits = {
        "train": n_train,
        "val": n_val,
    }

    for split, total in splits.items():
        units_path = out_dir / f"race_{split}_units.bin"
        idx_path = out_dir / f"race_{split}_units.idx.npy"
        labels_path = out_dir / f"race_{split}_labels.jsonl"
        meta_path = out_dir / f"race_{split}_meta.jsonl"

        offsets: array.array = array.array("Q", [0])
        labels_file = open(labels_path, "w")
        meta_file = open(meta_path, "w")
        units_file = open(units_path, "wb")

        generated = 0
        attempts = 0
        positives_target = total // 2
        negatives_target = total - positives_target
        positives = 0
        negatives = 0

        with tempfile.TemporaryDirectory(dir=out_dir, prefix=f"build_{split}_") as tmp:
            tmp_path = Path(tmp)
            while generated < total and attempts < total * 10:
                attempts += 1
                want_positive = positives < positives_target or (
                    negatives >= negatives_target and positives < total
                )
                tmpl = rng.choice(TEMPLATES)
                program = tmpl["positive" if want_positive else "negative"]
                name = f"F{generated:05d}"
                program = program.replace("func F()", f"func {name}()")
                program = program.replace("main() { F() }", f"main() {{ {name}() }}")

                src = tmp_path / f"{name}.go"
                src.write_text(program)

                try:
                    rc, _, stderr = run_race_check(src)
                except subprocess.TimeoutExpired:
                    continue

                has_race = "WARNING: DATA RACE" in stderr
                if want_positive and not has_race:
                    continue
                if not want_positive and has_race:
                    continue

                # Tokenize the generated file.
                try:
                    _, _, _, name_pos_map = tokenize_go_file(src, tokenizer)
                    units = extract_function_units(src, tokenizer)
                except Exception as exc:
                    print(f"warning: tokenization failed for {src}: {exc}", file=sys.stderr)
                    continue

                target = next((u for u in units if u[0] == name), None)
                if target is None:
                    continue

                _, target_tokens, target_start, _ = target
                shared_name = tmpl["shared"]
                # Find all global token positions for the shared identifier that
                # lie inside the target function.
                risk_positions = [
                    int(global_pos) - target_start
                    for global_pos, orig_name in name_pos_map.items()
                    if orig_name == shared_name and target_start <= int(global_pos) < target_start + len(target_tokens)
                ]

                if want_positive and not risk_positions:
                    continue

                unit_ids = [
                    min(TOKEN_TO_ID.get(t, PAD_ID), VOCAB_SIZE - 1)
                    for t in target_tokens
                ]
                arr = np.array(unit_ids, dtype=np.uint16)
                arr.tofile(units_file)
                offsets.append(offsets[-1] + len(unit_ids))

                labels = {
                    "race_risks": sorted(risk_positions) if want_positive else [],
                    "nil_risks": [],
                    "bounds_risks": [],
                }
                labels_file.write(json.dumps(labels) + "\n")
                meta_file.write(json.dumps({"func": name, "callers": []}) + "\n")

                generated += 1
                if want_positive:
                    positives += 1
                else:
                    negatives += 1

        units_file.close()
        labels_file.close()
        meta_file.close()
        np.save(idx_path, np.frombuffer(offsets, dtype=np.uint64))
        print(
            f"{split}: generated {generated} units "
            f"({positives} positive, {negatives} negative) in {attempts} attempts"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data_race_synth"))
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_corpus(args.n_train, args.n_val, args.out_dir, args.tokenizer, args.seed)


if __name__ == "__main__":
    main()
