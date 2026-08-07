"""Manually generate a small, verified race-condition eval set.

Uses deterministic templates instead of an LLM so the eval can be reproduced
quickly. Each generated program is verified with `go run -race`.

Usage:
    python -m astrolabe.synth_race_manual --out-dir synth_eval_race_manual
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from astrolabe.synth_eval import (
    REPO_ROOT,
    DEFAULT_HELPER,
    build_program,
    classify_panic,
    find_risk_token_position,
    parse_panic_line,
    run_program,
    tokenize_generated,
)

RACE_TEMPLATES = [
    (
        "func {func_name}(m map[string]int) int",
        "m[\"x\"]++; return 0",
        [],
        """m := map[string]int{{}}
var wg sync.WaitGroup
wg.Add(1)
go func() {{ defer wg.Done(); {func_name}(m) }}()
{func_name}(m)
wg.Wait()""",
    ),
    (
        "func {func_name}(s *[]int) int",
        "*s = append(*s, 1); return 0",
        [],
        """s := make([]int, 0)
var wg sync.WaitGroup
wg.Add(1)
go func() {{ defer wg.Done(); {func_name}(&s) }}()
{func_name}(&s)
wg.Wait()""",
    ),
    (
        "func {func_name}(n *int) int",
        "*n++; return 0",
        [],
        """n := 0
var wg sync.WaitGroup
wg.Add(1)
go func() {{ defer wg.Done(); {func_name}(&n) }}()
{func_name}(&n)
wg.Wait()""",
    ),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("synth_eval_race_manual"))
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--helper", type=Path, default=DEFAULT_HELPER)
    args = p.parse_args()

    if not args.helper.exists():
        sys.exit(f"ast-tokenize not found at {args.helper}; build it first")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    rng = random.Random(0)

    for idx in range(args.count):
        signature_template, behavior, structs, main_call_template = rng.choice(RACE_TEMPLATES)
        func_name = f"RiskRace{idx:04d}"
        signature = signature_template.format(func_name=func_name)
        main_call = main_call_template.format(func_name=func_name)

        code = build_program(signature, func_name, behavior, structs, main_call, imports=["sync"])
        work_dir = args.out_dir / f"race_{idx:04d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        generated_path = work_dir / "generated.go"
        generated_path.write_text(code)

        try:
            rc, stdout, stderr = run_program(generated_path, race=True, timeout=15)
        except subprocess.TimeoutExpired:
            print(f"  {func_name}: timeout")
            continue

        panic_kind = classify_panic(stderr)
        if panic_kind != "race":
            print(f"  {func_name}: no race detected (rc={rc})")
            continue

        panic_line = parse_panic_line(stderr)
        if panic_line is None:
            print(f"  {func_name}: could not parse race line")
            continue

        try:
            tokens, pos_map, name_pos_map, types_map = tokenize_generated(generated_path, args.helper)
        except Exception as exc:
            print(f"  {func_name}: tokenization failed: {exc}")
            continue

        token_pos = find_risk_token_position(tokens, pos_map, panic_line, "race")
        if token_pos is None:
            print(f"  {func_name}: could not align race line {panic_line} to token")
            continue

        print(f"  {func_name} OK: race at generated.go:{panic_line} token={tokens[token_pos]}")
        labels.append({
            "id": f"race_{idx:04d}",
            "kind": "race",
            "func_name": func_name,
            "source": str(generated_path),
            "panic_kind": panic_kind,
            "panic_line": panic_line,
            "token_pos": token_pos,
            "token_name": tokens[token_pos],
            "orig_name": name_pos_map.get(token_pos),
            "type_cat": types_map.get(token_pos, "unknown"),
            "signature": signature,
            "behavior": behavior,
        })

    labels_path = args.out_dir / "labels.jsonl"
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    print(f"\nGenerated {len(labels)} verified race samples")
    print(f"Saved to {labels_path}")


if __name__ == "__main__":
    main()
