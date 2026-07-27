"""Risk label miner: static analysis over CFG annotations to flag unguarded
nil dereferences and out-of-bounds accesses.

Reads the paired unit token stream + CFG annotations and writes one JSON line
per unit listing token positions that are at risk.

Usage:
    python -m astrolabe.risk_miner \
        --units data/train_units.bin \
        --idx   data/train_units.idx.npy \
        --ann   data/train_ann.jsonl \
        --out   data/risk_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astrolabe.vocab import ID_TO_TOKEN, TOKEN_TO_ID


def load_units(units_bin: Path, idx_path: Path) -> tuple[np.memmap, np.ndarray]:
    data = np.memmap(Path(units_bin), dtype=np.uint16, mode="r")
    offsets = np.load(Path(idx_path))
    return data, offsets


def get_token_names(data: np.memmap, start: int, end: int) -> list[str]:
    return [ID_TO_TOKEN[t] for t in data[start:end].tolist()]


def _is_external_use(pos: int, use_set: set[int], du: dict[int, int], local_defs: set[int]) -> bool:
    """Return True if the token at position `pos` is a use of a name
    that is defined *outside* the current function (package import, global,
    or any name without a local def-use chain).
    """
    if pos not in use_set:
        return False  # not a tracked use → keep as candidate

    if pos in du:
        # Has a DU chain.  Skip if the def is outside the function.
        return du[pos] not in local_defs
    else:
        # Use with no DU chain → defined outside (package import, global, etc.)
        return True


def _extract_if_conditions(tokens: list[str]) -> list[list[str]]:
    """Return a list of condition token lists for every OPEN_IF block."""
    conditions: list[list[str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == "OPEN_IF":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if tokens[j] == "OPEN_IF":
                    depth += 1
                elif tokens[j] == "CLOSE_IF":
                    depth -= 1
                j += 1
            conditions.append(tokens[i + 1 : j - 1])
            i = j
        else:
            i += 1
    return conditions


def _collect_nil_guards(conditions: list[list[str]]) -> set[str]:
    """Build a set of NAME_N slots that are compared to BI_NIL in any condition."""
    guarded: set[str] = set()
    for cond in conditions:
        for k, tok in enumerate(cond):
            if not tok.startswith("NAME_") or tok in ("NAME_BLANK", "NAME_UNK"):
                continue
            # name OP_EQL|OP_NEQ BI_NIL
            if k + 2 < len(cond) and cond[k + 1] in ("OP_EQL", "OP_NEQ") and cond[k + 2] == "BI_NIL":
                guarded.add(tok)
            # BI_NIL OP_EQL|OP_NEQ name
            if k >= 2 and cond[k - 2] == "BI_NIL" and cond[k - 1] in ("OP_EQL", "OP_NEQ"):
                guarded.add(tok)
    return guarded


def _collect_bounds_guards(conditions: list[list[str]]) -> set[tuple[str, str]]:
    """Build a set of (idx_slot, arr_slot) pairs guarded by a bounds check."""
    guarded: set[tuple[str, str]] = set()
    for cond in conditions:
        # Find all len(arr_slot) expressions
        len_pairs: list[tuple[int, str]] = []  # (position in cond, arr_slot)
        for k, tok in enumerate(cond):
            if tok == "OPEN_CALL" and k + 3 < len(cond):
                if cond[k + 1] == "BI_LEN" and cond[k + 2].startswith("NAME_"):
                    len_pairs.append((k, cond[k + 2]))
        if not len_pairs:
            continue
        # Find idx_slot with comparison op anywhere in the condition
        for k, tok in enumerate(cond):
            if not tok.startswith("NAME_") or tok in ("NAME_BLANK", "NAME_UNK"):
                continue
            op = None
            if k + 1 < len(cond) and cond[k + 1] in ("OP_GEQ", "OP_LSS", "OP_LEQ", "OP_GTR", "OP_EQL", "OP_NEQ"):
                op = cond[k + 1]
            elif k >= 1 and cond[k - 1] in ("OP_GEQ", "OP_LSS", "OP_LEQ", "OP_GTR", "OP_EQL", "OP_NEQ"):
                op = cond[k - 1]
            if op is not None:
                # Any arr_slot that has a len() check in this condition is considered guarded
                # for this idx_slot (approximation used by POC).
                for _, arr_slot in len_pairs:
                    guarded.add((tok, arr_slot))
    return guarded


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


def find_nil_risks(tokens: list[str], ann: dict | None) -> list[int]:
    """Find token positions where a pointer/slice is dereferenced without
    a dominating nil check.

    Patterns:
        OPEN_STAR   NAME_N          ← risky
        OPEN_SELECTOR NAME_N        ← risky

    Only NAME_N slots whose coarse type is ptr or interface are considered risky.
    """
    risks: list[int] = []
    if ann is None:
        return risks

    use_set = set(ann.get("use", []))
    du = {int(k): v for k, v in ann.get("du", {}).items()}
    local_defs = set(ann.get("def", []))
    guarded = _collect_nil_guards(_extract_if_conditions(tokens))
    types_map = ann.get("types", {})
    risky_types = {"ptr", "interface"}

    for i, tok in enumerate(tokens):
        # Pattern A: OPEN_STAR followed by NAME_N
        if tok == "OPEN_STAR" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.startswith("NAME_") and nxt not in ("NAME_BLANK", "NAME_UNK"):
                if _is_external_use(i + 1, use_set, du, local_defs):
                    continue
                if _slot_type(nxt, types_map) not in risky_types:
                    continue
                if nxt not in guarded:
                    risks.append(i + 1)

        # Pattern B: OPEN_SELECTOR followed by NAME_N
        if tok == "OPEN_SELECTOR" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.startswith("NAME_") and nxt not in ("NAME_BLANK", "NAME_UNK"):
                if _is_external_use(i + 1, use_set, du, local_defs):
                    continue
                if _slot_type(nxt, types_map) not in risky_types:
                    continue
                if nxt not in guarded:
                    risks.append(i + 1)

    return risks


def find_bounds_risks(tokens: list[str], ann: dict | None) -> list[int]:
    """Find token positions where an array/slice/string is indexed without a
    dominating bounds check.

    Pattern:
        OPEN_INDEX
          NAME_A          ← array
          NAME_B          ← index variable ← we label THIS

    Only NAME_A slots whose coarse type is slice, array, or string are considered risky.
    """
    risks: list[int] = []
    if ann is None:
        return risks

    use_set = set(ann.get("use", []))
    du = {int(k): v for k, v in ann.get("du", {}).items()}
    local_defs = set(ann.get("def", []))
    guarded = _collect_bounds_guards(_extract_if_conditions(tokens))
    types_map = ann.get("types", {})
    risky_types = {"slice", "array", "string"}

    for i, tok in enumerate(tokens):
        if tok != "OPEN_INDEX":
            continue
        if i + 2 >= len(tokens):
            continue
        arr_tok = tokens[i + 1]
        idx_tok = tokens[i + 2]
        if not (arr_tok.startswith("NAME_") and idx_tok.startswith("NAME_")):
            continue
        if _is_external_use(i + 1, use_set, du, local_defs):
            continue
        if _slot_type(arr_tok, types_map) not in risky_types:
            continue
        if (idx_tok, arr_tok) not in guarded:
            risks.append(i + 2)

    return risks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=Path, required=True)
    p.add_argument("--idx", type=Path, required=True)
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-units", type=int, default=None,
                   help="Process only first N units (for quick POC iteration)")
    args = p.parse_args()

    data, offsets = load_units(args.units, args.idx)
    n_units = len(offsets) - 1
    if args.max_units is not None:
        n_units = min(n_units, args.max_units)

    nil_count = 0
    bounds_count = 0
    total_nil = 0
    total_bounds = 0

    with open(args.ann) as fin, open(args.out, "w") as fout:
        for unit_idx in range(n_units):
            line = fin.readline()
            ann = json.loads(line) if line.strip() and line.strip() != "null" else None

            start = int(offsets[unit_idx])
            end = int(offsets[unit_idx + 1])
            tokens = get_token_names(data, start, end)

            nil_risks = find_nil_risks(tokens, ann)
            bounds_risks = find_bounds_risks(tokens, ann)

            total_nil += len(nil_risks)
            total_bounds += len(bounds_risks)
            nil_count += 1 if nil_risks else 0
            bounds_count += 1 if bounds_risks else 0

            record = {
                "nil_risks": nil_risks,
                "bounds_risks": bounds_risks,
            }
            fout.write(json.dumps(record) + "\n")

    print(f"Wrote {args.out}")
    print(f"  Units with nil risks:      {nil_count:,} / {n_units:,}")
    print(f"  Total nil risk positions:  {total_nil:,}")
    print(f"  Units with bounds risks:   {bounds_count:,} / {n_units:,}")
    print(f"  Total bounds risk pos:     {total_bounds:,}")


if __name__ == "__main__":
    main()
