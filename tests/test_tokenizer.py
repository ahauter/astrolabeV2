"""Smoke tests that lock the Go→token scheme.

These pin down the expected output for small, unambiguous Go snippets so any
change to the tokenizer that alters the stream has to be explicit.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "ast-tokenize"


def tokenize(src: str, tmp_path: Path) -> list[str]:
    if not HELPER.exists():
        pytest.skip(f"helper not built at {HELPER}")
    p = tmp_path / "snippet.go"
    p.write_text(src)
    out = subprocess.check_output([str(HELPER), str(p)], text=True)
    return [
        line for line in out.splitlines()
        if line and not line.startswith(("ANN ", "POSMAP ", "PKGS ", "NAMEPOSMAP "))
    ]


def test_brackets_balance(tmp_path):
    src = """package p
import "fmt"
func F(p *int, arr []int, i int) error {
    if p == nil {
        return fmt.Errorf("nil")
    }
    if i >= len(arr) { return nil }
    _ = arr[i]
    return nil
}
"""
    toks = tokenize(src, tmp_path)
    stack = []
    for t in toks:
        if t.startswith("OPEN_"):
            stack.append(t[5:])
        elif t.startswith("CLOSE_"):
            assert stack and stack[-1] == t[6:], f"mismatch on {t}, stack={stack}"
            stack.pop()
    assert stack == [], f"unclosed: {stack}"


def test_selector_and_index_emission(tmp_path):
    src = """package p
func F(p *int, arr []int, i int) {
    _ = arr[i]
}
"""
    toks = tokenize(src, tmp_path)
    idx_open = toks.index("OPEN_INDEX")
    # Walk the INDEX subtree and assert its shape: NAME refs to arr and i.
    slab: list[str] = []
    depth = 0
    for t in toks[idx_open:]:
        if t == "OPEN_INDEX":
            depth += 1
        elif t == "CLOSE_INDEX":
            depth -= 1
            slab.append(t)
            if depth == 0:
                break
        elif depth > 0:
            slab.append(t)
    name_refs = [t for t in slab if t.startswith("NAME_")]
    assert len(name_refs) == 2, f"expected 2 NAME refs inside INDEX, got {slab}"
    # Both refs must resolve to slots (neither should be UNK or BLANK).
    assert all(r.split("_")[1].isdigit() for r in name_refs), name_refs


def test_builtins_and_predeclared(tmp_path):
    src = """package p
func F(arr []int) int {
    if len(arr) == 0 { return 0 }
    return 1
}
"""
    toks = tokenize(src, tmp_path)
    assert "BI_LEN" in toks
    assert "T_INT" in toks


def test_vocab_covers_all_emitted(tmp_path):
    from astrolabe.vocab import TOKEN_TO_ID
    src = """package p
import (
    "fmt"
    "os"
)

type Point struct { X, Y int }

var globalCount int = 0
const name = "bob"

func (p *Point) Method() error {
    defer fmt.Println("bye")
    go func() { _ = p }()
    ch := make(chan int, 2)
    ch <- 1
    select {
    case v := <-ch:
        _ = v
    default:
    }
    switch x := globalCount; x {
    case 0:
        return nil
    default:
        return fmt.Errorf("nope")
    }
}

func F[T any](xs []T) (out []T) {
    for i, x := range xs {
        if i > 0 { out = append(out, x) }
    }
    m := map[string]int{name: 1}
    _ = m
    arr := [3]int{1,2,3}
    _ = arr[:2]
    return
}
"""
    toks = tokenize(src, tmp_path)
    missing = [t for t in toks if t not in TOKEN_TO_ID]
    assert not missing, f"tokens emitted but not in vocab: {set(missing)}"


def test_type_tokens_emitted(tmp_path):
    src = """package p
func F(p *int, arr []int, i int, m map[string]int, s string, ch chan int) {
    _ = *p
    _ = arr[i]
    _ = m["k"]
    _ = s[i]
    <-ch
}
"""
    toks = tokenize(src, tmp_path)
    # Each parameter should be followed by a TYPE_* token.
    assert "TYPE_PTR" in toks
    assert "TYPE_SLICE" in toks
    assert "TYPE_BASIC" in toks
    assert "TYPE_MAP" in toks
    assert "TYPE_STRING" in toks
    assert "TYPE_CHAN" in toks
    # Spot-check relative ordering: parameter name followed by its type token.
    ptr_idx = toks.index("TYPE_PTR")
    assert toks[ptr_idx - 1].startswith("NAME_")
    slice_idx = toks.index("TYPE_SLICE")
    assert toks[slice_idx - 1].startswith("NAME_")
