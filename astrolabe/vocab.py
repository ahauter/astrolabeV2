"""Token vocabulary for the Go AST-token language.

This file is the single source of truth. It must stay in sync with
`cmd/ast-tokenize/main.go`: every token the Go emitter can produce must appear
here, and the NAME/FIELD slot counts must match.
"""
from __future__ import annotations


NAME_SLOTS = 64
FIELD_SLOTS = 64


# Node open/close pairs. Each entry adds two tokens: OPEN_X and CLOSE_X.
_NODE_PAIRS = [
    "FUNC_DECL", "RECV",
    "FUNC_TYPE", "TYPE_PARAMS", "PARAMS", "RESULTS",
    "GEN_DECL", "VAR_DECL", "CONST_DECL", "TYPE_DECL",
    "VALUE_SPEC", "TYPE_SPEC",
    "STRUCT_TYPE", "INTERFACE_TYPE",
    "ARRAY_TYPE", "MAP_TYPE", "CHAN_TYPE", "ELLIPSIS_TYPE",
    "FIELD",
    "BLOCK", "EXPR_STMT",
    "ASSIGN", "INCDEC", "RETURN", "BRANCH",
    "IF", "ELSE", "FOR", "RANGE",
    "SWITCH", "TYPE_SWITCH", "CASE",
    "SELECT", "COMM_CLAUSE",
    "GO", "DEFER", "SEND", "LABELED",
    "BINOP", "UNARY", "STAR",
    "SELECTOR", "INDEX", "INDEX_LIST", "SLICE", "TYPE_ASSERT",
    "CALL", "KV", "COMPOSITE_LIT", "FUNC_LIT",
]

# Standalone structural / leaf tokens that are not open/close paired.
_ATOMS = [
    # control
    "BOS", "EOF", "PAD",
    # literal kinds (content-free)
    "INT_LIT", "FLOAT_LIT", "IMAG_LIT", "CHAR_LIT", "STRING_LIT", "LIT_UNK",
    # statement markers
    "EMPTY_STMT", "ELLIPSIS",
    # channel direction (inside OPEN_CHAN_TYPE)
    "CHAN_SEND", "CHAN_RECV", "CHAN_BI",
    # branch kinds (inside OPEN_BRANCH)
    "BR_BREAK", "BR_CONTINUE", "BR_GOTO", "BR_FALLTHROUGH", "BR_UNK",
    # name slot specials
    "NAME_BLANK", "NAME_UNK", "NAME_OVF",
    "FIELD_OVF",
    # operators
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_QUO", "OP_REM",
    "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR", "OP_ANDNOT",
    "OP_LAND", "OP_LOR", "OP_ARROW", "OP_INC", "OP_DEC",
    "OP_EQL", "OP_NEQ", "OP_LSS", "OP_GTR", "OP_LEQ", "OP_GEQ", "OP_NOT",
    "OP_ASSIGN", "OP_DEFINE",
    "OP_ADD_ASSIGN", "OP_SUB_ASSIGN", "OP_MUL_ASSIGN",
    "OP_QUO_ASSIGN", "OP_REM_ASSIGN",
    "OP_AND_ASSIGN", "OP_OR_ASSIGN", "OP_XOR_ASSIGN",
    "OP_SHL_ASSIGN", "OP_SHR_ASSIGN", "OP_ANDNOT_ASSIGN",
    "OP_UNK",
    # builtins
    "BI_LEN", "BI_CAP", "BI_PANIC", "BI_RECOVER", "BI_MAKE", "BI_NEW",
    "BI_APPEND", "BI_COPY", "BI_DELETE", "BI_CLOSE",
    "BI_COMPLEX", "BI_REAL", "BI_IMAG", "BI_PRINT", "BI_PRINTLN",
    "BI_NIL", "BI_TRUE", "BI_FALSE", "BI_IOTA",
    # predeclared types
    "T_BOOL", "T_BYTE", "T_RUNE", "T_STRING",
    "T_INT", "T_INT8", "T_INT16", "T_INT32", "T_INT64",
    "T_UINT", "T_UINT8", "T_UINT16", "T_UINT32", "T_UINT64", "T_UINTPTR",
    "T_FLOAT32", "T_FLOAT64", "T_COMPLEX64", "T_COMPLEX128",
    "T_ERROR", "T_ANY", "T_COMPARABLE",
]


def _build_vocab() -> tuple[list[str], dict[str, int]]:
    tokens: list[str] = []
    tokens += _ATOMS
    for kind in _NODE_PAIRS:
        tokens.append(f"OPEN_{kind}")
        tokens.append(f"CLOSE_{kind}")
    for i in range(NAME_SLOTS):
        tokens.append(f"NAME_{i}")
    for i in range(FIELD_SLOTS):
        tokens.append(f"FIELD_{i}")
    if len(set(tokens)) != len(tokens):
        dupes = [t for t in tokens if tokens.count(t) > 1]
        raise RuntimeError(f"duplicate vocab entries: {set(dupes)}")
    return tokens, {tok: i for i, tok in enumerate(tokens)}


ID_TO_TOKEN, TOKEN_TO_ID = _build_vocab()
VOCAB_SIZE = len(ID_TO_TOKEN)

PAD_ID = TOKEN_TO_ID["PAD"]
BOS_ID = TOKEN_TO_ID["BOS"]
EOF_ID = TOKEN_TO_ID["EOF"]

# Open/close ID pairs for the bracket-balance validity check.
OPEN_CLOSE_PAIRS: dict[int, int] = {
    TOKEN_TO_ID[f"OPEN_{k}"]: TOKEN_TO_ID[f"CLOSE_{k}"] for k in _NODE_PAIRS
}
OPEN_IDS = frozenset(OPEN_CLOSE_PAIRS.keys())
CLOSE_IDS = frozenset(OPEN_CLOSE_PAIRS.values())


def encode(token: str) -> int:
    """Encode a single token string. Raises KeyError on unknown tokens so the
    caller can report a vocab-emitter drift bug instead of silently mapping
    to a pad/unknown slot."""
    return TOKEN_TO_ID[token]


def bracket_balance_rate(ids: list[int]) -> float:
    """Fraction of open tokens whose matching close appears correctly nested.
    Used as a validity metric during pretraining eval."""
    stack: list[int] = []
    ok = 0
    total = 0
    for tok in ids:
        if tok in OPEN_IDS:
            stack.append(OPEN_CLOSE_PAIRS[tok])
            total += 1
        elif tok in CLOSE_IDS:
            if stack and stack[-1] == tok:
                ok += 1
                stack.pop()
            else:
                # mismatched close — count it against the nearest open, if any
                if stack:
                    stack.pop()
    return ok / total if total else 1.0


if __name__ == "__main__":
    print(f"vocab size: {VOCAB_SIZE}")
    print(f"open/close pairs: {len(OPEN_CLOSE_PAIRS)}")
    print(f"name slots: {NAME_SLOTS}, field slots: {FIELD_SLOTS}")
