"""Synthetic eval-set generator for nil-deref and bounds-check risks.

Starts a local llama-server, prompts it for buggy Go functions with explicit
behavior, extracts the function body, rebuilds a guaranteed-panic harness,
tokenizes, and labels the panic site.

Usage:
    python -m astrolabe.synth_eval \
        --out-dir synth_eval \
        --count-nil 100 \
        --count-bounds 100 \
        --llama-model unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HELPER = REPO_ROOT / "ast-tokenize"

LLAMA_SERVER_BIN = shutil.which("llama-server") or "/home/austin/Repositories/Personal/llama.cpp/build/bin/llama-server"
LLAMA_PORT = 8080
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"

PROGRAM_PROMPT = """Write a complete Go program in package main.

The program must contain:
1. Any type definitions needed for the function below.
2. A function named {func_name} with this exact signature:
   {signature}
   It should {behavior}
   Important: do NOT add nil checks, bounds checks, length checks, or recover calls inside {func_name}.
   The function body should be a single return statement.
3. A main function that calls {func_name} with arguments guaranteed to cause a {kind} panic.

Only output the Go code. No explanation, no markdown fences.
"""

NIL_TEMPLATES = [
    ("func {func_name}(p *Record) int", "return p.Value", ["type Record struct { Value int }"], "{func_name}(nil)"),
    ("func {func_name}(p *Record) string", "return p.Name", ["type Record struct { Name string }"], "{func_name}(nil)"),
    ("func {func_name}(p *Node) int", "return p.Next.Value", ["type Node struct { Value int; Next *Node }"], "{func_name}(nil)"),
    ("func {func_name}(a, b *Pair) int", "return a.Left + b.Right", ["type Pair struct { Left, Right int }"], "{func_name}(&Pair{{1, 2}}, nil)"),
    ("func {func_name}(p *Point) float64", "return p.X * p.Y + p.Z", ["type Point struct { X, Y, Z float64 }"], "{func_name}(nil)"),
    ("func {func_name}(cfg *Config) string", "return cfg.Settings.Timeout", ["type Config struct { Settings struct { Timeout string } }"], "{func_name}(nil)"),
    ("func {func_name}(user *User) string", "return user.Profile.Email", ["type User struct { Profile struct { Email string } }"], "{func_name}(nil)"),
    ("func {func_name}(head *ListNode) int", "return head.Next.Value", ["type ListNode struct { Value int; Next *ListNode }"], "{func_name}(nil)"),
]

BOUNDS_TEMPLATES = [
    ("func {func_name}(arr []int, idx int) int", "return arr[idx]", [], "{func_name}([]int{{1, 2}}, 10)"),
    ("func {func_name}(data string, idx int) byte", "return data[idx]", [], "{func_name}(\"hi\", 10)"),
    ("func {func_name}(tokens []Token, pos int) string", "return tokens[pos].Literal", ["type Token struct { Literal string }"], "{func_name}(nil, 0)"),
    ("func {func_name}(records []Record, idx int) string", "return records[idx].Name", ["type Record struct { Name string }"], "{func_name}(nil, 0)"),
    ("func {func_name}(s []byte, n int) byte", "return s[n]", [], "{func_name}([]byte{{1, 2}}, 10)"),
    ("func {func_name}(arr [5]int, idx int) int", "return arr[idx]", [], "{func_name}([5]int{{1,2,3,4,5}}, 10)"),
    ("func {func_name}(matrix [][]int, row, col int) int", "return matrix[row][col]", [], "{func_name}([][]int{{{{1,2}}}}, 10, 0)"),
]

RACE_TEMPLATES = [
    ("func {func_name}(m map[string]int) int", "m[\"x\"]++; return 0", [], """m := map[string]int{{}}
var wg sync.WaitGroup
wg.Add(1)
go func() {{ defer wg.Done(); {func_name}(m) }}()
{func_name}(m)
wg.Wait()"""),
    ("func {func_name}(s *[]int) int", "*s = append(*s, 1); return 0", [], """s := make([]int, 0)
var wg sync.WaitGroup
wg.Add(1)
go func() {{ defer wg.Done(); {func_name}(&s) }}()
{func_name}(&s)
wg.Wait()"""),
    ("func {func_name}(n *int) int", "*n++; return 0", [], """n := 0
var wg sync.WaitGroup
wg.Add(1)
go func() {{ defer wg.Done(); {func_name}(&n) }}()
{func_name}(&n)
wg.Wait()"""),
]


# ---------------------------------------------------------------------------
# llama-server management
# ---------------------------------------------------------------------------

class LlamaServer:
    def __init__(self, model: str, port: int = LLAMA_PORT, n_gpu_layers: int = 99):
        self.model = model
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.is_running():
            print("llama-server already running; reusing it.")
            return
        cmd = [
            LLAMA_SERVER_BIN,
            "-hf", self.model,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "--host", "127.0.0.1",
        ]
        print(f"Starting llama-server: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        atexit.register(self.stop)
        self._wait_for_ready(timeout=180)
        print("llama-server ready.")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            print("Stopping llama-server...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _wait_for_ready(self, timeout: int = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return
            if self.proc is not None and self.proc.poll() is not None:
                out, err = self.proc.communicate()
                print("llama-server exited early:")
                print(out[-2000:] if out else "")
                print(err[-2000:] if err else "")
                raise RuntimeError("llama-server failed to start")
            time.sleep(1)
        raise RuntimeError("llama-server did not become ready in time")


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def chat_completion(messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048) -> str:
    payload = {
        "model": "local",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def extract_go_code(text: str) -> str | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if "package main" not in text or "func " not in text:
        return None
    return text


def extract_function_body(code: str, func_name: str) -> str | None:
    """Extract the function body from a Go program."""
    pattern = re.compile(
        rf"func\s+{re.escape(func_name)}\s*\([^)]*\)\s*\w*\s*{{",
        re.DOTALL,
    )
    m = pattern.search(code)
    if not m:
        return None
    start = m.end()
    depth = 1
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[start:i].strip()
    return None


def body_is_valid(body: str, kind: str) -> bool:
    """Reject bodies that are empty or contain forbidden safety checks."""
    body = body.strip()
    lowered = body.lower()
    if kind == "race":
        # Race bodies perform an unsynchronized write to a shared variable.
        if not ("++" in body or "append" in lowered or "[" in body):
            return False
        forbidden = {"sync.", "atomic", "make(chan", "mutex", "rwmutex"}
        for tok in forbidden:
            if tok in lowered:
                return False
        return True

    if not body.startswith("return"):
        return False
    forbidden = {"if", "for", "len", "recover", "==", "!=", "<", ">", "<=", ">="}
    for tok in forbidden:
        if tok in lowered:
            return False
    if kind == "nil" and "." not in body and "*" not in body:
        return False
    if kind == "bounds" and "[" not in body:
        return False
    return True


def build_program(
    signature: str,
    func_name: str,
    body: str,
    structs: list[str],
    main_call: str,
    imports: list[str] | None = None,
) -> str:
    structs_block = ("\n\n".join(structs) + "\n\n") if structs else ""
    import_block = ""
    if imports:
        import_block = "import (\n    " + "\n    ".join(f'"{imp}"' for imp in imports) + "\n)\n\n"
    return f"""package main

{import_block}{structs_block}{signature} {{
    {body}
}}

func main() {{
    {main_call}
}}
"""


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------

def run_program(go_file: Path, timeout: int = 10, race: bool = False) -> tuple[int, str, str]:
    cmd = ["go", "run"]
    if race:
        cmd.append("-race")
    # Run from the file's directory and refer to the file by name so relative
    # paths resolve correctly.
    proc = subprocess.run(
        cmd + [go_file.name],
        cwd=go_file.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def classify_panic(stderr: str) -> str | None:
    if "WARNING: DATA RACE" in stderr:
        return "race"
    if "invalid memory address or nil pointer dereference" in stderr:
        return "nil"
    if "index out of range" in stderr or "slice bounds out of range" in stderr:
        return "bounds"
    return None


def parse_panic_line(stderr: str) -> int | None:
    m = re.search(r"generated\.go:(\d+)", stderr)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Tokenization and alignment
# ---------------------------------------------------------------------------

def tokenize_generated(generated_path: Path, helper: Path) -> tuple[list[str], list[int], dict[int, str], dict[int, str]]:
    out = subprocess.check_output([str(helper), str(generated_path)], text=True)
    tokens: list[str] = []
    pos_map: list[int] = []
    name_pos_map: dict[int, str] = {}
    types_map: dict[int, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("POSMAP "):
            pos_map = json.loads(line[len("POSMAP "):])
            continue
        if line.startswith("NAMEPOSMAP "):
            raw = json.loads(line[len("NAMEPOSMAP "):])
            name_pos_map = {int(k): v for k, v in raw.items()}
            continue
        if line.startswith("ANN "):
            ann = json.loads(line[4:])
            if ann and "types" in ann:
                types_map = {int(k): v for k, v in ann["types"].items()}
            continue
        tokens.append(line)
    return tokens, pos_map, name_pos_map, types_map


def find_risk_token_position(tokens: list[str], pos_map: list[int], panic_line: int, kind: str) -> int | None:
    indices = [i for i, l in enumerate(pos_map) if l == panic_line]
    if not indices:
        return None
    if kind == "nil":
        for i in indices:
            if i + 1 < len(tokens) and tokens[i] in ("OPEN_STAR", "OPEN_SELECTOR"):
                nxt = tokens[i + 1]
                if nxt.startswith("NAME_") and nxt not in ("NAME_BLANK", "NAME_UNK"):
                    return i + 1
    elif kind == "bounds":
        for i in indices:
            if tokens[i] == "OPEN_INDEX":
                if i + 2 < len(tokens):
                    idx_tok = tokens[i + 2]
                    if idx_tok.startswith("NAME_") and idx_tok not in ("NAME_BLANK", "NAME_UNK"):
                        return i + 2
    else:  # race
        for i in indices:
            if tokens[i].startswith("NAME_") and tokens[i] not in ("NAME_BLANK", "NAME_UNK"):
                return i
    for i in indices:
        if tokens[i].startswith("NAME_") and tokens[i] not in ("NAME_BLANK", "NAME_UNK"):
            return i
    return None


# ---------------------------------------------------------------------------
# Generator loop
# ---------------------------------------------------------------------------

def generate_one(
    out_dir: Path,
    kind: str,
    idx: int,
    helper: Path,
    temperature: float = 0.7,
    max_retries: int = 5,
) -> dict[str, Any] | None:
    if kind == "nil":
        templates = NIL_TEMPLATES
    elif kind == "bounds":
        templates = BOUNDS_TEMPLATES
    else:
        templates = RACE_TEMPLATES
    signature_template, behavior, structs, main_call_template = random.choice(templates)
    func_name = f"Risk{kind.capitalize()}{idx:04d}"
    signature = signature_template.format(func_name=func_name)
    main_call = main_call_template.format(func_name=func_name)

    kind_desc = {
        "nil": "nil pointer dereference",
        "bounds": "index out of range",
        "race": "data race",
    }[kind]
    prompt = PROGRAM_PROMPT.format(
        func_name=func_name,
        signature=signature,
        behavior=behavior,
        kind=kind_desc,
    )

    work_dir = out_dir / f"{kind}_{idx:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            response = chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=temperature + attempt * 0.05,
                max_tokens=2048,
            )
        except Exception as e:
            print(f"  {func_name} LLM call failed: {e}")
            continue

        code = extract_go_code(response)
        if code is None:
            print(f"  {func_name} attempt {attempt+1}: no Go code extracted")
            (work_dir / f"attempt_{attempt}_raw.txt").write_text(response)
            continue

        body = extract_function_body(code, func_name)
        if body is None or not body_is_valid(body, kind):
            print(f"  {func_name} attempt {attempt+1}: body rejected")
            (work_dir / f"attempt_{attempt}_raw.go").write_text(code)
            continue

        # Rebuild with our own guaranteed-panic/race harness.
        imports = ["sync"] if kind == "race" else None
        code = build_program(signature, func_name, body, structs, main_call, imports=imports)
        generated_path = work_dir / "generated.go"
        generated_path.write_text(code)

        try:
            rc, stdout, stderr = run_program(generated_path, race=(kind == "race"))
        except subprocess.TimeoutExpired:
            print(f"  {func_name} attempt {attempt+1}: execution timed out")
            continue

        if rc == 0 and kind != "race":
            print(f"  {func_name} attempt {attempt+1}: no panic (safe)")
            continue
        if rc == 0 and kind == "race":
            print(f"  {func_name} attempt {attempt+1}: no race detected")
            continue

        panic_kind = classify_panic(stderr)
        if panic_kind != kind:
            print(f"  {func_name} attempt {attempt+1}: wrong panic kind ({panic_kind})")
            continue

        panic_line = parse_panic_line(stderr)
        if panic_line is None:
            print(f"  {func_name} attempt {attempt+1}: could not parse panic line")
            continue

        try:
            tokens, pos_map, name_pos_map, types_map = tokenize_generated(generated_path, helper)
        except Exception as e:
            print(f"  {func_name} tokenization failed: {e}")
            continue

        token_pos = find_risk_token_position(tokens, pos_map, panic_line, kind)
        if token_pos is None:
            print(f"  {func_name}: could not align panic line {panic_line} to token")
            continue

        token_name = tokens[token_pos]
        orig_name = name_pos_map.get(token_pos)
        type_cat = types_map.get(token_pos, "unknown")

        print(f"  {func_name} OK: {kind} at generated.go:{panic_line} token={token_name}")
        return {
            "id": f"{kind}_{idx:04d}",
            "kind": kind,
            "func_name": func_name,
            "source": str(generated_path),
            "panic_kind": panic_kind,
            "panic_line": panic_line,
            "token_pos": token_pos,
            "token_name": token_name,
            "orig_name": orig_name,
            "type_cat": type_cat,
            "signature": signature,
            "behavior": behavior,
        }

    print(f"  {func_name}: exhausted {max_retries} attempts")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("synth_eval"))
    p.add_argument("--count-nil", type=int, default=100)
    p.add_argument("--count-bounds", type=int, default=100)
    p.add_argument("--count-race", type=int, default=0)
    p.add_argument("--llama-model", default="unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M")
    p.add_argument("--llama-port", type=int, default=LLAMA_PORT)
    p.add_argument("--helper", type=Path, default=DEFAULT_HELPER)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-attempts", type=int, default=600)
    p.add_argument("--no-start-server", action="store_true")
    args = p.parse_args()

    if not args.helper.exists():
        sys.exit(f"ast-tokenize helper not found at {args.helper}; build it first")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    server = LlamaServer(args.llama_model, port=args.llama_port)
    if not args.no_start_server:
        server.start()
    else:
        if not server.is_running():
            sys.exit(f"llama-server not running on port {args.llama_port}")

    global LLAMA_URL
    LLAMA_URL = f"http://127.0.0.1:{args.llama_port}/v1/chat/completions"

    labels: list[dict[str, Any]] = []
    counters: dict[str, int] = {"nil": 0, "bounds": 0, "race": 0}
    target = {"nil": args.count_nil, "bounds": args.count_bounds, "race": args.count_race}
    total_attempts = 0

    kinds = ["nil"] * args.count_nil + ["bounds"] * args.count_bounds + ["race"] * args.count_race
    random.shuffle(kinds)

    idx = {"nil": 0, "bounds": 0, "race": 0}
    for kind in kinds:
        if counters[kind] >= target[kind]:
            continue
        if total_attempts >= args.max_attempts:
            print("Reached max attempts cap.")
            break
        total_attempts += 1

        label = generate_one(
            args.out_dir,
            kind,
            idx[kind],
            args.helper,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        idx[kind] += 1

        if label:
            labels.append(label)
            counters[kind] += 1

    labels_path = args.out_dir / "labels.jsonl"
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    print(f"\nGenerated {len(labels)} labels ({counters['nil']} nil, {counters['bounds']} bounds, {counters['race']} race)")
    print(f"Saved to {labels_path}")

    if not args.no_start_server:
        server.stop()


if __name__ == "__main__":
    main()
