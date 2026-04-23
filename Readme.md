# gopanic-zero

A tiny model that rewrites Go code to propagate nil pointer and array out-of-bounds errors up the call stack instead of panicking.

-----

## The Problem

Go programs panic at runtime when code dereferences a nil pointer or accesses an array out of bounds. These bugs compile silently, appear only at runtime, and often surface in production.

```go
// this compiles fine, panics at runtime if ptr is nil
value := ptr.Field

// this compiles fine, panics at runtime if i >= len(arr)
value := arr[i]
```

The fix is always the same — check before accessing, return an error if the check fails. The problem is that developers forget, skip it under time pressure, or inherit code that never had the checks in the first place.

-----

## The Approach

A small model trained to detect unguarded pointer dereferences and array accesses and rewrite them with proper Go error propagation:

```go
// nil pointer — before
value := ptr.Field

// nil pointer — after
if ptr == nil {
    return fmt.Errorf("ptr is nil")
}
value := ptr.Field
```

```go
// array bounds — before
value := arr[i]

// array bounds — after
if i >= len(arr) {
    return fmt.Errorf("index %d out of bounds (len %d)", i, len(arr))
}
value := arr[i]
```

The model doesn't just insert a local check — it propagates the error return up through every caller in the call chain until it reaches the top level function. Partial fixes that swallow errors silently are treated as failures.

-----

## AST Token Representation

The model does not consume raw Go source. Instead, each file is parsed with `go/parser` and emitted as a stream of **structural tokens** with open/close pairs for every AST node. This is the core architectural choice.

```go
ptr.Field                    // Go
```
```
OPEN_SELECTOR NAME_0 FIELD_1 CLOSE_SELECTOR
```

```go
arr[i]                       // Go
```
```
OPEN_INDEX NAME_2 NAME_3 CLOSE_INDEX
```

Identifiers are replaced with scoped indices rather than raw strings. The first distinct name introduced in the active scope chain is `NAME_0`, the next is `NAME_1`, and so on; references to a bound name emit that same index. Scope pop discards the innermost frame. Field/method names on the right side of a selector use a parallel `FIELD_N` index space assigned per file. Predeclared identifiers (`nil`, `len`, `panic`, `error`, `int`, …) have their own dedicated tokens and never occupy a NAME slot.

String and comment contents are dropped — pretraining is about **structure**, not lexical content.

The resulting vocabulary is **329 tokens** (vs. 32k+ for standard BPE). That shrinks the embedding table, forces the model to learn bracket balancing and scoping from the token stream itself, and makes it cheap to check generation validity: any sample with mismatched open/close tokens is structurally invalid by definition.

The Go→token converter lives in `cmd/ast-tokenize/`. Python code reads the stream through `astrolabe/vocab.py`, which is the single source of truth for the token table.

-----

## Architecture

The model is an **encoder-decoder transformer**:

- **Encoder** — bidirectional attention over the full AST token stream. Builds a rich structural representation of the code without causal masking, so every token attends to the full context around it.
- **Decoder** — autoregressive generation over the same AST token vocabulary, conditioned on the encoder's output. At inference the decoder emits the input stream back with `ASSERT_*` tokens inserted inline at the positions it identifies as unguarded.

Four assertion intent tokens are added to the vocabulary:

```
ASSERT_NIL       NAME_X               # NAME_X may be nil before dereference
ASSERT_BOUNDS    NAME_X  NAME_Y       # NAME_X may be >= len(NAME_Y)
ASSERT_NONZERO   NAME_X               # NAME_X may be zero (divisor, modulo)
ASSERT_INVARIANT NAME_X  ...          # general precondition on NAME_X
```

These tokens carry only structural intent. The actual variable names are recovered post-inference by looking up the `NAME_N` index in the AST scope stack — the model never needs to know that `NAME_3` is `userID`. The assertion is then rendered into real Go syntax by a lightweight template pass.

-----

## Pretraining

Pretraining has two stages:

**Stage 1 — Causal LM** (decoder-only, existing pipeline): next-token prediction over packed AST token streams. Gives the model a syntactic and scoping prior — bracket balancing, scope nesting, typical Go structural patterns — before any task-specific objective.

**Stage 2 — Masked span prediction** (encoder-decoder): nil-check and bounds-check spans in real code are identified and replaced with their corresponding `ASSERT_*` token in the target stream. The model learns to predict `ASSERT_NIL NAME_X` from the surrounding structural context, directly pretraining the assertion insertion capability on real Go code written by real engineers.

```
input:   ... OPEN_IF OPEN_BINOP NAME_3 OP_EQL BI_NIL CLOSE_BINOP ... OPEN_STAR NAME_3 ...
         (nil check span masked out)
target:  ... [ASSERT_NIL NAME_3] OPEN_STAR NAME_3 ...
```

Pipeline:

```
scrape.py            → scraped_code/<repo>/**/*.go      # raw Go files
ast-tokenize         → one token per line on stdout      # structural stream
astrolabe.prepare    → data/train.bin, data/val.bin     # packed uint16 IDs
astrolabe.train      → checkpoints/ckpt_<step>.pt       # stage 1: causal LM
astrolabe.pretrain2  → checkpoints/ckpt_enc_<step>.pt   # stage 2: masked spans
```

Eval during pretraining logs validation loss plus **bracket-balance rate** on sampled generations — the fraction of open tokens whose matching close appears correctly nested. That's a cheap, meaningful validity metric unique to this tokenization: random generations score ~0%, and a well-trained model should climb into the 90s.

Build and run:

```bash
go build ./cmd/ast-tokenize
export GITHUB_TOKEN=ghp_...          # strongly recommended
python scrape.py --max-pages 33 --output-dir scraped_code
python -m astrolabe.prepare --src scraped_code --dst data
python -m astrolabe.train  --data-dir data --out-dir checkpoints
```

-----

## Fine-Tuning Data

Fine-tuning data comes from two sources:

**Hand-labelled corpus samples** — a subset of the pretraining corpus is manually annotated with assertion sites and known unsafe patterns. Small volume, high signal.

**Synthetic generation** — a weak language model is deliberately prompted to write buggy code:

```
Write a Go function implementing {algorithm}.
Use arrays and pointers. Do not add nil checks or bounds checks.
```

Cheap models — small local models or low-tier API models — produce naturally buggy code at low cost. No human annotation required.

Each generated function is wrapped in a test harness:

```go
func main() {
    err := generatedFunction(fuzzedInputs)
    if err != nil {
        // reward: error correctly propagated to top level
        os.Exit(0)
    }
    // silent exit means error was swallowed somewhere — bad
    os.Exit(1)
}
```

The harness is run with randomized inputs using Go's built-in fuzzer to trigger latent panics. A program that panics before reaching main is a training example where the rewriter needs to insert a check. A program that exits cleanly via the error path is a successful rewrite.

-----

## Reward Signal

The Go runtime itself is the loss function. No human labels, no rule-based ground truth, no large model annotation.

```
panic before main    →  rewriter missed a dereference
clean exit via error →  rewriter correctly propagated the error (reward)
silent clean exit    →  rewriter swallowed the error (penalise)
```

This forces the model to learn complete error propagation, not just local nil checks. A rewrite only scores the reward if the error travels all the way up the call stack intact.

-----

## Usage

```bash
gopanic-zero ./mypackage/...
```

Outputs a diff. Review and commit.

Designed to run as a pre-commit hook or CI step. Runs on CPU in milliseconds. No GPU, no API, no internet connection required.
