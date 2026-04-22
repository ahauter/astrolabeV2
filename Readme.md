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

The model doesn’t just insert a local check — it propagates the error return up through every caller in the call chain until it reaches the top level function. Partial fixes that swallow errors silently are treated as failures.

-----

## Training Data

Training data is generated automatically using a weak language model deliberately prompted to write buggy code:

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

The harness is run with randomized inputs using Go’s built-in fuzzer to trigger latent panics. A program that panics before reaching main is a training example where the rewriter needs to insert a check. A program that exits cleanly via the error path is a successful rewrite.

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

## Rule-Based Validation

A conventional AST-based rule checker runs in parallel as a sanity filter. Any rewrite the model produces that the rule checker considers unnecessary is flagged as a suspected hallucination and withheld from output.

The rule checker is conservative by design — it only catches the obvious cases. The model catches the cases the rules miss. The rules prevent the model from inventing checks where none are needed.

-----

## Usage

```bash
gopanic-zero ./mypackage/...
```

Outputs a diff. Review and commit.

Designed to run as a pre-commit hook or CI step. Runs on CPU in milliseconds. No GPU, no API, no internet connection required.
