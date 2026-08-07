# Agent Notes — Astrolabe Race-Condition POC

This workspace is a proof-of-concept for detecting Go data races with a small
hierarchical Transformer. It sits on top of the existing nil-deref / bounds-check
risk model.

## Repository Layout

- `cmd/ast-tokenize/` — Go tokenizer that emits token streams, POSMAP, PKGS, NAMEPOSMAP, and per-function `ANN` metadata (sync events, def/use, types). Now also flags `sync/atomic.*` calls as sync events.
- `cmd/ast-callgraph/` — AST-only static call-graph helper; does **not** need `go.mod`.
- `astrolabe/model.py` — `GPT` backbone plus `HierarchicalRiskGPT` (L1 encoder + L2 context aggregator + race head).
- `astrolabe/dataset.py` — `CFGUnitMixDataset` for pretraining over declaration units.
- `astrolabe/risk_miner.py` — static miner for nil-deref and bounds-check risk positions.
- `astrolabe/risk_dataset.py` — `RiskUnitDataset` for nil/bounds fine-tuning.
- `astrolabe/race_dataset.py` — `RaceContextDataset`: target function + up to 8 depth-1 callers, with **online** sync removal. Uses line-offset sidecars (`.idx.npy`) to avoid loading annotation/meta/risk JSONL files into memory.
- `astrolabe/prepare_v2.py` — parallel tokenizer that builds the unified base corpus.
- `astrolabe/build_race_meta.py` — builds caller-index metadata for the race dataloader.
- `astrolabe/prepare_race.py` — legacy offline race-corpus builder (kept for reference; the active pipeline uses online mutation).
- `astrolabe/finetune_race.py` — multi-head fine-tuning script (race + nil + bounds).
- `astrolabe/detect.py` — unified inference CLI for nil, bounds, and race risks.
- `astrolabe/detect_race_project.py` — project-wide race scan with caller context.
- `astrolabe/eval_race_val.py` — evaluates a race checkpoint on the reserved validation corpus.
- `data_v1/` — unified base corpus: train/val units, annotations, callgraph, nil/bounds labels, and race caller metadata.
- `checkpoints_race/` — saved `HierarchicalRiskGPT` checkpoints.

## Build

```bash
go build ./cmd/ast-tokenize
go build ./cmd/ast-callgraph
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # torch, numpy
```

## Vocabulary Alignment

The pretrained risk checkpoints expect a **329-token** vocabulary. `astrolabe/vocab.py`
must match the tokenizer output. If you change the tokenizer, regenerate the corpus
and re-align `VOCAB_SIZE` / `TOKEN_TO_ID` before training.

## Unified Corpus (`data_v1/`)

The full scraped Go corpus is tokenized once into a shared base dataset that feeds
pretraining, nil/bounds fine-tuning, and race fine-tuning.

### 1. Tokenize

```bash
python -m astrolabe.prepare_v2 \
    --src scraped_code_remote/scraped_code \
    --dst data_v1 \
    --workers 16 \
    --batch 64 \
    --val-frac 0.05
```

This writes:

- `data_v1/train_units.bin`, `data_v1/val_units.bin`
- `data_v1/train_units.idx.npy`, `data_v1/val_units.idx.npy`
- `data_v1/train_ann.jsonl`, `data_v1/val_ann.jsonl`

### 2. Callgraph

```bash
./ast-callgraph scraped_code_remote/scraped_code > data_v1/callgraph.jsonl
```

### 3. Nil / Bounds Labels

```bash
python -m astrolabe.risk_miner \
    --units data_v1/train_units.bin \
    --idx   data_v1/train_units.idx.npy \
    --ann   data_v1/train_ann.jsonl \
    --out   data_v1/risk_train.jsonl

python -m astrolabe.risk_miner \
    --units data_v1/val_units.bin \
    --idx   data_v1/val_units.idx.npy \
    --ann   data_v1/val_ann.jsonl \
    --out   data_v1/risk_val.jsonl
```

### 4. Race Caller Metadata

```bash
python -m astrolabe.build_race_meta --data-dir data_v1
```

This writes `data_v1/race_train_meta.jsonl` and `data_v1/race_val_meta.jsonl`.
To keep memory low it builds a one-time SQLite lookup table at
`data_v1/callgraph.db` and streams the annotation files twice.

## Race Mutation Strategy

The race dataloader (`RaceContextDataset`) mutates sequences **online**:

- Every target function is loaded with its sync calls intact (negative).
- With probability 0.5, all sync calls are stripped and the previously-protected
  risky shared accesses are labeled as race positives.
- Non-concurrency functions (no sync events) are always negatives.
- Callers are kept original so the L2 context still sees goroutine spawns.

This avoids duplicating the entire corpus on disk and explicitly teaches the model
that locks, mutexes, and atomics make accesses safe.

## Training

Resume from the pretrained risk checkpoint:

```bash
python -m astrolabe.finetune_race \
    --data-dir data_v1 \
    --out-dir checkpoints_race \
    --resume checkpoints_risk/ckpt_risk_5000.pt
```

Resume an in-progress race checkpoint:

```bash
python -m astrolabe.finetune_race \
    --data-dir data_v1 \
    --out-dir checkpoints_race \
    --resume-ckpt checkpoints_race/ckpt_race_2000.pt \
    --max-steps 5000
```

The current best race checkpoint (`checkpoints_race/ckpt_race_10000.pt`) was
resumed from `ckpt_race_5400.pt` and trained to 10,000 steps on the full
`data_v1` corpus with online mutation.

## Evaluation

Reserved validation corpus:

```bash
python -m astrolabe.eval_race_val \
    --checkpoint checkpoints_race/ckpt_race_10000.pt \
    --data-dir data_v1 \
    --samples 2000
```

Live project scan:

```bash
python -m astrolabe.detect_race_project \
    --project ../../PolyScam/main \
    --checkpoint checkpoints_race/ckpt_race_10000.pt \
    --threshold 0.2
```

Single-file race inference:

```bash
python -m astrolabe.detect --kind race \
    --checkpoint checkpoints_race/ckpt_race_10000.pt \
    --file some_file.go \
    --project project_root \
    --threshold 0.2
```

## Important Caveats

- The model is trained on a **mutation corpus**: positives are accesses that used
  to be protected by sync primitives. It may miss races in functions with no
  internal synchronization but concurrent external callers.
- Caller context is depth-1 only and uses a static AST call graph, so indirect
  calls (interfaces, function values, reflection) are invisible.
- High-confidence findings should be treated as candidates, not confirmed bugs;
  verify with `go run -race` or static analyzers.
