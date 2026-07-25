#!/usr/bin/env bash
# Runs successive warm-restart training loops, halving peak LR each time.
# Stops when bracket_balance hits 1.000 on 3 of the last 5 evals, or LR < 5e-6.
set -euo pipefail

PYTHON=/home/austin/Repositories/Personal/astrolabev2/.venv/bin/python3
LOGFILE=/tmp/train.log
CKPT_DIR=/home/austin/Repositories/Personal/astrolabev2/checkpoints

LOOP_STEPS=50000
START_LR=${1:-7.5e-5}
START_LOOP=${2:-4}

log() { echo "[loop_mgr] $*" | tee -a "$LOGFILE" >&2; }

find_latest_ckpt() {
    ls -t "$CKPT_DIR"/ckpt_[0-9]*.pt 2>/dev/null | grep -v restart | head -1
}

lr_too_small() {
    $PYTHON -c "import sys; sys.exit(0 if float('$1') < 5e-6 else 1)" 2>/dev/null
}

halve_lr() {
    $PYTHON -c "print(f'{float(\"$1\")/2:.2e}')"
}

run_loop() {
    local lr=$1 loop=$2
    local latest restart_ckpt

    latest=$(find_latest_ckpt)
    [[ -z "$latest" ]] && { log "ERROR: no checkpoint found"; exit 1; }

    restart_ckpt="$CKPT_DIR/restart_loop${loop}.pt"
    log "=== LOOP $loop  peak_lr=$lr  base=$(basename "$latest") ==="

    $PYTHON -c "
import torch
ckpt = torch.load('$latest', map_location='cpu', weights_only=False)
ckpt['step'] = 0
torch.save(ckpt, '$restart_ckpt')
" >> "$LOGFILE" 2>&1
    log "restart ckpt saved: $(basename "$restart_ckpt")"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PYTHON -m astrolabe.train \
        --data-dir /home/austin/Repositories/Personal/astrolabev2/data \
        --out-dir "$CKPT_DIR" \
        --max-steps $LOOP_STEPS \
        --batch-size 16 \
        --lr "$lr" \
        --min-lr "$(halve_lr "$(halve_lr "$lr")")" \
        --warmup-steps 200 \
        --resume "$restart_ckpt" \
        >> "$LOGFILE" 2>&1 &
    echo $!   # only stdout: the PID
}

# ---- main ----
lr=$START_LR
loop=$START_LOOP

while true; do
    if lr_too_small "$lr"; then
        log "LR $lr below 5e-6 floor, stopping."
        break
    fi

    pid=$(run_loop "$lr" "$loop")
    log "loop $loop running as PID $pid"

    # wait for it
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    log "loop $loop (PID $pid) finished"

    # check last 5 evals for 3 perfect hits
    perfect=$(grep '\[eval\]' "$LOGFILE" | tail -5 \
              | grep -oP 'bracket_balance \K[0-9.]+' \
              | awk 'BEGIN{c=0} $1=="1.000"{c++} END{print c}' || echo 0)
    last_bal=$(grep '\[eval\]' "$LOGFILE" | tail -1 \
               | grep -oP 'bracket_balance \K[0-9.]+' || echo "?")
    log "loop $loop done. last_balance=$last_bal  perfect_in_last_5=$perfect"

    if [[ "$perfect" -ge 3 ]]; then
        log "=== GROKKED: 3/5 recent evals at bracket_balance 1.000 ==="
        break
    fi

    lr=$(halve_lr "$lr")
    loop=$((loop + 1))
done

log "loop manager exiting."
