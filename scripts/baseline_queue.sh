#!/usr/bin/env bash
# Run the baseline trainings one at a time on the single GPU:
#   MDLM (diffusion, from scratch) → NCM (next-concept, from scratch) → Dream-7B GPU smoke test → Dream-7B fine-tune.
# Every run logs to runs/<id>/ via RunLogger, so it shows up in the web app's Training tab as soon as it starts.
# Restart-safe: finished runs are skipped, and a run whose process is still alive is waited for, never started twice.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-llm/bin/python
mkdir -p runs/_logs
log() { echo "$(date '+%F %T') $*"; }
state() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['state'])" "runs/$1/status.json" 2>/dev/null || echo missing; }
alive() { pgrep -f "^$PY -I -W ignore mougia_train/$1 --run-id $2( |$)" >/dev/null; }

run() {  # run <id> <script> <args...>
  local id=$1 script=$2; shift 2
  while alive "$script" "$id"; do sleep 60; done
  if [ "$(state "$id")" != "finished" ]; then
    log "start $id"
    $PY -I -W ignore "mougia_train/$script" --run-id "$id" "$@" > "runs/_logs/$id.log" 2>&1
  fi
  log "$id: $(state "$id")"
}

run mdlm-v3 mdlm_train.py --data data/processed/train_v3 --pretrain-tokens 200e6 --max-hours 4
run ncm-v3 ncm_train.py --max-hours 5

while alive dream_finetune.py dream-7b; do sleep 60; done
if [ "$(state dream-7b)" != "finished" ]; then
  SMOKE=$(mktemp -d)
  log "dream GPU smoke test in $SMOKE"
  if MOUGIA_RUNS=$SMOKE $PY -I -W ignore mougia_train/dream_finetune.py --run-id dream-smoke --max-steps 10 --max-val 8 \
       --eval-replies 2 --sample-every-min 9999 > runs/_logs/dream-smoke.log 2>&1 \
     && grep -q '"finished"' "$SMOKE/dream-smoke/status.json"; then
    rm -rf "$SMOKE"
    run dream-7b dream_finetune.py --epochs 1.5
  else
    log "dream smoke test FAILED — see runs/_logs/dream-smoke.log; full run skipped"
  fi
fi
log "baseline queue done"
