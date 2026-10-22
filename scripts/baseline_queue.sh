#!/usr/bin/env bash
# Run the baseline trainings one at a time on the single GPU, after the NCM run started by its own builder:
#   MDLM from scratch → Dream-7B GPU smoke test → Dream-7B fine-tune → NCM (once its builder has committed the code).
# Every run logs to runs/<id>/ via RunLogger, so it shows up in the web app's Training tab as soon as it starts.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-llm/bin/python
mkdir -p runs/_logs
log() { echo "$(date '+%F %T') $*"; }
state() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['state'])" "runs/$1/status.json" 2>/dev/null || echo missing; }

if [ "$(state mdlm-v3)" != "finished" ]; then
  log "start mdlm-v3"
  $PY -I -W ignore mougia_train/mdlm_train.py --run-id mdlm-v3 --data data/processed/train_v3 --pretrain-tokens 200e6 --max-hours 4 > runs/_logs/mdlm-v3.log 2>&1
  log "end mdlm-v3 ($(state mdlm-v3))"
fi

if [ "$(state dream-7b)" != "finished" ]; then
  SMOKE=$(mktemp -d)
  log "dream GPU smoke test in $SMOKE"
  if MOUGIA_RUNS=$SMOKE $PY -I -W ignore mougia_train/dream_finetune.py --run-id dream-smoke --max-steps 10 --max-val 8 \
       --eval-replies 2 --sample-every-min 9999 > runs/_logs/dream-smoke.log 2>&1 \
     && grep -q '"finished"' "$SMOKE/dream-smoke/status.json"; then
    rm -rf "$SMOKE"
    log "smoke ok — start dream-7b"
    $PY -I -W ignore mougia_train/dream_finetune.py --run-id dream-7b --epochs 1.5 > runs/_logs/dream-7b.log 2>&1
    log "end dream-7b ($(state dream-7b))"
  else
    log "dream smoke test FAILED — see runs/_logs/dream-smoke.log; full run skipped"
  fi
fi
if [ "$(state ncm-v3)" != "finished" ] && [ -f runs/_logs/ncm-v3.cmd ]; then
  log "start ncm-v3: $(cat runs/_logs/ncm-v3.cmd)"
  bash runs/_logs/ncm-v3.cmd > runs/_logs/ncm-v3.log 2>&1
  log "end ncm-v3 ($(state ncm-v3))"
fi
log "baseline queue done"
