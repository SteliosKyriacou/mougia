#!/usr/bin/env bash
# Train the fly-brain poets back to back on the single GPU. Each run logs to runs/<id>/ (see mougia_train/runlog.py);
# console output goes to runs/_logs/<id>.log. Re-running skips runs that already finished.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/_logs
FEMALE_EDGES=2700429   # Kyriakou's full-brain connection count, for density-matched Markos

run() {
  local id=$1; shift
  if grep -qs '"finished"' "runs/$id/status.json"; then echo "skip $id (finished)"; return; fi
  echo "$(date '+%F %T') start $id"
  python3 -I -W ignore mougia_train/train.py --run-id "$id" --data data/processed/train_v3 --batch 64 "$@" > "runs/_logs/$id.log" 2>&1
  echo "$(date '+%F %T') end $id ($(cat runs/$id/status.json 2>/dev/null))"
}

# v3: data with 600 synthetic duels + clean real duels; early stopping, LM replay, learnable time constants,
# context drive. Markos-matched comes second so the fair Kyriakou-vs-Markos pair finishes first.
run v3-kyriakou-full          --sex female --region full
run v3-markos-full-matched    --sex male   --region full --max-edges $FEMALE_EDGES
run v3-kyriakou-full-random   --sex female --region full --wiring random
run v3-markos-full            --sex male   --region full
run v3-markos-full-random     --sex male   --region full --wiring random --max-edges $FEMALE_EDGES
