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
  python3 -I -W ignore mougia_train/train.py --run-id "$id" --data data/old/processed/train_v3 --batch 64 "$@" > "runs/_logs/$id.log" 2>&1
  echo "$(date '+%F %T') end $id ($(cat runs/$id/status.json 2>/dev/null))"
}

# v4: same data as v3 (600 synthetic duels + clean real duels) + EARS: the call enters through 567 auditory
# neurons, own words through 2,210 olfactory neurons, reply read from 1,192 descending neurons (fair per-type counts).
run v4-kyriakou-full          --sex female --region full
run v4-markos-full-matched    --sex male   --region full --max-edges $FEMALE_EDGES
run v4-kyriakou-full-random   --sex female --region full --wiring random
run v4-markos-full            --sex male   --region full
run v4-markos-full-random     --sex male   --region full --wiring random --max-edges $FEMALE_EDGES

# v5: v4 + MUSHROOM-BODY MEMORY — the 46–51k Kenyon-cell→MBON synapses become fast plastic weights, rewritten every
# syllable by dopamine through the real DAN→MBON wiring (scripts/build_mushroom_body.py, FlyBrainLM mb=True).
# Same data and settings as v4, so v4 vs v5 isolates the memory; the random control rewires the memory circuit too.
run v5-kyriakou-full-mb       --sex female --region full --mb
run v5-kyriakou-full-random-mb --sex female --region full --wiring random --mb
run v5-markos-full-mb         --sex male   --region full --mb
