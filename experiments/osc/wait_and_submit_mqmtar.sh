#!/usr/bin/env bash
# Waits for the MQMTAR dataset DATA_DONE marker, then submits the full mqmtar array.
# Run on a login node under nohup; polls every 5 min for up to 12 h.
source /fs/scratch/PAS2836/$USER/asentmax/experiments/osc/env.sh
MARK="$DATA_ROOT/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4.DATA_DONE"
for i in $(seq 1 144); do
  if [ -f "$MARK" ]; then
    cd "$REPO" && TASKS="mqmtar" bash experiments/osc/submit.sh full 2>&1
    echo "[$(date)] submitted mqmtar full array"; exit 0
  fi
  sleep 300
done
echo "[$(date)] TIMEOUT waiting for $MARK"; exit 1
