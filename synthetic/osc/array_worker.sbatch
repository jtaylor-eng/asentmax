#!/bin/bash
# Array worker: reads line $SLURM_ARRAY_TASK_ID of $MANIFEST -> "task method seed lr" -> run_one.sh
set -uo pipefail
module reset >/dev/null 2>&1 || true
source /fs/scratch/PAS2836/$USER/asentmax/synthetic/osc/env.sh
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
read -r task method seed lr <<< "$line"
echo "job $SLURM_JOB_ID / array $SLURM_ARRAY_TASK_ID on $(hostname): $line  (MAX_STEPS_OVERRIDE='${MAX_STEPS_OVERRIDE:-}')"
nvidia-smi -L || true
exec bash "$REPO/synthetic/osc/run_one.sh" "$task" "$method" "$seed" "$lr" ${MAX_STEPS_OVERRIDE:+"$MAX_STEPS_OVERRIDE"}
