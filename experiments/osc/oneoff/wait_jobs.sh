#!/usr/bin/env bash
# wait_jobs.sh <jobid>... : poll Slurm on ascend every 10 min until none of the given job ids are queued/running.
FILTER='^\s*\*|authorized users|without authority|activities on|recorded by|improperly|the activities|expressly|such monitoring|officials'
ids=$(IFS=,; echo "$*")
while :; do
  n=$(ssh ascend-agent "squeue --me -h -j $ids 2>/dev/null | wc -l" 2>/dev/null | grep -v -E "$FILTER" | tail -1)
  ts=$(date '+%m-%d %H:%M')
  if [ -z "$n" ]; then echo "$ts ssh failed, retrying"; sleep 300; continue; fi
  echo "$ts $n array elements still queued/running"
  [ "$n" -eq 0 ] && break
  sleep 600
done
echo "ALL DONE: jobs $ids finished at $(date)"
ssh ascend-agent "sacct -j $ids -X --format=JobID%16,JobName%16,State%12,Elapsed -P 2>/dev/null | grep -v -E 'COMPLETED' " 2>/dev/null | grep -v -E "$FILTER"
