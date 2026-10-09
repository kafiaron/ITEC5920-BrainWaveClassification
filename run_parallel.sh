#!/usr/bin/env bash
# Run every "python ..." line of a job file as its own process, N at a time.
#   bash run_parallel.sh experiments_tier1.sh 4
# Logs go to logs/job_<n>.log. Re-running resumes: finished folds are skipped.
set -u
JOBS_FILE=$1
MAX=${2:-4}
export TF_FORCE_GPU_ALLOW_GROWTH=true   # lets several processes share one GPU
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p logs
i=0
while IFS= read -r cmd; do
  [[ $cmd == python* ]] || continue
  i=$((i + 1))
  ( $cmd > "logs/job_$i.log" 2>&1 && echo "done   $i: $cmd" || echo "FAILED $i: $cmd" ) &
  while (( $(jobs -rp | wc -l) >= MAX )); do sleep 5; done
done < "$JOBS_FILE"
wait
echo "All jobs finished. Check for FAILED lines above, then re-run to resume any of them."
