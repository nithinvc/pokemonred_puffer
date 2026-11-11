#!/bin/bash
# Periodically push offline wandb runs to the server from a login node, for
# clusters whose compute nodes have no internet (e.g. Barfoot).
#
# Usage (from the repo root, on a login node):
#   setsid nohup cluster/wandb_sync_loop.sh <pbs_job_id> > logs/wandb_sync.log 2>&1 < /dev/null &
#
# Exits after one final sync once the job leaves the queue.
set -uo pipefail
JOB_ID="${1:?pbs job id}"
INTERVAL="${INTERVAL:-600}"

cd "$(dirname "$0")/.."
source .venv/bin/activate
WANDB_API_KEY="$(<"${HOME}/.config/pokered/wandb_key")"
export WANDB_API_KEY

sync_all() {
  for run in wandb/offline-run-*; do
    [[ -d "${run}" ]] || continue
    wandb sync "${run}" 2>&1 | tail -n 2
  done
}

while qstat "${JOB_ID}" >/dev/null 2>&1; do
  echo "=== $(date) syncing"
  sync_all
  sleep "${INTERVAL}"
done
echo "=== $(date) job ${JOB_ID} finished; final sync"
sync_all
