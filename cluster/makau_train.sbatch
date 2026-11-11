#!/bin/bash
# Pokemon Red PPO training on one Makau H100 NVL node (1 GPU, 256 CPUs).
#
# Usage (from the repo root on Makau):
#   EXP_NAME=smoke-1 TRAIN_SECONDS=1200 sbatch -t 00:45:00 cluster/makau_train.sbatch
#   EXP_NAME=gym2-1 sbatch -t 24:00:00 cluster/makau_train.sbatch
#
# Env vars:
#   EXP_NAME       wandb run id/name (required). Reusing it only reattaches the
#                  wandb run; training state is NOT resumed from checkpoints.
#   CONFIG         config file (default configs/repro-gym2.yaml)
#   WRAPPERS       wrapper set (default no_stream: no public websocket streaming)
#   WANDB_GROUP    wandb group (default: none)
#   TRAIN_SECONDS  if set, send SIGINT after this many seconds so the trainer
#                  saves a checkpoint and exits cleanly (used for smoke tests)
#SBATCH --account=ODEFN5169CYFZ
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_nvl:1
#SBATCH --exclusive
#SBATCH --job-name=pokered
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

set -eo pipefail
: "${EXP_NAME:?set EXP_NAME}"
CONFIG="${CONFIG:-configs/repro-gym2.yaml}"
WRAPPERS="${WRAPPERS:-no_stream}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} exp=${EXP_NAME} config=${CONFIG}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nproc
python -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"

# Fall back to offline wandb if compute nodes cannot reach wandb.
if curl -s -o /dev/null -m 10 https://api.wandb.ai; then
  echo "wandb: online"
else
  echo "wandb: compute node has no egress, logging offline (sync later from login node)"
  export WANDB_MODE=offline
fi
export WANDB_DIR="${SLURM_SUBMIT_DIR}"
# The default ~/.netrc key on Makau cannot write to the nithinc entity; use a
# project-specific key file (mode 600) instead.
if [[ -f "${HOME}/.config/pokered/wandb_key" ]]; then
  WANDB_API_KEY="$(<"${HOME}/.config/pokered/wandb_key")"
  export WANDB_API_KEY
fi

cmd=(python -m pokemonred_puffer.train train
  --config "${CONFIG}"
  --wrappers-name "${WRAPPERS}"
  --exp-name "${EXP_NAME}"
  --rom-path red.gb
  --track)

if [[ -n "${WANDB_GROUP:-}" ]]; then
  export WANDB_RUN_GROUP="${WANDB_GROUP}"
fi

if [[ -n "${TRAIN_SECONDS:-}" ]]; then
  # SIGINT -> KeyboardInterrupt -> trainer __exit__ saves checkpoint and uploads model.
  timeout -s INT -k 300 "${TRAIN_SECONDS}" "${cmd[@]}" || echo "train exited with $?"
else
  "${cmd[@]}"
fi

# Checkpoints go to runs/pokemon-red-<uuid> (train.py:180), not runs/${EXP_NAME}.
echo "=== checkpoints"
latest="$(ls -td runs/pokemon-red-* 2>/dev/null | head -1)"
if [[ -n "${latest}" ]]; then
  du -sh "${latest}"
  ls -la "${latest}" | head -20
fi
echo "=== done $(date)"
