#!/bin/bash
# debug_run.sh - Verify training workflow using 2 nodes (8 GPUs) without saving checkpoints

# 1. Set working directory
cd /eagle/lc-mpi/Zhiqing/Megatron-DeepSpeed
export PBS_O_WORKDIR=$(pwd)

# 2. Activate virtual environment - use existing environment
source /eagle/lc-mpi/Zhiqing/Megatron-DeepSpeed/venvs/2024-04-29/bin/activate

# 3. Disable wandb (avoid login errors)
export WANDB_DISABLED=1

# 4. Set debug-optimized training parameters
# Basic parallelism settings - reduce parallelism to accommodate fewer GPUs
export TP=1                        # Tensor parallelism
export PP=1                        # Pipeline parallelism
export SP=1                        # Sequence parallelism

# Model configuration - use GPT-7B but reduce layers to speed up debugging
export MODEL_SIZE_KEY="GPT6_7B"
export NLAYERS=24                   # Reduce layers to speed up debugging
export HIDDEN=4096
export HEADS=32
export NUM_KV_HEAD=8
export FFN_HIDDEN_SIZE=11008

# Batch settings - small scale suitable for 8 GPUs
export MICRO_BATCH=1               # Smaller batch size to avoid memory issues
export GAS=1                       # Gradient accumulation steps
export DATA_CACHE_PATH="${PBS_O_WORKDIR}/data_cache"
# Optimizer and training settings
export ZERO_STAGE=1
export OPT="adamw"
export LR=0.0002
export LR_WARMUP_FRAC=0.01
export WEIGHT_DECAY=0.1

# Sequence length - reduce sequence length to speed up training
export SEQ=1024                    # Reduce sequence length to speed up training

# Iteration count - run only a few iterations to verify system
export TRAIN_ITERS=10              # Very few iterations for debugging
export SAVE_INTERVAL=0             # Set to 0 to disable checkpoint saving
export EVAL_INTERVAL=100           # Set interval for evaluation
export LOG_INTERVAL=1              # Keep log output

# Other options
export USE_ACTIVATION_CHECKPOINTING=1
export DTYPE="fp16"
export TIMING_LOG_LEVEL=2          # Increase log verbosity for debugging

# Data path
export DATA_FILE_LIST="${PBS_O_WORKDIR}/ALCF/data-lists/polaris/books.txt"
# 5. Configure communication environment
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
# 5. Configure communication environment
source ${PBS_O_WORKDIR}/ALCF/aws_ofi_nccl_plugin.sh

# 6. Create log directory
LOG_DIR="${PBS_O_WORKDIR}/logs/debug_run_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

# 7. Set up temporary directory as load and save directory, disable checkpoint saving
TEMP_DIR="/tmp/no_checkpoint_$(date +%s%N)"
mkdir -p "${TEMP_DIR}"
export SAVE="${TEMP_DIR}"
export LOAD="${TEMP_DIR}"
export NO_SAVE_OPTIM=1           # Don't save optimizer state
export NO_LOAD_OPTIM=1           # Don't load optimizer state

# 8. Output current configuration
echo "=== DEBUG RUN CONFIGURATION ===" | tee "${LOG_DIR}/config.log"
echo "Number of nodes: $(wc -l < ${PBS_NODEFILE})" | tee -a "${LOG_DIR}/config.log"
echo "GPUs per node: 4" | tee -a "${LOG_DIR}/config.log"
echo "Total GPUs: $(wc -l < ${PBS_NODEFILE}) * 4 = 8" | tee -a "${LOG_DIR}/config.log"
echo "Model: ${MODEL_SIZE_KEY} (modified to ${NLAYERS} layers)" | tee -a "${LOG_DIR}/config.log"
echo "Sequence length: ${SEQ}" | tee -a "${LOG_DIR}/config.log"
echo "Micro batch size: ${MICRO_BATCH}" | tee -a "${LOG_DIR}/config.log"
echo "Global batch size: approximately 8" | tee -a "${LOG_DIR}/config.log"
echo "Training iterations: ${TRAIN_ITERS}" | tee -a "${LOG_DIR}/config.log"
echo "Data file: ${DATA_FILE_LIST}" | tee -a "${LOG_DIR}/config.log"
echo "Checkpoint saving: Disabled" | tee -a "${LOG_DIR}/config.log"
echo "=================================" | tee -a "${LOG_DIR}/config.log"

# 9. Custom arguments passed to train_aGPT_7B.sh
CUSTOM_ARGS="--no-load-optim --no-save-optim --no-save-rng --mmap-warmup"
# 10. Start training
echo "Starting debug run..." | tee -a "${LOG_DIR}/config.log"
bash train_aGPT_7B.sh ${CUSTOM_ARGS} 2>&1 | tee "${LOG_DIR}/training.log"

# 11. Clean up temporary directory
rm -rf "${TEMP_DIR}"

# 12. Verify successful run
if grep -q "iteration.*${TRAIN_ITERS}" "${LOG_DIR}/training.log"; then
  echo "Debug successful! Training reached expected iteration count ${TRAIN_ITERS}" | tee -a "${LOG_DIR}/config.log"
else
  echo "Debug may not have completed expected iterations, please check logs" | tee -a "${LOG_DIR}/config.log"
fi