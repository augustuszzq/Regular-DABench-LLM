#!/bin/bash
# large_scale_run.sh - Scale up training to 20 nodes (80 GPUs)

# 1. Set working directory
cd /eagle/lc-mpi/Zhiqing/Megatron-DeepSpeed
export PBS_O_WORKDIR=$(pwd)

# 2. Activate virtual environment
source /eagle/lc-mpi/Zhiqing/Megatron-DeepSpeed/venvs/2024-04-29/bin/activate

# 3. Disable wandb (avoid login errors)
export WANDB_DISABLED=1

# 4. Set optimized training parameters for large-scale training
# Increased parallelism for 80 GPUs
export TP=4                        # Tensor parallelism
export PP=4                        # Pipeline parallelism
export SP=1                        # Sequence parallelism

# Full model configuration - use full-sized GPT-7B 
export MODEL_SIZE_KEY="GPT_7B"
export NLAYERS=32                  # Standard number of layers
export HIDDEN=4096
export HEADS=32
export NUM_KV_HEAD=8
export FFN_HIDDEN_SIZE=11008

# Batch settings - optimized for 80 GPUs
export MICRO_BATCH=4               # Increased micro batch size
export GAS=2                       # Gradient accumulation steps

# Optimizer and training settings
export ZERO_STAGE=1                # Zero-redundancy optimizer stage
export OPT="adamw"                 # AdamW optimizer
export LR=0.0001                   # Learning rate
export LR_WARMUP_FRAC=0.01         # Warmup fraction
export WEIGHT_DECAY=0.1            # Weight decay

# Sequence length
export SEQ=2048                    # Full sequence length

# Iteration count
export TRAIN_ITERS=10000           # Increased for proper training
export SAVE_INTERVAL=1000          # Save checkpoints every 1000 iterations
export EVAL_INTERVAL=1000          # Evaluate every 1000 iterations
export LOG_INTERVAL=10             # Log output frequency

# Other options
export USE_ACTIVATION_CHECKPOINTING=1
export DTYPE="bf16"                # Use bfloat16 for better performance/stability
export TIMING_LOG_LEVEL=1          # Standard timing logs

# Data path
export DATA_FILE_LIST="${PBS_O_WORKDIR}/ALCF/data-lists/polaris/books.txt"

# 5. Configure communication environment
source ${PBS_O_WORKDIR}/ALCF/aws_ofi_nccl_plugin.sh

# 6. Create log directory
LOG_DIR="${PBS_O_WORKDIR}/logs/large_scale_run_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

# 7. Set up checkpoint directory for saving model
CHECKPOINT_DIR="${PBS_O_WORKDIR}/checkpoints/large_scale_run_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${CHECKPOINT_DIR}"
export SAVE="${CHECKPOINT_DIR}"
export LOAD="${CHECKPOINT_DIR}"

# 8. Output current configuration
echo "=== LARGE-SCALE TRAINING CONFIGURATION ===" | tee "${LOG_DIR}/config.log"
echo "Number of nodes: 20" | tee -a "${LOG_DIR}/config.log"
echo "GPUs per node: 4" | tee -a "${LOG_DIR}/config.log"
echo "Total GPUs: 80" | tee -a "${LOG_DIR}/config.log"
echo "Model parallelism: TP=${TP}, PP=${PP}, SP=${SP}" | tee -a "${LOG_DIR}/config.log"
echo "Model: ${MODEL_SIZE_KEY} (${NLAYERS} layers)" | tee -a "${LOG_DIR}/config.log"
echo "Sequence length: ${SEQ}" | tee -a "${LOG_DIR}/config.log"
echo "Micro batch size: ${MICRO_BATCH}" | tee -a "${LOG_DIR}/config.log"
echo "Global batch size: ${MICRO_BATCH} * ${GAS} * (80/(${TP}*${PP}))" | tee -a "${LOG_DIR}/config.log"
echo "Training iterations: ${TRAIN_ITERS}" | tee -a "${LOG_DIR}/config.log"
echo "Data file: ${DATA_FILE_LIST}" | tee -a "${LOG_DIR}/config.log"
echo "Checkpoint directory: ${CHECKPOINT_DIR}" | tee -a "${LOG_DIR}/config.log"
echo "Checkpoint saving interval: ${SAVE_INTERVAL}" | tee -a "${LOG_DIR}/config.log"
echo "=========================================" | tee -a "${LOG_DIR}/config.log"

# 9. Custom arguments passed to train_aGPT_7B.sh
CUSTOM_ARGS="--mmap-warmup"

# 10. Start training
echo "Starting large-scale training..." | tee -a "${LOG_DIR}/config.log"
bash train_aGPT_7B.sh ${CUSTOM_ARGS} 2>&1 | tee "${LOG_DIR}/training.log"

# 11. Verify training progress
echo "Training completed. Logs are available at: ${LOG_DIR}" | tee -a "${LOG_DIR}/config.log"
if grep -q "iteration.*${TRAIN_ITERS}" "${LOG_DIR}/training.log"; then
  echo "Training successfully completed all ${TRAIN_ITERS} iterations!" | tee -a "${LOG_DIR}/config.log"
else
  echo "Training may not have completed all iterations, please check logs for details." | tee -a "${LOG_DIR}/config.log"
fi