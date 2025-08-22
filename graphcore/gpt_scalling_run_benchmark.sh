#!/bin/bash

# Date: $(date +%Y-%m-%d)

set -e  # Exit on error

# ====== CONFIGURATION ======
# Base directory
BASE_DIR="/home/$USER/graphcore/examples/nlp/gpt2/pytorch"
# New output directories
RESULTS_DIR="$BASE_DIR/gpt2_optimized_benchmarks"
LOG_DIR="$RESULTS_DIR/logs"
DATA_DIR="$RESULTS_DIR/metrics"

# Create timestamped run identifier
RUN_ID=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR" "$DATA_DIR"

# Create a CSV file to store all metrics
METRICS_FILE="$DATA_DIR/metrics_${RUN_ID}.csv"
echo "run_id,timestamp,model,n_layer,ipus_per_replica,replication_factor,layers_per_ipu,total_ipus,batch_size,gradient_accumulation,embedding_serialization_factor,effective_batch,max_len,throughput,loss,accuracy,compilation_time,execution_time,seed" > "$METRICS_FILE"

# Random seed for reproducibility
SEED=42

# ====== UTILITY FUNCTIONS ======

extract_metrics() {
    local log_file=$1
    local model=$2
    local n_layer=$3
    local ipus_per_replica=$4
    local replication_factor=$5
    local layers_per_ipu=$6
    local total_ipus=$7
    local batch_size=$8
    local gradient_accumulation=$9
    local embedding_serialization_factor=${10}
    local max_len=${11:-1024}
    local effective_batch=$((batch_size * gradient_accumulation * replication_factor))
    
    # Extract metrics from log file
    local throughput=$(grep -oP "throughput: \K[0-9.]+" "$log_file" | tail -1)
    local loss=$(grep -oP "loss: \K[0-9.]+" "$log_file" | tail -1)
    local accuracy=$(grep -oP "acc: \K[0-9.]+" "$log_file" | tail -1)
    local compilation_time=$(grep -oP "Compiled/Loaded model in \K[0-9.]+" "$log_file" || echo "N/A")
    
    # Calculate execution time from log timestamps
    local start_time=$(grep -oP "Benchmark started at: \K.*" "$log_file")
    local end_time=$(grep -oP "Benchmark completed at: \K.*" "$log_file")
    local execution_time="N/A"
    if [[ -n "$start_time" && -n "$end_time" ]]; then
        start_sec=$(date -d "$start_time" +%s)
        end_sec=$(date -d "$end_time" +%s)
        execution_time=$((end_sec - start_sec))
    fi
    
    # Append metrics to CSV file
    echo "$RUN_ID,$(date +%Y-%m-%d_%H:%M:%S),$model,$n_layer,$ipus_per_replica,$replication_factor,\"$layers_per_ipu\",$total_ipus,$batch_size,$gradient_accumulation,$embedding_serialization_factor,$effective_batch,$max_len,$throughput,$loss,$accuracy,$compilation_time,$execution_time,$SEED" >> "$METRICS_FILE"
    
    echo "Extracted metrics from $log_file and saved to $METRICS_FILE"
}

# ====== BENCHMARK FUNCTION ======
run_benchmark() {
    local model=$1
    local config_file=$2
    local layers_per_ipu=$3
    local ipus_per_replica=$4
    local replication_factor=$5
    local matmul_proportion=$6
    local batch_size=$7
    local gradient_accumulation=$8
    local embedding_serialization_factor=${9:-6}
    local max_len=${10:-1024}
    local run_number=1
    local effective_batch=$((batch_size * gradient_accumulation * replication_factor))

    # Determine n_layer based on model
    local n_layer
    if [[ "$model" == "gpt2-test" ]]; then
        n_layer=12
    elif [[ "$model" == "gpt2-medium" ]]; then
        n_layer=24
    elif [[ "$model" == "gpt2-large" ]]; then
        n_layer=36
    elif [[ "$model" == "gpt2-xl" ]]; then
        n_layer=48
    else
        n_layer=12  # Default to small
    fi


    if (( gradient_accumulation < 5 )); then
        echo "Warning: gradient_accumulation must be at least 5 for pipelined execution. Adjusting to 5."
        gradient_accumulation=5

        effective_batch=$((batch_size * gradient_accumulation * replication_factor))
    fi

    # Ensure an even number of IPUs is requested
    local total_ipus=$((ipus_per_replica * replication_factor))
    if (( total_ipus % 2 != 0 )); then
        echo "Warning: Requested $total_ipus IPUs which is an odd number. Adjusting to use an even number."
        ipus_per_replica=$((ipus_per_replica + 1))
        total_ipus=$((ipus_per_replica * replication_factor))
        # Append a zero allocation for the extra IPU
        layers_per_ipu="$layers_per_ipu 0"
        matmul_proportion="$matmul_proportion 0.15"
        echo "Adjusted to use $total_ipus IPUs with layer distribution: $layers_per_ipu"
    fi

    # Replace spaces in layers_per_ipu with underscores to form a layout tag
    local layout_tag
    layout_tag=$(echo "$layers_per_ipu" | tr ' ' '_')

    # Create unique identifiers for this benchmark, including layout information
    local config_id="${model}_${n_layer}layers_bs${batch_size}_ga${gradient_accumulation}_effbs${effective_batch}_ml${max_len}_run${run_number}_${layout_tag}"
    local log_file="$LOG_DIR/${config_id}_${RUN_ID}.log"

    echo "==============================================="
    echo "Running benchmark with model: $model ($n_layer layers)"
    echo "Batch size: $batch_size"
    echo "Gradient accumulation: $gradient_accumulation"
    echo "Replication factor: $replication_factor"
    echo "Effective batch size: $effective_batch"
    echo "Max sequence length: $max_len"
    echo "IPUs per replica: $ipus_per_replica"
    echo "Layers per IPU: $layers_per_ipu"
    echo "Matmul proportion: $matmul_proportion"
    echo "Embedding serialization factor: $embedding_serialization_factor"
    echo "Total IPUs: $total_ipus"
    echo "Log file: $log_file"
    echo "==============================================="

    # Write initial information to the log file
    {
        echo "==============================================="
        echo "Benchmark started at: $(date)"
        echo "Configuration ID: $config_id"
        echo "Run ID: $RUN_ID"
        echo "Parameters:"
        echo "  - Model: $model"
        echo "  - Config: $config_file"
        echo "  - Layers: $n_layer"
        echo "  - Batch size: $batch_size" 
        echo "  - Gradient accumulation: $gradient_accumulation"
        echo "  - Replication factor: $replication_factor"
        echo "  - Effective batch size: $effective_batch"
        echo "  - Max sequence length: $max_len"
        echo "  - IPUs per replica: $ipus_per_replica"
        echo "  - Layers per IPU: $layers_per_ipu"
        echo "  - Matmul proportion: $matmul_proportion"
        echo "  - Embedding serialization factor: $embedding_serialization_factor"
        echo "  - Total IPUs: $total_ipus"
        echo "  - Seed: $SEED"
        echo "==============================================="

        echo "Running benchmark command at: $(date)"

        # Run the benchmark command with all outputs redirected to the log file
        /opt/slurm/bin/srun --ipus=$total_ipus python $BASE_DIR/train_gpt2.py \
          --model $model \
          --ipus-per-replica $ipus_per_replica \
          --replication-factor $replication_factor \
          --gradient-accumulation $gradient_accumulation \
          --device-iterations 8 \
          --batch-size $batch_size \
          --layers-per-ipu $layers_per_ipu \
          --matmul-proportion $matmul_proportion \
          --max-len $max_len \
          --optimizer AdamW \
          --learning-rate 0.00015 \
          --lr-schedule cosine \
          --lr-warmup 0.01 \
          --remap-logit True \
          --enable-sequence-serialized True \
          --embedding-serialization-factor $embedding_serialization_factor \
          --recompute-checkpoint-every-layer True \
          --enable-half-partials True \
          --replicated-tensor-sharding True \
          --dataset 'generated' \
          --epochs 1 \
          --seed $SEED

        echo "Benchmark command completed at: $(date)"

        echo "==============================================="
        echo "Benchmark completed at: $(date)"
        echo "==============================================="
    } &> "$log_file"

    # Extract and save metrics from the log file
    extract_metrics "$log_file" "$model" "$n_layer" "$ipus_per_replica" "$replication_factor" "$layers_per_ipu" "$total_ipus" "$batch_size" "$gradient_accumulation" "$embedding_serialization_factor" "$max_len"

    echo "Benchmark $config_id completed. Output saved to $log_file"
    echo ""
}


# ====== MAIN EXECUTION ======

echo "===== GPT-2 Model Scaling Benchmark Suite ====="
echo "Run ID: $RUN_ID"
echo "Starting at $(date)"
echo "Results will be saved to:"
echo "  - Logs: $LOG_DIR"
echo "  - Metrics: $METRICS_FILE"
echo "=========================================="
echo ""


echo "===== Running benchmarks with effective batch size approx. 16 ====="

run_benchmark "gpt2-test" "config/config_test.json" "0 4 4 4" 4 4 "0.4 0.15 0.15 0.15" 1 5 6 1024

run_benchmark "gpt2-medium" "config/config_medium.json" "0 3 3 3 3 4 4 4" 8 2 "0.4 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1 8 6 1024

run_benchmark "gpt2-large" "config/config_large.json" "0 2 2 2 2 2 2 3 3 3 3 3 3 2 2 2" 16 1 "0.4 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1 16 6 1024

echo "===== Running benchmarks with effective batch size approx. 32 ====="

run_benchmark "gpt2-test" "config/config_test.json" "0 4 4 4" 4 4 "0.4 0.15 0.15 0.15" 2 5 6 1024

run_benchmark "gpt2-medium" "config/config_medium.json" "0 3 3 3 3 4 4 4" 8 2 "0.4 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 2 8 6 1024

run_benchmark "gpt2-large" "config/config_large.json" "0 2 2 2 2 2 2 3 3 3 3 3 3 2 2 2" 16 1 "0.4 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 2 16 6 1024


echo "===== Running benchmarks with effective batch size approx. 64 ====="

run_benchmark "gpt2-test" "config/config_test.json" "0 4 4 4" 4 4 "0.4 0.15 0.15 0.15" 3 5 6 1024  # 3 * 5 * 4 = 60，接近64

run_benchmark "gpt2-medium" "config/config_medium.json" "0 3 3 3 3 4 4 4" 8 2 "0.4 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 4 8 6 1024

run_benchmark "gpt2-large" "config/config_large.json" "0 2 2 2 2 2 2 3 3 3 3 3 3 2 2 2" 16 1 "0.4 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 4 16 6 1024

echo "===== Benchmark Suite Completed ====="
echo "Finished at $(date)"
echo "All results saved to $RESULTS_DIR"
echo "Metrics file: $METRICS_FILE"

# Generate a simple summary report
echo ""
echo "===== Summary Report ====="
echo "Model Performance Summary by Effective Batch Size:"
echo ""
echo "Model | Eff. BS | BS | GA | Replica | IPUs/Rep | Total IPUs | Throughput | Loss"
echo "-----------------------------------------------------------------------------------------"

awk -F, 'NR>1 {print $2","$12","$8","$9","$5","$4","$7","$14","$15}' "$METRICS_FILE" | sort | uniq | while IFS=, read -r model eff_bs bs ga repl ipus total_ipus throughput loss; do
    if (( eff_bs <= 20 )); then
        group_bs="~16"
    elif (( eff_bs <= 40 )); then
        group_bs="~32"
    else
        group_bs="~64"
    fi
    printf "%-10s| %-7s| %-3s| %-3s| %-7s| %-8s| %-10s| %-10.2f| %-9.4f\n" "$model" "$group_bs" "$bs" "$ga" "$repl" "$ipus" "$total_ipus" "$throughput" "$loss"
done | sort -k1,1 -k2,2 -k3,3n

echo "-----------------------------------------------------------------------------------------"
echo "See $METRICS_FILE for complete results"
echo "===== End of Summary ====="