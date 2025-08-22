#!/bin/bash
# GPT-2 Comprehensive Scaling Benchmark Script
# Systematic evaluation of GPT-2 model scaling with different layer configurations
# Author: [Your Name]
# Date: $(date +%Y-%m-%d)

set -e  # Exit on error

# ====== CONFIGURATION ======
# Base directory
BASE_DIR="/home/$USER/graphcore/examples/nlp/gpt2/pytorch"
# Output directories - CHANGED TO A NEW DIRECTORY NAME
RESULTS_DIR="$BASE_DIR/scaling_benchmark_results_new_full"
LOG_DIR="$RESULTS_DIR/logs"
REPORT_DIR="$RESULTS_DIR/popvision_reports"
DATA_DIR="$RESULTS_DIR/metrics"

# Create timestamped run identifier
RUN_ID=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$DATA_DIR"

# Create a CSV file to store all metrics
METRICS_FILE="$DATA_DIR/metrics_${RUN_ID}.csv"
echo "run_id,timestamp,n_layer,model,ipus_per_replica,replication_factor,layers_per_ipu,total_ipus,batch_size,gradient_accumulation,embedding_serialization_factor,throughput,loss,accuracy,compilation_time,execution_time,memory_utilization,compute_utilization,seed" > "$METRICS_FILE"

# Random seed for reproducibility
SEED=42

# ====== UTILITY FUNCTIONS ======

modify_config() {
    local config_file=$1
    local n_layer=$2
    local temp_file="${config_file}.tmp"
    
    # Create a copy of the original file
    cp "$config_file" "$config_file.orig"
    
    # Use sed to replace n_layer value
    cat "$config_file" | sed "s/\"n_layer\": [0-9]*,/\"n_layer\": $n_layer,/" > "$temp_file"
    mv "$temp_file" "$config_file"
    
    echo "Modified $config_file to use $n_layer layers"
}

restore_config() {
    local config_file=$1
    if [ -f "$config_file.orig" ]; then
        mv "$config_file.orig" "$config_file"
        echo "Restored original $config_file"
    fi
}

extract_metrics() {
    local log_file=$1
    local n_layer=$2
    local model=$3
    local ipus_per_replica=$4
    local replication_factor=$5
    local layers_per_ipu=$6
    local total_ipus=$7
    local batch_size=1
    local gradient_accumulation=2048
    local embedding_serialization_factor=4
    
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
    
    # Extract memory and compute utilization (placeholder - actual extraction would depend on PopVision report format)
    local memory_utilization="N/A"  # Would need to extract from PopVision report
    local compute_utilization="N/A"  # Would need to extract from PopVision report
    
    # Append metrics to CSV file
    echo "$RUN_ID,$(date +%Y-%m-%d_%H:%M:%S),$n_layer,$model,$ipus_per_replica,$replication_factor,\"$layers_per_ipu\",$total_ipus,$batch_size,$gradient_accumulation,$embedding_serialization_factor,$throughput,$loss,$accuracy,$compilation_time,$execution_time,$memory_utilization,$compute_utilization,$SEED" >> "$METRICS_FILE"
    
    echo "Extracted metrics from $log_file and saved to $METRICS_FILE"
}

# ====== BENCHMARK FUNCTION ======
run_benchmark() {
    local n_layer=$1
    local model=$2
    local config_file=$3
    local layers_per_ipu=$4
    local ipus_per_replica=$5
    local replication_factor=$6
    local matmul_proportion=$7
    local run_number=1

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
    local config_id="${model}_${n_layer}layers_run${run_number}_${layout_tag}"
    local log_file="$LOG_DIR/${config_id}_${RUN_ID}.log"
    local report_path="$REPORT_DIR/${config_id}_${RUN_ID}"
    mkdir -p "$report_path"

    echo "==============================================="
    echo "Running benchmark with $n_layer layers (Run $run_number)"
    echo "Model: $model"
    echo "IPUs per replica: $ipus_per_replica"
    echo "Replication factor: $replication_factor"
    echo "Layers per IPU: $layers_per_ipu"
    echo "Matmul proportion: $matmul_proportion"
    echo "Total IPUs: $total_ipus"
    echo "Log file: $log_file"
    echo "PopVision report: $report_path"
    echo "==============================================="

    # Write initial information to the log file
    {
        echo "==============================================="
        echo "Benchmark started at: $(date)"
        echo "Configuration ID: $config_id"
        echo "Run ID: $RUN_ID"
        echo "Parameters:"
        echo "  - Layers: $n_layer"
        echo "  - Model: $model"
        echo "  - IPUs per replica: $ipus_per_replica"
        echo "  - Replication factor: $replication_factor"
        echo "  - Layers per IPU: $layers_per_ipu"
        echo "  - Matmul proportion: $matmul_proportion"
        echo "  - Total IPUs: $total_ipus"
        echo "  - Seed: $SEED"
        echo "==============================================="

        # Modify the configuration file for this benchmark
        modify_config "$BASE_DIR/$config_file" "$n_layer"

        # Configure PopVision reporting options
        # Modified: capture only the Graph Profile (compilation time information) and disable execution profile to reduce report size
        local poplar_options='{"autoReport.outputGraphProfile": "true", "autoReport.outputExecutionProfile": "false", "autoReport.directory": "'"$report_path"'"}'
        echo "Running benchmark command at: $(date)"

        # Export the PopVision options for this run
        export POPLAR_ENGINE_OPTIONS="$poplar_options"

        # Run the benchmark command with all outputs redirected to the log file
        /opt/slurm/bin/srun --ipus=$total_ipus python $BASE_DIR/train_gpt2.py \
          --model $model \
          --ipus-per-replica $ipus_per_replica \
          --replication-factor $replication_factor \
          --gradient-accumulation 2048 \
          --device-iterations 8 \
          --batch-size 1 \
          --layers-per-ipu $layers_per_ipu \
          --matmul-proportion $matmul_proportion \
          --max-len 1024 \
          --optimizer AdamW \
          --learning-rate 0.00015 \
          --lr-schedule cosine \
          --lr-warmup 0.01 \
          --remap-logit True \
          --enable-sequence-serialized True \
          --embedding-serialization-factor 4 \
          --recompute-checkpoint-every-layer True \
          --enable-half-partials False \
          --replicated-tensor-sharding True \
          --dataset 'generated' \
          --epochs 3 \
          --seed $SEED

        echo "Benchmark command completed at: $(date)"

        # Restore the original configuration file
        restore_config "$BASE_DIR/$config_file"

        echo "==============================================="
        echo "Benchmark completed at: $(date)"
        echo "==============================================="
    } &> "$log_file"

    # Extract and save metrics from the log file
    extract_metrics "$log_file" "$n_layer" "$model" "$ipus_per_replica" "$replication_factor" "$layers_per_ipu" "$total_ipus"

    echo "Benchmark $config_id completed. Output saved to $log_file"
    echo "PopVision report saved to $report_path"
    echo ""
}


# ====== MAIN EXECUTION ======

echo "===== GPT-2 Scaling Benchmark Suite ====="
echo "Run ID: $RUN_ID"
echo "Starting at $(date)"
echo "Results will be saved to:"
echo "  - Logs: $LOG_DIR"
echo "  - PopVision reports: $REPORT_DIR"
echo "  - Metrics: $METRICS_FILE"
echo "=========================================="
echo ""

# ===== SYSTEMATIC BENCHMARK EXPERIMENTS =====
# Organized by model size (n_layer) and then by IPU allocation strategy
# Each configuration uses only the allowed IPU counts: 2, 4, 8, or 16

# ===== IPU-CENTRIC BENCHMARK EXPERIMENTS =====
# Organized by IPU count first, then exploring different n_layer configurations
# This approach prioritizes available IPU resources over fixed model sizes

# ---------------------------------------
# 2 IPU CONFIGURATIONS
# ---------------------------------------
# Small model (4 layers)
run_benchmark 4 "gpt2-test" "config/config_test.json" "0 4" 2 1 "0.15 0.15" 1

# Medium-small model (6 layers) - Commented out in your code
#run_benchmark 6 "gpt2-test" "config/config_test.json" "0 6" 2 1 "0.15 0.15" 1

# Medium model (8 layers)
run_benchmark 8 "gpt2-test" "config/config_test.json" "0 8" 2 1 "0.15 0.15" 1

# Testing layer capacity limit for 2 IPUs (10 layers)
run_benchmark 10 "gpt2-test" "config/config_test.json" "0 10" 2 1 "0.15 0.15" 1

# ---------------------------------------
# 4 IPU CONFIGURATIONS
# ---------------------------------------
# Small model with different distributions
run_benchmark 4 "gpt2-test" "config/config_test.json" "0 1 1 2" 4 1 "0.15 0.15 0.15 0.15" 1
run_benchmark 4 "gpt2-test" "config/config_test.json" "0 2 1 1" 4 1 "0.15 0.15 0.15 0.15" 1

# Medium model (8 layers) with different distributions
run_benchmark 8 "gpt2-test" "config/config_test.json" "0 2 3 3" 4 1 "0.15 0.15 0.15 0.15" 1
run_benchmark 8 "gpt2-test" "config/config_test.json" "0 4 2 2" 4 1 "0.15 0.15 0.15 0.15" 1

# Standard GPT-2 (12 layers) with different distributions - CHANGED CONFIG
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 4 4 4" 4 1 "0.15 0.15 0.15 0.15" 1
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 6 3 3" 4 1 "0.15 0.15 0.15 0.15" 1

# Medium GPT-2 (16 layers) - dense configuration - CHANGED CONFIG
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 6 5 5" 4 1 "0.15 0.15 0.15 0.15" 1

# ---------------------------------------
# 8 IPU CONFIGURATIONS
# ---------------------------------------
# Small model with excessive parallelism (testing overhead)
run_benchmark 4 "gpt2-test" "config/config_test.json" "0 1 1 1 1 0 0 0" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Medium model (8 layers) with full parallelism
run_benchmark 8 "gpt2-test" "config/config_test.json" "0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1    

# Standard GPT-2 (12 layers) with different distributions - CHANGED CONFIG
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 1 2 1 2 2 2 2" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 2 2 2 2 1 1 2" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Medium GPT-2 (16 layers) with different distributions - CHANGED CONFIG
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 2 2 2 2 2 3 3" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 3 3 2 2 2 2 2" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Large GPT-2 (24 layers) - dense configuration - CHANGED CONFIG
run_benchmark 24 "gpt2-test" "config/config_test.json" "0 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# ---------------------------------------
# 16 IPU CONFIGURATIONS
# ---------------------------------------
# Standard GPT-2 (12 layers) with maximum parallelism - CHANGED CONFIG
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Medium GPT-2 (16 layers) with maximum parallelism - CHANGED CONFIG
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1    

# Large GPT-2 (24 layers) with different distributions - CHANGED CONFIG
run_benchmark 24 "gpt2-test" "config/config_test.json" "0 1 2 1 2 1 2 1 2 1 2 1 2 2 2 2" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1
run_benchmark 24 "gpt2-test" "config/config_test.json" "0 2 2 2 2 2 2 2 2 1 1 1 1 1 1 2" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# ---------------------------------------
# ADDITIONAL EXPERIMENTS WITH NEW LAYER LAYOUTS
# ---------------------------------------

# ---------------------------------------
# 8 IPU CONFIGURATIONS - NEW LAYOUTS
# ---------------------------------------
# Medium model (8 layers) with new distributions for 8 IPUs
run_benchmark 8 "gpt2-test" "config/config_test.json" "0 2 2 1 1 1 1 0" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Standard GPT-2 (12 layers) with new distributions - CHANGED CONFIG
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 2 2 2 2 2 1 1" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1
run_benchmark 12 "gpt2-test" "config/config_test.json" "0 1 1 2 2 2 2 2" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Medium GPT-2 (16 layers) with new distributions - CHANGED CONFIG
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 2 2 2 2 3 3 2" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 2 3 3 2 2 2 2" 8 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# ---------------------------------------
# 16 IPU CONFIGURATIONS - NEW LAYOUTS
# ---------------------------------------
# Medium GPT-2 (16 layers) with more balanced distribution - CHANGED CONFIG
run_benchmark 16 "gpt2-test" "config/config_test.json" "0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

# Large GPT-2 (24 layers) with reversed distributions - CHANGED CONFIG
run_benchmark 24 "gpt2-test" "config/config_test.json" "0 2 2 2 2 1 1 1 1 1 1 2 2 2 2 2" 16 1 "0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15" 1

echo "===== Benchmark Suite Completed ====="
echo "Finished at $(date)"
echo "All results saved to $RESULTS_DIR"
echo "Metrics file: $METRICS_FILE"

# Generate a simple summary report
echo ""
echo "===== Summary Report ====="
echo "Model Scaling Performance Summary:"
echo ""
echo "Layer Count | Avg Throughput | Avg Loss | Avg Accuracy"
echo "-------------------------------------------------------"

# Extract and display summary statistics
for layers in 4 8 10 12 16 24; do
    avg_throughput=$(awk -F, -v l="$layers" '$3==l {sum+=$12; count++} END {print sum/count}' "$METRICS_FILE")
    avg_loss=$(awk -F, -v l="$layers" '$3==l {sum+=$13; count++} END {print sum/count}' "$METRICS_FILE")
    avg_accuracy=$(awk -F, -v l="$layers" '$3==l {sum+=$14; count++} END {print sum/count}' "$METRICS_FILE")
    printf "%-12s| %-14.2f| %-9.4f| %-12.4f\n" "$layers" "$avg_throughput" "$avg_loss" "$avg_accuracy"
done

echo "-------------------------------------------------------"
echo "See $METRICS_FILE for complete results"
echo "===== End of Summary ====="