# Artifact Evaluation README for [Your Paper Title]

## Overview
This repository contains the artifacts for the paper titled  Regular-DABench-LLM: Standardized and In-Depth Benchmarking of Post-Moore Dataflow AI Accelerators for LLMs by Ziyu Hu and Zhiqing Zhong. These artifacts are submitted for Artifact Evaluation (AE) to demonstrate the reproducibility of the results presented in the paper.


The artifacts include source code, datasets, scripts, and documentation necessary to replicate the experiments. We aim for the **Available**, **Functional**, and **Reproduced** badges .

## Artifact Description
### Abstract
This artifact provides a comprehensive collection of test scripts designed to evaluate performance across all our hardware configurations, including Graphcore IPU, Cerebras, and SambaNova systems. It also includes references to key articles and papers that informed the development and validation of these tests, enabling reproducibility of the experiments described in the paper.

### Scope

- **Limitations**: Limitations: All hardware is provided by Argonne National Laboratory(ALCF), and all environments are based on the current hardware setups.

### Hardware and Software Requirements
### Hardware and Software Requirements
- **Hardware**:
  - Graphcore Bow Pod64: 64 Bow-class Intelligence Processing Units (IPUs), providing aggregate performance of 22 Petaflops/s in half precision, with 57.6 GB total In-Processor-Memory and 94,208 IPU cores across four servers for data processing.
  - Cerebras CS-2: 850,000 processing cores, each with 48 KB dedicated SRAM (totaling 40 GB on-chip memory), configured as a Wafer-Scale Cluster with one CS-2 system and supporting nodes (MemoryX for weight storage, SwarmX for gradient accumulation, management, and input workers).
  - SambaNova DataScale SN30: Eight nodes across four full racks, each node featuring eight Reconfigurable Dataflow Units (RDUs) interconnected for model and data parallelism.

- **Software**:
  - OS: Linux-based (e.g., Ubuntu 20.04 LTS or later, as managed in the ALCF environment).
  - Programming Language: Python 3.10+ (integrated with machine learning frameworks).
  - Dependencies: Listed in `requirements.txt` or below. Key dependencies include:
    - Graphcore: Poplar SDK (for graph software and ML applications), PopTorch (PyTorch wrapper optimized for IPUs), PopLibs (for tensor and graph operations), and integrations with TensorFlow/PyTorch.
    - Cerebras: PyTorch framework integration for model compilation and execution; managed as an appliance for data preprocessing, streaming, and orchestration.
    - SambaNova: SambaFlow software stack (for optimizing and mapping dataflow graphs to RDUs), with integrations to PyTorch.
## Installation
1. Clone the repository:
   ```
   git clone https://github.com/[YourUsername]/[YourRepoName].git
   cd [YourRepoName]
   ```

2. Install dependencies:
   - For Python-based projects:
     ```
     python -m venv env
     source env/bin/activate
     pip install -r requirements.txt
     ```
   - List key packages: [e.g., numpy==1.26.0, torch==2.0.1, etc.]

3. Download datasets (if not included):
   - [Instructions, e.g., "Run `scripts/download_data.sh` or download from [URL]."]
   - Datasets are stored in `/data/` directory.

## Usage and Reproduction Steps
### Quick Start
To run a basic test:
```
python main.py --test
```
Expected output: [Describe, e.g., "Prints 'Test successful' and generates a sample plot in /outputs/.]

### Reproducing Key Results
Follow these steps to reproduce the main experiments. Estimated time: [e.g., 2 hours on recommended hardware].

1. **Prepare Environment**:
   - Ensure all dependencies are installed.
   - Set environment variables if needed: [e.g., `export CUDA_VISIBLE_DEVICES=0`]

2. **Run Experiment 1: [Description, e.g., Baseline Comparison]**:
   ```
   python scripts/experiment1.py --input data/input1.csv --output results/exp1/
   ```
   - Input: [Describe files/params]
   - Output: [e.g., Generates CSV and plots in /results/exp1/]
   - Expected Results: [Match paper, e.g., "Accuracy: 92.5% ± 0.5%"]
   - Time: [e.g., 30 minutes]

3. **Run Experiment 2: [Description]**:
   ```
   bash scripts/run_exp2.sh
   ```
   - [Detailed steps]
   - Verification: Compare with paper Figure X using `python visualize.py results/exp2/`

4. **Full Reproduction**:
   - Run all experiments: `./run_all.sh`
   - Generate report: `python generate_report.py`

### Customizing Experiments
- Parameters: Edit `config.yaml` or use command-line flags (see `--help`).
- Troubleshooting: [Common issues, e.g., "If CUDA error: Install compatible drivers."]

## Evaluation Criteria
- **Functionality**: All scripts run without errors.
- **Reproducibility**: Results within [e.g., 5%] of paper values (due to randomness; use `--seed 42` for determinism).
- **Documentation**: This README and inline code comments.
- **Ease of Use**: Designed for one-command reproduction where possible.

## Additional Notes
- **License**: [e.g., MIT License; see LICENSE file]
- **Citations**: If using this artifact, cite the paper as: [BibTeX entry]
- **Feedback for Evaluators**: [Any specific instructions, e.g., "Contact us if issues arise."]
- **Known Issues**: [List any, e.g., "Windows support limited; use WSL."]

For questions, open an issue on GitHub or email [your.email@example.com].

Last Updated: August 21, 2025