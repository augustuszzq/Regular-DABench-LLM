# Artifact Evaluation README for Regular-DABench-LLM: Standardized and In-Depth Benchmarking of Post-Moore Dataflow AI Accelerators for LLMs

## Overview
This repository contains the artifacts for the paper titled  Regular-DABench-LLM: Standardized and In-Depth Benchmarking of Post-Moore Dataflow AI Accelerators for LLMs by Ziyu Hu and Zhiqing Zhong. These artifacts are submitted for Artifact Evaluation (AE) to demonstrate the reproducibility of the results presented in the paper.


The artifacts include source code, datasets, scripts, and documentation necessary to replicate the experiments. We aim for the **Available**, **Functional**, and **Reproduced** badges .

## Artifact Description
### Abstract
This artifact provides a comprehensive collection of test scripts designed to evaluate performance across all our hardware configurations, including Graphcore IPU, Cerebras, and SambaNova systems. It also includes references to key articles and papers that informed the development and validation of these tests, enabling reproducibility of the experiments described in the paper.
Additionally, the models used in this artifact are sourced from the official GitHub repository at. This ensures that all experiments are based on the standard, unmodified implementations provided by the developers.

#### Models and Sources
The models evaluated in this artifact are derived from the following official GitHub repositories, corresponding to the supported hardware configurations:

- **Graphcore IPU**: Sourced from https://github.com/graphcore/examples.git. This repository provides example models, tutorials, and optimized implementations for IPU hardware, including pre-trained models for various machine learning tasks.
- **SambaNova**: Sourced from https://github.com/sambanova/tutorials/tree/main. This branch includes tutorials and model examples tailored for SambaNova's Reconfigurable Dataflow Units (RDUs), enabling dataflow-optimized executions.
- **Cerebras**: Sourced from https://github.com/Cerebras/modelzoo.git. This repository contains a collection of models (e.g., Transformer-based architectures) designed for the Cerebras CS-2 wafer-scale engine, with scripts for training and inference.

These sources ensure that all experiments use standard, unmodified official implementations. Specific commits or tags used: [If applicable, add details like "main branch" or a commit hash; otherwise, evaluators can use the latest stable version]. Models are loaded by cloning the respective repositories and following the provided setup instructions.


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

Since this artifact relies on hardware provided by Argonne National Laboratory (ALCF) AI Testbed, installation and setup are performed within the ALCF environment. Users must have an ALCF account with Multi-Factor Authentication (MFA) enabled (e.g., via MobilePASS+ or CRYPTOCard). The process involves accessing the respective hardware nodes, cloning the repository, and configuring the environment specific to each hardware type (Graphcore IPU, Cerebras CS-2, SambaNova SN30). Below are detailed steps for each.

### General Prerequisites
- **ALCF Account**: Request an account via the ALCF "Get Started" guide if you don't have one. Enable MFA for secure access.
- **SSH Access**: Use SSH from your local machine with your ALCF user ID and MFA-generated passcode.
- **Repository Cloning**: All setups start by cloning this GitHub repository on the ALCF nodes.
- **Dependencies**: Python-based; key packages include those for ML frameworks (e.g., PyTorch, TensorFlow). Specific versions and installations vary by hardware.

### 1. Access ALCF Systems
Access follows a two-step SSH process for most hardware:
   - From your local machine: `ssh ALCFUserID@ai.alcf.anl.gov` (replace `ALCFUserID` with your ID; enter MFA passcode as password). Use `-v` for debugging (e.g., `ssh -v ALCFUserID@ai.alcf.anl.gov`).
   - Once on the login node, SSH to the specific hardware node (details below).

### 2. Clone the Repository
On the target hardware node (after accessing it):
   ```
   git clone https://github.com/augustuszzq/Regular-DABench-LLM.git
   cd Regular-DABench-LLM
   ```

### 3. Hardware-Specific Setup and Installation

#### Graphcore IPU
- **Access Nodes**: From the login node, SSH to an accessible Graphcore node, e.g., `ssh gc-poplar-02.ai.alcf.anl.gov`, `ssh gc-poplar-03.ai.alcf.anl.gov`, or `ssh gc-poplar-04.ai.alcf.anl.gov`. Note: `gc-poplar-01.ai.alcf.anl.gov` is not directly accessible; its IPU resources are allocated via Slurm.
- **Environment Setup**: The Poplar SDK and related tools (e.g., PopTorch for PyTorch integration, PopLibs for tensor operations) are pre-configured in the ALCF environment. No manual setup is needed beyond login.
- **Install Dependencies**:
  - Activate a Python virtual environment if desired: 
    ```
    python -m venv env
    source env/bin/activate
    ```
  - Install project-specific packages: `pip install -r requirements.txt`.
  - Key packages: numpy==1.26.0, torch==2.0.1 (optimized via PopTorch), and Graphcore-specific libraries (pre-installed via Poplar SDK).
- **Job Submission**: Use Slurm for running jobs, e.g., `sbatch your_script.slurm` to allocate IPU resources.
- **Troubleshooting**: If SSH issues arise, use `-v` flag. For software, refer to Poplar SDK documentation.

#### Cerebras CS-2
- **Access Nodes**: From the login node, SSH to a Cerebras login node (specific node names like cs2-login-01.ai.alcf.anl.gov; use random assignment if multiple). Authentication uses MFA passcode.
- **Environment Setup**: The Cerebras software stack is managed as an appliance, including PyTorch integration for model compilation and execution. Environment variables and tools are auto-set upon login. Includes support for data preprocessing, streaming, and orchestration via MemoryX and SwarmX nodes.
- **Install Dependencies**:
  - Create a virtual environment:
    ```
    python -m venv env
    source env/bin/activate
    ```
  - Install via: `pip install -r requirements.txt`.
  - Key packages: torch==2.0.1 (Cerebras-optimized), numpy==1.26.0, and dependencies for wafer-scale processing (pre-integrated).
- **Job Submission**: Submit workflows via Cerebras-specific commands or Slurm; details in ALCF workflows section (e.g., for training/inference on the wafer-scale cluster).
- **Troubleshooting**: Check MFA setup; refer to ALCF support for node access issues.

#### SambaNova SN30
- **Access Nodes**: From the login node, SSH to a SambaNova node using aliases like `ssh sn30-r1-h1` (format: sn30-r[1-4]-h[1-2], where r=rack, h=host).
- **Environment Setup**: The SambaFlow software stack is automatically set up upon login, including environmental variables. SambaFlow optimizes dataflow graphs for RDUs and integrates with PyTorch.
- **Install Dependencies**:
  - Virtual environment (optional):
    ```
    python -m venv env
    source env/bin/activate
    ```
  - Run: `pip install -r requirements.txt`.
  - Key packages: torch==2.0.1 (SambaNova-integrated), numpy==1.26.0, and SambaFlow-specific tools (pre-installed).
- **Job Submission**: Use standard commands for model parallelism; refer to SambaNova tutorials for running on RDUs.
- **Troubleshooting**: Debug SSH with `-v`; consult ALCF for account issues.

### 4. Download Datasets (if not included)
- Run the provided script: `bash scripts/download_data.sh` (or similar, depending on your repo structure).
- Alternatively, download manually from official sources (e.g., via wget or curl) and place in `/data/` directory.
- Datasets are stored in `/data/`. For large datasets, use ALCF storage systems to avoid node limits.

This setup ensures compatibility with ALCF's hardware. For full details, refer to ALCF documentation: [Cerebras](https://docs.alcf.anl.gov/ai-testbed/cerebras/getting-started/), [Graphcore](https://docs.alcf.anl.gov/ai-testbed/graphcore/getting-started/), [SambaNova](https://docs.alcf.anl.gov/ai-testbed/sambanova/getting-started/). If issues arise, contact ALCF support.

## Usage and Reproduction Steps
### Quick Start
To run a basic test on graphcore:
```
cd graphcore
./full_run_benchmark_scalling.sh
python ana.py
```




For questions, open an issue on GitHub.

Last Updated: August 21, 2025