# Compile and Train GPT Model

## Quick Start

```bash
# Clone repository
cd $HOME
git clone https://github.com/augustuszzq/dataflow-arch-test.git
cd dataflow-arch-test/sambaNova/Gpt

# Set environment variables (example only, adjust to your paths)
export PEF=<path/to/pef>
export DATADIR=<path/to/dataset>

# Run training
SAMBA_SEED=256 python3 generative_train.py run   --max_seq_length 1024   --batch-size 16   --config_name configuration/gpt2_small_config.json   --weight_decay 0.1   --max_grad_norm_clip 1.0   --data_dir $DATADIR/generative_tuning_sst2/hdf5/   --checkpoint_name $OUTPUT_FOLDER/train_movie_review_checkpoint.pt   --model_name_or_path gpt2   --steps 800   --min_eval_acc 0.87   --log-level error   --pef $PEF
```

---

## 1. Check Python Version
Requires **Python 3.9** or later.

```bash
python3 --version
```

---

## 2. Check SambaFlow Installation
You must have the `sambaflow` package installed (**v1.17 or later**).

**Check installation:**
- On **Ubuntu**:
  ```bash
  dpkg -s sambaflow
  ```
- On **RHEL**:
  ```bash
  rpm -qi sambaflow
  ```

If `sambaflow` is missing, please contact your system administrator.  
Ensure the installed version matches the documentation.

---

## 3. Clone the Repository
```bash
cd $HOME
git clone https://github.com/augustuszzq/dataflow-arch-test.git
cd dataflow-arch-test/sambaNova/Gpt
```

**Important files/folders:**
- `generative_train.py` — Python code for training the model  
- `configuration/` — contains configs for training & inference  
  - Example: `gpt2_small_config.json` overrides Hugging Face settings for better RDU performance  

---

## 4. Download the Dataset
Follow instructions here:  
👉 [Generative Data Prep](https://github.com/sambanova/generative_data_prep)

---

## 5. Train the Model

Before starting, ensure you have:
- A dataset compatible with the model  
- A config file (e.g., `gpt2_small_config.json`)  
- The training script: `generative_train.py`  

> ⚡ SambaFlow’s `run` command supports:
> - **Training** (default)  
> - **Inference** (`--inference`)  

---

### Step 1. Set environment variables
*(values depend on your environment, do not copy directly)*

```bash
export PEF=<path/to/pef>
export DATADIR=<path/to/dataset>
```

---

### Step 2. Run the training script
```bash
SAMBA_SEED=256 python3 generative_train.py run   --max_seq_length 1024   --batch-size 16   --config_name configuration/gpt2_small_config.json   --weight_decay 0.1   --max_grad_norm_clip 1.0   --data_dir $DATADIR/generative_tuning_sst2/hdf5/   --checkpoint_name $OUTPUT_FOLDER/train_movie_review_checkpoint.pt   --model_name_or_path gpt2   --steps 800   --min_eval_acc 0.87   --log-level error   --pef $PEF
```

---

## 6. Key Training Arguments
- **`--pef`**: Required; specifies PEF file generated during compilation.  
- **`--data_dir`**: Location of dataset (after preprocessing).  
- **`--checkpoint_name`**: Checkpoint filename for saving/restoring.  
  - Saved per step (e.g., `800.pt`)  
  - Useful for fault tolerance & incremental training  
- **`--steps`**: Number of iterations (forward + backward + optimization).  
- **`--min_eval_acc`**: Minimum accuracy for evaluation runs to validate correctness.  

---
