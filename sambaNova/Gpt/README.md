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
SAMBA_SEED=256 python3 generative_train.py run \
  --max_seq_length 1024 \
  --batch-size 16 \
  --config_name configuration/gpt2_small_config.json \
  --weight_decay 0.1 \
  --max_grad_norm_clip 1.0 \
  --data_dir $DATADIR/generative_tuning_sst2/hdf5/ \
  --checkpoint_name $OUTPUT_FOLDER/train_movie_review_checkpoint.pt \
  --model_name_or_path gpt2 \
  --steps 800 \
  --min_eval_acc 0.87 \
  --log-level error \
  --pef $PEF
