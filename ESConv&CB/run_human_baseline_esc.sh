#!/bin/bash
# Human baseline for ESConv valid: replay gold dialogues through ollama critic.
# Run: bash ESConv\&CB/run_human_baseline_esc.sh > ESConv\&CB/human_baseline_esc.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

python run_human_baseline_esc.py \
    --data_path data/esc-valid.txt \
    --max_turn 8 \
    --ollama_url http://localhost:11434/v1 \
    --ollama_model llama3.2:latest \
    --n_critic_samples 10 \
    --max_tokens 16 \
    --temperature 1.1 \
    --out_log human_baseline_esc_perdialog.log
