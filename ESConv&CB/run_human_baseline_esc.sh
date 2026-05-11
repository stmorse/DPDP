#!/bin/bash
# Human baseline for ESConv valid: replay gold dialogues through the critic.
#
# Backend is selected via env vars (matches the env-based eval scripts):
#   - Default (no env vars): local ollama at localhost:11434, model llama3.2:latest
#   - OpenAI:
#       export OPENAI_API_KEY=sk-...
#       export LLM_BASE_URL=https://api.openai.com/v1
#       export LLM_MODEL=gpt-4o-mini      # or gpt-5-nano, gpt-3.5-turbo, ...
#
# Run: bash ESConv\&CB/run_human_baseline_esc.sh > ESConv\&CB/human_baseline_esc.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

python run_human_baseline_esc.py \
    --data_path data/esc-valid.txt \
    --max_turn 8 \
    --n_critic_samples 10 \
    --max_tokens 16 \
    --temperature 1.1 \
    --out_log human_baseline_esc_perdialog.log
