#!/bin/bash
# Human baseline on ESConv TEST split (n=130) using llama3.2 via local ollama.
# Mirrors the existing valid-split run methodology (early-stop on mean reward >= 0.1)
# but with new file paths so logs don't collide. Concurrent-safe with the OAI variant.
#
# Optional env overrides:
#   LLAMA_URL=http://localhost:11434/v1
#   LLAMA_MODEL=llama3.2:latest
#
# Run:
#   bash ESConv\&CB/run_human_baseline_esc_test_llama.sh \
#     > ESConv\&CB/human_baseline_esc_test_llama.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

export LLM_BASE_URL=${LLAMA_URL:-http://localhost:11434/v1}
export LLM_MODEL=${LLAMA_MODEL:-llama3.2:latest}

python run_human_baseline_esc.py \
    --data_path data/esc-test.txt \
    --max_turn 8 \
    --n_critic_samples 10 \
    --max_tokens 16 \
    --temperature 1.1 \
    --out_log human_baseline_esc_test_llama_perdialog.log
