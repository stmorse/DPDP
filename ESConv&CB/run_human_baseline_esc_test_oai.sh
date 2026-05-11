#!/bin/bash
# Human baseline on ESConv TEST split (n=130) using gpt-4o-mini.
# Mirrors the existing valid-split run methodology (early-stop on mean reward >= 0.1)
# but with new file paths so logs don't collide. Concurrent-safe with the llama variant.
#
# Required env (set before invocation):
#   OPENAI_API_KEY=sk-...
#
# Run:
#   export OPENAI_API_KEY=sk-...
#   bash ESConv\&CB/run_human_baseline_esc_test_oai.sh \
#     > ESConv\&CB/human_baseline_esc_test_oai.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini

python run_human_baseline_esc.py \
    --data_path data/esc-test.txt \
    --max_turn 8 \
    --n_critic_samples 10 \
    --max_tokens 16 \
    --temperature 1.1 \
    --out_log human_baseline_esc_test_oai_perdialog.log
