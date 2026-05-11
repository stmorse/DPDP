#!/bin/bash
# Cross-model judge calibration on ESConv valid (130 dialogues, all 8 turns each).
# Compares gpt-4o-mini (OpenAI) and llama3.2 (local ollama) on the SAME
# conversation prefixes, scored with the SAME judge prompt. See JUDGE_SPEC.md
# for what's being measured.
#
# Required env:
#   OPENAI_API_KEY=sk-...
#   LLAMA_URL=http://localhost:80/v1   (override default if your ollama listens elsewhere)
#   LLAMA_MODEL=llama3.2:latest        (override if you want a different ollama tag)
#
# Run: bash ESConv\&CB/run_calibrate_judge_esc.sh > ESConv\&CB/judge_calibration_esc.run.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

LLAMA_URL=${LLAMA_URL:-http://localhost:11434/v1}
LLAMA_MODEL=${LLAMA_MODEL:-llama3.2:latest}
DATA_PATH=${DATA_PATH:-data/esc-valid.txt}
LIMIT=${LIMIT:--1}
# TAG suffixes the output files so v1 (no template) results aren't clobbered
# when re-running with --critic_template variants.
TAG=${TAG:-v1}
# LLAMA_TEMPLATE: optional path to a critic-prompt template for the llama
# judge only. Empty = use codebase default (apples-to-apples with v1 pilot).
LLAMA_TEMPLATE=${LLAMA_TEMPLATE:-}

LLAMA_SPEC="name=llama,url=${LLAMA_URL},model=${LLAMA_MODEL},api_key=ollama,loop=true"
if [ -n "$LLAMA_TEMPLATE" ]; then
  LLAMA_SPEC="${LLAMA_SPEC},template=${LLAMA_TEMPLATE}"
fi

python calibrate_judge_esc.py \
  --data_path "$DATA_PATH" \
  --limit "$LIMIT" \
  --out_jsonl "judge_calibration_esc_${TAG}.jsonl" \
  --out_summary "judge_calibration_esc_${TAG}.summary.txt" \
  --judge "name=gpt,url=https://api.openai.com/v1,model=gpt-4o-mini,api_key_env=OPENAI_API_KEY,loop=false" \
  --judge "$LLAMA_SPEC"
