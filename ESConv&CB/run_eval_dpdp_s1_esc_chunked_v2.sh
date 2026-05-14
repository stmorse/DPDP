#!/bin/bash
# v2: standalone DPDP-S1-only eval on ESConv test (130), gpt-4o-mini backend.
# The 0.846 SR figure was sourced from Stage 1 SFT per-epoch eval at Epoch 3,
# which evaluated the full test set correctly (bug doesn't affect start_index=0
# runs) — but this script gives us a fresh, parity-matching standalone eval for
# direct comparison with the other v2 numbers and AvgT/SD breakdown.
#
# Loads Stage 1 best_checkpoint from sft/esc/roberta_10.0_400_..._-1.0/best_checkpoint/
# (saved at Epoch 3 by run_pt_esc_ollama.sh).
#
# Action-selection mode: --mcts_applied_ratio 0.0 -> policy classifier picks the
# action directly, MCTS is never invoked. NOT --skip_policy_load: we need the
# trained policy. Per-process VRAM ~3.3 GB for the roberta load.
#
# Writes to *_v2_{tag}.log and uses --remark dpdp_s1_v2_${tag} to keep clean of
# any earlier (nonexistent — this combo was never run standalone) artifacts.
#
# Run (expects OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL already exported):
#   export OPENAI_API_KEY=sk-...
#   export LLM_BASE_URL=https://api.openai.com/v1
#   export LLM_MODEL=gpt-4o-mini
#   bash ESConv\&CB/run_eval_dpdp_s1_esc_chunked_v2.sh \
#     > ESConv\&CB/eval_dpdp_s1_esc_v2_chunked.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

# Sanity check: refuse if LLM_BASE_URL points at localhost. This script targets
# the OpenAI gpt-4o-mini path; use run_eval_dpdp_s1_esc_llama_chunked_v2.sh for
# the ollama path.
if [[ -z "$LLM_BASE_URL" ]]; then
  echo "ERROR: LLM_BASE_URL is not set. Expected https://api.openai.com/v1" >&2
  exit 1
fi
if [[ "$LLM_BASE_URL" == *"localhost"* ]]; then
  echo "ERROR: LLM_BASE_URL points at localhost ($LLM_BASE_URL)." >&2
  echo "  This v2 script is for OpenAI gpt-4o-mini. Use the _llama_ v2 for ollama." >&2
  exit 1
fi
echo "Using LLM_BASE_URL=$LLM_BASE_URL  LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}"

CHUNKS=(
  "0 33 c0"
  "33 33 c1"
  "66 32 c2"
  "98 32 c3"
)

PIDS=()
for c in "${CHUNKS[@]}"; do
  read start n tag <<< "$c"
  echo "[$(date +%T)] launching chunk $tag: start=$start n=$n"
  python -m ppdpp.run \
      --mode train \
      --do_eval \
      --epochs 50000 \
      --gamma 0.999 \
      --lmbda 0.95 \
      --eps 0.2 \
      --learning_rate 1e-6 \
      --data_name esc \
      --system ollama \
      --user ollama \
      --critic ollama \
      --planner ollama \
      --max_turn 8 \
      --max_seq_length 512 \
      --model_name roberta \
      --model_name_or_path roberta-large \
      --model_path lmsys/vicuna-7b-v1.5 \
      --start_step 0 \
      --max_steps 5 \
      --sample_times 100 \
      --eval_start_index $start \
      --eval_sample_times $n \
      --eval_num 1 \
      --save_num 1 \
      --output_dir ppdpp \
      --sft_dir roberta_10.0_400_6e-06_0.001_1e-08_1.0_1.0_0.1_-1.0 \
      --zero_shot \
      --device_id 0 \
      --num_gpus 1 \
      --use_mcts_sys_resp \
      --use_mcts_usr_resp \
      --dropout 0.25 \
      --mcts_applied_ratio 0.0 \
      --num_mcts_sims 2 \
      --max_realizations 1 \
      --max_conv_turns 9 \
      --max_hist_num_turns 8 \
      --resp_max_new_tokens 64 \
      --reward_max_new_tokens 16 \
      --action_temperature 1.0 \
      --resp_temperature 0.7 \
      --reward_temperature 1.1 \
      --action_num_return_sequences 15 \
      --reward_num_return_sequences 10 \
      --train_batch_size 4 \
      --target_update_count -1 \
      --critic_loss_w 1.0 \
      --remark dpdp_s1_v2_${tag} \
      > eval_dpdp_s1_esc_v2_${tag}.log 2>&1 &
  PIDS+=($!)
  echo "  pid=$!"
  sleep 5
done

echo "[$(date +%T)] waiting on chunks: ${PIDS[*]}"
EXIT_CODES=()
for pid in "${PIDS[@]}"; do
  wait $pid
  rc=$?
  EXIT_CODES+=($rc)
  echo "[$(date +%T)] pid $pid exited rc=$rc"
done

echo "[$(date +%T)] all chunks done. exit codes: ${EXIT_CODES[*]}"
echo "[$(date +%T)] aggregating..."

python aggregate_chunks_esc.py \
  "eval_dpdp_s1_esc_v2_c0.log:33" \
  "eval_dpdp_s1_esc_v2_c1.log:33" \
  "eval_dpdp_s1_esc_v2_c2.log:32" \
  "eval_dpdp_s1_esc_v2_c3.log:32"
