#!/bin/bash
# v2: re-run with env.py:70 fix (test_num now honors --eval_start_index).
# Original chunked logs had each chunk re-evaluating dataset[0..32] / [0..31]
# four times instead of slicing 0..32 / 33..65 / 66..97 / 98..129.
#
# Eval: DPDP-S1+S2 (policy prior + MCTS) on ESConv test (130), gpt-4o-mini
# backend via OpenAI API.
#
# Loads Stage 1 best_checkpoint via --sft_dir and Stage 2 RL checkpoint via
# --checkpoint_path (bypasses the auto-built filename which would embed
# mcts_applied_ratio=1.0 — but Stage 2 was trained with 0.0).
#
# Writes to *_v2_{tag}.log so we don't append to the buggy v1 logs.
# --remark dpdp_e5_v2_${tag} so tmp/.../eval_result dirs don't collide.
#
# Run (expects OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL already exported):
#   export OPENAI_API_KEY=sk-...
#   export LLM_BASE_URL=https://api.openai.com/v1
#   export LLM_MODEL=gpt-4o-mini
#   bash ESConv\&CB/run_eval_dpdp_esc_chunked_v2.sh \
#     > ESConv\&CB/eval_dpdp_esc_v2_chunked.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

# Sanity check: refuse if LLM_BASE_URL points at localhost. This script targets
# the OpenAI gpt-4o-mini path; use the _llama_ v2 for ollama.
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

# Stage 2 checkpoint to evaluate (epoch 5, the final one).
RL_CKPT="esc-roberta_10.0_400_6e-06_0.001_1e-08_1.0_1.0_0.1_-1.0-ollama-ollama-ollama-0.0-5-5-0.25-0.2-epoch-5"

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
      --checkpoint_path $RL_CKPT \
      --zero_shot \
      --device_id 0 \
      --num_gpus 1 \
      --use_mcts_sys_resp \
      --use_mcts_usr_resp \
      --use_policy_prior \
      --dropout 0.25 \
      --mcts_applied_ratio 1.0 \
      --num_mcts_sims 10 \
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
      --target_update_count 5 \
      --critic_loss_w 1.0 \
      --remark dpdp_e5_v2_${tag} \
      > eval_dpdp_esc_v2_e5_${tag}.log 2>&1 &
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
  "eval_dpdp_esc_v2_e5_c0.log:33" \
  "eval_dpdp_esc_v2_e5_c1.log:33" \
  "eval_dpdp_esc_v2_e5_c2.log:32" \
  "eval_dpdp_esc_v2_e5_c3.log:32"
