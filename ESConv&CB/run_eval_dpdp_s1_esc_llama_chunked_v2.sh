#!/bin/bash
# v2: re-run with env.py:70 fix (test_num now honors --eval_start_index).
# Original chunked logs had each chunk re-evaluating dataset[0..32] / [0..31]
# four times instead of slicing 0..32 / 33..65 / 66..97 / 98..129.
#
# Eval: DPDP-S1 only (policy alone, no MCTS) on ESConv test, llama3.2 backend
# via ollama:11437. Hardcoded port.
#
# Uses sft/esc/llama_best_v1 symlink -> sft_llama/esc/roberta_10.0_400_..._-1.0
# which is the llama-trained Stage 1 best_checkpoint (Epoch 2, SR=0.885 valid).
#
# Action-selection mode: --mcts_applied_ratio 0.0 -> policy classifier picks the
# action directly, MCTS is never invoked.
#
# Writes to *_v2_{tag}.log so we don't append to the buggy v1 logs.
# --remark s1_llama_v2_${tag} so tmp/.../eval_result dirs don't collide.
#
# Run:
#   bash ESConv\&CB/run_eval_dpdp_s1_esc_llama_chunked_v2.sh \
#     > ESConv\&CB/eval_dpdp_s1_esc_llama_v2_chunked.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

export LLM_BASE_URL=http://localhost:11437/v1
export LLM_MODEL=llama3.2:latest
export OPENAI_API_KEY=ollama

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
      --sft_dir llama_best_v1 \
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
      --remark s1_llama_v2_${tag} \
      > eval_dpdp_s1_esc_llama_v2_${tag}.log 2>&1 &
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
  "eval_dpdp_s1_esc_llama_v2_c0.log:33" \
  "eval_dpdp_s1_esc_llama_v2_c1.log:33" \
  "eval_dpdp_s1_esc_llama_v2_c2.log:32" \
  "eval_dpdp_s1_esc_llama_v2_c3.log:32"
