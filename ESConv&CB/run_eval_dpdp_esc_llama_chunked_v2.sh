#!/bin/bash
# v2: re-run with env.py:70 fix (test_num now honors --eval_start_index).
# Original chunked logs had each chunk re-evaluating dataset[0..32] / [0..31]
# four times instead of slicing 0..32 / 33..65 / 66..97 / 98..129.
#
# Eval: DPDP-S1+S2 (policy prior + MCTS) on ESConv test, llama3.2 backend
# via ollama:11437. Hardcoded port. (Switched from 11438 to 11437 since the
# earlier DPDP-S1 llama v2 run vacated 11437 and 11438 has the busier load.)
#
# Loads Stage 2 RL checkpoint via --checkpoint_path (bypasses the auto-built
# filename which would embed mcts_applied_ratio=1.0 — but Stage 2 was trained
# with 0.0, so the filename doesn't line up via --load_rl_epoch).
#
# Writes to *_v2_{tag}.log so we don't append to the buggy v1 logs.
# --remark with_pt_llama_v2_${tag} so each chunk gets its own tmp/.../eval_result
# directory (the v1 script used a bare remark with no tag — fixing in v2 for
# parity with the gpt-side chunked script).
#
# RL_EPOCH defaults to 5 (final epoch); override via env var LOAD_RL_EPOCH.
#
# Run:
#   bash ESConv\&CB/run_eval_dpdp_esc_llama_chunked_v2.sh \
#     > ESConv\&CB/eval_dpdp_esc_llama_v2_chunked.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

export LLM_BASE_URL=http://localhost:11437/v1
export LLM_MODEL=llama3.2:latest
export OPENAI_API_KEY=ollama

RL_EPOCH=${LOAD_RL_EPOCH:-5}
RL_CKPT="esc-llama_best_v1-ollama-ollama-ollama-0.0-5-5-0.25-0.2-epoch-${RL_EPOCH}"
echo "Using RL epoch $RL_EPOCH from with_pt_llama checkpoint"

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
      --remark with_pt_llama_v2_${tag} \
      --checkpoint_path "$RL_CKPT" \
      > eval_dpdp_esc_llama_v2_e${RL_EPOCH}_${tag}.log 2>&1 &
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
  "eval_dpdp_esc_llama_v2_e${RL_EPOCH}_c0.log:33" \
  "eval_dpdp_esc_llama_v2_e${RL_EPOCH}_c1.log:33" \
  "eval_dpdp_esc_llama_v2_e${RL_EPOCH}_c2.log:32" \
  "eval_dpdp_esc_llama_v2_e${RL_EPOCH}_c3.log:32"
