#!/bin/bash
# Stage 2: MCTS-guided self-play training for ESConv on llama3.2 via ollama:11437.
# Seeds from sft/esc/llama_best_v1 -> sft_llama Stage 1 best_checkpoint (Epoch 2).
#
# Paper hyperparameters (Table 6, ESConv column) preserved:
#   --learning_rate 1e-6, --critic_loss_w 1.0, --max_steps 5,
#   --target_update_count 5, --sample_times 100, --num_mcts_sims 10
#
# eval_num=999 effectively disables in-training valid eval (train_step in {1..5}
# never satisfies train_step % 999 == 0). Final test-split eval is a separate
# run after this finishes. Estimated wallclock: ~24-30h.
#
# Output checkpoint directory tag: with_pt_llama
#
# Run:
#   bash ESConv\&CB/run_spt_esc_llama_p11437.sh \
#     > ESConv\&CB/spt_train_esc_llama_p11437.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

export LLM_BASE_URL=http://localhost:11437/v1
export LLM_MODEL=llama3.2:latest
export OPENAI_API_KEY=ollama

python -m ppdpp.run \
    --mode train \
    --do_train \
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
    --eval_start_index 0 \
    --eval_sample_times 130 \
    --eval_num 999 \
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
    --mcts_applied_ratio 0.0 \
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
    --remark with_pt_llama
