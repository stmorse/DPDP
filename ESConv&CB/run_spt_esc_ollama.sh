#!/bin/bash
# Stage 2: MCTS-guided self-play training for ESConv with Ollama backend.
# Paper hyperparameters (Table 6, ESConv column):
#   --learning_rate 1e-6, --critic_loss_w 1.0, --max_steps 5
#   --target_update_count 5
# Uses Stage 1 SFT checkpoint at:
#   sft/esc/roberta_10_400_6e-06_0.001_1e-08_1.0_1.0_0.1_-1.0/best_checkpoint/
# Run: bash ESConv\&CB/run_spt_esc_ollama.sh > ESConv\&CB/spt_train_esc.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

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
    --eval_num 1 \
    --save_num 1 \
    --output_dir ppdpp \
    --sft_dir roberta_10_400_6e-06_0.001_1e-08_1.0_1.0_0.1_-1.0 \
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
    --remark with_pt
