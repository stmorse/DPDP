#!/bin/bash
# Stage 2 v3: MCTS-guided Self-Play Training for CIMA with Ollama
# Matches paper's Table 6 CIMA hyperparameters:
#   --learning_rate 1e-5 (was 6e-6) — paper's CIMA SPT rate
#   --critic_loss_w 10.0 (was 1.0) — paper's lambda_2 for CIMA
#   --max_steps 3 (was 10) — paper uses 3 epochs for CIMA
#   --target_update_count 5 (kept from v2)
#   --remark with_pt_v3 — separate output directory
# Run: bash CIMA/run_spt_ollama_v2.sh > CIMA/spt_train_v3.log 2>&1 &

cd /home/stmorse/projects/DPDP/CIMA
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

python -m ppdpp.run \
    --mode train \
    --do_train \
    --epochs 50000 \
    --gamma 0.999 \
    --lmbda 0.95 \
    --eps 0.2 \
    --learning_rate 1e-5 \
    --data_name cima \
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
    --max_steps 3 \
    --sample_times 100 \
    --eval_start_index 0 \
    --eval_sample_times 113 \
    --eval_num 1 \
    --save_num 1 \
    --output_dir ppdpp \
    --sft_dir roberta_10.0_200_6e-06_0.01_1e-08_1.0_10.0_1.0_0.1_-1_None \
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
    --success_base 1.0 \
    --critic_loss_w 10.0 \
    --remark with_pt_v3
