#!/bin/bash
# Stage 1: SFT pretraining for ESConv with Ollama backend.
# Paper hyperparameters (Table 6, ESConv column):
#   --learning_rate 6e-6, --warmup_steps 400, --num_train_epochs 10
#   --critic_loss_w 1.0, --dropout 0.1, --weight_decay 0.001
# Run: bash ESConv\&CB/run_pt_esc_ollama.sh > ESConv\&CB/sft_train_esc.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

python -m ppdpp.sft \
    --data_name esc \
    --set_name valid \
    --system ollama \
    --user ollama \
    --critic ollama \
    --planner ollama \
    --model_name roberta \
    --model_name_or_path roberta-large \
    --output_dir sft \
    --data_dir ./data \
    --cache_dir ./plm \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --dropout 0.1 \
    --seed 42 \
    --gpu 0 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 1 \
    --eval_start_index 0 \
    --eval_sample_times 130 \
    --start_epoch 0 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 400 \
    --learning_rate 6e-6 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --local_rank -1 \
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
    --critic_loss_w 1.0
