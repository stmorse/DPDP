#!/bin/bash
# Stage 1 (PT/SFT pretraining) for ESConv with llama3.2 via ollama on port 11438.
# Mirrors run_pt_esc_ollama.sh hyperparameters (Table 6, ESConv) but:
#   - routes per-epoch eval LLM (system/user/critic/planner) to ollama:11438
#   - writes checkpoint to sft_llama/ so existing gpt-trained sft/ checkpoint is preserved
#
# Output checkpoint path:
#   sft_llama/esc/roberta_10_400_6e-06_0.001_1e-08_1.0_1.0_0.1_-1.0/best_checkpoint/
#
# Port and model are hardcoded here (no fallback to outer LLAMA_URL env var) so this
# wrapper is collision-safe even if a concurrent run has LLAMA_URL set in the shell.
# To override, edit this script.
#
# Run:
#   bash ESConv\&CB/run_pt_esc_llama_p11438.sh \
#     > ESConv\&CB/sft_train_esc_llama_p11438.log 2>&1 &

cd "/home/stmorse/projects/DPDP/ESConv&CB"
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)

export LLM_BASE_URL=http://localhost:11438/v1
export LLM_MODEL=llama3.2:latest
export OPENAI_API_KEY=ollama

python -m ppdpp.sft \
    --data_name esc \
    --set_name valid \
    --system ollama \
    --user ollama \
    --critic ollama \
    --planner ollama \
    --model_name roberta \
    --model_name_or_path roberta-large \
    --output_dir sft_llama \
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
