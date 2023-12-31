#!/bin/bash
set -x

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
export CUDA_VISIBLE_DEVICES=4,5,6,7
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port src/run_s2s.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path /data/private/tianrunchu/ckpt/t5-lm-3b/ \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --add_method heuristic \
    --tk_instruct False \
    --data_dir data/splits/plus_alpaca \
    --task_dir data/tasks \
    --output_dir ../models/t5-lm-3b-d-heuristic-alpaca/ \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 5000 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5-experiment \
    --report_to none
