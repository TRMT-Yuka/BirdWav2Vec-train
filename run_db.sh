#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
for i in 8
    do
        echo "Running training with batch size: ${i}"
        accelerate launch run_wav2vec2_pretraining_no_trainer.py \
            --dataset_name="kojima-lab/bird-jp-excl-test" \
            --dataset_split_names train \
            --dataset_config_names clean \
            --validation_split_percentage 10 \
            --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
            --output_dir="./wav2vec2-birddb/batch_${i}_5" \
            --max_train_steps="30000" \
            --num_warmup_steps="48000" \
            --learning_rate="0.00005" \
            --weight_decay="0.01" \
            --max_duration_in_seconds="100.0" \
            --min_duration_in_seconds="0.2" \
            --logging_steps="100" \
            --saving_steps="100000" \
            --per_device_train_batch_size="${i}" \
            --per_device_eval_batch_size="${i}" \
            --adam_beta1="0.9" \
            --adam_beta2="0.98" \
            --adam_epsilon="1e-06" \
            --gradient_checkpointing
    done

