#!/bin/bash

# Define the range of batch sizes you want to test
batch_sizes=(1 2 4 8)

# Loop over each batch size
for batch_size in "${batch_sizes[@]}"
do
    echo "Running training with batch size: $batch_size"
    accelerate launch run_wav2vec2_pretraining_no_trainer.py \
        --dataset_name="kojima-lab/bird-jp-excl-test" \
        --dataset_split_names train \
        --dataset_config_names clean \
        --validation_split_percentage 10 \
        --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
        --output_dir="./wav2vec2-birddb/batch_${batch_size}" \
        --max_train_steps="20000" \
        --num_warmup_steps="32000" \
        --learning_rate="0.005" \
        --weight_decay="0.01" \
        --max_duration_in_seconds="100.0" \
        --min_duration_in_seconds="0.2" \
        --logging_steps="100" \
        --saving_steps="100000" \
        --per_device_train_batch_size="$batch_size" \
        --per_device_eval_batch_size="$batch_size" \
        --adam_beta1="0.9" \
        --adam_beta2="0.98" \
        --adam_epsilon="1e-06" \
        --gradient_checkpointin
done