MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=512
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage1_no_offloading.conf \
    --model_name_or_path TencentARC/LLaMA-Pro-8B \
    --tokenizer_name TencentARC/LLaMA-Pro-8B \
    --use_fast_tokenizer False \
    --train_file /data/ssz/cpm/CPM-9G-8B/9G-Train/datasets.json\
    --max_seq_length 4096 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --preprocessing_num_workers 16 \
    --use_flash_attn \
    --use_checkpointing \
    --output_dir output/LLaMA-Pro-8B-evol-codealpaca \
    --bf16 \
    --tf32 True \
    --overwrite_output_dir \
    --report_to "none" 
