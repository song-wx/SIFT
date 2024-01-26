export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3 

deepspeed --num_gpus=4 --master_port=6000 train.py \
    --model_name_or_path /data/share/pretrained/llama-30b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./tmp/llama-30b/lora \
    --logging_dir ./tmp/llama-30b/lora \
    --logging_strategy steps \
    --logging_first_step True \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --finetuning_type "lora" \
    --lora_r 128 \
    --lora_alpha 256 \
    --deepspeed "./configs/default_offload_opt_param.json" 