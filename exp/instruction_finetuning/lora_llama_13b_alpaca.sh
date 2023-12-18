export CUDA_VISIBLE_DEVICES=2 

python train.py \
    --model_name_or_path /data/share/pretrained/llama-13b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./tmp/llama-13b/lora \
    --logging_dir ./tmp/llama-13b/lora \
    --logging_strategy steps \
    --logging_first_step True \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --finetuning_type "lora" \
    --lora_r 128 \
    --lora_alpha 256 