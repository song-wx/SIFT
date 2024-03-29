export TASK=qnli
export CUDA_VISIBLE_DEVICES=0

python run_glue.py \
    --model_name_or_path /data/share/pretrained/roberta-large \
    --task_name $TASK \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --output_dir ./result/$TASK/roberta-large/ \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --weight_decay 0.1 \
    --warmup_ratio 0.06 \
    --finetuning_type sup \
    --sparse_rate 0.008 \
    --sparse_module query value key attention.output.dense \
    --sparse_exception classifier \
    --seed 42 \
    # --gradient_checkpointing True \
