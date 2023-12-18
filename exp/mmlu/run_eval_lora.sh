export CUDA_VISIBLE_DEVICES=0,1,2,3 #,1,2,3,4,5,6,7

python eval_mmlu.py \
    -m  /data/share/pretrained/llama-30b \
    -s ./results/llama-30b/lora1-0shot \
    --use_peft True \
    --peft_dir /data/share/vc/stanford_alpaca/tmp/llama-30b/lora1 \
    -k 0