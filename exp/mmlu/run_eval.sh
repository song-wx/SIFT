export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7

python eval_mmlu.py \
    -m  /data/share/vc/stanford_alpaca/tmp/llama-7b/sparse \
    -s ./results/llama-7b/sparse-0shot \
    -k 0 
