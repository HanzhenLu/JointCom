# 1. Finetune the Retriever
```
# For java
lang=java
python finetune_retriever.py \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/unixcoder-base \
    --train_data_file ../../dataset/$lang/train.jsonl \
	--eval_data_file ../../dataset/$lang/valid.jsonl \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_train_epochs 10 \
    --output_dir saved_models/$lang
```