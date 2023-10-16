# 1. train the Retriever
```
# For java
lang=java
python train_retriever.py \
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

# 2. Retrieve relevant code and comment
```
# For java
lang=java
python retrieval.py \
    --dataset $lang \
    --model_name_or_path microsoft/unixcoder-base \
    --nl_length 128 \
    --code_length 256 \
    --retrieve_batch_size 256 \
    --output_dir dataset
```

# 3. Train the generator
```
# For java
lang=java
python train_generator.py \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/$lang \
    --train_filename dataset/$lang/train.jsonl \
    --dev_filename dataset/$lang/valid.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --beam_size 10 \
    --num_train_epochs 10
```

# 4. Generate the predictions
```
# For java
lang=java
python train_generator.py \
    --do_test \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/$lang \
    --test_filename dataset/$lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32
```
