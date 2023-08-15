# Retrieval-augmented-Code-Summarization
## 1. Finetune retriever
```
python finetune_retriever.py \
    --output_dir saved_models/retriever/ \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file CSN/java/train.jsonl \
    --eval_data_file CSN/java/valid.jsonl \
    --codebase_file CSN/java/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```
## 2. Retrieval
```
python retrieval.py \
    --model_dir saved_models/retriever \
    --output_dir RAdataset \
    --model_name_or_path microsoft/unixcoder-base  \
    --train_data_file dataset/java/train.jsonl \
    --eval_data_file dataset/java/valid.jsonl \
    --test_data_file dataset/java/test.jsonl \
    --code_length 256 \
    --nl_length 64 \
    --retrieval_batch_size 64
```
## 3. Train Retrieval-Augmented Comment Generator
```
python RA_generator.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename RAdataset/train_DR.jsonl \
	--dev_filename RAdataset/valid_DR.jsonl \
	--output_dir saved_models/RA_Generator \
	--max_source_length 512 \
	--max_target_length 64 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
```
## 4. Test Retrieval-Augmented Comment Generator
```
python RA_generator.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename RAdataset/test_DR.jsonl \
	--output_dir saved_models/RA_Generator \
	--max_source_length 512 \
	--max_target_length 64 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 
```