# Retrieval-augmented-Code-Summarization

## 0. Delete the duplicate datas in the dataset
```
python process.py
```


## 1. Train retriever and generator jointly
```
python integrated_generator.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/java/train.jsonl \
	--dev_filename dataset/java/valid.jsonl \
	--output_dir saved_models/integrate \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--train_batch_size 24 \
	--eval_batch_size 24 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 10 \
	--passage_number 4
```

## 2. Eval retriever and generator
```
python integrated_generator.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/java/train.jsonl \
	--test_filename dataset/java/test.jsonl \
	--output_dir saved_models/integrate \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--train_batch_size 24 \
	--eval_batch_size 24 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 10 \
	--passage_number 4
```
