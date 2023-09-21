# Retrieval-augmented-Code-Summarization

The dataset comes from [JCSD](#https://github.com/xing-hu/TL-CodeSum) and [PCSD](#https://github.com/EdinburghNLP/code-docstring-corpus)

## 0. Delete the duplicate datas in the dataset
```
python process.py
```


## 1. Train retriever and generator jointly
```
# For JCSD
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/JCSD/train.jsonl \
	--dev_filename dataset/JCSD/valid.jsonl \
	--output_dir saved_models/JCSD \
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

# For PCSD
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/PCSD/train.jsonl \
	--dev_filename dataset/PCSD/valid.jsonl \
	--output_dir saved_models/PCSD \
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
# For JCSD
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/JCSD/train.jsonl \
	--test_filename dataset/JCSD/test.jsonl \
	--output_dir saved_models/JCSD \
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

# For PCSD
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/PCSD/train.jsonl \
	--test_filename dataset/PCSD/test.jsonl \
	--output_dir saved_models/PCSD \
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


Hopefully, you will get a pretty good result, 23.67% for JCSD and 31.66% for PCSD.