# Retrieval-augmented-Code-Summarization

The dataset comes from [CodeSearchNet](#https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text), [JCSD](#https://github.com/xing-hu/TL-CodeSum) and [PCSD](#https://github.com/EdinburghNLP/code-docstring-corpus)

## 1. Train retriever and generator jointly
```
# For go
lang=go
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/$lang/train.jsonl \
	--dev_filename dataset/$lang/valid.jsonl \
	--output_dir saved_models/$lang \
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
# For go
lang=go
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/$lang/train.jsonl \
	--test_filename dataset/$lang/test.jsonl \
	--output_dir saved_models/$lang \
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

