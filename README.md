# Retrieval-augmented-Code-Summarization

The dataset comes from [JCSD](https://github.com/xing-hu/TL-CodeSum) provided by Hu and PCSD(origin version is [code-docstring-corpus](https://github.com/EdinburghNLP/code-docstring-corpus), but for fair comparision we use filtered version provided by [DECOM](https://github.com/ase-decom/ASE22_DECOM/tree/master/dataset/PCSD))

## 0. Data preprocess
```
# For JCSD and PCSD
python process.py
```

## 1. Train retriever and generator jointly
```
# For java
lang=JCSD
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

## 2. Generate predictions for test set
```
# For java
lang=JCSD
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

## 3. Evaluate the result
```
# You should use python2.7 to run the evaluation program
# Set the path as the dir containing test.output and test.gold
path=saved_models/JCSD
cd eval
python evaluate.py $path
```
