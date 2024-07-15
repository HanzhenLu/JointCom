# Retrieval-augmented-Code-Summarization

The dataset comes from [JCSD](https://github.com/xing-hu/TL-CodeSum) provided by Hu and [PCSD](https://github.com/EdinburghNLP/code-docstring-corpus). But for fair and convenient comparision we use filtered version of JCSD provided by [DECOM](https://github.com/ase-decom/ASE22_DECOM/tree/master/dataset/JCSD) and PCSD provided by [SG-Trans](https://github.com/shuzhenggao/SG-Trans/tree/master/python/data)

## Data preprocess
```
python process.py
```
## For who lack patience
```
# first parameter is the dataset
# second parameter is the GPU_ids
# third parameter is the number of exemplars
bash run.sh JCSD 0,1 4
```

## 1. Train retriever and generator jointly
```
# For java and 4 exemplars
lang=JCSD
number=4
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path Salesforce/codet5-base \
	--train_filename dataset/${lang}/train.jsonl \
	--dev_filename dataset/${lang}/valid.jsonl \
	--output_dir saved_models/${lang}${number} \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--train_batch_size 32 \
	--eval_batch_size 24 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 10 \
	--passage_number ${number} \
	--GPU_ids 0,1
```

## 2. Generate predictions for test set
```
# For java
python run.py \
	--do_test \
	--model_name_or_path Salesforce/codet5-base \
	--train_filename dataset/${lang}/train.jsonl \
	--test_filename dataset/${lang}/test.jsonl \
	--output_dir saved_models/${lang}${number} \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--eval_batch_size 24 \
	--GPU_ids 0,1
```

## 3. Evaluate the result
```
# You should use python2.7 to run the evaluation program
# Set the path as the dir containing test.output and test.gold
path=saved_models/JCSD4
cd eval
python evaluate.py $path
```
