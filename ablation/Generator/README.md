# 1. Training
```
# For JCSD
lang=JCSD
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename ../../dataset/$lang/train.jsonl \
	--dev_filename ../../dataset/$lang/valid.jsonl \
	--output_dir saved_models/$lang \
	--max_source_length 256 \
	--max_target_length 64 \
	--beam_size 10 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 10 
```
	
# 2. Predicting	
```
# For JCSD
lang=JCSD
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename ../../dataset/$lang/test.jsonl \
	--output_dir saved_models/$lang \
	--max_source_length 256 \
	--max_target_length 64 \
	--beam_size 10 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 10 	
```
