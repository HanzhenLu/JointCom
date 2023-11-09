#!/bin/bash

display(){
    echo "===================="
    echo $1
    echo "====================" 
}

result(){
    if [ $1 -eq 0 ];then
        display "Finish"
    else 
        display "$2"
        exit
    fi 
}

display "Target Dataset is : $1"
lang=$1
display "GPUs are : $2"

display "Start training the first generator"
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
	--num_train_epochs 10 \
    --GPU_ids $2
result $? "Training first generator failed"

display "Start retrieval"
python retrieval.py \
    --dataset $lang \
    --model_name_or_path microsoft/unixcoder-base \
    --nl_length 128 \
    --code_length 256 \
    --retrieve_batch_size 256 \
    --output_dir dataset \
    --GPU_ids $2
result $? "Retrieval failed"

display "Start training the second generator"
python train_generator.py \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/${lang}_retrieval \
    --train_filename dataset/$lang/train.jsonl \
    --dev_filename dataset/$lang/valid.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --beam_size 10 \
    --num_train_epochs 10 \
    --GPU_ids $2
result $? "Training generator failed"

display "Start predicting"
python train_generator.py \
    --do_test \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/${lang}_retrieval \
    --test_filename dataset/$lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --GPU_ids $2
result $? "Predicting failed"