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

display "Start training retriever"
python train_retriever.py \
    --do_train \
    --do_eval \
    --model_name_or_path Salesforce/codet5-base \
    --train_data_file ../../dataset/$lang/train.jsonl \
    --eval_data_file ../../dataset/$lang/valid.jsonl \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --output_dir saved_models/$lang \
    --GPU_ids $2
result $? "Training retriever failed"

display "Start retrieval"
python retrieval.py \
    --dataset $lang \
    --model_name_or_path Salesforce/codet5-base \
    --nl_length 128 \
    --code_length 256 \
    --retrieve_batch_size 64 \
    --output_dir dataset \
    --GPU_ids $2
result $? "Retrieval failed"

display "Start training generator"
python train_generator.py \
    --do_train \
    --do_eval \
    --model_name_or_path Salesforce/codet5-base \
    --output_dir saved_models/$lang \
    --train_filename dataset/$lang/train.jsonl \
    --dev_filename dataset/$lang/valid.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --beam_size 10 \
    --num_train_epochs 10 \
    --GPU_ids $2
result $? "Training generator failed"

display "Start predicting"
python train_generator.py \
    --do_test \
    --model_name_or_path Salesforce/codet5-base \
    --output_dir saved_models/$lang \
    --test_filename dataset/$lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --eval_batch_size 16\
    --GPU_ids $2
result $? "Predicting failed"
