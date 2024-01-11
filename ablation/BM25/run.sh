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
number=1
display "GPUs are : $2"

# display "Start retrieval"
# python retrieval.py ${lang} ${number}
# result $? "Retrieval failed"

display "Start training!"
python train_generator.py \
	--do_train \
	--do_eval \
	--model_name_or_path Salesforce/codet5-base \
	--train_filename dataset/$lang/train.jsonl \
	--dev_filename dataset/$lang/valid.jsonl \
	--output_dir saved_models/${lang} \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--train_batch_size 24 \
	--eval_batch_size 6 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 10 \
	--passage_number $number \
	--GPU_ids $2
result $? "Training failed!"

display "Start prediction!"
python run.py \
	--do_test \
	--model_name_or_path Salesforce/codet5-base \
	--train_filename dataset/$lang/train.jsonl \
	--test_filename dataset/$lang/test.jsonl \
	--output_dir saved_models/${lang} \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--eval_batch_size 16 \
	--GPU_ids $2
result $? "Prediction failed!"
