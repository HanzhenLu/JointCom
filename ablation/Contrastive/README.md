# For the impatient
```
# For JCSD
# The first parameter is the target dataset
# The sencond parameter is the GPU_ids you want to used
bash run.sh JCSD 0,1
```

# For who want to know each steps
## 1. Train the Retriever
```
# For JCSD
lang=JCSD
python train_retriever.py \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/unixcoder-base \
    --train_data_file ../../dataset/$lang/train.jsonl \
    --eval_data_file ../../dataset/$lang/valid.jsonl \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_train_epochs 10 \
    --output_dir saved_models/$lang \
    --GPU_ids 0
```

## 2. Retrieve relevant code and comment
```
# For JCSD
lang=JCSD
python retrieval.py \
    --dataset $lang \
    --model_name_or_path microsoft/unixcoder-base \
    --nl_length 128 \
    --code_length 256 \
    --retrieve_batch_size 256 \
    --output_dir dataset \
    --GPU_ids 0
```

## 3. Train the generator
```
# For JCSD
lang=JCSD
python train_generator.py \
    --do_train \
    --do_eval \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/$lang \
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
    --GPU_ids 0
```

## 4. Generate predictions for test set
```
# For JCSD
lang=JCSD
python train_generator.py \
    --do_test \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/$lang \
    --test_filename dataset/$lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --GPU_ids 0
```

# Evaluate the result
```
# You should use python2.7 to run the evaluation program
# go to the dir containing evaluate.py
cd ../../
# Set the path as the dir containing test.output and test.gold
path=ablation/Independence/saved_models/java
python evaluate.py $path
```