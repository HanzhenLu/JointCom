# 1. Train the generator
```
# For java
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
    --num_train_epochs 10
```

# 2. Generate predictions for test set
```
# For java
lang=JCSD
python train_generator.py \
    --do_test \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir saved_models/$lang \
    --test_filename dataset/$lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32
```
# 3. Evaluate the result
```
# You should use python2.7 to run the evaluation program
# go to the dir containing evaluate.py
cd ../../
# Set the path as the dir containing test.output and test.gold
path=ablation/BM25/saved_models/JCSD
python evaluate.py $path
```