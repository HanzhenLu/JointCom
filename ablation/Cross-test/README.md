# Test Untrain + Integrated_Generator
'''
lang=JCSD
python predict.py \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir test1/$lang \
    --model_dir ../../saved_models/$lang/checkpoint-best-bleu/generator_model.bin \
    --test_filename ../Untrain/dataset/$lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --eval_batch_size 32
'''

# Test Integrated_retriever + BM25_Generator
```
lang=JCSD
# retrieve first
python retrieval.py \
    --dataset $lang \
    --model_name_or_path microsoft/unixcoder-base \
    --output . \
    --retrieve_batch_size 32

# test
python predict.py \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir test2/$lang \
    --model_dir ../Untrain/saved_models/$lang/checkpoint-best-result/generator.bin \
    --test_filename $lang/test.jsonl \
    --max_source_length 512 \
    --max_target_length 64 \
    --eval_batch_size 32
```
