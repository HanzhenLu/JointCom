import torch
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from model import Retriever
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)  
from finetune_retriever import TextDataset

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--train_data_file", default=None, type=str, 
                    help="The input training data file (a json file).")
parser.add_argument("--model_dir", default=None, type=str, required=True,
                    help="The model directory stores the retriever's parameters.")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the new dataset will be written.")
parser.add_argument("--eval_data_file", default=None, type=str,
                    help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
parser.add_argument("--test_data_file", default=None, type=str,
                    help="An optional input test data file to test the MRR(a josnl file).") 

parser.add_argument("--model_name_or_path", default=None, type=str,
                    help="The model checkpoint for weights initialization.")

parser.add_argument("--nl_length", default=64, type=int,
                    help="Optional NL input sequence length after tokenization.")    
parser.add_argument("--code_length", default=256, type=int,
                    help="Optional Code input sequence length after tokenization.") 

parser.add_argument("--retrieval_batch_size", default=256, type=int,
                    help="Batch size for training.")

#print arguments
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
config = RobertaConfig.from_pretrained('microsoft/unixcoder-base')
model = RobertaModel.from_pretrained('microsoft/unixcoder-base')

model=Retriever(model)
checkpoint_prefix = 'checkpoint-best-mrr'
model_dir = os.path.join(args.model_dir, '{}'.format(checkpoint_prefix))
model_dir = os.path.join(model_dir, '{}'.format('model.bin'))
model_to_load = model.module if hasattr(model, 'module') else model
model_to_load.load_state_dict(torch.load(model_dir))
model.to(device)
    
train_dataset = TextDataset(tokenizer, args, args.train_data_file)
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.retrieval_batch_size)

valid_dataset = TextDataset(tokenizer, args, args.eval_data_file)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.retrieval_batch_size)

test_dataset = TextDataset(tokenizer, args, args.test_data_file)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.retrieval_batch_size)

model.eval()
train_vecs = [] 
valid_vecs = []
test_vecs = []
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with torch.no_grad():
    for batch in tqdm(train_dataloader):  
        code_inputs = batch[0].to(device)
        code_vec = model(code_inputs=code_inputs)
        train_vecs.append(code_vec.cpu().numpy())
        
    for batch in tqdm(valid_dataloader):  
        code_inputs = batch[0].to(device)
        code_vec = model(code_inputs=code_inputs) 
        valid_vecs.append(code_vec.cpu().numpy()) 
            
    for batch in tqdm(test_dataloader):
        code_inputs = batch[0].to(device)
        code_vec = model(code_inputs=code_inputs)
        test_vecs.append(code_vec.cpu().numpy())

train_vecs = np.concatenate(train_vecs,0)
valid_vecs = np.concatenate(valid_vecs,0)
test_vecs = np.concatenate(test_vecs,0)

train_scores = np.zeros((train_vecs.shape[0],5),dtype=float)
train_sort_ids = np.zeros((train_vecs.shape[0],5),dtype=int)
for i in tqdm(range(int((train_vecs.shape[0] / 5000) + 1))):
    scores = np.matmul(train_vecs[i*5000:min(train_vecs.shape[0], (i+1)*5000), :], train_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    for j in range(len(sort_ids)):
        for k in range(5):
            train_scores[i*5000+j][k]=scores[j][sort_ids[j][k]]
            train_sort_ids[i*5000+j][k]=sort_ids[j][k]
    del(scores)
    del(sort_ids)
    
train_dir = os.path.join(args.output_dir, 'train_DR.jsonl')
train_output = open(train_dir, 'w')
for i in range(len(train_dataset)):
    js = {}
    js['idx'] = i
    js['source_code'] = train_dataset.data[i]['code_tokens']
    js['source_comment'] = train_dataset.data[i]['docstring_tokens']
    if i == train_sort_ids[i][0]:
        j = 1
    else:
        j = 0
    js['similar_code_0'] = train_dataset.data[train_sort_ids[i][j]]['code_tokens']
    js['similar_comment_0'] = train_dataset.data[train_sort_ids[i][j]]['docstring_tokens']
    js['score_0'] = train_scores[i][j]

    string = json.dumps(js)
    train_output.write(string)
    train_output.write('\n')
train_output.close()
del(train_sort_ids)
del(train_scores)

valid_scores = np.zeros((valid_vecs.shape[0],5),dtype=float)
valid_sort_ids = np.zeros((valid_vecs.shape[0],5),dtype=int)
scores = np.matmul(valid_vecs, train_vecs.T)
sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
for i in tqdm(range(len(sort_ids))):
    for k in range(5):
        valid_scores[i][k]=scores[i][sort_ids[i][k]]
        valid_sort_ids[i][k]=sort_ids[i][k]
del(scores)
del(sort_ids)

valid_dir = os.path.join(args.output_dir, 'valid_DR.jsonl')
valid_output = open(valid_dir, 'w')
for i in range(len(valid_dataset)):
    js = {}
    js['idx'] = i
    js['source_code'] = valid_dataset.data[i]['code_tokens']
    js['source_comment'] = valid_dataset.data[i]['docstring_tokens']
    for j in range(1):
        js['similar_code_{}'.format(j)] = train_dataset.data[valid_sort_ids[i][j]]['code_tokens']
        js['similar_comment_{}'.format(j)] = train_dataset.data[valid_sort_ids[i][j]]['docstring_tokens']
        js['score_{}'.format(j)] = valid_scores[i][j]
    string = json.dumps(js)
    valid_output.write(string)
    valid_output.write('\n')
valid_output.close()
del(valid_sort_ids)
del(valid_scores)

test_scores = np.zeros((test_vecs.shape[0],5),dtype=float)
test_sort_ids = np.zeros((test_vecs.shape[0],5),dtype=int)
scores = np.matmul(test_vecs, train_vecs.T)
sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
for i in tqdm(range(len(sort_ids))):
    for k in range(5):
        test_scores[i][k]=scores[i][sort_ids[i][k]]
        test_sort_ids[i][k]=sort_ids[i][k]
del(scores)
del(sort_ids)

test_dir = os.path.join(args.output_dir, 'test_DR.jsonl')
test_output = open(test_dir, 'w')
for i in range(len(test_dataset)):
    js = {}
    js['idx'] = i
    js['source_code'] = test_dataset.data[i]['code_tokens']
    js['source_comment'] = test_dataset.data[i]['docstring_tokens']
    for j in range(4):
        js['similar_code_{}'.format(j)] = train_dataset.data[test_sort_ids[i][j]]['code_tokens']
        js['similar_comment_{}'.format(j)] = train_dataset.data[test_sort_ids[i][j]]['docstring_tokens']
        js['score_{}'.format(j)] = test_scores[i][j]
    string = json.dumps(js)
    test_output.write(string)
    test_output.write('\n')
test_output.close()
del(test_sort_ids)
del(test_scores)