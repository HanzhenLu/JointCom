import sys 
import os
import argparse
import torch
import json
import logging
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)  
# Import model
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
grandpa_path = os.path.dirname(parent_path)
sys.path.append(grandpa_path)
from model import Retriever

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str, 
                    help="The dataset we are going to process")
parser.add_argument("--model_name_or_path", default=None, type=str,
                    help="The model checkpoint for weights initialization.")
parser.add_argument("--nl_length", default=128, type=int,
                    help="Optional NL input sequence length after tokenization.")
parser.add_argument("--code_length", default=256, type=int,
                    help="Optional Code input sequence length after tokenization.") 
parser.add_argument("--retrieve_batch_size", default=256, type=int,
                    help="Batch size for retrieval.")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_file_name = '../../dataset/{}/train.jsonl'.format(args.dataset)
test_file_name = '../../dataset/{}/test.jsonl'.format(args.dataset)
batch_size = args.retrieve_batch_size

logger.info('  ' + '*'*20)
logger.info('begin loading parameters from pretrained')
logger.info('  ' + '*'*20)
tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
config = RobertaConfig.from_pretrained(args.model_name_or_path)
model = RobertaModel.from_pretrained(args.model_name_or_path)
model=Retriever(model)
model_to_load = model.module if hasattr(model, 'module') else model
model_to_load.load_state_dict(torch.load('../../saved_models/{}/checkpoint-best-bleu/retriever_model.bin'.format(args.dataset)))
model.to(device)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_ids,
                 comment_ids,
                 idx,
                 source_code,
                 source_comment

    ):
        self.code_ids = code_ids
        self.comment_ids = comment_ids
        self.idx = idx
        self.source_code = source_code
        self.source_comment = source_comment

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())           
            examples.append(
                Example(
                        idx = idx,
                        source = code,
                        target = nl
                        ) 
            )
    return examples

def convert_examples_to_features(examples, tokenizer):
    features = []
    for example_index, example in enumerate(examples):
        #source
        code_tokens = tokenizer.tokenize(example.source)[:args.code_length-4]
        source_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.code_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        target_tokens = tokenizer.tokenize(example.target)[:args.nl_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.nl_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length
       
        features.append(
            InputFeatures(
                 source_ids,
                 target_ids,
                 example_index,
                 example.source,
                 example.target,
            )
        )
    return features

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        examples = read_examples(file_path)
        self.features = convert_examples_to_features(examples, tokenizer)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return torch.tensor(self.features[i].code_ids), torch.tensor(self.features[i].comment_ids)
    
train_dataset = TextDataset(tokenizer, train_file_name)
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

test_dataset = TextDataset(tokenizer, test_file_name)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model.eval()
train_vecs = [] 
test_vecs = []
comment_vecs = []

with torch.no_grad():
    logger.info('encode train data')
    for batch in tqdm(train_dataloader):  
        code_inputs = batch[0].to(device)
        comment_inputs = batch[1].to(device)
        code_vec = model(code_inputs=code_inputs)
        comment_vec = model(nl_inputs=comment_inputs)
        train_vecs.append(code_vec.cpu().numpy())
        comment_vecs.append(comment_vec.cpu().numpy())
    
    logger.info('encode tset data')        
    for batch in tqdm(test_dataloader):
        code_inputs = batch[0].to(device)
        code_vec = model(code_inputs=code_inputs)
        test_vecs.append(code_vec.cpu().numpy())

train_vecs = np.concatenate(train_vecs,0)
test_vecs = np.concatenate(test_vecs,0)
comment_vecs = np.concatenate(comment_vecs,0)

output_path = os.path.join(args.output_dir, args.dataset)
if not os.path.exists(output_path):
    os.makedirs(output_path)

logger.info('write test data')
test_scores = np.zeros((test_vecs.shape[0],1),dtype=float)
test_sort_ids = np.zeros((test_vecs.shape[0],1),dtype=int)
scores = np.matmul(test_vecs, train_vecs.T)
sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
for i in tqdm(range(len(sort_ids))):
    test_scores[i][0]=scores[i][sort_ids[i][0]]
    test_sort_ids[i][0]=sort_ids[i][0]
del(scores)
del(sort_ids)

test_output = open(os.path.join(output_path, 'test.jsonl'), 'w')
for i in range(len(test_dataset)):
    js = {}
    js['idx'] = i
    js['source_code'] = test_dataset.features[i].source_code
    js['source_comment'] = test_dataset.features[i].source_comment
    js['similar_code'] = train_dataset.features[test_sort_ids[i][0]].source_code
    js['similar_comment'] = train_dataset.features[test_sort_ids[i][0]].source_comment
    string = json.dumps(js)
    test_output.write(string)
    test_output.write('\n')
test_output.close()
del(test_sort_ids)
del(test_scores)
