import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
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

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(js['code'])[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl_tokens = tokenizer.tokenize(js['docstring'])[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length  
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)
    
class ValidDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line)
                code=' '.join(js['code_tokens']).replace('\n',' ')
                code=' '.join(code.strip().split())
                nl=' '.join(js['docstring_tokens']).replace('\n',' ')
                nl=' '.join(nl.strip().split())
                data.append({'code':code, 'docstring':nl})

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))    

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = ValidDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_result = 0,0,1000000000000 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        result = evaluate(args, model, tokenizer, args.eval_data_file, eval_when_training=True)
            
        #save best model
        if result < best_result:
            best_result = result
            logger.info("  "+"*"*20)
            logger.info("  Best loss:%s",round(result,4))
            logger.info("  "+"*"*20)

            checkpoint_prefix = 'checkpoint-best-result'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('retriever.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
        else:
            logger.info("  "+"*"*20)
            logger.info("  Current loss:%s",round(result,4))
            logger.info("  "+"*"*20)


def evaluate(args, model, tokenizer, file_name, eval_when_training=False):
    query_dataset = ValidDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    loss = 0

    with torch.no_grad():
        for _,batch in enumerate(query_dataloader):
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss += loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device)).item()

    return loss

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")        

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--GPU_ids", default=None, type=str, 
                    help="The ids of GPUs which will be used")
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    os.environ['CUDA_VISIBLE_DEVICES']=args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    
    model = Retriever(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    if args.do_train:
        train(args, model, tokenizer)


if __name__ == "__main__":
    main()
    