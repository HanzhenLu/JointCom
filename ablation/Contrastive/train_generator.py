from __future__ import absolute_import
import os
import sys
import torch
import json
import random
import logging
import argparse
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from io import open
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
# Import model
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
grandpa_path = os.path.dirname(parent_path)
sys.path.append(grandpa_path)
from model import build_model
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 similar_code,
                 similar_comment,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.similar_code = similar_code
        self.similar_comment = similar_comment

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            examples.append(
                Example(
                        idx = idx,
                        source = js['source_code'],
                        target = js['source_comment'],
                        similar_code=js['similar_code'],
                        similar_comment=js['similar_comment']
                        ) 
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids     
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)+['\n', '#']+tokenizer.tokenize(example.similar_comment)+['\n']+tokenizer.tokenize(example.similar_code)
        source_tokens = [tokenizer.cls_token]+source_tokens[:args.max_source_length-2]+[tokenizer.eos_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.eos_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
            )
        )
    return features



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--GPU_ids", default=None, type=str, 
                    help="The ids of GPUs which will be used")
    
    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    os.environ['CUDA_VISIBLE_DEVICES']=args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    _, model, _, tokenizer = build_model(args)
    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long) 
        train_data = TensorDataset(all_source_ids,all_target_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)


        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        patience, best_bleu, losses, dev_dataset = 0, 0, [], {}
        for epoch in range(args.num_train_epochs):
            for idx, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                inputs, outputs = batch
                source_mask = inputs.ne(tokenizer.pad_token_id)
                target_mask = outputs.ne(tokenizer.pad_token_id)
                results = model(input_ids=inputs, attention_mask=source_mask,
                                    labels=outputs, decoder_attention_mask=target_mask)
                loss = results.loss

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
            if args.do_eval:
                #Eval model with dev dataset                   

                logger.info("\n***** Running evaluation *****")
                logger.info("  Batch size = %d", args.eval_batch_size)  

                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long) 
                    eval_data = TensorDataset(all_source_ids)   
                    dev_dataset['dev_bleu'] = eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]
                    with torch.no_grad():
                        inputs = source_ids
                        source_mask = inputs.ne(tokenizer.pad_token_id)
                        preds = model(inputs,
                                       attention_mask=source_mask,
                                       is_generate=True)
                        top_preds = list(preds.cpu().numpy())
                        p.extend(top_preds)
                p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
                model.train()
                predictions, refs = [], []
                with open(args.output_dir+"/dev.output",'w') as f, open(args.output_dir+"/dev.gold",'w') as f1:
                    for pred,gold in zip(p,eval_examples):
                        predictions.append(pred.strip().split(' '))
                        refs.append([gold.target.strip().split(' ')])
                        f.write(str(gold.idx)+'\t'+pred+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                dev_bleu=round(corpus_bleu(refs, predictions),4)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-result')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "generator.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience =0
                else:
                    patience +=1
                    if patience ==2:
                        break
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-result/generator.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))                

        eval_examples = read_examples(args.test_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval() 
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]                  
            with torch.no_grad():
                inputs = source_ids
                source_mask = inputs.ne(tokenizer.pad_token_id)
                preds = model(inputs,
                                attention_mask=source_mask,
                                is_generate=True)
                top_preds = list(preds.cpu().numpy())
                p.extend(top_preds)
        p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]            
        model.train()
        predictions, refs = [], []
        with open(args.output_dir+"/test.output",'w') as f, open(args.output_dir+"/test.gold",'w') as f1:
            for pred,gold in zip(p,eval_examples):
                predictions.append(pred.strip().split(' '))
                refs.append([gold.target.strip().split(' ')])
                f.write(str(gold.idx)+'\t'+pred+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

        dev_bleu = round(corpus_bleu(refs, predictions),4)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    

                
if __name__ == "__main__":
    main()

