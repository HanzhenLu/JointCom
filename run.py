from __future__ import absolute_import
import os
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
import torch.nn as nn
from model import DataBase, build_model
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from nltk.translate.bleu_score import corpus_bleu

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code = ' '.join(js['code_tokens']).replace('\n',' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n','')
            nl = ' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source = code,
                        target = nl,
                        ) 
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 query_ids
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.query_ids = query_ids 
        
def convert_examples_to_features(examples:Example, tokenizer, args, stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_str = example.source.replace('</s>', '<unk>')
        source_tokens = tokenizer.tokenize(source_str)
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens[:args.code_length]) 
        query_tokens = [tokenizer.cls_token]+source_tokens[:args.code_length-2]+[tokenizer.eos_token]
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        padding_length = args.code_length - len(query_ids)
        query_ids += [tokenizer.pad_token_id]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_str = example.target.replace('</s>', '<unk>')
            target_tokens = tokenizer.tokenize(target_str)[:args.nl_length]         
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 query_ids
            )
        )
    return features

class MyDataset(Dataset):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]
    
    def BuildIndex(self, retriever):
        with torch.no_grad():
            input = [feature.query_ids for feature in self.features]
            input = torch.tensor(input, dtype=torch.long)
            dataset = TensorDataset(input)
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            query_vecs = []
            retriever.eval()
            for batch in tqdm(dataloader):
                code_inputs = torch.tensor(batch[0]).to(device)
                code_vec = retriever(code_inputs)
                query_vecs.append(code_vec.cpu().numpy())
            query_vecs = np.concatenate(query_vecs,0)
            index = DataBase(query_vecs)
        return index
    
def DoNothingCollator(batch):
    return batch

        
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
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--passage_number", default=4, type=int,
                        help="the number of passages being retrieved back.")
    
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
    parser.add_argument('--GPU_ids', type=str, default='0',
                        help="The ids of GPUs will be used")
    
    # print arguments
    args = parser.parse_args()
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    logger_path = os.path.join(args.output_dir, 'train.log') if args.do_train else os.path.join(args.output_dir, 'test')
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    # build model
    config, generator, retriever, tokenizer = build_model(args)
    
    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)  
    retriever.to(args.device) 
    if args.n_gpu > 1:
        generator = torch.nn.DataParallel(generator)
        retriever = torch.nn.DataParallel(retriever)  
        
    prefix = [tokenizer.cls_token_id]
    postfix = [tokenizer.sep_token_id]
    sep = tokenizer.convert_tokens_to_ids(["\n", "#"])
    sep_ = tokenizer.convert_tokens_to_ids(["\n"])
    def Cat2Input(code, similar_comment, similar_code):
        input = code + sep + similar_comment + sep_ + similar_code
        input = prefix + input[:args.max_source_length-2] + postfix
        padding_length = args.max_source_length - len(input)
        input += padding_length * [tokenizer.pad_token_id]
        return input
    def Cat2Output(comment):
        output = prefix + comment[:args.max_target_length-2] + postfix
        padding_length = args.max_target_length - len(output)
        output += padding_length * [tokenizer.pad_token_id]
        return output
    
    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_dataset = MyDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps, 
                                      collate_fn=DoNothingCollator)
        index = train_dataset.BuildIndex(retriever)
        
        # Prepare optimizer and schedule (linear warmup and decay) for generator
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': retriever.parameters(), 'eps': 1e-8}
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
        
        patience, best_bleu, losses, dev_dataset = 0, 0, [], {}
        
        for epoch in range(args.num_train_epochs):
            for idx, batch in enumerate(train_dataloader):
                retriever.train()
                generator.train()
                query = [feature.query_ids for feature in batch]
                query = torch.tensor(query, dtype=torch.long).to(device)
                query_vec = retriever(query)
                query_vec_cpu = query_vec.detach().cpu().numpy()
                i = index.search(query_vec_cpu, args.passage_number, 'train')
                document = [train_dataset.features[idx].query_ids for idxs in i for idx in idxs]
                document = torch.tensor(document, dtype=torch.long).to(device)
                document_vec = retriever(document)
                document_vec = document_vec.view(len(batch), args.passage_number, -1)
                score = torch.einsum('bd,bpd->bp', query_vec, document_vec)
                softmax = nn.Softmax(dim=-1)
                score = softmax(score)
                score = score.view(-1)
                
                # Cat the ids of code, relevant document to form the final input
                inputs, outputs = [], []
                for no, feature in enumerate(batch):
                    for j in i[no]:
                        relevant = train_dataset.features[j]
                        inputs.append(Cat2Input(feature.source_ids, relevant.target_ids, relevant.source_ids))
                        outputs.append(Cat2Output(feature.target_ids))
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                outputs = torch.tensor(outputs, dtype=torch.long).to(device)
                source_mask = inputs.ne(tokenizer.pad_token_id)
                target_mask = outputs.ne(tokenizer.pad_token_id)
                results = generator(input_ids=inputs, attention_mask=source_mask,
                                    labels=outputs, decoder_attention_mask=target_mask, score=score)
                loss = results.loss
                
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
                    
                    # Update Vector
                    with torch.no_grad():
                        retriever.eval()
                        history = index.get_history()
                        for i in history:
                            document = [train_dataset.features[idx].query_ids for idxs in i for idx in idxs]
                            document = torch.tensor(document, dtype=torch.long).to(device)
                            document_vec = retriever(document)
                            update_id = [id for j in i for id in j]

                            index.update(update_id, document_vec.cpu().numpy())
                            
                        retriever.train()

            print(score.view(-1, args.passage_number))
            
            if args.do_eval:
                index = train_dataset.BuildIndex(retriever) 
                #Eval model with dev dataset                   
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    eval_data = MyDataset(eval_features)
                    dev_dataset = (eval_examples, eval_data)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                             collate_fn=DoNothingCollator)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                
                retriever.eval()
                generator.eval()
                p=[]
                with torch.no_grad():
                    for batch in eval_dataloader:
                        query = [feature.query_ids for feature in batch]
                        query = torch.tensor(query, dtype=torch.long).to(device)
                        query_vec = retriever(query)
                        query_vec_cpu = query_vec.detach().cpu().numpy()
                        i = index.search(query_vec_cpu, 1)
                        inputs = []
                        for no, feature in enumerate(batch):
                            relevant = train_dataset.features[i[no][0]]
                            inputs.append(Cat2Input(feature.source_ids, relevant.target_ids, relevant.source_ids))
                        
                        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                        source_mask = inputs.ne(tokenizer.pad_token_id)
                        preds = generator(inputs,
                                        attention_mask=source_mask,
                                        is_generate=True)
                        top_preds = list(preds.cpu().numpy())
                        p.extend(top_preds)
                p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
                generator.train()
                retriever.train()
                predictions, refs = [], []
                with open(args.output_dir+"/dev.output",'w') as f, open(args.output_dir+"/dev.gold",'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(ref.strip().split(' '))
                        refs.append([gold.target.strip().split(' ')])
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                dev_bleu=round(corpus_bleu(refs, predictions),4)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = generator.module if hasattr(generator, 'module') else generator  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "generator_model.bin")
                    if os.path.exists(output_model_file):
                        os.remove(output_model_file)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save = retriever.module if hasattr(retriever, 'module') else retriever  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "retriever_model.bin")
                    if os.path.exists(output_model_file):
                        os.remove(output_model_file)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience =0
                else:
                    patience += 1
                    if patience == 2:
                        break
                
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-bleu/retriever_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
        model_to_load = retriever.module if hasattr(retriever, 'module') else retriever  
        model_to_load.load_state_dict(torch.load(output_dir))                
        
        checkpoint_prefix = 'checkpoint-best-bleu/generator_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
        model_to_load = generator.module if hasattr(generator, 'module') else generator  
        model_to_load.load_state_dict(torch.load(output_dir))                
        
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args)
        train_dataset = MyDataset(train_features)
        index = train_dataset.BuildIndex(retriever)
        eval_examples = read_examples(args.test_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
        eval_data = MyDataset(eval_features)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                        collate_fn=DoNothingCollator)

        logger.info("\n***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        retriever.eval()
        generator.eval()
        p=[]
        for batch in tqdm(eval_dataloader):
            query = [feature.query_ids for feature in batch]
            query = torch.tensor(query, dtype=torch.long).to(device)
            query_vec = retriever(query)
            query_vec_cpu = query_vec.detach().cpu().numpy()
            i = index.search(query_vec_cpu, 1)
            inputs = []
            for no, feature in enumerate(batch):
                relevant = train_dataset.features[i[no][0]]
                inputs.append(Cat2Input(feature.source_ids, relevant.target_ids, relevant.source_ids))
            with torch.no_grad():
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                source_mask = inputs.ne(tokenizer.pad_token_id)
                preds = generator(inputs,
                                attention_mask=source_mask,
                                is_generate=True)
                top_preds = list(preds.cpu().numpy())
                p.extend(top_preds)
        p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
        generator.train()
        retriever.train()
        predictions, refs = [], []
        with open(args.output_dir+"/test.output",'w') as f, open(args.output_dir+"/test.gold",'w') as f1:
            for pred,gold in zip(p,eval_examples):
                predictions.append(pred.strip().split(' '))
                refs.append([gold.target.strip().split(' ')])
                f.write(str(gold.idx)+'\t'+pred+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

        dev_bleu=round(corpus_bleu(refs, predictions),4)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    

if __name__ == '__main__':
    main()  
