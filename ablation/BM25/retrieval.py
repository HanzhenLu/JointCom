import fastbm25
import json
import sys
import os
from tqdm import tqdm

args = sys.argv[1:]

# build the database
data_path = '../../dataset/'
codes = []
nls = []
tokenized_codes = []
with open(os.path.join(data_path, args[0], 'train.jsonl'), 'r') as f:
    for line in f:
        js = json.loads(line)
        codes.append(' '.join(js['code_tokens']))
        nls.append(' '.join(js['docstring_tokens']))
        tokenized_codes.append([token.lower() for token in js['code_tokens']])

database = fastbm25.fastbm25(tokenized_codes)

if not os.path.exists('dataset'):
    os.mkdir('dataset')
if not os.path.exists('dataset/{}'.format(args[0])):
    os.mkdir('dataset/{}'.format(args[0]))

for phase in ['train', 'valid', 'test']:
    source_path = os.path.join(data_path, args[0], '{}.jsonl'.format(phase))
    write_path = os.path.join('dataset', args[0], '{}.jsonl'.format(phase))
    with  open(source_path, 'r') as s, open(write_path, 'w') as processed:
        for idx, line in tqdm(enumerate(s)):
            js = json.loads(line)
            code = ' '.join(js['code_tokens'])
            result = database.top_k_sentence([token.lower() for token in js['code_tokens']], k=int(args[1])+1)
            dic = {
                'source_code':code.strip(),
                'source_comment':' '.join(js['docstring_tokens']).strip()
            }
            count = 0
            for (_, index, score) in result:
                if index == idx:
                    continue
                dic['similar_{}'.format(count)] = index
                dic['score_{}'.format(count)] = score
                count += 1
            
            string = json.dumps(dic)
            processed.write(string)
            processed.write('\n')