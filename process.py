import json
import re
import os

# Process JCSD
phase = ['train', 'valid', 'test']
database = []
for n in phase:
    code_path = './dataset/JCSD/{}.token.code'.format(n)
    nl_path = './dataset/JCSD/{}.token.nl'.format(n)
    write_path = './dataset/JCSD/{}.jsonl'.format(n)
    count = 0
    with open(code_path, 'r') as codes, open(nl_path, 'r') as nls, open(write_path, 'w') as file:
        for code, nl in zip(codes, nls):
            code = re.sub('[0-9]+\t', '', code, 1)
            # If the data has already appeared in the training set, 
            # delete it from the validation and testing sets
            if n == 'train':
                database.append(code)
            elif n == 'test':
                if code in database:
                    continue
            code = code.strip().split(' ')
            nl = re.sub('[0-9]+\t', '', nl, 1)
            nl = nl.strip().split(' ')
            line = {
                'code_tokens':code,
                'docstring_tokens':nl
            }
            line = json.dumps(line)
            file.write(line)
            file.write('\n')
            count += 1
        print('length of JCSD-{} : {}'.format(n, count))

# Process PCSD
database = []
phase = ['train', 'dev', 'test']
for n in phase:
    code_path = './dataset/PCSD/{}/code.original_subtoken'.format(n)
    nl_path = './dataset/PCSD/{}/javadoc.original'.format(n)
    write_path = './dataset/PCSD/{}.jsonl'.format(n)
    count = 0
    with open(code_path, 'r') as codes, open(nl_path, 'r') as nls, open(write_path, 'w') as file:
        for code, nl in zip(codes, nls):
            code = code.strip().split(' ')
            nl = nl.strip().split(' ')
            if n == 'train':
                database.append(code)
            elif n == 'test':
                if code in database:
                    continue
            line = {
                'code_tokens':code,
                'docstring_tokens':nl
            }
            line = json.dumps(line)
            file.write(line)
            file.write('\n')
            count += 1
        print('length of PCSD-{} : {}'.format(n, count))
        
os.rename('dataset/PCSD/dev.jsonl', 'dataset/PCSD/valid.jsonl')
