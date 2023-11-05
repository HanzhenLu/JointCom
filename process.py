import json
import re
import os
import tokenize

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

# split compound words
def split_word(word):
    words = []
    
    if len(word) <= 1:
        return word

    word_parts = re.split('[^0-9a-zA-Z]', word)
    for part in word_parts:
        part_len = len(part)
        if part_len == 1:
            words.append(part)
            continue
        word = ''
        for index, char in enumerate(part):
            # condition : level|A
            if index == part_len - 1 and char.isupper() and part[index-1].islower():
                if word != '':
                    words.append(word)
                words.append(char)
                word = ''
                
            elif(index != 0 and index != part_len - 1 and char.isupper()):
                # condition 1 : FIRST|Name
                # condition 2 : first|Name
                condition1 = part[index-1].isalpha() and part[index+1].islower()
                condition2 = part[index-1].islower() and part[index+1].isalpha()
                if condition1 or condition2:
                    if word != '':
                        words.append(word)
                    word = ''
                else:
                    word += char
            
            else:
                word += char
        
        if word != '':
            words.append(word)
            
    return [word.lower() for word in words]

# Process PCSD
database = []
phase = ['train', 'dev', 'test']
for n in phase:
    code_path = './dataset/PCSD/{}_originalcode'.format(n)
    nl_path = './dataset/PCSD/{}.comment'.format(n)
    write_path = './dataset/PCSD/{}.jsonl'.format(n)
    count = 0
    if n == 'test':
        repete = open('repeted', 'w')
    with open(code_path, 'r') as codes, open(nl_path, 'r') as nls, open(write_path, 'w') as file:
        for code, nl in zip(codes, nls):
            code = re.sub('DCNL\s+', '\n', code)
            code = re.sub('DCSP\s+', '\t', code)
            # tokenize
            with open('temp', 'w') as w:
                w.write(code)
            sub_tokens = []
            with tokenize.open('temp') as w:
                tokens = tokenize.generate_tokens(w.readline)
                for token in tokens:
                    split = split_word(token.string)
                    for i in split:
                        sub_tokens.append(i.lower())
            processed_code = ' '.join(sub_tokens).split()
            nl = nl.strip().split()
            if n == 'train':
                database.append(processed_code)
            elif n == 'test':
                if processed_code in database:
                    repete.write(str(count)+str(processed_code)+'\n')
                    continue
            line = {
                'code_tokens':processed_code,
                'docstring_tokens':nl
            }
            line = json.dumps(line)
            file.write(line)
            file.write('\n')
            count += 1
        print('length of PCSD-{} : {}'.format(n, count))
        
os.rename('dataset/PCSD/dev.jsonl', 'dataset/PCSD/valid.jsonl')
os.remove('temp')
repete.close()