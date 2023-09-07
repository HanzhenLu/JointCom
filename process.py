import json
import re

phase = ['train', 'valid', 'test']
database = []
for n in phase:
    code_path = './dataset/java/{}.token.code'.format(n)
    nl_path = './dataset/java/{}.token.nl'.format(n)
    write_path = './dataset/java/{}.jsonl'.format(n)
    with open(code_path, 'r') as codes, open(nl_path, 'r') as nls, open(write_path, 'w') as file:
        for code, nl in zip(codes, nls):
            code = re.sub('[0-9]+\t', '', code, 1)
            if n == 'train':
                database.append(code)
            else:
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