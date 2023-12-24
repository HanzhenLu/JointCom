import json

stage = ['train', 'valid', 'test']
for s in stage:
    with open('dataset/PCSD/{}.jsonl'.format(s), 'r') as f, \
        open('dataset/PCSD/{}/source.code'.format(s), 'w') as sc, open('dataset/PCSD/{}/source.comment'.format(s), 'w') as sn, \
        open('dataset/PCSD/{}/similar.code'.format(s), 'w') as rc, open('dataset/PCSD/{}/similar.comment'.format(s), 'w') as rn:
        for line in f:
            js = json.loads(line)
            sc.write(js['source_code']+'\n')
            sn.write(js['source_comment']+'\n')
            rc.write(js['similar_code']+'\n')
            rn.write(js['similar_comment']+'\n')
            