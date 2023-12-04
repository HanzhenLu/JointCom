import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import sys
import re

arguments = sys.argv

def main(hyp, ref):
    hyps = [[]*10]
    refs = [[]*10]
    with open(hyp, 'r') as Fh, open(ref, 'r') as Fr:
        for h, r in zip(Fh, Fr):
            h = re.sub('[0-9]+\t', '', h.strip(), 1)
            index = len(r) // 5
            if index >= len(refs):
                continue
            r = re.sub('[0-9]+\t', '', r.strip(), 1)
            hyps[index].append(h.split(' '))
            refs[index].append([r.split(' ')])
    for i in range(len(hyps)):
        score = corpus_bleu(refs[i], hyps[i])
        print('{}-{} : {}'.format(5*i, 5*(i+1), round(score, 4)))

if __name__ == '__main__':
    hyp_address = '{}/test.output'.format(arguments[1])
    ref_address = '{}/test.gold'.format(arguments[1])
    main(hyp_address, ref_address)