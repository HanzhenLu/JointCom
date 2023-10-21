from eval.bleu.bleu import Bleu
from eval.meteor.meteor import Meteor
from eval.rouge.rouge import Rouge
from eval.cider.cider import Cider
from eval.meteor.meteor import Meteor
import numpy as np
import sys
import re

arguments = sys.argv

def main(hyp, ref):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        # Delete the index number at the beginning of the line
        # and lower every characters
        res = {k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()] for k, v in enumerate(references)}

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
    print("Corpus-level Bleu_1: "), score_Bleu[0]
    print("Corpus-level Bleu_2: "), score_Bleu[1]
    print("Corpus-level Bleu_3: "), score_Bleu[2]
    print("Corpus-level Bleu_4: "), score_Bleu[3]
    print("Sentence-level Bleu_1: "), np.mean(scores_Bleu[0])
    print("Sentence-level Bleu_2: "), np.mean(scores_Bleu[1])
    print("Sentence-level Bleu_3: "), np.mean(scores_Bleu[2])
    print("Sentence-level Bleu_4: "), np.mean(scores_Bleu[3])

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor
    with open('Meteor_result.txt', 'w') as f:
        for i in range(len(scores_Meteor)):
            if scores_Meteor[i] == 0.0:
                f.write(str({'idx':i, 'ref':gts[i], 'pred':res[i]}))
                f.write('\n')

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("Rouge: "), score_Rouge

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: "), score_Cider


if __name__ == '__main__':
    hyp_address = '{}/test.output'.format(arguments[1])
    ref_address = '{}/test.gold'.format(arguments[1])
    main(hyp_address, ref_address)


