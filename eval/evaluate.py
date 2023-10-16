from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
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
    print(score_Bleu)
    print("Bleu_1: "), np.mean(scores_Bleu[0])
    print("Bleu_2: "), np.mean(scores_Bleu[1])
    print("Bleu_3: "), np.mean(scores_Bleu[2])
    print("Bleu_4: "), np.mean(scores_Bleu[3])

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("Rouge: "), score_Rouge

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: "), score_Cider


if __name__ == '__main__':
    hyp_address = '../saved_models/{}/test.output'.format(arguments[1])
    ref_address = '../saved_models/{}/test.gold'.format(arguments[1])
    main(hyp_address, ref_address)

