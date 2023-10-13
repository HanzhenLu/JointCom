from eval import bleu, rouge, meteor, cider
import re
import sys

arguments = sys.argv

comment_path = './saved_models/{}/test.output'.format(arguments[1])
ref_comment_path = './saved_models/{}/test.gold'.format(arguments[1])

comments, refs = [], []
with open(comment_path, 'r') as Cfile, open(ref_comment_path, 'r') as Rfile:
    for comment, ref in zip(Cfile, Rfile):
        comment = re.sub('[0-9]+\t', '', comment, 1)
        comments.append(comment.strip().split(' '))
        ref = re.sub('[0-9]+\t', ' ', ref, 1)
        refs.append([ref.strip().split(' ')])
ids = [i for i in range(len(comments))]

bleu_score = bleu.corpus_bleu(ids, comments, refs)
print('corpus level score : {}'.format(bleu_score[0]))
print('sentence level score : {}'.format(bleu_score[1]))

r = rouge.Rouge()
rouge_score = r.compute_score(ids, comments, refs)
print('rouge-L score : {}'.format(rouge_score[0]))

res = {k: [" ".join(v[0])] for k, v in enumerate(refs)}
gts = {k: [" ".join(v)] for k, v in enumerate(comments)}

c = cider.Cider()
cider_score = c.compute_score(gts, res)
print('cider score : {}'.format(cider_score[0]))

m = meteor.Meteor()
meteor_score = m.compute_score(gts, res)
print('meteor score : {}'.format(meteor_score[0]))