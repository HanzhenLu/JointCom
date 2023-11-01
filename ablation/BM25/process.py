import json
import os

# For JCSD
path = './dataset/JCSD'
for phase in ['train', 'valid', 'test']:
    similar_code_path = os.path.join(path, phase, 'similar.code')
    similar_comment_path = os.path.join(path, phase, 'similar.comment')
    source_code_path = os.path.join(path, phase, 'source.code')
    source_comment_path = os.path.join(path, phase, 'source.comment')
    write_path = os.path.join(path, '{}.jsonl'.format(phase))
    with  open(similar_code_path, 'r') as Scode, open(similar_comment_path, 'r') as Snl:
        with open(source_code_path, 'r') as Ocode, open(source_comment_path, 'r') as Onl:
            with open(write_path, 'w') as processed:
                for similar_code, similar_comment, source_code, source_comment in zip(Scode, Snl, Ocode, Onl):
                    dic = {
                        'source_code':source_code.strip(),
                        'source_comment':source_comment.strip(),
                        'similar_code':similar_code.strip(),
                        'similar_comment':similar_comment.strip()
                    }
                    string = json.dumps(dic)
                    processed.write(string)
                    processed.write('\n')