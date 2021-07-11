import json
import random


def split(source_file, rate=0.1):
    train_file = source_file + '.train'
    dev_file = source_file + '.dev'
    with open(source_file, 'r', encoding='utf-8') as fin, open(
            train_file, 'w', encoding='utf-8') as ftrain, open(dev_file, 'w',
                                                               encoding='utf-8') as fdev:
        docs = []
        n_sents = 0
        for line in fin:
            doc = json.loads(line.strip())
            docs.append(doc)
            n_sents += len(doc['sentences'])
        random.shuffle(docs)
        max_n_sampled_sents = int(n_sents * rate)
        n_sampled_sents = 0
        idx = 0
        while n_sampled_sents <= max_n_sampled_sents:
            n_sampled_sents += len(docs[idx]['sentences'])
            idx += 1
        for doc in docs[:idx]:
            print(json.dumps(doc), file=fdev)
        for doc in docs[idx:]:
            print(json.dumps(doc), file=ftrain)

        print(f"the number of documents: {len(docs)}")
        print(f"the number of training documents: {len(docs) - idx}")
        print(f"the number of dev documents: {idx}")
        print(f"the number of sentences: {n_sents}")
        print(f"the number of training sentences: {n_sents - n_sampled_sents}")
        print(f"the number of dev sentences: {n_sampled_sents}")


if __name__ == '__main__':
    split("0.json")
    split("1.json")
    split("2.json")
    split("3.json")
    split("4.json")
