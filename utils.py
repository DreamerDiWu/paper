from collections import defaultdict
import re
import random

def gen_train_test(raw_file='news_fasttext_train.txt', n_testset=1000, n_trainset=2000)
    datasets = defaultdict(list)
    label_pattern = re.compile('__label__(.+)')

    with open(raw_file, 'r') as f:
        for line in f:
            label = label_pattern.findall(line)[0]
            datasets[label].append(line)

    permutation = random.sample(list(range(n_testset + n_trainset)), n_testset + n_trainset)
    idx_train = set(permutation[:n_trainset])
    idx_test = set(permutation[n_trainset: n_trainset + n_testset])

    with open("train.txt", 'w') as tr:
        with open("test.txt", 'w') as tt:
            for label, lines in datasets.items():
                for i, line in enumerate(lines):
                    if i in idx_train:
                        tr.write(line)
                    if i in idx_test:
                        tt.write(line)
                    if i > n_testset + n_trainset:
                        break

    return "train.txt", "test.txt"