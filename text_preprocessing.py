from collections import defaultdict
import os, sys
import random
import jieba

def get_dataset(max_per_label=10000):
    dataset_dir = "/home/wood/Download/BaiduDownload/692899593_MaxPayne居然/THUCNews"
    dataset = defaultdict(list)
    for topic in os.listdir(dataset_dir):
        topic_dir = "{}/{}".format(dataset_dir, topic)
        text_files = os.listdir(topic_dir)
        for i, file in enumerate(text_files):
            if i < max_per_label:
                file_path = "{}/{}".format(topic_dir, file)
                with open(file_path, 'r') as f:
                    dataset[topic].append(''.join(f.readlines()))
            else:
                break
    return dataset

def get_stop_words_set():
    stop_words_set = set()
    with open("stop_words.txt", 'r') as f:
        for w in f:
            stop_words_set.add(w.strip())
    stop_words_set.add('\n')
    stop_words_set.add('\u3000')
    stop_words_set.add(' ')
    return stop_words_set


def remove_stop_words(tokens, stop_words_set):
    ret_str = ''
    for tok in tokens:
        if tok not in stop_words_set:
            ret_str += tok + ' '
    return ret_str.strip()


def make_train_valid_test(datasets, n_train=2000, n_valid=1000, n_test=1000, seed=0):
    dir_name = "tokened"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    train = "train_token.txt"
    valid = "valid_token.txt"
    test = "test_token.txt"
    path = lambda x: "{}/{}".format(dir_name, x)

    random.seed(seed)
    stop_words_set = get_stop_words_set()
    with open(path(train), 'w') as tr, open(path(test), 'w') as tt, open(path(valid), 'w') as va:
        for topic, docs in datasets.items():
            perm_doc_idx = random.sample(range(len(docs)), len(docs))
            for i, doc_idx in enumerate(perm_doc_idx):
                tokens = jieba.cut(docs[doc_idx])
                tokenized = remove_stop_words(tokens, stop_words_set) + '\t__label__{}\n'.format(topic)
                if i < n_train:
                    tr.write(tokenized)
                elif n_train <= i < n_train + n_valid:
                    va.write(tokenized)
                elif n_train + n_valid <= i < n_train + n_valid + n_test:
                    tt.write(tokenized)
                else:
                    break
    return path(train), path(valid), path(test)



