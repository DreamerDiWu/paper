from random_fasttext import *
import time
from sklearn.metrics import precision_recall_fscore_support
import fastText
no_label = re.compile("(.+)\t__label__")
label_pt = re.compile("__label__.+")

import numpy as np

def report(preds, true_labels, labels):
    precisions = []
    recalls = []
    fscores = []
    for label in labels:
        preds_bin = [pred == label for pred in preds]
        true_labels_bin = [true_l == label for true_l in true_labels]
        p, r, f, _ = precision_recall_fscore_support(y_pred=preds_bin, y_true=true_labels_bin, pos_label=True,\
                                                    average='binary')
        precisions.append(p)
        recalls.append(r)
        fscores.append(f)
        print("{l}: \tprecision = {p:.3f}\trecall = {r:.3f}\tfscore = {f:.3f}".format(l=label, p=p, r=r , f=f))
    print("Average Precision: {:.3f}\tAverage Recall: {:.3f}\tAverage Fscore: {:.3f}"\
          .format(np.mean(precisions), np.mean(recalls), np.mean(fscores)))

def gen_word_vec_file(model,vec_file_name):
    with open(vec_file_name, 'w',encoding='UTF-8') as f:
        f.write("{} {}\n".format(len(model.get_words()), model.get_dimension()))
        for word in model.get_words():
            vec = model.get_word_vector(word)
            f.write("{} {}\n".format(word, ' '.join(map(str, vec))))
    return vec_file_name

if __name__ == "__main__":
    s = time.time()
    dim = 100
    # cbow = fastText.train_unsupervised("tokened/train_token.txt",dim=100,model='skipgram',epoch=20)
    # word_vec_file = gen_word_vec_file(cbow, "tmp.vec")
    word_vec_file="tmp.vec"
    cbow_t = time.time()
    print("word embedding Time: {:.2f}s".format(cbow_t-s))


    rft = RandomFastText(dim_ratio=0.2, n_classifier=200, dim=dim,
                         ws=5,lr=2,epoch=1,min_count=5)
    rft.train("tokened/train_token.txt", word_vec_file)
    train_t = time.time()
    print("Trainning Time: {:.2f}s".format(train_t-cbow_t))

    single = fastText.train_supervised("tokened/train_token.txt", ws=5,
                                    epoch=5, dim=dim)

    with open("tokened/test_token.txt", 'r',encoding='UTF-8') as f:
        lines = f.readlines()
        preds = rft.predict([no_label.findall(line)[0] for line in lines])
        pred_t = time.time()
        print("Prediction Time: {:.2f}s".format(pred_t - train_t))
        true_labels = [label_pt.findall(line)[0] for line in lines]
        preds_sin, _ = single.predict([no_label.findall(line)[0] for line in lines])
    labels = set(true_labels)

    print("RandomFastText Report:")
    report(preds, true_labels, labels)
    print("\n\n")
    print("SingleFastText Report:")
    report(preds_sin, true_labels, labels)


