from random_fasttext import *
import time
from sklearn.metrics import precision_recall_fscore_support
import fasttext
no_label = re.compile("(.+)__label__")
label_pt = re.compile("__label__(.+)")

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

if __name__ == "__main__":
    s = time.time()
    rft = RandomFastText(dim_ratio=0.2, n_classifier=30,dim=100,
                         ws=5,lr=0.5,epoch=5)
    rft.train("tokened/train_token.txt", "train_token.vec")
    train_t = time.time()
    print("Trainning Time: {:.2f}s".format(train_t-s))

    single = fasttext.supervised("tokened/train_token.txt", "single",ws=5, lr=0.5,
                                    epoch=5,)

    with open("tokened/test_token.txt", 'r') as f:
        lines = f.readlines()
        preds = rft.predict([no_label.findall(line)[0] for line in lines])
        pred_t = time.time()
        print("Prediction Time: {:.2f}s".format(pred_t - train_t))
        true_labels = [label_pt.findall(line)[0] for line in lines]
        preds_sin = [pred[0] for pred in
                     single.predict([no_label.findall(line)[0] for line in lines])]
    labels = set(true_labels)

    print("RandomFastText Report:")
    report(preds, true_labels, labels)
    print("\n\n\n")
    print("SingleFastText Report:")
    report(preds_sin, true_labels, labels)


