import fastText
import random
import re
from collections import defaultdict, Counter
import multiprocessing as mp
import sys, os


class RandomFastText():
    def __init__(self, n_classifier=100, dim=100, ws=5, lr=0.1, epoch=5, dim_ratio=0.8, min_count=1
                 ):
        self.n_classifier = n_classifier
        self.dim_ratio = dim_ratio
        self.dim = dim
        self.ws = ws
        self.lr = lr
        self.epoch = epoch
        self.min_count = min_count


    def train(self, text_file, word_vec_file):

        print("Trainning Start:")
        file_stats = self._get_file_stats(text_file)
        pool = mp.Pool()
        for i in range(self.n_classifier):
            pool.apply_async(func=self._train_one,
                           args=(text_file, file_stats, word_vec_file, i+1),
                           )
        pool.close()
        pool.join()


    def predict(self, text_list):
        predict_output = "pred.out"
        n_preds = len(text_list)
        label_pattern = re.compile('__label__.+')
        with open(predict_output, 'w', encoding='UTF-8') as f:
            pool = mp.Pool()
            # shared_text = mp.Manager().list(text_list)
            shared_text = text_list
            for i in range(self.n_classifier):
                res = pool.apply_async(func=self._predict_one, args=(shared_text,i+1))
                line = ' '.join([label_pattern.findall(pred)[0]
                                 for pred in res.get()])
                f.write(line+'\n')
            pool.close()
            pool.join()
        preds = self._merge_and_del(predict_output, n_preds)
        return preds


    def _merge_and_del(self, file, n_cols):
        counters = [Counter() for _ in range(n_cols)]
        with open(file, 'r', encoding='UTF-8') as f:
            for line in f:
                for i, label in enumerate(line.split(' ')):
                    counters[i][label] += 1
        #os.remove(file)
        return [cnt.most_common(1)[0][0] for cnt in counters]


    def _train_one(self, text_file, text_file_stats, word_vec_file, job_id):
        path = 'models'
        if not os.path.exists(path):
            os.mkdir(path)
        print("Processing Job[{}/{}]...\n".format(job_id, self.n_classifier))
        sampled_vec = self._get_sample_vec(word_vec_file, job_id)
        sub_text = self._get_sample_text(text_file, text_file_stats, job_id)
        model = fastText.train_supervised(sub_text,
                                    dim=int(self.dim_ratio*self.dim),
                                    ws=self.ws, lr=self.lr,
                                    epoch=self.epoch,
                                    pretrainedVectors=sampled_vec,
                                    bucket=0
                                    )

        model.save_model("{}/{}.bin".format(path, job_id))
        os.remove(sampled_vec)
        os.remove(sub_text)
        print("Job[{}/{}] Done.\n".format(job_id, self.n_classifier))


    def _predict_one(self, text_list, job_id):
        model = fastText.load_model("models/{}.bin".format(job_id))
        preds, _ = model.predict(list(text_list))
        return preds


    def _get_file_stats(self, file):
        pattern = re.compile(b"__label__(.+)")
        labels = defaultdict(list)
        count = 0
        with open(file, 'rb') as thefile:
            for i, line in enumerate(thefile):
                label = pattern.findall(line)[0].decode('utf8')
                labels[label].append(i)
                count += 1
        return count, labels


    def _get_sample_vec(self, word_vec_file, job_id):
        output_file = "sample_vec_{}.vec".format(job_id)
        with open(word_vec_file, 'r', encoding='UTF-8') as wvf:
            line = wvf.readline()
            n_words, dim = line.split(' ')
            n_sam_dims = int(int(dim) * self.dim_ratio)
            with open(output_file, 'w', encoding='UTF-8') as f:
                f.write("{} {}\n".format(n_words, n_sam_dims))
                col_idx = random.sample(range(1, int(dim)+1), n_sam_dims)
                for line in wvf:
                    sp = line.split(' ')
                    f.write(sp[0] + ' ')
                    f.write(' '.join([sp[idx] for idx in col_idx]))
                    f.write('\n')
        return output_file


    def _bootstrap_by_dist(self, n, label_dict):
        buckets = [0] * n
        for label, indice in label_dict.items():
            rand = (random.choice(indice) for _ in range(len(indice)))
            for num in rand:
                buckets[num] += 1
        return buckets

    def _bootstrap_by_dist2(self, n, label_dict):
        buckets = [0] * n
        for label, indice in label_dict.items():
            for num in random.sample(indice, int(len(indice))):
                buckets[num] = 1
        return buckets


    def _get_sample_text(self, text_file, text_file_stats, job_id):
        buckets = self._bootstrap_by_dist(*text_file_stats)
        output_file = "sub_text_{}.txt".format(job_id)
        with open(text_file, 'rb') as tf:
            with open(output_file, 'wb') as of:
                for lno, line in enumerate(tf):
                    while buckets[lno]:
                        of.write(line)
                        buckets[lno] -= 1
        return output_file

    def _gen_word_vec_file(self, model, vec_file_name):
        with open(vec_file_name, 'w',encoding='UTF-8') as f:
            f.write("{} {}\n".format(len(model.get_words()), model.get_dimension()))
            for word in model.get_words():
                vec = model.get_word_vector(word)
                f.write("{} {}\n".format(word, ' '.join(map(str, vec))))
        return vec_file_name