import pandas as pd
from common import config
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self):
        self.result = pd.read_csv("new_entity.csv")
        self.categories = set(self.result["calss"].values)
        with open("new_600.txt", "r") as f:
            self.data = [sentences[:-1] for sentences in f.readlines()]
        self.dic = self.vocab_dict()
        tag2label = {"B-DES": 1, "I-DES": 2, "B-SYD": 3, "I-SYD": 4, "B-DIS": 5, "I-DIS": 6, "B-MED": 7, "I-MED": 8,
                     "B-SUR": 9, "I-SUR": 10, "B-ANA": 11, "I-ANA": 12, "O": 13, "< PAD >": 0}
        _, self.seq_len = self.convert_label()
        X = self.padding_sentences([self.convert_data(x, self.dic) for x in self.data], config.padding_size)
        y = np.array([self.convert_data(tag, tag2label) for tag in self.convert_label()[0]])
        self.X, self.x_val, self.y, self.y_val = train_test_split(
            np.concatenate([X, np.array(self.seq_len)[:, np.newaxis]], axis=1), y, test_size=config.val_size)

    def vocab_dict(self):
        vocab = set([word for sentence in self.data for word in sentence])
        vocab_dict = dict(zip(vocab, range(2, len(vocab), 1)))
        vocab_dict["< PAD >"], vocab_dict["< UNK >"] = 0, 1
        return vocab_dict

    @staticmethod
    def convert_data(sentence, vocab_dic):
        data = np.zeros((len(sentence)))
        for idx, word in enumerate(sentence):
            if word in vocab_dic:
                data[idx] = vocab_dic[word]
            else:
                data[idx] = vocab_dic["< UNK >"]
        return data

    @staticmethod
    def padding_sentences(sentences, length):
        paddles = np.zeros((len(sentences), length))
        for i, sentence in enumerate(sentences):
            if length > sentence.shape[0]:
                paddle = np.concatenate((sentence, np.zeros((length - sentence.shape[0]))), axis=0)
                paddles[i, :] = paddle
            elif length < sentence.shape[0]:
                paddle = sentence[:length]
                paddles[i, :] = paddle
            else:
                paddles[i, :] = sentence
        return paddles

    def pad_sequences(self, sequences, pad_mark="< PAD >"):
        """

        :param sequences:
        :param pad_mark:
        :return:
        """

        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:config.padding_size] + [pad_mark] * max(config.padding_size - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), config.padding_size))
        return seq_list, seq_len_list

    def convert_label(self):
        label = [["O"] * len(i) for i in self.data]
        mapentity = {"独立症状": "DES", '症状描述': 'SYD', '疾病': 'DIS', '药物': 'MED', '手术': 'SUR', '解剖部位': 'ANA'}
        for index, row in self.result.iterrows():
            if int(row["begin"]) >= len(label[int(row["case_id"])]):
                print(row["case_id"], int(row["begin"]))
            label[int(row["case_id"])][int(row["begin"])] = "B-" + mapentity[row["calss"]]
            for i in range(int(row["begin"]) + 1, int(row["end"]), 1):
                label[int(row["case_id"])][i] = "I-" + mapentity[row["calss"]]
        return self.pad_sequences(label)

    def one_batch_train(self, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        num_batches_per_epoch = int((config.train_size - 1) / config.batch_size) + 1
        for epoch in range(config.nr_epoch):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(config.train_size))
                shuffled_data_X, shuffled_data_y, shuffle_seq_lenth = self.X[shuffle_indices, :-1], self.y[
                                                                                                    shuffle_indices, :], \
                                                                      self.X[shuffle_indices, -1]
            else:
                shuffled_data_X, shuffled_data_y, shuffle_seq_lenth = self.X[:, :-1], self.y, self.X[:, -1]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * config.batch_size
                end_index = min((batch_num + 1) * config.batch_size, config.train_size)
                yield shuffled_data_X[start_index:end_index], shuffled_data_y[start_index:end_index], shuffle_seq_lenth[
                                                                                                      start_index:end_index]

    def one_batch_val(self, shuffle=False):
        """
        Generates a batch iterator for a dataset.
        """
        num_batches_per_epoch = int((config.val_size - 1) / config.batch_size) + 1
        for epoch in range(config.nr_epoch):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(config.val_size))
                shuffled_data_X, shuffled_data_y, shuffle_seq_lenth = self.x_val[shuffle_indices, :-1], self.y_val[
                    shuffle_indices], self.x_val[shuffle_indices, -1]
            else:
                shuffled_data_X, shuffled_data_y, shuffle_seq_lenth = self.x_val[:, :-1], self.y_val, self.x_val[:, -1]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * config.batch_size
                end_index = min((batch_num + 1) * config.batch_size, config.train_size)
                yield shuffled_data_X[start_index:end_index], shuffled_data_y[start_index:end_index], shuffle_seq_lenth[
                                                                                                      start_index:end_index]


if __name__ == "__main__":
    t = Dataset()
    t.one_batch_train()
