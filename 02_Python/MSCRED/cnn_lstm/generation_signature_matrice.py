"""
To represent the inter-correlations between different pairs of time series om a multivariate
time  series segment from t-w to t, we construct an n * n signature matrix M based upon the
pairwise inner-product of two time series within the segment.

Construct s (s = 3) signature matrices with different lengths(w = 10, 30, 60)
"""

import numpy as np
import pandas as pd
import cnn_lstm.utils as util
import os


class SignatureMatrices:
    def __init__(self):

        self.raw_data = pd.read_csv(util.raw_data_path, header=None)
        self.series_number = self.raw_data.shape[0]
        self.series_length = self.raw_data.shape[1]
        self.signature_matrices_number = int(self.series_length / util.gap_time)

        print("series_number is", self.series_number)
        print("series_length is", self.series_length)
        print("signature_matrices_number is", self.signature_matrices_number)

    def signature_matrices_generation(self, win):
        """
        Generation signature matrices according win_size and gap_time, the size of raw_data is n * T, n is the number of
        time series, T is the length of time series.
        To represent the inter-correlations between different pairs of time series in a multivariate time series segment
        from t − w to t, we construct an n × n signature matrix Mt based upon the pairwise inner-product of two time series
        within this segment.
        :param win: the length of the time series segment
        :return: the signature matrices
        """

        if win == 0:
            print("The size of win cannot be 0")

        raw_data = np.asarray(self.raw_data)
        signature_matrices = np.zeros((self.signature_matrices_number, self.series_number, self.series_number))

        for t in range(win, self.signature_matrices_number):
            raw_data_t = raw_data[:, t - win:t]
            signature_matrices[t] = np.dot(raw_data_t, raw_data_t.T) / win

        return signature_matrices

    def generate_train_test(self, signature_matrices):
        """
        Generate train and test dataset, and store them to ../data/train/train.npy and ../data/test/test.npy
        :param signature_matrices:
        :return:
        """
        train_dataset = []
        test_dataset = []

        for data_id in range(self.signature_matrices_number):
            index = data_id - util.step_max + 1
            if data_id < util.train_start_id:
                continue
            index_dataset = signature_matrices[:, index:index + util.step_max]
            if data_id < util.test_start_id:
                train_dataset.append(index_dataset)
            else:
                test_dataset.append(index_dataset)

        train_dataset = np.asarray(train_dataset)
        train_dataset = np.reshape(train_dataset, [-1, util.step_max, self.series_number, self.series_number,
                                                   signature_matrices.shape[0]])
        test_dataset = np.asarray(test_dataset)
        test_dataset = np.reshape(test_dataset, [-1, util.step_max,self.series_number, self.series_number,
                                                signature_matrices.shape[0]])

        print("train dataset shape is", train_dataset.shape)
        print("test dataset shape is", test_dataset.shape)

        train_path = "../data/train/"
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_path = train_path + "train.npy"

        test_path = "../data/test/"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        test_path = test_path + "test.npy"

        np.save(train_path, train_dataset)
        np.save(test_path, test_dataset)


if __name__ == '__main__':
    Matrices = SignatureMatrices()
    signature_matrices = []

    # Generation signature matrices according the win size w
    for w in util.win_size:
        signature_matrices.append(Matrices.signature_matrices_generation(w))

    signature_matrices = np.asarray(signature_matrices)
    print("the shape of signature_matrices is", signature_matrices.shape)

    # Generate train and test dataset
    Matrices.generate_train_test(signature_matrices)
