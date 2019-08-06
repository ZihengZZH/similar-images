import numpy as np
import matplotlib.pyplot as plt


class Dataset(object):
    def __init__(self):
        self.image_train = np.array([])
        self.image_test = np.array([])
        self.label_train = np.array([])
        self.label_test = np.array([])
        self.unique_train_label = np.array([])
        self.map_train_label_indices = dict()

    def _get_siamese_similar_pair(self):
        label =np.random.choice(self.unique_train_label)
        l, r = np.random.choice(self.map_train_label_indices[label], 2, replace=False)
        return l, r, 1

    def _get_siamese_dissimilar_pair(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        l = np.random.choice(self.map_train_label_indices[label_l])
        r = np.random.choice(self.map_train_label_indices[label_r])
        return l, r, 0

    def _get_siamese_pair(self):
        if np.random.random() < 0.5:
            return self._get_siamese_similar_pair()
        else:
            return self._get_siamese_dissimilar_pair()

    def get_siamese_batch(self, n):
        idxs_left, idxs_right, labels = [], [], []
        for _ in range(n):
            l, r, x = self._get_siamese_pair()
            idxs_left.append(l)
            idxs_right.append(r)
            labels.append(x)
        return self.image_train[idxs_left,:], self.image_train[idxs_right, :], np.expand_dims(labels, axis=1)
