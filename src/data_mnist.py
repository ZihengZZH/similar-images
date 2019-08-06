import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from src.dataset import Dataset


class DataMNIST(Dataset):
    def __init__(self):
        print("--"*20)
        print("loading MNIST dataset")
        print("--"*20)
        (self.image_train, self.label_train), (self.image_test, self.label_test) = mnist.load_data()
        self.image_train = np.expand_dims(self.image_train, axis=3) / 255.0
        self.image_test = np.expand_dims(self.image_test, axis=3) / 255.0
        self.label_train = np.expand_dims(self.label_train, axis=1)
        self.unique_train_label = np.unique(self.label_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.label_train == label) for label in self.unique_train_label}
        print("image train: ", self.image_train.shape)
        print("label train: ", self.label_train.shape)
        print("image test: ", self.image_test.shape)
        print("label test: ", self.label_test.shape)
        print("unique label", self.unique_train_label)
        print("--"*20)
    
    def vis(self):
        batch_size = 5
        ls, rs, xs = self.get_siamese_batch(batch_size)
        fig, axarr = plt.subplots(batch_size, 2, figsize=(10,10))
        for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
            print("row", idx, "label:", "similar" if x else "dissimilar")
            axarr[idx, 0].axis('off')
            axarr[idx, 1].axis('off')
            axarr[idx, 0].imshow(np.squeeze(l, axis=2))
            axarr[idx, 1].imshow(np.squeeze(r, axis=2))
        plt.savefig('images/mnist.png')
