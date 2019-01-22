'''
Data generator reference:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
You only need to change __data_generation method to navigate where images are stored.
'''

import numpy as np
import keras
from augment import seq
import pickle
import os
from PIL import Image

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

with open('features.p', 'rb') as handle:
    features = pickle.load(handle)

with open('normalizer.p', 'rb') as handle:
    mms = pickle.load(handle)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, labels, n_classes, batch_size=128, shape=(75,75,3),
                 feat_shape=(16,), shuffle=True, augment=False):
        'Initialization'
        self.shape = shape
        self.batch_size = batch_size
        self.feat_shape = feat_shape
        self.labels = labels
        self.paths = paths
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        if augment:
            self.seq = seq
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        paths = [self.paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(paths, labels)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, paths, labels):

        X_img = np.empty((self.batch_size, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        X_feat = np.empty((self.batch_size, self.feat_shape[0]))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, (path, label) in enumerate(zip(paths, labels)):
            padded = Image.open(path)
            padded = np.array(padded, dtype=np.uint8)
            if self.augment:
                aug = self.seq.augment_image(padded)
                X_img[i,] = preprocess_input(np.array(aug, dtype=np.float32))
            else:
                X_img[i,] = preprocess_input(np.array(padded, dtype=np.float32))
            _, im_name = os.path.split(path)
            X_feat[i,] = features[im_name]
            y[i,] = label

        X_feat = mms.transform(X_feat)
        X = [X_img, X_feat]

        return X, y
