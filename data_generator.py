'''
Data generator reference:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
You only need to change __data_generation method to navigate where images are stored.
'''

import numpy as np
import keras
from keras.applications.inception_v3 import preprocess_input
from augment import seq
import pickle
import cv2
import os

with open('normalizer.pickle', 'rb') as handle:
    mms = pickle.load(handle)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, labels, n_classes, batch_size=128, shape=(75,75,3),
                 feat_shape=(16,), shuffle=True, augment=False, multi=False):
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
        if multi:
            with open('features_multi.pickle', 'rb') as handle:
                self.features = pickle.load(handle)
        else:
            with open('features.pickle', 'rb') as handle:
                self.features = pickle.load(handle)
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

        X_img = np.empty((self.batch_size, self.shape[0], self.shape[1], self.shape[2]), dtype=np.uint8)
        X_feat = np.empty((self.batch_size, self.feat_shape[0]))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, (path, label) in enumerate(zip(paths, labels)):
            padded = cv2.imread(path, -1)
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            if self.augment:
                aug = self.seq.augment_image(padded)
                X_img[i,] = preprocess_input(aug)
            else:
                X_img[i,] = preprocess_input(padded)
            _, im_name = os.path.split(path)
            X_feat[i,] = self.features[im_name]
            y[i,] = label

        X_feat = mms.transform(X_feat)
        X = [X_img, X_feat]

        return X, y
