# Data generator adapted and modified from
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# with edits to generate image-gt pairs from shuffled dictionary

import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img2gt_paths, batch_size=4, dim=(360,480), n_channels=3,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.img2gt_paths = img2gt_paths
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.img_paths = list(img2gt_paths.keys())
        self.gt_paths = list(img2gt_paths.values())
        self.indexes = np.arange(len(self.img_paths))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of paths and labels
        img_paths_temp = [self.img_paths[k] for k in indexes]
        gt_paths_temp = [self.gt_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_paths_temp, gt_paths_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_temp, gt_paths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 352, 480, 30))

        # Generate data
        for i, path in enumerate(img_paths_temp):
            img = load_img(path, target_size=self.dim)
            img_arr = img_to_array(img)
            gt = load_img(gt_paths_temp[i], grayscale=True, target_size=(352,480))
            gt_arr = img_to_array(gt)
            np.set_printoptions(threshold=10000)
            # Crate one hot encoding H x W to H x W x Classes
            gt_arr_onehot = np.squeeze(np.arange(30) == gt_arr[..., None])
            # Normalization
            img_arr = img_arr/255.0
            # Store sample
            X[i,] = img_arr
            # Store one-hot label img
            y[i] = gt_arr_onehot

        return X, y
