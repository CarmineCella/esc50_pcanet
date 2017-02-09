import matplotlib
import numpy as np
import scipy as scp
import pywt
import matplotlib.pyplot as plt
import os
import pickle
import h5py

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Activation, Dense, Permute, Merge
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils.io_utils import HDF5Matrix


from transform_scatt import load_transform, Concat_scat_tree, Concat_scal, JointScat
from compute_features import compute_features, load_features
from plot_utils import plot_confusion_matrix



import pdb
import logging
logging.basicConfig(filename='log.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')



class Convolution1D_4Tensor(Layer):
    def __init__(self, dim1_shape, nb_filter, filter_length, activation=None, **kwargs):
        self.dim1_shape = dim1_shape
        self.convolution1D = Convolution1D(nb_filter, filter_length, activation=activation)
        output_dim = Convolution1D.output_shape
        super(Convolution1D_4Tensor, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape_conv = (input_shape[0],input_shape[2],input_shape[3])
        self.convolution1D.build(input_shape_conv)
        self.trainable_weights = self.convolution1D.trainable_weights
        super(Convolution1D_4Tensor, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        x_conv = []
        for i in range(self.dim1_shape):
            x_conv_i = self.convolution1D(x[:,i])
            x_conv_i = K.expand_dims(x_conv_i, dim=1)
            x_conv.append(x_conv_i)
        x = K.concatenate(x_conv, axis=1)
        return x

    def get_output_shape_for(self, input_shape):
        input_shape3D = (input_shape[0], input_shape[2], input_shape[3])
        output_conv_shape = self.convolution1D.get_output_shape_for(input_shape3D)
        output_shape = (output_conv_shape[0], self.dim1_shape,
                        output_conv_shape[1], output_conv_shape[2])
        return output_shape


#n_samples = 256
#filter_factor = 2
#nClasses = 40


def learn_joint_scat_model(nOctaves, nfo, nfo2, nClasses=50, n_samples=256, filter_factor=2):
    input_scat0 = Input(shape=(1,))
    input_scat1 = Input(shape=(nfo*nOctaves,))
    inputs_scat2 = [Input(shape=(j2*nfo , nfo2, n_samples)) for j2 in range(1, nOctaves)]
    conv_1 = [Convolution1D_4Tensor(n_samples, j2*filter_factor, 2*j2, activation="relu") \
              for j2 in range(1, nOctaves)]
    x_list = [Permute((3,1,2))(x) for x in inputs_scat2]
    x_list = [conv(x) for (conv, x) in zip(conv_1, x_list)]
    x_list = [Permute((3, 2, 1))(x) for x in x_list]
    pool2D = GlobalAveragePooling2D()

    x_pooled_list = [input_scat0, input_scat1]

    x_pooled_list.extend([pool2D(x) for x in x_list])
    x_merged = merge(x_pooled_list, mode='concat', concat_axis=1)

    representation = Dense(nClasses)(x_merged)
    output = Activation("softmax")(representation)

    inputs = [input_scat0, input_scat1]
    inputs.extend(inputs_scat2)
    model = Model(input=inputs, output=output)
    return model


def get_train_test_generator(h5location, test_size=0.20, nsamples=2000, **kwargs):
    h5file = h5py.File(h5location, "r")
    y = h5file['labels']
    labels_train, labels_test, _, _ = \
        train_test_split(np.arange(2000), np.array(y), test_size=test_size,
                         random_state=42, stratify=y)
    gen_train = generator_scat_h5(h5location, int((1-test_size)*nsamples), labels=labels_train, **kwargs)
    gen_test  = generator_scat_h5(h5location,     int(test_size*nsamples), labels=labels_test,  **kwargs)
    return gen_train, gen_test

class generator_scat_h5(object):
    def __init__(self, h5location, nsamples, batch_size=32, labels=None, nclasses=50, nOctaves=10):
        self.h5file = h5py.File(h5location, "r")
        self.scat0 = self.h5file['scat0']
        self.scat1 = self.h5file['scat1']
        self.scat2 = self.h5file['scat2']
        self.Y = self.h5file['labels']
        self.batch_size = batch_size
        self.labels = labels
        if self.labels is None:
            self.labels = np.arange(nsamples)
        self.count = 0
        self.nclasses = nclasses
        self.nOctaves = nOctaves
        self.nsamples = nsamples



    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def normalizer_mean(self, X):
        log_eps = 0.0001
        return np.mean(np.log(log_eps+np.abs(X)), axis=-1)


    def next(self):
        c = self.count
        #lab = sorted(self.labels[c:c+self.batch_size])
        lab = sorted(self.labels[c:c+self.batch_size])


        X0 = np.array(self.scat0[lab,:])
        X0 = self.normalizer_mean(X0)
        X1 = np.array(self.scat1[lab,:,:])
        X1 = self.normalizer_mean(X1)
        X2 = np.array(self.scat2[lab,:,:,:])
        X2_list = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:] for j2 in range(1, self.nOctaves)]

        out = [X0, X1]
        out.extend(X2_list)

        y = self.Y[lab]
        y_binarized = label_binarize(y, np.arange(self.nclasses))

        #logging.debug((c, lab, self.nsamples, y))
        self.count += self.batch_size
        if self.count + self.batch_size > self.nsamples:
            self.count = 0

        return (out, y_binarized)



if __name__ == "__main__":
    params = {'channels': (84,12), 'hops': (512,4),
              'fmin':32.7, 'fmax':11001,
              'alphas':(6,6),'Qs':(12,12), # only used for flex scattering
              'nclasses': 50, 'n_itemsbyclass':40, 'max_sample_size':2**17,
              'audio_ext':'*.ogg'}

    nOctaves=10
    nfo=12
    nfo2=12

    #directory = "/users/data/blier/features_esc50/scat_10_12_12/"
    #X, y = load_features(directory, params['nclasses'], params['n_itemsbyclass'])






    #h5file = h5py.File(h5location, "r")
    h5location = "/users/data/blier/features_esc50/scat_10_12_12.h5"
    #X0 = HDF5Matrix(h5location, 'scat0', normalizer=normalizer_mean)
    #X1 = HDF5Matrix(h5location, 'scat1', normalizer=normalizer_mean)
    #inputs = [X0, X1]
    #X2_list = [HDF5Matrix(h5location, 'scat2', normalizer=lambda x: normalizer_scatt2(x, j2)) \
    #           for j2 in range(1, nOctaves)]
    #Y = HDF5Matrix(h5location, 'labels', normalizer=lambda y: label_binarize(y, np.arange(params['nclasses'])))
    #inputs.extend(X2_list)

    #def scat_to_list(X):
    #    X0, X1, X2 = [[x[i] for x in X] for i in range(3)]
    #    X2_list = [np.stack([x2[:j2*nfo,j2*nfo2:(j2+1)*nfo2,:] for x2 in X2]) \
    #               for j2 in range(1, nOctaves)]
    #    return X0, X1, X2_list


    model = learn_joint_scat_model(nOctaves, nfo, nfo2, filter_factor=2,
                                   nClasses=params['nclasses'], n_samples=256)

    model.compile(optimizer='rmsprop', metrics=['categorical_accuracy'], loss='categorical_crossentropy')
    print(model.summary())
    #model.load_weights("weights_models/model1.npy")
    test_size = 0.2
    samples_train = int((1-test_size)*2000)
    samples_test = 200
    gen_train, gen_test = get_train_test_generator(h5location, test_size=test_size, batch_size=32)
    #y_binarized = label_binarize(y, np.arange(params['nclasses']))
    #model.fit(inputs, Y, nb_epoch=500, batch_size=32, validation_split=0.20)

    model.fit_generator(gen_train, samples_per_epoch=samples_train,
                        nb_epoch=200, validation_data=gen_test, nb_val_samples=samples_test)
