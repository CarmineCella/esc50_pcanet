import matplotlib
import numpy as np
import scipy as scp
import pywt
import fnmatch
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
from keras.layers.core import Activation, Dense, Permute, Merge, Lambda, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, ZeroPadding1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling1D
from keras.utils.io_utils import HDF5Matrix


from transform_scatt import load_transform, Concat_scat_tree, Concat_scal, JointScat
from compute_features import compute_features, load_features, compute_features_listfiles
from plot_utils import plot_confusion_matrix



import pdb
import logging
logging.basicConfig(filename='log.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')




class Convolution1D_4Tensor(Layer):
    def __init__(self, dim1_shape, nb_filter, filter_length,
                 activation=None, subsample_length=1, **kwargs):
        self.dim1_shape = dim1_shape
        self.convolution1D = Convolution1D(nb_filter, filter_length,
                                           subsample_length=subsample_length,
                                           activation=activation)
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


class GlobalAveragePooling1D_4Tensor(Layer):
    def __init__(self, dim1_shape, **kwargs):
        self.dim1_shape = dim1_shape
        self.pooling_layer = GlobalAveragePooling1D()
        #output_dim = Convolution1D.output_shape
        super(GlobalAveragePooling1D_4Tensor, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape_pooling = (input_shape[0],input_shape[2],input_shape[3])
        self.pooling_layer.build(input_shape_pooling)
        super(GlobalAveragePooling1D_4Tensor, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        x_pooled = []
        for i in range(self.dim1_shape):
            x_pooled_i = self.pooling_layer(x[:,i])
            x_pooled_i = K.expand_dims(x_pooled_i, dim=1)
            x_pooled.append(x_pooled_i)
        x = K.concatenate(x_pooled, axis=1)
        return x

    def get_output_shape_for(self, input_shape):
        input_shape3D = list(input_shape)
        input_shape3D.pop(1)
        input_shape3D = tuple(input_shape3D)

        output_pool_shape = self.pooling_layer.get_output_shape_for(input_shape3D)

        output_shape = list(output_pool_shape)
        output_shape.insert(1, self.dim1_shape)
        output_shape = tuple(output_shape)

        return output_shape



def learn_joint_scat_model(nOctaves, nfo, nfo2, nClasses=50, n_samples=256,
                           filter_factor=4, length_factor=2):
    input_scat0 = Input(shape=(1,))
    input_scat1 = Input(shape=(nfo*nOctaves,))

    # Treat the real part
    inputs_scat2_real = [Input(shape=(j2*nfo , nfo2, n_samples)) for j2 in range(1, nOctaves)]
    conv_1_real = [Convolution1D_4Tensor(n_samples, j2*filter_factor, j2*length_factor,
                                         activation=None, subsample_length=3) \
                   for j2 in range(1, nOctaves)]

    x_list_real = [Permute((3,1,2))(x) for x in inputs_scat2_real]
    x_list_real = [conv(x) for (conv, x) in zip(conv_1_real, x_list_real)]
    #x_list_real = [Permute((3,2,1))(x) for x in x_list_real]

    # Treat the imaginary part
    inputs_scat2_imag = [Input(shape=(j2*nfo , nfo2, n_samples)) \
                         for j2 in range(1, nOctaves)]
    conv_1_imag = [Convolution1D_4Tensor(n_samples, j2*filter_factor, j2*length_factor,
                                         activation=None, subsample_length=3) \
                   for j2 in range(1, nOctaves)]

    x_list_imag = [Permute((3,1,2))(x) for x in inputs_scat2_imag]
    x_list_imag = [conv(x) for (conv, x) in zip(conv_1_imag, x_list_imag)]
    #x_list_imag = [Permute((3,2,1))(x) for x in x_list_imag]

    # Merge the real and imaginary parts.
    def complex_modulus(x_real, x_imag):
        pow2 = Lambda((lambda x: x**2))
        sqrt = Lambda((lambda x: x**(1./2)))
        return sqrt(merge([pow2(x_real), pow2(x_imag)], mode='sum'))

    x_list = [complex_modulus(x_real, x_imag) for (x_real, x_imag) in zip(x_list_real, x_list_imag)]

    # Average
    pool2D = GlobalAveragePooling1D_4Tensor(n_samples)
    x_pooled_list = [pool2D(x) for x in x_list]
    x = merge(x_pooled_list, mode='concat', concat_axis=2)

    # Conv1
    x = ZeroPadding1D(padding=8)(x)
    x = Convolution1D(32, 16, activation='relu', subsample_length=2)(x)
    x = MaxPooling1D(pool_length=2, stride=2)(x)

    # Conv2
    x = ZeroPadding1D(padding=4)(x)
    x = Convolution1D(64, 8, activation='relu', subsample_length=2)(x)
    x = MaxPooling1D(pool_length=2, stride=2)(x)

    # Conv3
    #x = ZeroPadding1D(padding=8)(x)
    #x = Convolution1D(128, 16, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=8, stride=8)(x)

    # Conv4
    x = ZeroPadding1D(padding=2)(x)
    x = Convolution1D(128, 4, activation='relu', subsample_length=2)(x)

    # Conv5
    x = ZeroPadding1D(padding=2)(x)
    x = Convolution1D(256, 4, activation='relu', subsample_length=4)(x)

    # Flatten
    x = Flatten()(x)

    x_merged_list = [input_scat0, input_scat1, x]


    #x_pooled_list.extend([pool2D(x) for x in x_list])
    x_merged = merge(x_merged_list, mode='concat', concat_axis=1)

    # Clasiffy
    representation = Dense(128)(x_merged)
    representation = Dropout(0.5)(representation)
    representation = Dense(128)(representation)
    #representation = Dropout(0.5)(representation)
    representation = Dense(nClasses)(representation)
    output = Activation("softmax")(representation)

    inputs = [input_scat0, input_scat1]
    inputs.extend(inputs_scat2_real)
    inputs.extend(inputs_scat2_imag)
    model = Model(input=inputs, output=output)
    return model


def get_train_test_generator_h5(h5location, test_size=0.20, nsamples=2000, **kwargs):
    h5file = h5py.File(h5location, "r")
    y = h5file['labels']
    labels_train, labels_test, _, _ = \
        train_test_split(np.arange(2000), np.array(y), test_size=test_size,
                         random_state=42, stratify=y)
    gen_train = generator_scat_h5(h5location, int((1-test_size)*nsamples), labels=labels_train, **kwargs)
    gen_test  = generator_scat_h5(h5location,     int(test_size*nsamples), labels=labels_test,  **kwargs)
    return gen_train, gen_test


def get_train_test_generator_scat(directory, test_size=0.20, nsamples=2000,
                                  **kwargs):
    classes = 0
    listfiles = []
    labels = []
    for root, dir, files in os.walk(directory):
        files_filtered = fnmatch.filter(files, "*.ogg")
        if len(files_filtered) == 0:
            continue
        listfiles.extend([os.path.join(root, f) for f in files_filtered])
        labels.extend([classes for _ in range(len(files_filtered))])
        classes += 1

    files_train, files_test, labels_train, labels_test = \
        train_test_split(listfiles, np.array(labels), test_size=test_size,
                         random_state=42, stratify=labels)
    gen_train = generator_scat(files_train, labels_train, nclasses=classes,
                               **kwargs)
    gen_test = generator_scat(files_test,   labels_test,  nclasses=classes,
                               **kwargs)
    return gen_train, gen_test



class generator_scat(object):
    def __init__(self, listfiles, listlabels, batch_size=32, nclasses=50,
                 nOctaves=8, nfo1=12, nfo2=12, params=None):
        self.listfiles = listfiles
        self.Y = listlabels
        self.batch_size = batch_size
        #self.labels = labels
        #if self.labels is None:
        #    self.labels = np.arange(nsamples)
        self.count = 0
        self.nclasses = nclasses
        self.nOctaves = nOctaves
        self.nfo1 = nfo1
        self.nfo2 = nfo2
        self.nsamples = len(listfiles)
        if params is None:
            self.params = params = {'channels': (84,12), 'hops': (512,4),
              'fmin':32.7, 'fmax':11001,
              'alphas':(6,6),'Qs':(12,12), # only used for flex scattering
              'nclasses': 50, 'max_sample_size':2**17,
              'audio_ext':'*.ogg'}
        else:
            self.params = params

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
        files = self.listfiles[c:c+self.batch_size]
        #lab = sorted(self.labels[c:c+self.batch_size])

        scat_results = compute_features_listfiles(files, "plain_scat_2_tree",
                                self.params, nOctaves = self.nOctaves,
                                nfo1 = self.nfo1, nfo2 = self.nfo2)

        X0, X1, X2 = [[x[i] for x in scat_results] for i in range(3)]

        X0 = self.normalizer_mean(np.stack(X0))
        X1 = self.normalizer_mean(np.stack(X1))

        X2_list = [np.stack([x2[:j2*self.nfo1,j2*self.nfo2:(j2+1)*self.nfo2,:] \
                             for x2 in X2]) \
                    for j2 in range(1, self.nOctaves)]
        X2_list_real = [x2.real for x2 in X2_list]
        X2_list_imag = [x2.imag for x2 in X2_list]

        out = [X0, X1]
        out.extend(X2_list_real)
        out.extend(X2_list_imag)

        y = self.Y[c:c+self.batch_size]
        y_binarized = label_binarize(y, np.arange(self.nclasses))

        self.count += self.batch_size
        if self.count + self.batch_size > self.nsamples:
            self.count = 0

        return (out, y_binarized)



class generator_scat_h5(object):
    def __init__(self, h5location, nsamples, batch_size=32, labels=None,
                 nclasses=50, nOctaves=10):
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
        X2_list_real = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:].real for j2 in range(1, self.nOctaves)]
        X2_list_imag = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:].real for j2 in range(1, self.nOctaves)]

        out = [X0, X1]
        out.extend(X2_list_real)
        out.extend(X2_list_imag)

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




    model = learn_joint_scat_model(nOctaves, nfo, nfo2,
                                   nClasses=params['nclasses'], n_samples=256)

    model.compile(optimizer='rmsprop', metrics=['categorical_accuracy'],
                  loss='categorical_crossentropy')
    print(model.summary())
    #model.load_weights("weights_models/model1.npy")
    test_size = 0.2
    samples_train = int((1-test_size)*2000)
    samples_test = int(test_size*2000)
    gen_train, gen_test = get_train_test_generator_h5(h5location,
                                                   test_size=test_size,
                                                   batch_size=4)
    #gen_train, gen_test = \
    #    get_train_test_generator_scat("/users/data/blier/ESC-50", test_size=0.20,
    #                                  nsamples=2000, batch_size=1, nOctaves=nOctaves,
    #                                  nfo1=nfo, nfo2=nfo2)
    #y_binarized = label_binarize(y, np.arange(params['nclasses']))
    #model.fit(inputs, Y, nb_epoch=500, batch_size=32, validation_split=0.20)

    model.fit_generator(gen_train, samples_per_epoch=samples_train,
                        nb_epoch=200, validation_data=gen_test,
                        nb_val_samples=samples_test, max_q_size=1, nb_worker=1)
