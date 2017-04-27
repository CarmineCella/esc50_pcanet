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
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline

from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Activation, Dense, Permute, Lambda, Dropout, Flatten, Reshape
from keras.layers.merge import Average, Concatenate
from keras.layers.convolutional import Convolution1D, ZeroPadding1D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D
from keras.utils.io_utils import HDF5Matrix
from keras.optimizers import Adagrad, Adam


from transform_scatt import load_transform, Concat_scat_tree, Concat_scal, JointScat
from compute_features import compute_features, load_features, compute_features_listfiles
from plot_utils import plot_confusion_matrix



import pdb
import logging
logging.basicConfig(filename='log.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
#import Oscar




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


class Flatten_4Tensor(Layer):
    def __init__(self, **kwargs):
        self.layer = Flatten()
        #output_dim = Convolution1D.output_shape
        super(Flatten_4Tensor, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape3D = (input_shape[0],input_shape[2],input_shape[3])
        self.layer.build(input_shape3D)
        super(Flatten_4Tensor, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        x_list = []
        dim1_shape = K.int_shape(x)[1]
        for i in range(dim1_shape):
            x_i = self.layer(x[:,i])
            x_i = K.expand_dims(x_i, dim=1)
            x_list.append(x_i)
        x = K.concatenate(x_list, axis=1)
        return x

    def get_output_shape_for(self, input_shape):
        input_shape3D = list(input_shape)
        dim1_shape = input_shape3D.pop(1)
        input_shape3D = tuple(input_shape3D)

        output_pool_shape = self.layer.get_output_shape_for(input_shape3D)

        output_shape = list(output_pool_shape)
        output_shape.insert(1, dim1_shape)
        output_shape = tuple(output_shape)

        return output_shape


def model_joint_scat(nOctaves, nfo, nfo2, nClasses=50, n_samples=256,
                           filter_factor=4, length_factor=2):
    input_scat0 = Input(shape=(1,))
    input_scat1 = Input(shape=(nfo*nOctaves,))
    inputs_scat2_real = [Input(shape=(j2*nfo , nfo2, n_samples)) \
                         for j2 in range(1, nOctaves)]
    inputs_scat2_imag = [Input(shape=(j2*nfo , nfo2, n_samples)) \
                         for j2 in range(1, nOctaves)]

    # Treat the real part
    conv_1_real = [Convolution1D_4Tensor(n_samples, j2*filter_factor, j2*length_factor,
                                         activation=None, subsample_length=3) \
                   for j2 in range(1, nOctaves)]
    x_list_real = [Permute((3,1,2))(x) for x in inputs_scat2_real]
    x_list_real = [conv(x) for (conv, x) in zip(conv_1_real, x_list_real)]
    #x_list_real = [Permute((3,2,1))(x) for x in x_list_real]

    # Treat the imaginary part
    conv_1_imag = [Convolution1D_4Tensor(n_samples, j2*filter_factor, j2*length_factor,
                                         activation=None, subsample_length=3) \
                   for j2 in range(1, nOctaves)]
    x_list_imag = [Permute((3,1,2))(x) for x in inputs_scat2_imag]
    x_list_imag = [conv(x) for (conv, x) in zip(conv_1_imag, x_list_imag)]
    #x_list_imag = [Permute((3,2,1))(x) for x in x_list_imag]

    # 42 the real and imaginary parts.
    def complex_modulus(x_real, x_imag):
        pow2 = Lambda((lambda x: x**2))
        sqrt = Lambda((lambda x: x**(1./2)))
        return sqrt(merge([pow2(x_real), pow2(x_imag)], mode='sum'))

    x_list = [complex_modulus(x_real, x_imag) for (x_real, x_imag) in zip(x_list_real, x_list_imag)]

    def conv1D(x, nfilters=1, sizefilters=1, pool_length=1, subsample_length=2,
                    symetricpooling = True):
        #pdb.set_trace()

        if symetricpooling:
            padding=2**(int(sizefilters)-1)
        else:
            padding=(2**(int(sizefilters)-1),2**(int(sizefilters)-1)-1)
        x = ZeroPadding1D(padding)(x)
        x = Convolution1D(2**int(nfilters), 2**int(sizefilters), activation='relu',
                          subsample_length=subsample_length)(x)
        x = MaxPooling1D(pool_length=int(pool_length), stride=int(pool_length))(x)
        return x


    def conv2D(x, nfilters, nb_row, nb_col, subsample=(1,1),
                      pool_size=(2,2)):
        #pdb.set_trace()
        x = ZeroPadding2D((nb_row//2,nb_col//2))(x)
        x = Convolution2D(nfilters, nb_row, nb_col,
                          activation='relu',subsample=subsample)(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        return x


    # Conv joint 2
    filter_factor2 = 2
    #length_factor2 = 3
    x_list = [conv2D(x, (j2+1)*filter_factor2, 8, 4, pool_size=(8,2)) \
              for (j2, x) in enumerate(x_list)]

    # Conv joint 3
    filter_factor2 = 3
    #length_factor2 = 3
    x_list = [conv2D(x, (j2+1)*filter_factor2, 4, 4, pool_size=(4,2)) \
              for (j2, x) in enumerate(x_list)]

    # Conv joint 4
    #filter_factor2 = 4
    #length_factor2 = 3
    #x_list = [conv2D(x, (j2+1)*filter_factor2, 4, 4, pool_size=(4,2)) \
    #          for (j2, x) in enumerate(x_list)]

    # Average and merge
    # pool2D = GlobalAveragePooling1D_4Tensor(n_samples)
    # x_pooled_list = [pool2D(x) for x in x_list]
    # x = merge(x_pooled_list, mode='concat', concat_axis=2)
    x_list = [Flatten_4Tensor()(x) for x in x_list]
    x = Concatenate(axis=2)(x_list)

    # Conv1
    #x = ZeroPadding1D(padding=8)(x)
    #x = Convolution1D(32, 16, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=2, stride=2)(x)

    # Conv2
    #x = ZeroPadding1D(padding=4)(x)
    #x = Convolution1D(64, 8, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=2, stride=2)(x)

    # Conv3
    #x = ZeroPadding1D(padding=8)(x)
    #x = Convolution1D(128, 16, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=8, stride=8)(x)

    # Conv4
    x = ZeroPadding1D(padding=2)(x)
    x = Convolution1D(128, 4, activation='relu')(x)
    x = MaxPooling1D(pool_length=2)(x)

    # Conv5
    x = ZeroPadding1D(padding=2)(x)
    x = Convolution1D(256, 4, activation='relu')(x)
    x = MaxPooling1D(pool_length=2)(x)

    # Flatten
    x = Flatten()(x)

    x_merged_list = [input_scat0, input_scat1, x]


    #x_pooled_list.extend([pool2D(x) for x in x_list])
    x_merged = Concatenate(axis=1)(x_merged_list)

    # Clasiffy
    representation = Dense(128, activation='relu')(x_merged)
    representation = Dropout(0.5)(representation)
    representation = Dense(128, activation='relu')(representation)
    #representation = Dropout(0.5)(representation)
    representation = Dense(nClasses)(representation)
    output = Activation("softmax")(representation)

    inputs = [input_scat0, input_scat1]
    inputs.extend(inputs_scat2_real)
    inputs.extend(inputs_scat2_imag)
    model = Model(input=inputs, output=output)
    return model


def model_scat2(nOctaves, nfo, nfo2, hyperparams, nClasses=50, n_samples=256,
                filter_factor=4, length_factor=2):
    input_scat0 = Input(shape=(n_samples,))
    scat0 = Lambda((lambda x: K.expand_dims(x, dim=1)))(input_scat0)
    scat0 = Permute((2,1))(scat0)

    input_scat1 = Input(shape=(nfo*nOctaves,n_samples))
    scat1 = Permute((2,1))(input_scat1)


    inputs_scat2_real = [Input(shape=(j2*nfo , nfo2, n_samples)) \
                         for j2 in range(1, nOctaves)]
    inputs_scat2_imag = [Input(shape=(j2*nfo , nfo2, n_samples)) \
                         for j2 in range(1, nOctaves)]

    # Merge the real and imaginary part
    def complex_modulus(x_real, x_imag):
        pow2 = Lambda((lambda x: x**2))
        sqrt = Lambda((lambda x: x**(1./2)))
        return sqrt(merge([pow2(x_real), pow2(x_imag)], mode='sum'))

    x2_list = [complex_modulus(x_real, x_imag) for (x_real, x_imag) \
              in zip(inputs_scat2_real, inputs_scat2_imag)]



    # Merge
    x2_list = [Reshape((j2*nfo*nfo2, n_samples))(x) for (j2, x) \
              in zip(range(1, nOctaves), x2_list)]
    x2_list = [Permute((2,1))(x) for x in x2_list]

    #x_list = []
    #x_list.extend([scat0, scat1])
    x2 = Concatenate(axis=2)(x2_list)

    def convolution(x, nfilters=1, sizefilters=1, pool_length=1, subsample_length=1,
                    symetricpooling = True):
        #pdb.set_trace()

        if symetricpooling:
            padding=2**(int(sizefilters)-1)
        else:
            padding=(2**(int(sizefilters)-1),2**(int(sizefilters)-1)-1)
        x = ZeroPadding1D(padding)(x)
        x = Convolution1D(2**int(nfilters), 2**int(sizefilters), activation='relu',
                          subsample_length=subsample_length)(x)
        x = MaxPooling1D(pool_length=int(pool_length), stride=int(pool_length))(x)
        return x

    def get_conv_params(hyperparams, n_layers):
        conv_params = []
        for i in range(n_layers):
            prefix = "conv"+str(i+1)+"_"
            keys = [k for k in hyperparams.keys() if k.startswith(prefix)]
            params = dict((k.partition(prefix)[-1],hyperparams[k]) for k in keys)
            conv_params.append(params)
        return conv_params

    conv_params = get_conv_params(hyperparams, 5)


    # Conv1
    x = scat0
    x = convolution(x, symetricpooling=False,# subsample_length=1,
                    **conv_params[0])
    #x = ZeroPadding1D(padding=(8,7))(x)
    #x = Convolution1D(4, 16, activation='relu')(x)
    #x = MaxPooling1D(pool_length=2, stride=2)(x)
    x = Concatenate(axis=2)([x, scat1])
    ### #x = Dropout(0.5)(x)

    # Conv 2
    x = convolution(x, symetricpooling=False,# subsample_length=1,
                    **conv_params[1])
    #x = ZeroPadding1D(padding=(8,7))(x)
    #x = Convolution1D(8, 16, activation='relu')(x)
    #x = MaxPooling1D(pool_length=2, stride=2)(x)
    x = Concatenate(axis=2)([x, x2])
    #x = Dropout(0.5)(x)


    # Conv3
    #x = convolution(x, **conv_params[2])
    #x = ZeroPadding1D(padding=4)(x)
    #x = Convolution1D(16, 8, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=8, stride=8)(x)
    #x = Dropout(0.5)(x)

    # Conv4
    #x = convolution(x, **conv_params[3])
    #x = ZeroPadding1D(padding=2)(x)
    #x = Convolution1D(32, 4, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=4, stride=4)(x)
    #x = Dropout(0.5)(x)

    # Conv5
    #x = convolution(x, **conv_params[4])
    #x = ZeroPadding1D(padding=2)(x)
    #x = Convolution1D(64, 4, activation='relu', subsample_length=2)(x)
    #x = MaxPooling1D(pool_length=4, stride=4)(x)
    #x = Dropout(0.5)(x)

    # Conv6
    #x = ZeroPadding1D(padding=2)(x)
    #x = Convolution1D(128, 4, activation='relu', subsample_length=2)(x)
    #x = Dropout(0.5)(x)

    # Flatten
    x = GlobalAveragePooling1D()(x)
    #x = Flatten()(x)

    #x_merged_list = [input_scat0, input_scat1, x]


    #x_pooled_list.extend([pool2D(x) for x in x_list])
    #x_merged = merge(x_merged_list, mode='concat', concat_axis=1)

    # Clasiffy
    representation = Dense(2**int(hyperparams["dense1"]), activation='relu')(x)
    representation = Dropout(hyperparams["dropout"])(representation)
    #representation = Dense(2**(hyperparams["dense2"]), activation='relu')(representation)
    #representation = Dropout(hyperparams["dropout"])(representation)
    representation = Dense(nClasses)(representation)
    output = Activation("softmax")(representation)

    inputs = [input_scat0, input_scat1]
    inputs.extend(inputs_scat2_real)
    inputs.extend(inputs_scat2_imag)
    model = Model(input=inputs, output=output)
    return model


def model_scat2_av(nOctaves, nfo, nfo2, nClasses=50):


    input_scat0 = Input(shape=(1,))
    input_scat1 = Input(shape=(nfo*nOctaves,))


    inputs_scat2 = [Input(shape=(j2*nfo , nfo2)) \
                         for j2 in range(1, nOctaves)]


    # Merge
    x2_list = [Flatten()(x) for x in inputs_scat2]
    x2 = Concatenate()(x2_list)

    x = Concatenate()([input_scat0, input_scat1, x2])





    # Clasiffy
    #representation=x
    representation = Dense(4096, activation='relu')(x)
    representation = Dropout(0.5)(representation)
    representation = Dense(4096, activation='relu')(representation)
    representation = Dropout(0.5)(representation)
    representation = Dense(4096, activation='relu')(representation)
    representation = Dropout(0.5)(representation)
    representation = Dense(nClasses)(representation)
    output = Activation("softmax")(representation)

    inputs = [input_scat0, input_scat1]
    inputs.extend(inputs_scat2)
    model = Model(input=inputs, output=output)
    return model





def get_train_test_generator_h5(h5location, test_size=0.20, nsamples=2000,
                                dataaugmentation = False, **kwargs):
    h5file = h5py.File(h5location, "r")
    y = h5file['labels']
    labels_train, labels_test, _, _ = \
        train_test_split(np.arange(2000), np.array(y), test_size=test_size,
                         random_state=42, stratify=y)
    gen_train = generator_scat_h5(h5location, int((1-test_size)*nsamples),
                                  labels=labels_train,
                                  dataaugmentation=dataaugmentation,
                                  **kwargs)
    gen_test  = generator_scat_h5(h5location,     int(test_size*nsamples),
                                  labels=labels_test,  **kwargs)
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
                 nclasses=50, nOctaves=10, dataaugmentation = False):
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
        self.dataaugmentation = dataaugmentation



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

        log_eps = 0.0001
        X0 = np.array(self.scat0[lab,:])
        X0 = np.log(log_eps + X0[:,0])
        #X0 = self.normalizer_mean(X0)
        X1 = np.array(self.scat1[lab,:,:])
        X1 = np.log(log_eps + X1[:,:,0])
        #X1 = self.normalizer_mean(X1)

        #First version
        X2 = np.array(self.scat2[lab,:,:,:])
        #X2_list_real = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:].real for j2 in range(1, self.nOctaves)]
        #X2_list_imag = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:].real for j2 in range(1, self.nOctaves)]
        X2 = np.log(log_eps + np.abs(X2[:,:,:,0]))
        X2_list = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2] for j2 in range(1, self.nOctaves)]
        out = [X0, X1]
        #out.extend(X2_list_real)
        #out.extend(X2_list_imag)
        out.extend(X2_list)

        if self.dataaugmentation:
            shift = np.random.randint(0, high=X0.shape[-1])
            out = [np.roll(x, shift, axis=-1) for x in out]

        y = self.Y[lab]
        y_binarized = label_binarize(y, np.arange(self.nclasses))

        #logging.debug((c, lab, self.nsamples, y))
        self.count += self.batch_size
        if self.count + self.batch_size > self.nsamples:
            self.count = 0

        return (out, y_binarized)


def _normalizer_mean(self, X):
    log_eps = 0.0001
    return np.mean(np.log(log_eps+np.abs(X)), axis=-1)

def load_h5(h5location, nOctaves, nclasses=50, log_eps=0.0001):
    h5file = h5py.File(h5location, "r")
    scat0 = h5file['scat0']
    scat1 = h5file['scat1']
    scat2 = h5file['scat2']
    y = np.array(h5file['labels'])

    X0 = np.array(scat0)
    X0 = np.log(log_eps + X0[:,0])

    #X0 = self.normalizer_mean(X0)
    X1 = np.array(scat1)
    X1 = np.log(log_eps + X1[:,:,0])
    #X1 = self.normalizer_mean(X1)

    #First version
    X2 = np.array(scat2)
    #X2_list_real = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:].real for j2 in range(1, self.nOctaves)]
    #X2_list_imag = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2,:].real for j2 in range(1, self.nOctaves)]
    X2 = np.log(log_eps + np.abs(X2[:,:,:,0]))
    X2_list = [X2[:,:j2*nfo,j2*nfo2:(j2+1)*nfo2] for j2 in range(1, nOctaves)]
    out = [X0, X1]
    #out.extend(X2_list_real)
    #out.extend(X2_list_imag)
    out.extend(X2_list)
    pdb.set_trace()


    #y_binarized = label_binarize(y, np.arange(nclasses))

    return out, y
    #return (out, y)

def flatten_concat(l):
    n = l[0].shape[0]
    l = [x if x.ndim > 1 else x[:,np.newaxis] for x in l]
    return np.hstack([u.reshape((n,-1)) for u in l])

def joint_scat(X):
    eps = 0.0001
    scat2 = X[2:]
    wavelet = pywt.Wavelet('db2')

    jointcoeff = []
    for x in X[2:]:
        wdec = pywt.wavedec(x, wavelet, axis=1)
        wdec = [np.log(eps+np.abs(c)) for c in wdec]
        #wdec = [np.reshape(c, (x.shape[0], -1)) for c in wdec]
        #wdec = np.concatenate(wdec, axis=1)
        jointcoeff.extend(wdec)

    coeffs = [X[0], X[1]]
    coeffs.extend(jointcoeff)

    # jointcoeff = np.concatenate(jointcoeff, axis=1)
    # #pdb.set_trace()
    # scat1 = X[1].reshape((X[1].shape[0],-1))
    # Xout = np.hstack([X[0][:,np.newaxis], X[1], jointcoeff])
    return flatten_concat(coeffs)


if __name__ == "__main__":
    load_generator = False

    # params = {'channels': (84,12), 'hops': (512,4),
    #           'fmin':32.7, 'fmax':11001,
    #           'alphas':(6,6),'Qs':(12,12), # only used for flex scattering
    #           'nclasses': 50, 'n_itemsbyclass':40, 'max_sample_size':2**17,
    #           'audio_ext':'*.ogg'}

    nOctaves=16
    nfo=12
    nfo2=4
    n_samples = 1 #1024

    #directory = "/users/data/blier/features_esc50/scat_10_12_12/"
    #X, y = load_features(directory, params['nclasses'], params['n_itemsbyclass'])

    #h5file = h5py.File(h5location, "r")

    h5location = "features_esc50/scat_18_12_4.h5"
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





    #model.load_weights("weights_models/model1.npy")
    if load_generator:
        test_size = 0.2
        samples_train = int((1-test_size)*2000)
        samples_test = int(test_size*2000)
        gen_train, gen_test = get_train_test_generator_h5(h5location,
                                                       test_size=test_size,
                                                       batch_size=16,
                                                       nOctaves=nOctaves,
                                                       dataaugmentation=False)
        #gen_train, gen_test = \
        #    get_train_test_generator_scat("/users/data/blier/ESC-50", test_size=0.20,
        #                                  nsamples=2000, batch_size=1, nOctaves=nOctaves,
        #                                  nfo1=nfo, nfo2=nfo2)
        #y_binarized = label_binarize(y, np.arange(params['nclasses']))
        #model.fit(inputs, Y, nb_epoch=500, batch_size=32, validation_split=0.20)
        samples_train = 16000
        samples_test = 800

    else:
        test_size = 400
        X, y = load_h5(h5location, nOctaves)
        indexes = np.arange(y.shape[0])

        X_trans = flatten_concat(X)
        normalizer = StandardScaler()
        X_trans = normalizer.fit_transform(X_trans)

        # idx_train, idx_test, y_train, y_test = \
        #     train_test_split(indexes, y, test_size=test_size,
        #                      random_state=42, stratify=y)
        X_train, X_test, y_train, y_test = \
            train_test_split(X_trans, y, test_size=test_size,
                             random_state=44, stratify=y)

        #
        y_binarized = label_binarize(y, np.arange(50))
        y_binarized_train = label_binarize(y_train, np.arange(50))
        y_binarized_test  = label_binarize(y_test,  np.arange(50))

        # y_binarized_train = y_binarized[idx_train]
        # y_binarized_test  = y_binarized[idx_test]
        # X_train = [x[idx_train] for x in X]
        # X_test =  [x[idx_test]  for x in X]
        # Xtrans_train = joint_scat(X_train)
        # Xtrans_test  = joint_scat(X_test)
        #Xtrans_train = flatten_concat(X_train)
        #Xtrans_test = flatten_concat(X_test)


    ###########
    # # #To test with the SVM
    # #X_flat = np.vstack([Xtrans_train, Xtrans_test])
    # #y_flat = np.concatenate([y_train, y_test])
    # classifier = SVC(C=1., kernel='linear')
    # scores = cross_val_score(classifier, X_trans, y, cv=15)
    # print("--------------------")
    # print("--------------------")
    #
    # print(scores, scores.mean())
    # print("--------------------")
    # print("--------------------")


    #model = model_scat2_av(nOctaves, nfo, nfo2)
    input_model = Input(shape=(X_trans.shape[1],))
    x = input_model
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(50)(x)
    output = Activation("softmax")(x)


    model = Model(input=input_model, output=output)

    optimizer = Adam(lr=0.0001)
    #optimizer = "rmsprop"
    model.compile(optimizer=optimizer, metrics=['categorical_accuracy'],
                  loss='categorical_crossentropy')
    model.summary()

    if load_generator:
        model.fit_generator(gen_train, samples_per_epoch=samples_train, nb_epoch=20,
                            validation_data=gen_test, nb_val_samples=samples_test,
                            max_q_size=10, nb_worker=3)
    #trainloss, trainacc = history.history["loss"][-1], history.history["categorical_accuracy"][-1]

    else:




        model.fit(X_train, y_binarized_train, batch_size=32, epochs=400,
                  validation_data=(X_test, y_binarized_test))
    #        oscar_results = {"loss":testloss, "accuracy":testacc,
    #                         "train_loss":trainloss, "train_acc":trainacc}
    #        oscar.update(job,oscar_results)
    #    except (KeyboardInterrupt, SystemExit):
    #        raise
    #    except:
    #        pass
