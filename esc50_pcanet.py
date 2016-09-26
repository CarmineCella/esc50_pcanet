# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:42:36 2016

@author: cella
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:11:14 2016

ESC-50 dataset classification with PCA nets 

@author: Carmine E. Cella, 2016, ENS Paris
"""

import os
import fnmatch
import joblib
import librosa
import numpy as np
import os.path
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mel_scat import mel_scat

db_location = '../../datasets/ESC-50-master'
sample_len = 110250 # in samples is 5 sec @ 22050
standardize = True
log_features = False
log_eps = 0.01
nfolds = 1
split = 0.25

params = {'features':'cqt',  
          'channels': (84,12), 'hops': (1024,8),
          'fmin':32.7, 'fmax':16000,
          'nclasses': 2, 'nsamples':80}
 
def get_features (file, features, channels, hops, fmin, fmax):
    y = np.zeros(sample_len);   
    yt, sr = librosa.core.load (file, mono=True)
    
    if len(yt) == 0: 
        print ('*** warning: empty file -> ' + file + '! ***')
        return 0

    min_length = min(len(y), len(yt))
    y[:min_length] = yt[:min_length]
    
    if params['features'] == 'cqt':
        return np.abs(librosa.core.cqt (y=y, sr=sr, hop_length=hops[0], 
                                        n_bins=channels[0], real=False)) 
    elif params['features'] == 'mel_scat':
        return mel_scat(y=y, sr=sr, hop_lengths=hops, channels=channels, 
             fmin=fmax, fmax=fmin, fft_size=1024)
    elif params['features'] == 'cqt_scat':
        #cqtscat(y,sr,hop,bins)            
        pass
    else:
        raise ValueError('Unkonwn features requested')

cachedir = os.path.expanduser('~/esc50_pcanet_joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=1)
cached_get_features = memory.cache(get_features)

def compute_features (root_path, params):
    features = params['features']
    channels = params['channels']
    hops = params['hops']
    fmin = params['fmin']
    fmax = params['fmax']
        
    y_data = np.zeros ((params['nsamples']))
    
    classes = 0
    samples = 0
    
    X_list = []
    for root, dir, files in os.walk(root_path):
        waves = fnmatch.filter(files, "*.wav")
        if len(waves) != 0:
            print ("class: " + root.split("/")[-1])
            for item in waves:
                l = cached_get_features(os.path.join(root, item), features, 
                                        channels, hops, fmin, fmax)
                X_list.append([l])

            for item in waves:
                y_data[samples] = classes
                samples = samples + 1
            classes = classes + 1
    
    X_flat_list = [X_list[class_id][file_id]
                for class_id in range(len(X_list))
                for file_id in range(len(X_list[class_id]))]

    X_data = np.stack(X_flat_list, axis=2)
    X_data = np.transpose(X_data, (2,0,1))
    
    print ("classes = " + str (classes))
    print ("samples = " + str (samples))

    return X_data, y_data

def create_folds (X_data, y_data, nfolds, split):    
    cv = StratifiedShuffleSplit(y_data, n_iter=nfolds, 
                                test_size=split)
    return cv
        
def standardize (X_train, X_test):
    mu = np.mean (X_train, axis=0)
    de = np.std (X_train, axis=0)
    
    eps = np.finfo('float32').eps
    X_train = (X_train - mu) / (eps + de)
    X_test = (X_test - mu) / (eps + de)
    return X_train, X_test

def svm_classify(X_train, X_test, y_train, y_test):
    svm = SVC(C=1.)
    X_svm_train = X_train.view ()
    X_svm_train = np.reshape (X_svm_train, (X_svm_train.shape[0], -1))    
    X_svm_test = X_test.view ()
    X_svm_test = np.reshape (X_svm_test, (X_svm_test.shape[0], -1))  
    svm.fit(X_svm_train, y_train)
    y_pred = svm.predict(X_svm_test)
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)

    return score, cm
    

if __name__ == "__main__":
    print ("ESC-50 classification with Keras");
    print ("")    

    print ("computing features...")
    X_data, y_data = compute_features (db_location, params)

    X_data.astype('float32')
    y_data.astype('uint8')

    if log_features == True:
        print ("computing log data...")
        X_data = np.log (log_eps + X_data)

    print ("making folds...")
    cv = create_folds(X_data, y_data, nfolds, split)
        
    cnt = 1
    for train_index, test_index in cv:
        print ("----- fold: " + str (cnt) + " -----")
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

            
        if standardize == True:
            print ("standardizing data...")
            X_train, X_test = standardize(X_train, X_test)
            
        score, cm = svm_classify(X_train, X_test, y_train, y_test)
        print ("score " + str(score))
    
        cnt = cnt + 1
#eof
    
