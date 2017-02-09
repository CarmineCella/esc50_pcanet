import matplotlib
import numpy as np
import scipy as scp
import pywt
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from compute_features import compute_features, load_features

import pdb


def load_transform(directory, n_classes, n_itemsbyclass, trans_obj):
    files = [f for f in os.listdir(directory)]
    classes = 0
    X = []
    y = []
    for filename in sorted(files[:n_classes]):
        print(filename)
        with open(os.path.join(directory, filename), 'rb') as f:
            data = pickle.load(f)
        x_class = data[:n_itemsbyclass]
        
        
        x_class_transformed = trans_obj.transform(x_class)
        X.extend(x_class_transformed)
        y.extend([classes for _ in range( \
            min(n_itemsbyclass,len(data)))])
        classes += 1
    return X, np.array(y)


class Concat_scat_tree():
    def __init__(self, M=1, transf=None, log=True, nOctaves=10, nfo=12, nfo2 = 1, log_eps = 0.0001):
        self.M = M
        self.transf = transf
        self.log = log
        self.nfo = nfo
        self.nfo2 = nfo2
        self.nOctaves = nOctaves
        self.log_eps = log_eps
    
    def fit(self, X, y=None, *args):
        return self
    
    def transform(self, X, *args, **kwargs):
        #pdb.set_trace()
        X_transformed = [[x[i] for i in range(self.M)] for x in X]
        
        if self.M >= 3:
            for i, x in enumerate(X):
                X_i2 = [x[2][j1*self.nfo+q1,j2*self.nfo2+q2,:] \
                         for j1 in range(self.nOctaves) for q1 in range(self.nfo) 
                        for j2 in range(j1+1, self.nOctaves) for q2 in range(self.nfo2) \
                       ]
                X_transformed[i][2] = np.stack(X_i2)
        
        if self.log:
            fun = lambda x: np.log(self.log_eps + np.abs(x))
        else:
            fun = np.abs
            
        if self.transf == "max":
            fun1 = lambda x: fun(x).max(axis=-1)
        elif self.transf == "mean":
            fun1 = lambda x: fun(x).mean(axis=-1)
            
        elif self.transf == "decimate":
            fun1 = lambda x: scp.signal.decimate(fun(x), q=64, axis=-1)
        elif self.transf == "resample":
            fun1 = lambda x: scp.signal.resample(fun(x), num=16, axis=-1)
        else:
            fun1 = fun
        
        X_transformed = np.stack([np.concatenate(
                    [fun1(x[i]).ravel() for i in range(self.M)]
                ) for x in X_transformed ])
        
        #X_transformed = np.log(X_transformed)
        print("shape", X_transformed.shape)
        return X_transformed
    
class Concat_scal():
    def __init__(self, transf=None, log=True):
        self.transf = transf
    
    def fit(self, X, y=None, *args):
        return self
    
    def transform(self, X, *args, **kwargs):
        X_transformed = np.stack(X, axis=0)
        if self.transf == "max":
            X_transformed = np.max(X_transformed, axis=-1)
        elif self.transf == "mean":
            X_transformed = np.mean(X_transformed, axis=-1)
            
        X_transformed = X_transformed.reshape((X_transformed.shape[0],-1))
        #X_transformed = np.log(X_transformed)
        print("shape", X_transformed.shape)
        return X_transformed
    
class JointScat():
    def __init__(self, nOctaves=10, nfo=12, nfo2 = 1, log_eps = 0.0001):
        self.nOctaves = nOctaves
        self.nfo = nfo
        self.nfo2 = nfo2
        self.log_eps = log_eps
    
    def file_transform(self, U):
        U_transformed = []
        wavelet = pywt.Wavelet('db2')
        for j2 in range(1, self.nOctaves):
            U_j2 = U[:j2*self.nfo, j2*self.nfo2:(j2+1)*self.nfo2,:]
            
            
            wdec = pywt.wavedec(U_j2, wavelet, axis=0)
            wdec = np.stack([np.mean(np.log(self.log_eps+np.abs(c)), axis=0) for c in wdec])
            
            wdec = np.mean(wdec, axis=-1)
            U_transformed.append(wdec.ravel())
            
        return np.concatenate(U_transformed)
        
    
    def transform(self, X, *args):
        X_transformed = []
        
        for x in X:
            #pdb.set_trace()
            x_transformed = []
            x_transformed.append(np.mean(np.log(self.log_eps+np.abs(x[0])), axis=-1).ravel())
            x_transformed.append(np.mean(np.log(self.log_eps+np.abs(x[1])), axis=-1).ravel())
            x_transformed.append(self.file_transform(x[2]))
            X_transformed.append(np.concatenate(x_transformed))
        X_transformed = np.stack(X_transformed)
        print("shape", X_transformed.shape)
        return X_transformed
            
            
                
        return X
        
    
    def fit(self, X, y, *args):
        return self
        