# -*- coding: utf-8 -*-
import os 
import fnmatch
import joblib
import librosa
import numpy as np
import os.path
from mel_scat import mel_scat
from cqt_scat import cqt_scat
from flex_scat import flex_scat
from scattering import scattering
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import pdb
import pickle



num_cores = 15                      

def get_features (file, features, channels, hops, fmin, fmax, alphas, Qs,
                  max_sample_size):
    y = np.zeros(max_sample_size);   
    yt, sr = librosa.core.load (file)
    
    if len(yt) == 0: 
        print ('*** warning: empty file -> ' + file + '! ***')

    min_length = min(len(y), len(yt))
    if len(y) < len(yt):
        y = yt[:len(y)]
    else:
        p = len(y) // len(yt)
        r = len(y) % len(yt)
        for i in range(p):
            y[i*len(yt):(i+1)*len(yt)] = yt
        y[-r:] = yt[:r]
        
    #y[:min_length] = yt[:min_length]
    
    if features == 'cqt':
        return np.abs(librosa.core.cqt (y=y, sr=sr, hop_length=hops[0], 
                                        n_bins=channels[0], real=False))      
    if features == 'mfcc':
        return librosa.feature.mfcc(y=y, sr=sr)     
    elif features == 'mel_scat':
        s, m = mel_scat(y=y, sr=sr, hop_lengths=hops, channels=channels, 
                        fmin=fmin, fmax=fmax, fft_size=1024)
        return s
    elif features == 'cqt_scat':
        s, m = cqt_scat(y=y, sr=sr, hops=hops, bins=channels, fmin=fmin)
        return s        
    elif features == 'flex_scat':
        s = flex_scat(y=y, sr=sr, alphas=alphas, Qs=Qs,
                        hop_lengths=hops, channels=channels, 
                        fmin=fmin, fmax=fmax, fft_size=1024)
    elif features == 'plain_scat_1':
        S, _, _ = scattering(y, wavelet_filters=None,\
                    wavelet_filters_order2=None, M=1)
        return S
    elif features == 'plain_scat_2':
        S, U, _ = scattering(y, wavelet_filters=None,\
                    wavelet_filters_order2=None, M=2)
        return S
    elif features == 'plain_scat_2_tree':
        _, _, S_tree = scattering(y, wavelet_filters=None,\
                    wavelet_filters_order2=None, M=2, mod=False, cyclic=True)
        return S_tree
    else:
        raise ValueError('Unkonwn features requested')

cachedir = os.path.expanduser('~/esc50_pcanet_joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=1)
cached_get_features = memory.cache(get_features)

def parallel_wrapper_features(args):
    return cached_get_features(*args)
    
def compute_features_listfiles(files, features, params):
    channels = params['channels']
    hops = params['hops']
    fmin = params['fmin']
    fmax = params['fmax']
    alphas = params['alphas']
    Qs = params['Qs']
    max_sample_size = params['max_sample_size']
    
    arg_list = [(f, features, channels, hops, fmin, fmax,
                 alphas, Qs, max_sample_size) for f in files]
    results = Parallel(n_jobs=num_cores)(delayed(parallel_wrapper_features)(args) 
                                         for args in arg_list)
    # If without parallel, use this line instead of the line before
    #results = [get_features(*args) for args in arg_list]
    print(results[0][2].shape)
    
    return results
    
def compute_features(root_path, features, params, savedir = None):
    y_data = []
    classes = 0    
    X_list = []
    
    for root, dir, files in os.walk(root_path):
        waves = fnmatch.filter(files, params['audio_ext'])
        if len(waves) == 0:
            continue
        print ("class: " + root.split("/")[-1])
        results = compute_features_listfiles([os.path.join(root, f) for f in files], 
                                             features, params)

        if savedir is not None:
            savefile = open(os.path.join(savedir, root.split("/")[-1]+".pkl"), 'wb')
            #print("real : ", all([a.dtype == np.float64 for r in results for a in r.values()]))
            #with open(os.path.join(savedir, root.split("/")[-1]+".npy"), 'w') as savefile:
            pickle.dump(results, savefile)
            savefile.close()
        else :
            X_list.extend(results) 
            y_data.extend([classes for _ in range(len(waves))])
            
        classes = classes + 1
        if classes >= params['nclasses']:
            break
    
    if savedir is None:
        X_data = np.stack(X_list, axis=0)
        return X_data, np.array (y_data)
    else:
        return None


def load_features(directory, n_classes, n_itemsbyclass):
    files = [f for f in os.listdir(directory)]
    classes = 0
    X = []
    y = []
    for filename in sorted(files[:n_classes]):
        print(filename)
        with open(os.path.join(directory, filename), 'rb') as f:
            data = pickle.load(f)
        X.extend(data[:n_itemsbyclass])
        y.extend([classes for _ in range( \
            min(n_itemsbyclass,len(data)))])
        classes += 1
    return X, np.array(y)

if __name__ == "__main__":
    root_path = "/users/data/blier/ESC-50"
    features = "plain_scat_2_tree"
    savedir = "/users/data/blier/features_esc50/scat_10_12_6"
    params = {'channels': (84,12), 'hops': (512,4),
          'fmin':32.7, 'fmax':11001,
          'alphas':(6,6),'Qs':(12,12), # only used for flex scattering
          'nclasses': 50, 'max_sample_size':2**17,
          'audio_ext':'*.ogg'}

    compute_features(root_path, features, params, savedir)
                     
