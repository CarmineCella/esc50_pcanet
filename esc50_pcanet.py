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
from mel_scat import mel_scat

db_location = '../../datasets/ESC-50-master'
sample_len = 110250 # in samples is 5 sec @ 22050
log_features = True
log_eps = 0.01
nfolds = 50
split = 0.25
task = 'plain_scat'

params = {'features':'mel_scat',  
          'channels': (84,12), 'hops': (128,8),
          'fmin':32.7, 'fmax':18000,
          'nclasses': 5, 'nsamples':2000}
 
def get_features (file, features, channels, hops, fmin, fmax):
    y = np.zeros(sample_len);   
    yt, sr = librosa.core.load (file)
    
    if len(yt) == 0: 
        print ('*** warning: empty file -> ' + file + '! ***')
        return 0

    min_length = min(len(y), len(yt))
    y[:min_length] = yt[:min_length]
    
    if features == 'cqt':
        return np.abs(librosa.core.cqt (y=y, sr=sr, hop_length=hops[0], 
                                        n_bins=channels[0], real=False)) 
    elif features == 'mel_scat':
        return mel_scat(y=y, sr=sr, hop_lengths=hops, channels=channels, 
                        fmin=fmin, fmax=fmax, fft_size=1024)
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
        
    y_data = []
    classes = 0    
    X_list = []

    for root, dir, files in os.walk(root_path):
        waves = fnmatch.filter(files, "*.wav")
        if len(waves) != 0:
            print ("class: " + root.split("/")[-1])
            for item in waves:
                l = cached_get_features(os.path.join(root, item), features, 
                                        channels, hops, fmin, fmax)
                X_list.append([l])
                y_data.append (classes)

            classes = classes + 1
            if classes >= params['nclasses']:
            	break
    
    X_flat_list = [X_list[class_id][file_id]
                for class_id in range(len(X_list))
                for file_id in range(len(X_list[class_id]))]

    X_data = np.stack(X_flat_list, axis=0)
    #X_data = np.transpose(X_data, (2,0,1))
    
    print ("classes = " + str (classes))

    return X_data, np.array (y_data)

if __name__ == "__main__":
    print ("ESC-50 classification with PCA nets");
    print ("")    

    print ("computing features...")
    X_data, y_data = compute_features (db_location, params)

    if log_features == True:
        print ("computing log data...")
        X_data = np.log (log_eps + X_data)

    if task == 'pca_scat':
        print ('computing PCA connections...')
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.image import extract_patches
        patches = extract_patches(X_data, (1, X_data.shape[1], 1), (1, X_data.shape[1], 1))
        patches_reshaped = patches.reshape(np.prod(patches.shape[:3]),
                                           np.prod(patches.shape[3:]))
        pca = PCA(n_components=50)
        patches_transformed = pca.fit_transform(patches_reshaped)
        patches_transformed.shape = X_data.shape[0], -1, patches_transformed.shape[1]    
        X, y = patches_transformed.reshape(X_data.shape[0], -1), y_data.ravel()
    elif task == 'plain_scat':
        X, y = X_data.reshape(X_data.shape[0], -1), y_data.ravel()
    elif task == 'pca_1st_order':
        pass
    elif task == 'pca_per_channel':
        pass
    else:
        raise ValueError ('invalid task requested')
        
    print ('classifying...')
    from sklearn.cross_validation import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(y_data, n_iter=nfolds, test_size=split,)
                                #random_state=42)    
    from sklearn.pipeline import make_pipeline
    from sklearn.cross_validation import cross_val_score
    from sklearn.svm import SVC

    pipeline = make_pipeline(SVC(C=1., kernel='linear'))
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=10)
    print ('mean accuracy: {}+-{}'.format(scores.mean(), scores.std() / np.sqrt(nfolds)))
    
#eof
