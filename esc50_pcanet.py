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
from cqt_scat import cqt_scat
from flex_scat import flex_scat
from scattering import scattering
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

db_location = '../../datasets/ESC-50-WAV'
log_features = True
log_eps = 0.01
nfolds = 1
split = 0.25
pca_components = 50
pca_time_width = 9
pca_time_stride = 2
pca_freq_width = 9
pca_freq_stride = 4
connections = 'pca_net'

#update this list to test over different features
features_list = ['cqt' ]

params = {'channels': (84,12), 'hops': (1024,4),
          'fmin':32.7, 'fmax':11001,
          'alphas':(6,6),'Qs':(12,12), # only used for flex scattering
          'nclasses': 50, 'max_sample_size':2**17,
          'audio_ext':'*.wav'}

num_cores = 10                        

def get_features (file, features, channels, hops, fmin, fmax, alphas, Qs,
                  max_sample_size):
    y = np.zeros(max_sample_size);   
    yt, sr = librosa.core.load (file)
    
    if len(yt) == 0: 
        print ('*** warning: empty file -> ' + file + '! ***')

    min_length = min(len(y), len(yt))
    y[:min_length] = yt[:min_length]
    
    if features == 'cqt':
        return np.abs(librosa.core.cqt (y=y, sr=sr, hop_length=hops[0], 
                                        n_bins=channels[0], real=False))                           
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
        S, _, _ = scattering(y, wavelet_filters=None,\
                    wavelet_filters_order2=None, M=2)
        return S
    else:
        raise ValueError('Unkonwn features requested')

cachedir = os.path.expanduser('~/esc50_pcanet_joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=1)
cached_get_features = memory.cache(get_features)

def parallel_wrapper_features(args):
    return cached_get_features(*args)
    
def compute_features (root_path, features, params):
    channels = params['channels']
    hops = params['hops']
    fmin = params['fmin']
    fmax = params['fmax']
    alphas = params['alphas']
    Qs = params['Qs']
    max_sample_size = params['max_sample_size']
        
    y_data = []
    classes = 0    
    X_list = []
    
    for root, dir, files in os.walk(root_path):
        waves = fnmatch.filter(files, params['audio_ext'])
        if len(waves) != 0:
            print ("class: " + root.split("/")[-1])
            arg_list = [(os.path.join(root, item), features, 
                         channels, hops, fmin, fmax,
                         alphas, Qs, max_sample_size) for item in waves]
            results = Parallel(n_jobs=num_cores)(delayed(parallel_wrapper_features)(args) 
                        for args in arg_list)

            for i, item in enumerate(waves):
                X_list.append([results[i]]) #[l]
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
    os.sys.stdout.flush()
    scores_features = {}
    for feat in features_list:
        X_data, y_data = compute_features (db_location, feat, params)
    
    
        if log_features == True:
            print ("computing log data...")
            os.sys.stdout.flush()
            X_data = np.log (log_eps + X_data)
    
        if connections == 'pca_net':
            print ('computing PCA connections...')
            os.sys.stdout.flush()
            from sklearn.decomposition import PCA
            from sklearn.feature_extraction.image import extract_patches
            print (X_data.shape)
            patches = extract_patches(X_data, 
                                      (1, pca_freq_width, pca_time_width), 
                                      (1, pca_freq_stride, pca_time_stride))
            patches_reshaped = patches.reshape(np.prod(patches.shape[:3]),
                                               np.prod(patches.shape[3:]))
            #U, s, V = np.linalg.svd(patches_reshaped) 
            #V_cut = V[:pca_components, :]
            #patches_transformed = V_cut.dot(patches_reshaped.T).T                               
            #patches_transformed = patches_reshaped.dot (np.conj (V_cut.T))                               
            pca = PCA(n_components=pca_components)
            patches_transformed = pca.fit_transform(patches_reshaped)
            patches_transformed.shape = X_data.shape[0], -1, patches_transformed.shape[1]
            
            print (patches_transformed.shape)
            patches2 = np.abs (extract_patches(patches_transformed,
                                      (1, pca_freq_width, pca_time_width), 
                                      (1, pca_freq_stride, pca_time_stride)))
            patches_reshaped2 = patches2.reshape(np.prod(patches2.shape[:3]),
                                               np.prod(patches2.shape[3:]))
            #U, s, V = np.linalg.svd(patches_reshaped) 
            #V_cut = V[:pca_components, :]
            #patches_transformed = V_cut.dot(patches_reshaped.T).T                               
            #patches_transformed = patches_reshaped.dot (np.conj (V_cut.T))                               
            pca2 = PCA(n_components=pca_components/2)
            patches_transformed2 = pca2.fit_transform(patches_reshaped2)
            patches_transformed2.shape = patches_transformed.shape[0], -1, patches_transformed2.shape[1]
            X, y = np.abs(patches_transformed2.reshape(patches_transformed.shape[0], -1)), y_data.ravel()
            
        elif connections == 'none':
            X, y = X_data.reshape(X_data.shape[0], -1), y_data.ravel()
        else:
            raise ValueError ('invalid task requested')
    
        print ('classifying...')
        os.sys.stdout.flush()
        from sklearn.cross_validation import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(y_data, n_iter=nfolds, test_size=split,)
                                    #random_state=42)    
        from sklearn.pipeline import make_pipeline
        from sklearn.cross_validation import cross_val_score
        from sklearn.svm import SVC
        #from sklearn.ensemble import RandomForestClassifier
    
        pipeline = make_pipeline(SVC(C=1., kernel='linear'))
        #pipeline = make_pipeline(RandomForestClassifier(n_estimators=300))
        
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=10)
        print ('mean accuracy: {}+-{}'.format(scores.mean(), scores.std() / np.sqrt(nfolds)))
        #update scores for current feature
        scores_features[feat] = ((scores.mean(),scores.std()))
        
        if connections == 'pca_net':
            plt.figure ()
            for i in range (pca_components):
                plt.subplot (np.sqrt(pca_components) + 1, np.sqrt (pca_components) + 1, i + 1)
                plt.imshow (pca.components_[i].reshape(pca_freq_width, pca_time_width), aspect='auto')
            
            plt.show (False)
        
            plt.figure ()    
            for i in range (int(pca_components/2)):
                plt.subplot (np.sqrt(pca_components) + 1, np.sqrt (pca_components) + 1, i + 1)        
                plt.imshow (pca2.components_[i].reshape(pca_freq_width, pca_time_width), aspect='auto')
                plt.show (False)
    
    #displaying results as table
    print('\n')
    print("{:<12}, {:<12}, {:<12}".format('Features', 'Mean', 'Std'))
    prec = 5 
    for feat in scores_features:
        mean_val, std_val = scores_features[feat]
        print("{:<12}, {:<12}, {:<12}".format(feat, round(mean_val,prec), round(std_val,prec)))
    
#eof