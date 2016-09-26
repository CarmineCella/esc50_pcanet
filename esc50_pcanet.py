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
    
    SAMPLELEN = 110250 # in samples is 5 sec @ 22050
    
    params = {'plot':False, 'features':'cqt',  
              'standardize_data':True, 
              'ncoeff': (80,12), 'hop': (1024,8), \
              'nclasses': 50, 'nsamples':2000, \
              'nfolds': 1, 'split':.25}
 
    def get_features (file, hop, bins, params):
        y = np.zeros(SAMPLELEN);   
        yt, sr = librosa.core.load (file, mono=True)
        
        if len(yt) == 0: 
            print ('*** warning: empty file -> ' + file + '! ***')
            return 0
    
        min_length = min(len(y), len(yt))
        y[:min_length] = yt[:min_length]
        
        if params['features'] == 'cqt:
            return np.abs(librosa.core.cqt (y=yt, sr=sr, hop_length=hop[0], 
                                            n_bins=bins[0], real=False)) 
        elif params['features'] == 'mel_scat':
            
            pass
        elif params['features'] == 'cqt_scat:
            #cqtscat(y,sr,hop,bins)            
            pass
        else raise ValueError('Unkonwn features requested')
    
    cachedir = os.path.expanduser('~/esc50_pcanet_joblib')
    memory = joblib.Memory(cachedir=cachedir, verbose=1)
    cached_get_features = memory.cache(get_features)
    
    def compute_features (root_path, params):
        hop = params['hop']
        bins = params['ncoeff']
            
        y_data = np.zeros ((params['nsamples']))
        
        classes = 0
        samples = 0
        
        X_list = []
        for root, dir, files in os.walk(root_path):
            waves = fnmatch.filter(files, "*.wav")
            if len(waves) != 0:
                print ("class: " + root.split("/")[-1])
                X_list.append([
                    cached_get_features(os.path.join(root, item), hop, bins, 
                                        params)
                    for item in waves]) 
                for item in waves:
                    y_data[samples] = classes
                    samples = samples + 1
                classes = classes + 1
        
        X_flat_list = [X_list[class_id][file_id]
                    for class_id in range(len(X_list))
                    for file_id in range(len(X_list[class_id]))]
        
        for i in range (len(X_flat_list)):
            l = np.zeros((bins[0], SAMPLELEN//hop[0]))
            l[:min(X_flat_list[i].shape[0], l.shape[0]),
              :min(X_flat_list[i].shape[1], l.shape[1])] = \
            X_flat_list[i][:min(X_flat_list[i].shape[0], 
              l.shape[0]),:min(X_flat_list[i].shape[1], l.shape[1])]
            X_flat_list[i] =l
            

        X_data = np.stack(X_flat_list, axis=2)
        X_data = np.transpose(X_data, (2,0,1))
        
        print ("classes = " + str (classes))
        print ("samples = " + str (samples))
    
        return X_data, y_data
    
    def create_folds (X_data, y_data, params):    
        cv = StratifiedShuffleSplit(y_data, n_iter=params['nfolds'], 
                                    test_size=params['split'])
        return cv
            
    def standardize (X_train, X_test):
        mu = np.mean (X_train, axis=0)
        de = np.std (X_train, axis=0)
        
        eps = np.finfo('float32').eps
        X_train = (X_train - mu) / (eps + de)
        X_test = (X_test - mu) / (eps + de)
        return X_train, X_test
    
    def svm_classify(X_train, X_test, y_train, y_test, params):
        svm = SVC(C=1.)
        #rsvm = ShapeWrapper(svm)
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
        X_data, y_data = compute_features ("../../datasets/ESC-50-master", 
                                           params)
    
        X_data.astype('float32')
        y_data.astype('uint8')
    
         
        print ("making folds...")
        cv = create_folds(X_data, y_data, params)
            
        cnt = 1
        for train_index, test_index in cv:
            print ("----- fold: " + str (cnt) + " -----")
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
    
            if params["standardize_data"] == True:
                print ("standardizing data...")
                X_train, X_test = standardize(X_train, X_test)
                
            score, cm = svm_classify(X_train, X_test, y_train, y_test, params)
            print ("score " + str(score))
        
            cnt = cnt + 1
    #eof
        
