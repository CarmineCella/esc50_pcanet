import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys

def cqt_scat(y, sr, hops, bins, fmin=32.7):
    C = np.abs (librosa.core.cqt(y=y, sr=sr, hop_length=hops[0], n_bins=bins[0],
                                 bins_per_octave=12, fmin=fmin, real=False))
    freqs1 = librosa.core.cqt_frequencies(n_bins=bins[0], bins_per_octave=12,
                                          fmin=fmin)
    freqs2 = librosa.core.cqt_frequencies(n_bins=bins[1], bins_per_octave=12,
                                          fmin=fmin)
    mask = freqs1[:,np.newaxis] > freqs2
    
    frames2 = int(np.ceil (C.shape[1] / hops[1])) + 1
    print (frames2)
    scat = np.zeros ((mask.sum() + C.shape[0], frames2))
    layer2_dict = dict()
    scat_dict = {1: layer2_dict}
    ctn = 0
    for i, (c, m) in enumerate (zip(C[1:], mask[1:])):
        c1 = np.abs (librosa.core.cqt(y=c, sr=sr, hop_length=hops[1], 
                                      n_bins=bins[1], bins_per_octave=12,
                                      fmin=fmin, real=False))
        c1 = c1[m,:]
        print (c1.shape)
        sys.stdout.flush()
        scat[ctn:ctn+c1.shape[0]] = c1
        ctn += c1.shape[0]
        layer2_dict[i] = scat[ctn:len(c1) + ctn]

    ratio = int(np.ceil(C.shape[1] / scat.shape[1]))
    C1 = C[:,::ratio]
    scat[ctn:ctn+C1.shape[0],:C1.shape[1]]=C1
    scat_dict[0] = scat[ctn:ctn + len(C1)]
    return scat, scat_dict

if __name__ == '__main__':
    y, sr = librosa.load('1-26806-A.wav')    
    s, d = cqt_scat(y, sr, (16, 8), (48, 4))
    print (s.shape)
    