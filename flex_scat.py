import numpy as np

def get_fft_bins (freqs, num_fft_bands, sr):
	bin_step = (sr / 2.) / num_fft_bands;
	bins = []
	for l in freqs:
		bin = 0
		curr_freq = bin_step
		while l >= curr_freq:
			curr_freq += bin_step
			bin += 1
		bins.append (bin)

	return bins

def flex_kernel (alpha, Q, num_bands, freq_min, freq_max, num_fft_bands, sr):
	H = np.linspace (0, 1, num_bands)
	c1 = 1. / (1.0 - np.exp (alpha));
	lspace = c1 * (1.0 - np.exp (H * alpha));
	lspace = lspace * num_fft_bands;
	fc = (lspace * (freq_max - freq_min) / lspace[num_bands - 1]) + freq_min
	bw = fc / Q
	bins = get_fft_bins (fc, num_fft_bands, sr)
	bins_bw =  get_fft_bins (bw, num_fft_bands, sr)
	kernel = np.zeros ((num_bands, num_fft_bands))
	for i in range (len (bins)):
		k = np.hanning (bins_bw[i])
		kernel[i, bins[i]-bins_bw[i]:bins[i]-bins_bw[i]+k.shape[0]] = k
		i = i + 1    
	
	return kernel, fc

def flex_gram(y, hop_length, fft_size, flex_matrix):
    frames = np.ceil(y.shape[0] / hop_length).astype('int')
    s = np.zeros((flex_matrix.shape[0], frames))
    for i in range(frames):
        buff = np.zeros(fft_size)
        pin = i * hop_length
        pend = pin + fft_size
        interv = min(pend, y.shape[0]) - pin
        buff[:interv] = y[pin:pin + interv] * np.hanning(interv)
        mag = np.abs(np.fft.rfft(buff, fft_size))
        cq = np.dot(flex_matrix, mag)
        s[:, i] = cq[:flex_matrix.shape[0]]
    return s
    

def flex_scat(y, sr, hop_lengths, alphas, Qs, channels, fmin=32.7, fmax=22050,
              fft_size=1024):
     mmat1, c1 = flex_kernel (alphas[0], Qs[0], num_bands=channels[0],
                                 freq_min=fmin,
                                 freq_max=fmax,
                                 num_fft_bands=fft_size / 2 + 1,
                                 sr=sr)

     mmat2, c2 = flex_kernel (alphas[1], Qs[1], num_bands=channels[1],
                                 freq_min=fmin,
                                 freq_max=fmax,
                                 num_fft_bands=fft_size / 2 + 1,
                                 sr=sr)
     mask = (c1[:, np.newaxis] > c2)
     cq1 = flex_gram(y=y, hop_length=hop_lengths[0], fft_size=fft_size,
                    flex_matrix=mmat1)
     scat = []
     for i in range(cq1.shape[0]):
         b = cq1[i, :]
         cq2 = flex_gram(y=b, hop_length=hop_lengths[1], fft_size=fft_size,
                        flex_matrix=mmat2) / fft_size
         cq2 = cq2[mask[i], :]
         scat.append(cq2)

     ratio = int(np.ceil(cq1.shape[1] / scat[0].shape[1]))
     cq1ds = cq1[:, ::ratio]
     scat = np.vstack(scat)
     scat = np.vstack((scat, cq1ds))
     return scat


if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt
    
    plt.close ('all')
    k, f = flex_kernel (6, 3, 40, 32.7, 15000, 513, 44100)
    plt.figure()
    plt.imshow (k, aspect='auto', origin='lower')
    plt.show(False)
    
    y, sr = librosa.core.load("test1_22050.wav")
    gram = flex_gram (y, 512, 1024, k)
    plt.figure()
    plt.imshow(gram, aspect='auto', origin='lower')
    plt.show(False)
    
    cqt = np.abs (librosa.core.cqt (y=y, sr=sr,  hop_length=512, n_bins=40, 
    real=False))
    plt.figure()
    plt.imshow(cqt, aspect='auto', origin='lower')
    plt.show(False)

    scat = flex_scat(y=y, sr=sr, hop_lengths=(64,4), alphas=(9, 6), Qs=(12,12), channels=(84,12), fmin=32.7, fmax=sr/2)
    plt.figure()
    plt.imshow(np.log (.01 + scat), aspect='auto', origin='lower')
    plt.show(False)
    

