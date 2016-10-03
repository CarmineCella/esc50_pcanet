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

	cqt = np.abs (librosa.core.cqt (y=y, sr=sr,  hop_length=512, n_bins=40, real=False))
	plt.figure()
	plt.imshow(cqt, aspect='auto', origin='lower')
	plt.show(False)


