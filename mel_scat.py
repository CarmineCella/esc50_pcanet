import numpy as np


def hertz_to_mel(freq):
    return 2595.0 * np.log10(1 + (freq / 700.0))


def mel_to_hertz(mel):
    return 700.0 * (10 ** (mel / 2595.0)) - 700.0


def mel_frequencies(num_bands, freq_min, freq_max, num_fft_bands):
    mel_max = hertz_to_mel(freq_max)
    mel_min = hertz_to_mel(freq_min)
    delta_mel = abs(mel_max - mel_min) / (num_bands + 1.0)
    frequencies_mel = mel_min + delta_mel * np.arange(0, num_bands + 2)
    lower_edges_mel = frequencies_mel[:-2]
    upper_edges_mel = frequencies_mel[2:]
    center_frequencies_mel = frequencies_mel[1:-1]

    return center_frequencies_mel, lower_edges_mel, upper_edges_mel


def mel_matrix(num_mel_bands=40, freq_min=32.7, freq_max=16000,
               num_fft_bands=513, sample_rate=44100):
    center_frequencies_mel, lower_edges_mel, upper_edges_mel =  \
        mel_frequencies(
            num_mel_bands,
            freq_min,
            freq_max,
            num_fft_bands
        )

    center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
    lower_edges_hz = mel_to_hertz(lower_edges_mel)
    upper_edges_hz = mel_to_hertz(upper_edges_mel)
    freqs = np.linspace(0.0, sample_rate / 2.0, int(num_fft_bands))
    melmat = np.zeros((num_mel_bands, int(num_fft_bands)))

    for imelband, (center, lower, upper) in enumerate(zip(
            center_frequencies_hz, lower_edges_hz, upper_edges_hz)):

        left_slope = (freqs >= lower) == (freqs <= center)
        melmat[imelband, left_slope] = (
            (freqs[left_slope] - lower) / (center - lower)
        )

        right_slope = (freqs >= center) == (freqs <= upper)
        melmat[imelband, right_slope] = (
            (upper - freqs[right_slope]) / (upper - center)
        )
    
    return melmat, (center_frequencies_mel, freqs)


def mel_gram(y, hop_length, fft_size, mel_matrix):
    frames = np.ceil(y.shape[0] / hop_length).astype('int')
    s = np.zeros((mel_matrix.shape[0], frames))
    for i in range(frames):
        buff = np.zeros(fft_size)
        pin = i * hop_length
        pend = pin + fft_size
        interv = min(pend, y.shape[0]) - pin
        buff[:interv] = y[pin:pin + interv] * np.hanning(interv)
        mag = np.abs(np.fft.rfft(buff, fft_size))
        cq = np.dot(mel_matrix, mag)
        s[:, i] = cq[:mel_matrix.shape[0]]
    return s


def mel_scat(y, sr, hop_lengths, channels, fmin=32.7, fmax=22050,
             fft_size=1024):
    mmat1, (c1, f) = mel_matrix(num_mel_bands=channels[0], freq_min=fmin,
                                freq_max=fmax,
                                num_fft_bands=fft_size / 2 + 1,
                                sample_rate=sr)

    mmat2, (c2, f) = mel_matrix(num_mel_bands=channels[1], freq_min=fmin,
                                freq_max=fmax,
                                num_fft_bands=fft_size / 2 + 1,
                                sample_rate=sr)

    mask = (c1[:, np.newaxis] > c2)
    cq1 = mel_gram(y=y, hop_length=hop_lengths[0], fft_size=fft_size,
                   mel_matrix=mmat1)
    scat = []
    for i in range(cq1.shape[0]):
        b = cq1[i, :]
        cq2 = mel_gram(y=b, hop_length=hop_lengths[1], fft_size=fft_size,
                       mel_matrix=mmat2) / fft_size
        cq2 = cq2[mask[i], :]
        scat.append(cq2)

    ratio = int(np.ceil(cq1.shape[1] / scat[0].shape[1]))
    cq1ds = cq1[:, ::ratio]
    scat = np.vstack(scat)
    scat = np.vstack((scat, cq1ds))
    return scat, mask

if __name__ == '__main__':
    import librosa
    y, sr = librosa.core.load("../../datasets/ESC-50-toy/101 - Dog/4-207124-A.wav")
    s, m = mel_scat(y=y, sr=sr, hop_lengths=(16, 8), channels=(84, 12),
                 fmin=32.7, fmax=16000, fft_size=1024)
