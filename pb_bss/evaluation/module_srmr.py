import numpy as np
import scipy as sp
import math
from pb_bss.transform.gammatone import gammatone_filterbank, calculate_cfs
from paderbox.array.segment import segment_axis


def srmr(signal, sample_rate: int = 16000, n_cochlear_filters: int = 23, low_freq: int = 125):
    """
    Wrapper around the SRMR Metric to allow an independent axis.
    Note: The results of this implementation are slightly different from the Matlab implementation,
    because the ALS-adjustment of the signal is not implemented.
    However, the results of this implementation typically don't deviate more than 1e-3 from
    the Matlab implementation, so that a high
    correlation between the behavior of both implementations still is present.

        >>> import paderbox as pb
        >>> a = pb.testing.testfile_fetcher.get_file_path('speech_bab_0dB.wav')
        >>> a = pb.io.load_audio(a)
        >>> srmr(a, 16000)  # doctest: +ELLIPSIS
        np.float64(1.8561615800...)
        >>> srmr([a, a], 16000)
        array([1.85616158, 1.85616158])
    """

    signal = np.asarray(signal)
    if signal.ndim >= 2:
        for i in range(signal.ndim-1):
            assert signal.shape[i] < 30, (i, signal.shape)  # NOQA
        srmrs = []
        for i in np.ndindex(*signal.shape[:-1]):
            # TODO: Add option to also return the SRMR per gammatone filterbank (typically unused in evaluations)
            srmrs.append(SRMR(signal[i], sample_rate=sample_rate, n=n_cochlear_filters, low_freq=low_freq))
        return np.array(srmrs).reshape(signal.shape[:-1])
    elif signal.ndim == 1:
        # TODO: Add option to also return the SRMR per gammatone filterbank (typically unused in evaluations)
        return SRMR(signal, sample_rate=sample_rate, n=n_cochlear_filters, low_freq=low_freq)
    else:
        raise NotImplementedError(signal.ndim)


def SRMR(signal: np.ndarray, sample_rate: int = 16000, n: int = 23, low_freq: int = 125) -> float:
    """Python implementation of the SRMR metric.
    Matlab reference implementation: https://github.com/MuSAELab/SRMRToolbox
    Because results of other openly available SRMR python packages significantly deviate from
    the original evaluation tool, this function reimplements the Matlab functionality.
    An ASL-adjustment is not implemented, so that the calculated values still slightly differ from the Matlab implementation.
    For an exact reproduction of the matlab results, the usage of an ASL-adjustion is required. However the deviation of 
    this implmentation from the Matlab version typically is not larger than 1e-3.

    :param signal: signal on which the SRMR is calculated
    :param sample_rate: sample rate of signal
    :param n: number of gammatone filters used
    :param low_freq: lowest center frequency of the gammatone filterbank, highest frequency is half the sample rate
    :return: SRMR metric for given signal
    """
    #Preprocessing of the signal (Voice activity detection)
    signal = _preprocessing_vad(signal, sample_rate)
    signal = signal - np.mean(signal)
    signal /= np.std(signal, keepdims=True)

    #Gammatone filterbank (with n Filters)
    signal = gammatone_filterbank(signal, sample_rate=sample_rate, n=n, low_freq=low_freq)

    #Calculate temporal envelope of the signal
    for i in range(len(signal)):
        signal[i] = np.abs(sp.signal.hilbert(signal[i]))


    #Frequencies of the modulation filters
    modulation_filter_frequencies = [4.0, 6.5, 10.7, 17.6, 28.9, 47.5, 78.1, 128.0]


    #Using 8 modulation filters on the output of the gammatone filters
    E = []
    for j in range(len(signal)):
        E.append([])
        for k in range(8):
            W0 = math.tan(2 * math.pi * modulation_filter_frequencies[k] / (2 * sample_rate))
            B0 = W0 / 2

            b = np.ndarray((3,), dtype=float, buffer=np.array([B0 / (1 + B0 + W0 ** 2), 0, -B0 / (1 + B0 + W0 ** 2)]))
            a = np.ndarray((3,), dtype=float, buffer=np.array([1, (2 * W0 ** 2 - 2) / (1 + B0 + W0 ** 2), (1 - B0 + W0 ** 2) / (1 + B0 + W0 ** 2)]))

            E[j].append(sp.signal.lfilter(b, a, signal[j], axis = 0))


    #Calculation of the energy of the single bands
    energy = []
    for j in range(len(E)):
        energy.append([])
        for k in range(len(E[j])):
            energy[j].append([])

            #Segmentation of the signal
            temp = segment_axis(E[j][k], int(sample_rate/1000)*256, int(sample_rate/1000)*64)

            #Multiplication of a hamming window with each segment and summation of the result
            hamm_window = sp.signal.windows.hamming(int(sample_rate/1000)*256, sym=True)
            for window in temp:
                energy[j][k].append(np.sum(np.square(hamm_window*window)))


    #Calculation of the center frequencys (ERBS) and the corresponding ERBs
    cfs = calculate_cfs(low_freq, sample_rate/2, n)

    ERBs = []

    for i in range(len(cfs)):
        ERBs.append(cfs[i]/9.26449+24.7)


    #Calculation of the means of the single bands
    means = np.ndarray((len(energy), len(energy[0])))

    for j in range(len(energy)):
        for k in range(len(energy[j])):
            means[j][k] = np.mean(energy[j][k])


    #Calculation of the Bandwidth
    total_energy = np.sum(np.sum(means))
    AC_energy = np.sum(means, axis=1)
    AC_perc = AC_energy*100/total_energy

    sum = 0.0
    BW = 0.0

    for i in range(len(AC_perc)):
        sum += AC_perc[i]
        if(sum > 90):
            BW = ERBs[i]
            break


    #Calculate cutoffs
    cutoffs = []

    for cfs in modulation_filter_frequencies:
        w0 = 2*math.pi*cfs/sample_rate
        B0 = math.tan(w0/2)/2
        cutoffs.append(cfs - (B0*sample_rate / (2*math.pi)))


    #Calculation of the mean of the different bands wth regards to the cuffoff band
    numerator = np.sum(np.sum(means, axis=0)[:4])
    denominator = np.sum(means, axis=0)[4]

    for i in range(5, 8):
        denominator += np.sum(means, axis=0)[i]
        if cutoffs[i - 1] < BW < cutoffs[i]:
            break

    return numerator/denominator



def _preprocessing_vad(signal, sample_rate=16000):
    """Preprocesses the signal to remove silence parts 
    :param signal: input signal
    :param sample_rate: sample rate of the signal
    :return: Preprocessed signal
    """
    max_val = abs(signal).max()

    threshold = (max_val**2)/(10**5)

    L = np.where(abs(signal) > threshold)[0]

    window_width = 0.05*sample_rate

    #Checking, if there are some parts that has to be removed
    remove = []
    for i in range(len(L)-1):
        if L[i+1]-L[i] > window_width:
            remove.append((L[i], L[i+1]))

    #Removing the previously chosen parts of the signal
    if len(remove) > 0:
        ret = signal[:remove[0][0]+1]
        for i in range(0, len(remove)-1):
            ret = np.append(ret, signal[remove[i][1]:remove[i+1][0]+1])
        ret = np.append(ret, signal[remove[len(remove)-1][1]:])
    else:
        ret = signal
    return ret
