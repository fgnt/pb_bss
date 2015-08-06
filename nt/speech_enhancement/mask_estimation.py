import numpy as np


def _voicedUnvoicedSplitCharacteristic(numberOfFreqFrames):
    SplitBin = 200
    TransitionWidth = 99
    FastTransitionWidth = 5
    LowBin = 4
    HighBin = 500

    a = np.arange(0, TransitionWidth)  # Hilfsvariable
    a = np.pi / (TransitionWidth - 1) * a
    Transition = 0.5 * (1 + np.cos(a))

    b = np.arange(0, FastTransitionWidth)
    b = np.pi / (FastTransitionWidth - 1) * b
    FastTransition = (np.cos(b) + 1) / 2

    # TransitionVoicedStart = round(SplitBin-TransitionWidth/2);
    TransitionVoicedStart = int(SplitBin - TransitionWidth / 2)
    voiced = np.ones(numberOfFreqFrames)  # gibt Array von Einsen der Laenge F

    # High Edge
    voiced[TransitionVoicedStart - 1: (
        TransitionVoicedStart + TransitionWidth - 1)] = Transition
    voiced[TransitionVoicedStart - 1 + TransitionWidth: len(voiced)] = 0

    # Low Edge
    voiced[0: LowBin] = 0
    voiced[LowBin - 1: (LowBin + FastTransitionWidth - 1)] = 1 - FastTransition

    # Low Edge
    unvoiced = np.ones(numberOfFreqFrames)
    unvoiced[TransitionVoicedStart - 1: (
        TransitionVoicedStart + TransitionWidth - 1)] = 1 - Transition
    unvoiced[0: (TransitionVoicedStart)] = 0

    # High Edge
    unvoiced[HighBin - 1: (len(unvoiced))] = 0
    unvoiced[HighBin - 1: (HighBin + FastTransitionWidth - 1)] = FastTransition

    return (voiced, unvoiced)


def idealBinaryMask(X, N,
                    thresholdUnvoicedSpeech=5,  # default values
                    thresholdVoicedSpeech=0,
                    thresholdUnvoicedNoise=-10,
                    thresholdVoicedNoise=-10,
                    lowCut=5,
                    highCut=500):
    """Estimate an ideal binary mask given the speech and noise spectrum.

    :param X: speech signal in STFT domain with shape (Frames, Frequency-Bins)
    :param N: noise signal in STFT domain with shape (Frames, Frequency-Bins)
    :param thresholdUnvoicedSpeech:
    :param thresholdVoicedSpeech:
    :param thresholdUnvoicedNoise:
    :param thresholdVoicedNoise:
    :param lowCut: all values with frequency<lowCut are set to 0 in the
        speechMask ans set to 1 in the noiseMask
    :param highCut: all values with frequency>highCut are set to 0 in the
        speechMask ans set to 1 in the noiseMask
    :return: (speechMask, noiseMask): tuple containing the two arrays,
        which are the masks for X and N
    """
    (voiced, unvoiced) = _voicedUnvoicedSplitCharacteristic(X.shape[1])

    # calculate the thresholds
    threshold = thresholdVoicedSpeech * voiced + \
                thresholdUnvoicedSpeech * unvoiced
    threshold_new = thresholdUnvoicedNoise * voiced + \
                    thresholdVoicedNoise * unvoiced

    xPSD = X * X.conjugate()  # |X|^2 = Power-Spectral-Density

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    xPSD_threshold = np.divide(xPSD, c)
    c_new = np.power(10, (threshold_new / 10))
    xPSD_threshold_new = np.divide(xPSD, c_new)

    nPSD = N * N.conjugate()

    speechMask = (xPSD_threshold > nPSD)

    speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005))
    speechMask[:, 0:lowCut - 1] = 0
    speechMask[:, highCut:len(speechMask[0])] = 0

    noiseMask = (xPSD_threshold_new < nPSD)

    noiseMask = np.logical_or(noiseMask, (xPSD_threshold_new < 0.005))
    noiseMask[:, 0: lowCut - 1] = 1
    noiseMask[:, highCut: len(noiseMask[0])] = 1

    return (speechMask, noiseMask)
