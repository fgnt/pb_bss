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


def simple_ideal_soft_mask(*input, feature_dim=-2, source_dim=-1, tuple_output=False):
    """
    :param input: list of array_like or array_like
        These are the arrays like X, N or X_all.
        The arrays X and N will concanated on the last dim, if they have the same shape.
    :param featureDim: The sum diemension
    :param sourceDim: The dimension, where the sum is one.
    :param tuple_output:
    :return: ideal_soft_mask

    Examples:

    >>> F, T, D, K = 51, 31, 6, 2
    >>> X_all = np.random.rand(F, T, D, K)
    >>> X, N = (X_all[:, :, :, 0], X_all[:, :, :, 1])
    >>> simple_ideal_soft_mask(X_all).shape
    (51, 31, 2)
    >>> simple_ideal_soft_mask(X, N).shape
    (51, 31, 2)
    >>> simple_ideal_soft_mask(X_all, N).shape
    (51, 31, 3)
    >>> simple_ideal_soft_mask(X, N, feature_dim=-3).shape
    (51, 6, 2)
    >>> simple_ideal_soft_mask(X_all, feature_dim=-3, source_dim=1).shape
    (51, 6, 2)
    >>> simple_ideal_soft_mask(X_all, N, feature_dim=-2, source_dim=3, tuple_output=True)[0].shape
    (51, 31, 2)
    >>> simple_ideal_soft_mask(X_all, N, feature_dim=-2, source_dim=3, tuple_output=True)[1].shape
    (51, 31)
    """

    assert feature_dim != source_dim

    if len(input) != 1:
        num_dims_max = np.max([i.ndim for i in input])
        num_dims_min = np.min([i.ndim for i in input])
        if num_dims_max != num_dims_min:
            assert num_dims_max == num_dims_min+1
            # Expand dims, if necessary
            input = [np.expand_dims(i, source_dim) if i.ndim == num_dims_min else i for i in input]
        else:
            input = [np.expand_dims(i, num_dims_min+1) for i in input]
        X = np.concatenate(input, axis=source_dim)
    else:
        X = input[0]

    # Permute if nessesary
    # if feature_dim != -2 or source_dim != -1:
    #     r = list(range(np.ndim(X)))
    #     r[feature_dim], r[-2] = r[-2], r[feature_dim]
    #     r[source_dim], r[-1] = r[-1], r[source_dim]
    #     X = np.transpose(X, axes=r)


    power = np.sum(X.conjugate() * X, axis=feature_dim, keepdims=True)
    #power = np.einsum('...dk,...dk->...k', X.conjugate(), X)
    mask = (power / np.sum(power, axis=source_dim, keepdims=True)).real

    if not tuple_output:
        return np.squeeze(mask, axis=feature_dim)
    else:
        sizes = np.cumsum([o.shape[source_dim] for o in input])
        output = np.split(mask, sizes[:-1], axis=source_dim)

        for i in range(len(output)):
            if output[i].shape[source_dim] is 1:
                output[i] = np.squeeze(output[i])
                # ToDo: Determine, why the commented code is not working
                # output[i] = np.squeeze(output[i], axis=(source_dim,feature_dim))
            else:
                output[i] = np.squeeze(output[i], axis=feature_dim)

        return output


def quantile_mask(observations, quantile_fraction=0.98, quantile_weight=0.999):
    """ Calculate softened mask according to lorenz function criterion.

    :param observation: STFT of the the observed signal
    :param quantile_fraction: Fraction of observations which are rated down
    :param quantile_weight: Governs the influence of the mask
    :return: quantile_mask

    """
    power = (observations * observations.conj())
    sorted_power = np.sort(power, axis=None)[::-1]
    lorenz_function = np.cumsum(sorted_power)/np.sum(sorted_power)
    threshold = np.min(sorted_power[lorenz_function<quantile_fraction])
    mask = power > threshold
    mask = 0.5 + quantile_weight * (mask - 0.5)
    return mask


def estimate_IBM(X, N,
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
    (voiced, unvoiced) = _voicedUnvoicedSplitCharacteristic(X.shape[-1])

    # calculate the thresholds
    threshold = thresholdVoicedSpeech * voiced + \
                thresholdUnvoicedSpeech * unvoiced
    threshold_new = thresholdUnvoicedNoise * voiced + \
                    thresholdVoicedNoise * unvoiced

    xPSD = X * X.conjugate()  # |X|^2 = Power-Spectral-Density

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    xPSD_threshold = xPSD / c
    c_new = np.power(10, (threshold_new / 10))
    xPSD_threshold_new = xPSD / c_new

    nPSD = N * N.conjugate()

    speechMask = (xPSD_threshold > nPSD)

    speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005))
    speechMask[..., 0:lowCut - 1] = 0
    speechMask[..., highCut:len(speechMask[0])] = 0

    noiseMask = (xPSD_threshold_new < nPSD)

    noiseMask = np.logical_or(noiseMask, (xPSD_threshold_new < 0.005))
    noiseMask[..., 0: lowCut - 1] = 1
    noiseMask[..., highCut: len(noiseMask[0])] = 1

    return (speechMask, noiseMask)


if __name__ == '__main__':
    import nt.testing as tc

    '''
    ToDo:
        Move this to nt.tests
    Test for simple_ideal_soft_mask
    '''

    F, T, D, K = 51, 31, 6, 2
    X_all = np.random.rand(F, T, D, K) + 1j * np.random.rand(F, T, D, K)
    X, N = (X_all[:, :, :, 0], X_all[:, :, :, 1])

    def test1():
        M1 = simple_ideal_soft_mask(X_all)
        tc.assert_equal(M1.shape, (51, 31, 2))
        tc.assert_almost_equal(np.sum(M1, axis=2), 1)
        return M1

    def test2():
        M2 = simple_ideal_soft_mask(X, N)
        tc.assert_equal(M2.shape, (51, 31, 2))
        tc.assert_almost_equal(np.sum(M2, axis=2), 1)
        return M2

    tc.assert_equal(test1(), test2())

    def test3():
        M3 = simple_ideal_soft_mask(X_all, N)
        tc.assert_equal(M3.shape, (51, 31, 3))
        tc.assert_almost_equal(np.sum(M3, axis=2), 1)
    test3()

    def test4():
        M4 = simple_ideal_soft_mask(X, N, feature_dim=-3)
        tc.assert_equal(M4.shape, (51, 6, 2))
        tc.assert_almost_equal(np.sum(M4, axis=2), 1)
    test4()


