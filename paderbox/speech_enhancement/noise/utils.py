import numpy as np


def get_energy(x, axis=None, keepdims=False):
    return np.sum(np.abs(x * x.conj()), axis=axis, keepdims=keepdims)


def get_variance_for_zero_mean_signal(X, axis=None, keepdims=False):
    X = np.array(X)
    # Bug fix for https://github.com/numpy/numpy/issues/9679
    if np.iscomplexobj(X):
        return np.mean(X.real ** 2 + X.imag ** 2, axis=axis, keepdims=keepdims)
    else:
        return np.mean(X ** 2, axis=axis, keepdims=keepdims)


def get_snr(X, N, *, axis=None, keepdims=False):
    """
    Return SNR of time signals or STFT signals in dB.
    You can use any input dimension. It will always create the mean SNR of all
    channels, frames, ...

    The SNR in time domain or STFT domain is almost equal.

    :param X: Signal of target image.
    :param N: Signal of noise image.
    :return: SNR of time signals or STFT signals in dB.


    >>> get_snr([1, 2, 3], [1, 2, 3])
    0.0
    """
    power_X = get_variance_for_zero_mean_signal(X, axis=axis, keepdims=keepdims)
    power_N = get_variance_for_zero_mean_signal(N, axis=axis, keepdims=keepdims)
    return 10 * np.log10(power_X / power_N)


def set_snr(X, N, snr, current_snr=None, *, axis=None, inplace=True):
    """
    Set the SNR of two input images by rescaling the noise signal in place.

    This decision was made, because a quantization error should not deteriorate
    the target signal.

    TODO: Multi-Source
    The single source SNR is the ratio of each source to the noise channel.
    Multi-source environments are not yet implemented.

    :param X: STFT-signal of target image.
    :param N: STFT-signal of noise image (will be modified in place).
    :param snr: Single source SNR of STFT signals in dB.
    :return: None
    """

    if current_snr is None:
        current_snr = get_snr(X, N, axis=axis, keepdims=True)

    factor = 10 ** (-(snr - current_snr) / 20)

    if inplace:
        N *= factor
    else:
        return X, N * factor
