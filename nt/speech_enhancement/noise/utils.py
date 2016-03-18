import numpy


def get_power(x, axis=None):
    return numpy.sum(numpy.abs(x * x.conj()), axis=axis)


def get_snr(X, N):
    """
    Return SNR of time signals or STFT signals in dB.
    You can use any input dimension. It will always create the mean SNR of all
    channels, frames, ...

    The SNR in time domain or STFT domain is almost equal.

    :param X: Signal of target image.
    :param N: Signal of noise image.
    :return: SNR of time signals or STFT signals in dB.
    """
    energy_X = numpy.sum(numpy.abs(X * X.conj()))
    energy_N = numpy.sum(numpy.abs(N * N.conj()))
    return 10 * numpy.log10(energy_X / energy_N)


def set_snr(X, N, snr, current_snr=None):
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
        current_snr = get_snr(X, N)

    N *= 10 ** (-(snr - current_snr) / 20)
