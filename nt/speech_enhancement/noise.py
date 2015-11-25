import numpy as np

def get_snr(X, N):
    """
    Return SNR of STFT signals in dB.
    You can use any input dimension. It will always create the mean SNR of all
    channels, frames, ...

    :param X: STFT-signal of target image.
    :param N: STFT-signal of noise image.
    :return: SNR of STFT signals in dB.
    """
    energy_X = np.abs(np.sum(X * X.conj()))
    energy_N = np.abs(np.sum(N * N.conj()))
    return 10 * np.log10(energy_X / energy_N)

def set_snr(X, N, snr, current_snr=None):
    """
    Set the SNR of two input images by rescaling the noise signal.

    This decision was made, because a quantization error should not deteriorate
    the target signal.

    TODO: Multi-Source
    The single source SNR is the ratio of each source to the noise channel.
    Multi-source environments are not yet implemented.

    :param X: STFT-signal of target image.
    :param N: STFT-signal of noise image.
    :param snr: Single source SNR of STFT signals in dB.
    :return: Rescaled copies of both images.
    """

    if current_snr is None:
        current_snr = get_snr(X, N)

    return np.copy(X), N * 10 ** (-(snr - current_snr) / 20)
