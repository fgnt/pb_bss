import numpy as np
from functools import wraps


def constraints(f):
    @wraps(f)
    def wrapper(S, N, **kwargs):
        assert S.dtype == N.dtype
        assert S.dtype in (np.complex64, np.complex128)
        assert len(S.shape) == 3
        assert S.shape[1] == 1
        assert len(N.shape) == 3
        assert N.shape[1] == 1
        return f(S, N, **kwargs)
    return wrapper


def ideal_binary_mask(S, N):
    """ Calculates the ideal binary mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :return: Mask with shape (frames, 1, features)
    """
    out_type = S.real.dtype
    return (np.abs(S) > np.abs(N)).astype(out_type)


def ideal_ratio_mask(S, N, eps=1e-18):
    """ Calculates the ideal ratio mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    out_type = S.real.dtype
    return (np.abs(S) / (np.abs(S) + np.abs(N) + eps)).astype(out_type)


def wiener_like_mask(S, N, eps=1e-18):
    """ Calculates the "Wiener like" mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    out_type = S.real.dtype
    return (np.abs(S)**2 / (np.abs(S)**2 + np.abs(N)**2 + eps)).astype(out_type)


def ideal_amplitude_mask(S, N, eps=1e-18):
    """ Calculates the ideal amplitude mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    out_type = S.real.dtype
    return (np.abs(S) / (np.abs(S + N) + eps)).astype(out_type)


def phase_sensitive_mask(S, N, eps=1e-18):
    """ Calculates the phase sensitive real mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    out_type = S.real.dtype
    Y = S + N
    theta = np.angle(S) - np.angle(Y)
    return (np.abs(S) / (np.abs(Y) + eps) * np.cos(theta)).astype(out_type)


def ideal_complex_mask(S, N):
    """ Calculates the ideal complex mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    return S / (S + N)


def ideal_complex_mask_williamson(S, N, eps=1e-18):
    """ Calculates the ideal complex mask with real numbers.

    Williamson, Donald, et al. "Complex ratio masking for joint enhancement of
    magnitude and phase" ICASSP, 2016.

    This is simply a different implementation of ``ideal_complex_mask()``.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    Y = S + N
    denominator = Y.real**2 + Y.imag**2 + eps
    return (
        (Y.real * S.real + Y.imag * S.imag) / denominator +
        1j * (Y.real * S.imag - Y.imag * S.real) / denominator
    )


def apply_mask_clipping(mask, threshold=1):
    """ Clip a real or complex mask to a positive threshold.

    If the mask is complex, the amplitude of the mask will be clipped.

    :param mask: Real or complex valued mask.
    :param threshold:
    :return:
    """
    if mask.dtype in (np.complex64, np.complex128):
        return np.clip(np.abs(mask), 0, threshold) * np.exp(1j * np.angle(mask))
    elif mask.dtype in (np.float32, np.float64):
        return np.clip(mask, -threshold, threshold)
    else:
        raise TypeError('Desired mask.dtype not supported.')
