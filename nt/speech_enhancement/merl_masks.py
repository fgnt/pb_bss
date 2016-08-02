import numpy as np
from functools import wraps
import chainer.functions as functions
from chainer.functions.array.split_axis import split_axis
from chainer import link
from nt.speech_enhancement import mask_module


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
    return mask_module.ideal_binary_mask(
        np.concatenate([S, N], 1),
        source_axis=1
    )[:, [0], :]


def ideal_ratio_mask(S, N, eps=1e-18):
    """ Calculates the ideal ratio mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    return mask_module.ideal_ratio_mask(
        np.concatenate([S, N], 1),
        source_axis=1,
        eps=eps
    )[:, [0], :]


def wiener_like_mask(S, N, eps=1e-18):
    """ Calculates the "Wiener like" mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    return mask_module.wiener_like_mask(
        np.concatenate([S, N], 1),
        source_axis=1,
        eps=eps
    )[:, [0], :]


def ideal_amplitude_mask(S, N, eps=1e-18):
    """ Calculates the ideal amplitude mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    return mask_module.ideal_amplitude_mask(
        np.concatenate([S, N], 1),
        source_axis=1,
        eps=eps
    )[:, [0], :]


def phase_sensitive_mask(S, N, eps=1e-18):
    """ Calculates the phase sensitive real mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    return mask_module.phase_sensitive_mask(
        np.concatenate([S, N], 1),
        source_axis=1,
        eps=eps
    )[:, [0], :]


def ideal_complex_mask(S, N):
    """ Calculates the ideal complex mask.

    Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech
    separation using deep recurrent neural networks." ICASSP, 2015.

    :param S: Source image with shape (frames, 1, features)
    :param N: Noise image with shape (frames, 1, features)
    :param eps: Regularizing parameter to avoid division by zero.
    :return: Mask with shape (frames, 1, features)
    """
    return mask_module.ideal_complex_mask(
        np.concatenate([S, N], 1),
        source_axis=1,
    )[:, [0], :]


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


class PhaseSensitiveSpectrumApproximation(link.Link):
    def __call__(self, mask, noisy, clean):
        """  D_psa according to [Erdogan2015Masks]
        Args:
            mask: Real valued mask
            noisy: Stacked real and complex values of noisy observation
            clean: Stacked real and complex values of target clean signal

        Returns:

        """
        noisy_real, noisy_imag = split_axis(noisy, 2, -1)
        clean_real, clean_imag = split_axis(clean, 2, -1)

        return (
            functions.mean_squared_error(mask * noisy_real, clean_real) +
            functions.mean_squared_error(mask * noisy_imag, clean_imag)
        )
