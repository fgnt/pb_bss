"""
All provided masking functions expect the complex valued stft signal as input.
Each masking function should not take care of further convenience functions
than allowing arbitrary sensor_axis and any number of independent dimensions.

Only, when a multichannel signal is used to pool the power along channels,
the sensor_axis can be provided.

All other convenience should be dealt with from wrapper functions which possibly
take the masking function as a callback. If you want to use lists of arrays,
write an appropriate wrapper function.

If desired, concatenation of *ins can be done in a decorator.

When appropriate, functions assume that the target speaker is channel 0 and
noise is channel 1.

Optional axis parameters are:
 * ``source_axis`` with default ``0``.
 * ``sensor_axis`` with default ``None``. If given, it is used for pooling.
 * ``frequency_axis`` with default ``-2``.
 * ``time_axis`` with default ``-1``.

All other axes are regarded as independent dimensions.
"""

# TODO: Migrate and remove this files:
# TODO:  - tests/speech_enhancement_test/test_merl_masks.py
# TODO:  - nt/speech_enhancement/merl_masks.py
# TODO:  - nt/speech_enhancement/mask_estimation.py

# CB: Eventuell einen Dekorator nutzen für force signal np.ndarray?
# CB: Eventuell einen Dekorator nutzen für force signal.real.dtype == return.dtype?

import numpy as np
from typing import Union, Optional
EPS = 1e-18


def voiced_unvoiced_split_characteristic(
        frequency_bins, split_bin=None, width=None
):
    """ Use this to define different behavior for (un)voiced speech parts.

    Args:
        frequency_bins: Number of stft frequency bins, i.e. 513.
        split_bin: Depending on your stft parameters, this should be somewhere
            between voiced and unvoiced speech segments. For 16 kHz and
            an fft window size of 1024, this happens to be approximately 513//2.
        width: Depends on your stft parameters. For 16 kHz and
            an fft window size of 1024, this happens to be approximately 513//5.

    Returns: Tuple of voiced and unvoiced frequency mask.

    """
    if split_bin is None:
        split_bin = frequency_bins // 2
    if width is None:
        width = frequency_bins // 5

    transition = 0.5 * (1 + np.cos(np.pi / (width - 1) * np.arange(0, width)))
    start = int(split_bin - width / 2)

    voiced = np.ones(frequency_bins)
    voiced[start-1:(start + width - 1)] = transition
    voiced[start - 1 + width:len(voiced)] = 0

    unvoiced = 1 - voiced

    return voiced, unvoiced


def ideal_binary_mask(signal: np.ndarray,
                      source_axis: int=0,
                      sensor_axis: Optional[int]=None,
                      )-> np.ndarray:
    """
    The resulting masks are binary (Value is zero or one).
    Also the sum of all masks is one.

    Example:
        >>> rand = lambda *x: np.random.randn(*x).astype(np.complex)
        >>> M_x, M_n = ideal_binary_mask(rand(2, 3)).shape
        >>> ideal_binary_mask(rand(2, 3)).shape
        (2, 3)
        >>> ideal_binary_mask(rand(2, 3, 5)).shape
        (2, 3, 5)
        >>> ideal_binary_mask(rand(2, 3, 5), sensor_axis=1).shape
        (2, 5)
        >>> np.unique(ideal_binary_mask(rand(2, 3, 5), sensor_axis=1))
        array([ 0.,  1.])
        >>> np.unique(np.sum(ideal_binary_mask(rand(2, 3, 5), sensor_axis=1), \
            axis=0))
        array([ 1.])
    """
    components = signal.shape[source_axis]
    dtype = signal.real.dtype
    mask = signal.real ** 2 + signal.imag ** 2

    if sensor_axis is not None:
        mask = mask.sum(sensor_axis, keepdims=True)

    range_dimensions = signal.ndim * [1]
    range_dimensions[source_axis] = components
    mask = np.expand_dims(np.argmax(mask, axis=source_axis), source_axis)
    mask = mask == np.reshape(np.arange(components), range_dimensions)

    if sensor_axis is not None:
        mask = np.squeeze(mask, sensor_axis)

    return np.asarray(mask, dtype=dtype)


def wiener_like_mask(signal: np.ndarray,
                     source_axis: int=0,
                     sensor_axis: Optional[int]=None,
                     eps: float=EPS,
                     ) -> np.ndarray:
    """

    The resulting masks are soft (Value between zero and one).
    The mask values are source power / all power
    Also the sum of all masks is one.

    CB: Wo kommt der name her?

    Example:
        >>> rand = lambda *x: np.random.randn(*x).astype(np.complex)
        >>> M_x, M_n = wiener_like_mask(rand(2, 3)).shape
        >>> wiener_like_mask(rand(2, 3)).shape
        (2, 3)
        >>> wiener_like_mask(rand(2, 3, 5)).shape
        (2, 3, 5)
        >>> wiener_like_mask(rand(2, 3, 5), sensor_axis=1).shape
        (2, 5)
        >>> np.unique(np.sum(wiener_like_mask(rand(2, 3, 5), sensor_axis=1), \
            axis=0))
        array([ 1.])
    """
    mask = signal.real ** 2 + signal.imag ** 2

    if sensor_axis is not None:
        mask = mask.sum(sensor_axis, keepdims=True)

    mask /= mask.sum(source_axis, keepdims=True) + eps

    if sensor_axis is not None:
        mask = np.squeeze(mask, sensor_axis)

    return mask


def ideal_ratio_mask(signal: np.ndarray,
                     source_axis: int=0,
                     sensor_axis: Optional[int]=None,
                     eps: float=EPS,
                     ) -> np.ndarray:
    """

    The resulting masks are soft (Value between zero and one).
    The mask values are source magnitude / sum magnitude
    Also the sum of all masks is one.

    CB: Why is there a limit for only two sources?

    Example:
        >>> rand = lambda *x: np.random.randn(*x).astype(np.complex)
        >>> M_x, M_n = wiener_like_mask(rand(2, 3)).shape
        >>> ideal_ratio_mask(rand(2, 3)).shape
        (2, 3)
        >>> ideal_ratio_mask(rand(2, 3, 5)).shape
        (2, 3, 5)
        >>> ideal_ratio_mask(rand(2, 3, 5), sensor_axis=1).shape
        (2, 5)
        >>> np.unique(np.sum(ideal_ratio_mask(rand(2, 3, 5), sensor_axis=1), \
            axis=0))
        array([ 1.])
    """
    components = signal.shape[source_axis]
    assert components == 2, 'Only works for one speaker and noise.'

    if sensor_axis is not None:
        mask = signal.real ** 2 + signal.imag ** 2
        mask = mask.sum(sensor_axis, keepdims=True)
        mask = np.sqrt(mask)
    else:
        mask = np.abs(signal)

    mask /= mask.sum(source_axis, keepdims=True) + eps

    if sensor_axis is not None:
        mask = np.squeeze(mask, sensor_axis)

    return mask


def ideal_amplitude_mask(signal: np.ndarray,
                         source_axis: int=0,
                         sensor_axis: Optional[int]=None,
                         eps: float=EPS,
                         ) -> np.ndarray:
    """

    The resulting masks are soft (Value between zero and one).
    The mask values are source magnitude / sum magnitude
    Also the sum of all masks is one.

    CB: This is simmilar to ideal_ratio_mask.
        The different is in how sensor_axis is handeld.
        There is a sum over signal, which is complex ???

    Example:
        >>> rand = lambda *x: np.random.randn(*x).astype(np.complex)
        >>> M_x, M_n = wiener_like_mask(rand(2, 3)).shape
        >>> ideal_ratio_mask(rand(2, 3)).shape
        (2, 3)
        >>> ideal_ratio_mask(rand(2, 3, 5)).shape
        (2, 3, 5)
        >>> ideal_ratio_mask(rand(2, 3, 5), sensor_axis=1).shape
        (2, 5)
        >>> np.unique(np.sum(ideal_ratio_mask(rand(2, 3, 5), sensor_axis=1), \
            axis=0))
        array([ 1.])
    """
    if sensor_axis is not None:
        # TODO: Make sure, this is not an inplace operation.
        signal = np.sum(signal, sensor_axis, keepdims=True)

    amplitude = np.abs(signal)
    amplitude_of_sum = np.abs(np.sum(signal, source_axis, keepdims=True))
    mask = amplitude / (amplitude_of_sum + eps)

    if sensor_axis is not None:
        mask = np.squeeze(mask, sensor_axis)
    return mask


def phase_sensitive_mask(signal: np.ndarray,
                         source_axis: int=0,
                         sensor_axis: Optional[int]=None,
                         eps: float=EPS,
                         ) -> np.ndarray:
    """

    CB: Explanation, why to use this mask
        There is a sum over signal, which is complex ???

    """

    # ToDo: comment, why no sensor_axis
    assert sensor_axis is None

    observed_signal = np.sum(signal, axis=source_axis, keepdims=True)
    theta = np.angle(signal) - np.angle(observed_signal)

    mask = np.abs(signal)
    mask /= np.abs(observed_signal) + eps
    mask *= np.cos(theta)

    return mask


def ideal_complex_mask(signal: np.ndarray,
                       source_axis: int=0,
                       sensor_axis: Optional[int]=None,
                       ) -> np.ndarray:
    """

    CB: Explanation, why to use this mask
        There is a sum over signal, which is complex ???

    """
    assert sensor_axis is None

    return signal / np.sum(signal, axis=source_axis)


def lorenz_mask(signal: np.ndarray,
                sensor_axis: Optional[int]=None,
                frequency_axis: int=-2,
                time_axis: int=-1,
                lorenz_fraction: float=0.98,
                weight: float=0.999,
                ) -> np.adarray:
    """ Calculate softened mask according to Lorenz function criterion.

    To be precise, the lorenz_fraction is not actually a quantile
    although it is in the range [0, 1]. If it was the quantile fraction, it
    would the the fraction of the number of observations.

    Args:
        signal: Complex valued stft signal.
        sensor_axis:
        time_axis:
        frequency_axis:
        lorenz_fraction: Fraction of observations which are rated down
        weight: Governs the influence of the mask

    Returns:

    """
    assert frequency_axis == -2, 'Not yet implemented.'
    assert time_axis == -1, 'Not yet implemented.'
    assert sensor_axis is None, 'Not yet implemented.'

    shape = signal.shape
    dtype = signal.real.dtype

    # Only works, when last two dimensions are frequency and time.
    signal = np.reshape(signal, (-1, np.prod(shape[-2:])))
    mask = np.zeros_like(signal, dtype=dtype)

    def _mask(observations):
        power = (observations * observations.conj())
        sorted_power = np.sort(power, axis=None)[::-1]
        lorenz_function = np.cumsum(sorted_power) / np.sum(sorted_power)
        threshold = np.min(sorted_power[lorenz_function < lorenz_fraction])
        mask = power > threshold
        mask = 0.5 + weight * (mask - 0.5)
        return mask

    for i in range(signal.shape[0]):
        mask[i, :] = _mask(signal[i])

    return mask.reshape(shape)


def biased_binary_mask(signal: np.ndarray,
                       component_axis: int=0,
                       feature_axis: Optional[int]=None,
                       frequency_axis: int=-1,
                       threshold_unvoiced_speech: int=5,
                       threshold_voiced_speech: int=0,
                       threshold_unvoiced_noise: int=-10,
                       threshold_voiced_noise: int=-10,
                       low_cut: int=5,
                       high_cut: int=500,
                       ) -> np.ndarray:
    """

    CB: feature_axis und frequency_axis werden nicht verwendet

    """
    components = signal.shape[component_axis]
    assert components == 2, 'Only works for one speaker and noise.'

    voiced, unvoiced = voiced_unvoiced_split_characteristic(signal.shape[-1])

    # Calculate the thresholds
    threshold_speech = (
        threshold_voiced_speech * voiced +
        threshold_unvoiced_speech * unvoiced
    )
    threshold_noise = (
        threshold_unvoiced_noise * voiced +
        threshold_voiced_noise * unvoiced
    )

    power = signal.real ** 2 + signal.imag ** 2
    speech_power, noise_power = np.split(power, 2, axis=component_axis)

    power_threshold_speech = speech_power / 10 ** (threshold_speech / 10)
    power_threshold_noise = speech_power / 10 ** (threshold_noise / 10)

    speech_mask = (power_threshold_speech > noise_power)
    noise_mask = (power_threshold_noise < noise_power)

    # TODO: Where does 0.005 come from?
    # If this is not needed anymore, we may move the division from line 132f
    # to line 137 and make it a multiplication.
    speech_mask = np.logical_and(speech_mask, (power_threshold_speech > 0.005))
    noise_mask = np.logical_or(noise_mask, (power_threshold_noise < 0.005))

    speech_mask[..., 0:low_cut - 1] = 0
    speech_mask[..., high_cut:len(speech_mask[0])] = 0
    noise_mask[..., 0:low_cut - 1] = 1
    noise_mask[..., high_cut:len(noise_mask[0])] = 1

    return np.concatenate([speech_mask, noise_mask], axis=component_axis)
