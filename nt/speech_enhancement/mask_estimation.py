import numpy as np
import nt.speech_enhancement.mask_module
import warnings


def simple_ideal_soft_mask(*ins, feature_dim=-2, source_dim=-1,
                           tuple_output=False):
    warnings.warn(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.wiener_like_mask" '
        'for a replacement.',
        DeprecationWarning
    )
    raise NotImplementedError(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.wiener_like_mask" '
        'for a replacement.')


def quantile_mask(observations, lorenz_fraction=0.98, weight=0.999):
    warnings.warn(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.lorenz_mask" '
        'for a replacement.',
        DeprecationWarning
    )
    raise NotImplementedError(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.lorenz_mask" '
        'for a replacement.'
    )


def estimate_IBM(
        X, N,
        threshold_unvoiced_speech=5,
        threshold_voiced_speech=0,
        threshold_unvoiced_noise=-10,
        threshold_voiced_noise=-10,
        low_cut=5,
        high_cut=500
):
    warnings.warn(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.biased_binary_mask" '
        'for a replacement.',
        DeprecationWarning
    )
    raise NotImplementedError(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.biased_binary_mask" '
        'for a replacement.'
    )


def estimate_simple_IBM(
        X, N,
        threshold_speech=5,
        threshold_noise=-10,
        low_cut=5,
        high_cut=500
):
    warnings.warn(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.biased_binary_mask" '
        'for a replacement.',
        DeprecationWarning
    )
    raise NotImplementedError(
        'This function is gone. '
        'See "nt.speech_enhancement.mask_module.biased_binary_mask" '
        'for a replacement.'
    )
